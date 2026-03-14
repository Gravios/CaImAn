#!/usr/bin/env python

"""
Functions for implementing parallel scalable segmentation of two photon imaging data
"""

from copy import copy, deepcopy
import logging
import multiprocessing
import numpy as np
import os
import scipy
from sklearn.decomposition import NMF
import time

from caiman.cluster import (extract_patch_coordinates,
                             _collect_log_params,
                             _worker_logging_init,
                             flush_worker_log)
from caiman.mmapping import load_memmap
from caiman.source_extraction.cnmf.initialization import (
    precompute_corr_pnr_filtered_fov)
from caiman.shared_memory_utils import ShmHandle, attach_shared_frames
from caiman.cpu_topology import apply_affinity

def _worker_cuda_reset_if_available() -> None:
    """Pool initializer: reset CUDA context after fork.

    NOTE: patch pools now use the 'spawn' multiprocessing context so
    workers start with a clean process and no inherited CUDA state.
    This function is retained for any legacy fork-based callers.
    """
    try:
        import ctypes, ctypes.util
        _libcuda = ctypes.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        _libcuda.cuInit(0)
        _libcuda.cuDevicePrimaryCtxReset(0)
    except Exception:
        pass
    try:
        import cupy as cp
        cp.cuda.Device(0).use()
    except Exception:
        pass

def cnmf_patches(args_in):
    """Function that is run for each patches

         Will be called

        Args:
            file_name: string
                full path to an npy file (2D, pixels x time) containing the movie

            shape: tuple of three elements
                dimensions of the original movie across y, x, and time

            params:
                CNMFParams object containing all the parameters for the various algorithms

            rf: int
                half-size of the square patch in pixel

            stride: int
                amount of overlap between patches

            gnb: int
                number of global background components

            backend: string
                'ipyparallel' or 'single_thread'

            n_processes: int
                number of cores to be used (should be less than the number of cores started with ipyparallel)

            memory_fact: double
                unitless number accounting how much memory should be used.
                It represents the fraction of patch processed in a single thread.
                 You will need to try different values to see which one would work

            low_rank_background: bool
                if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)

        Returns:
            A_tot: matrix containing all the components from all the patches

            C_tot: matrix containing the calcium traces corresponding to A_tot

            sn_tot: per pixel noise estimate

            optional_outputs: set of outputs related to the result of CNMF ALGORITHM ON EACH patch

        Raises:
            Empty Exception
        """

    #FIXME Fix in-function imports
    from caiman.source_extraction.cnmf import CNMF
    logger = logging.getLogger("caiman")
    file_name, idx_, shapes, params = args_in

    # ── Exception barrier ─────────────────────────────────────────────────
    # multiprocessing.Pool serialises worker exceptions with pickle before
    # sending them back to the parent.  Some scipy/LAPACK exception types
    # (e.g. _flapack.error) are not importable in the parent process and
    # therefore cannot be pickled, causing a secondary MaybeEncodingError
    # that hides the real failure.  Wrapping here ensures any exception is
    # re-raised as a plain RuntimeError (always picklable) with the full
    # original traceback embedded as a string.
    try:
        result = _cnmf_patches_inner(file_name, idx_, shapes, params, CNMF, logger)
    except Exception as _e:
        import traceback as _tb
        flush_worker_log()
        raise RuntimeError(
            f"cnmf_patches failed on patch starting at idx={idx_[0]}:\n"
            + _tb.format_exc()
        ) from None
    flush_worker_log()
    return result


def _cnmf_patches_inner(file_name, idx_, shapes, params, CNMF, logger):

    # ── Support both legacy path strings and shared-memory handles ────────
    if isinstance(file_name, ShmHandle):
        # Name-log uses a synthetic filename for readability
        name_log = f"SHM_{file_name.name[:8]}_LOG_ {idx_[0]}_{idx_[-1]}"
    else:
        name_log = os.path.basename(
            file_name[:-5]) + '_LOG_ ' + str(idx_[0]) + '_' + str(idx_[-1])

    logger.debug(name_log + ' START')
    logger.debug(name_log + ' Read file')

    if isinstance(file_name, ShmHandle):
        # ── Zero-copy path: attach to shared memory ────────────────────────
        # The movie was loaded once in the parent process into a POSIX
        # shared-memory segment.  All workers map the same physical pages;
        # no data is copied through the OS IPC layer.
        #
        # Memory hierarchy notes:
        #   • All workers on the same socket share L3.  Because they all
        #     read from the same physical pages, the OS pulls each page into
        #     L3 exactly once regardless of how many workers access it.
        #   • Spatially adjacent patches are assigned to the same L3 group
        #     by ``run_CNMF_patches`` (via ``cache_aware_chunk_order``), so
        #     the patch data is likely warm in L3 when the second worker
        #     in the same group starts.
        handle = file_name
        # The mmap-style layout is (pixels, time) in Fortran order.
        # The SHM buffer was created with the full (T, d1, d2) C-order array
        # by SharedMovieBuffer.  We need to reconstruct the Yr view.
        full_movie = attach_shared_frames(handle)   # shape (T, d1, d2), C-order
        T_total    = full_movie.shape[0]
        dims       = full_movie.shape[1:]
        timesteps  = T_total

        # Reconstruct a Yr-compatible view: shape (d1*d2, T)
        # This is a reshape + transpose, still zero-copy as long as strides allow.
        Yr = full_movie.reshape(T_total, -1).T   # (pixels, T)
        images = full_movie  # already (T, d1, d2)
    else:
        # ── Legacy path: memory-mapped file ───────────────────────────────
        # np.memmap with mode='r' already leverages OS page sharing: multiple
        # processes opening the same .mmap file in read-only mode will share
        # the underlying physical pages via the OS page cache.  No explicit
        # shared-memory setup is needed here.
        Yr, dims, timesteps = load_memmap(file_name)
        images = np.reshape(Yr.T, [timesteps] + list(dims), order='F')

    # ── Spatial patch slicing ──────────────────────────────────────────────
    # Slice out the spatial patch for this worker (same logic as before).
    upper_left_corner = min(idx_)
    lower_right_corner = max(idx_)
    indices = np.unravel_index([upper_left_corner, lower_right_corner],
                               dims, order='F')  # indices as tuples
    slices = [slice(min_dim, max_dim + 1) for min_dim, max_dim in indices]
    # insert slice for timesteps, equivalent to :
    slices.insert(0, slice(timesteps))

    if not isinstance(file_name, ShmHandle):
        images = np.reshape(Yr.T, [timesteps] + list(dims), order='F')

    if params.get('patch', 'in_memory'):
        images = np.array(images[tuple(slices)], dtype=np.float32)
    else:
        images = images[tuple(slices)]

    logger.debug(name_log+'file loaded')

    if (np.sum(np.abs(np.diff(images.reshape(timesteps, -1).T)))) > 0.1:

        opts = copy(params)
        opts.set('patch', {'n_processes': 1, 'rf': None, 'stride': None})
        for group in ('init', 'temporal', 'spatial'):
            opts.set(group, {'nb': params.get('patch', 'nb_patch')})
        for group in ('preprocess', 'temporal'):
            opts.set(group, {'p': params.get('patch', 'p_patch')})

        cnm = CNMF(n_processes=1, params=opts)

        cnm.fit(images)
        return [idx_, shapes, scipy.sparse.coo_matrix(cnm.estimates.A),
                cnm.estimates.b, cnm.estimates.C, cnm.estimates.f,
                cnm.estimates.S, cnm.estimates.bl, cnm.estimates.c1,
                cnm.estimates.neurons_sn, cnm.estimates.g, cnm.estimates.sn,
                cnm.params.to_dict(), cnm.estimates.YrA]
    else:
        return None

def run_CNMF_patches(file_name, shape, params, gnb=1, dview=None,
                     memory_fact=1, border_pix=0, low_rank_background=True,
                     del_duplicates=False, indices=[slice(None)]*3):
    """Function that runs CNMF in patches

     Either in parallel or sequentially, and return the result for each.
     It requires that ipyparallel is running

     Will basically initialize everything in order to compute on patches then call a function in parallel that will
     recreate the cnmf object and fit the values.
     It will then recreate the full frame by listing all the fitted values together

    Args:
        file_name: string
            full path to an npy file (2D, pixels x time) containing the movie

        shape: tuple of three elements
            dimensions of the original movie across y, x, and time

        params:
            CNMFParams object containing all the parameters for the various algorithms

        gnb: int
            number of global background components

        dview: 
            TODO

        memory_fact: double
            unitless number accounting how much memory should be used.
            It represents the fraction of patch processed in a single thread.
             You will need to try different values to see which one would work

        border_pix: int
            TODO

        low_rank_background: bool
            if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)

        del_duplicates: bool
            if True keeps only neurons in each patch that are well centered within the patch.
            I.e. neurons that are closer to the center of another patch are removed to
            avoid duplicates, cause the other patch should already account for them.

        indices: List[slice]
            TODO

    Returns:

        A_tot: matrix containing all the components from all the patches

        C_tot: matrix containing the calcium traces corresponding to A_tot
        
        YrA_tot: TODO

        b: TODO

        f: TODO

        sn_tot: per pixel noise estimate

        optional_outputs: set of outputs related to the result of CNMF ALGORITHM ON EACH patch

    Raises:
        Empty Exception
    """
    logger = logging.getLogger("caiman")

    dims = shape[:-1]
    d = np.prod(dims)
    T = shape[-1]
    _method_init = params.get("init", "method_init") or "greedy_roi"

    rf = params.get('patch', 'rf')
    if rf is None:
        rf = 16
    if np.isscalar(rf):
        rfs = [rf] * len(dims)
    else:
        rfs = rf

    stride = params.get('patch', 'stride')
    if stride is None:
        stride = 4
    if np.isscalar(stride):
        strides = [stride] * len(dims)
    else:
        strides = stride

    params_copy = deepcopy(params)
    npx_per_proc = np.prod(rfs) // memory_fact
    params_copy.set('preprocess', {'n_pixels_per_process': npx_per_proc})
    params_copy.set('spatial', {'n_pixels_per_process': npx_per_proc})
    params_copy.set('temporal', {'n_pixels_per_process': npx_per_proc})

    idx_flat, idx_2d = extract_patch_coordinates(
        dims, rfs, strides, border_pix=border_pix, indices=indices[1:])

    # ── Optionally pre-load movie into shared memory ───────────────────────
    # For .mmap files the OS already shares physical pages between processes
    # that open them read-only.  However the first ``np.memmap`` call in each
    # worker still incurs page-fault overhead to build the per-process virtual
    # mapping.  Using a ``SharedMovieBuffer`` instead:
    #   1. Eliminates those per-worker page-fault storms.
    #   2. Forces the entire array to be faulted into physical RAM once,
    #      ensuring L3 is warm before workers start.
    #   3. Lays the data out in C order (T, d1, d2), making temporal slices
    #      contiguous and cache-friendly for the CNMF inner loops.
    _shm_buf = None
    file_name_or_handle = file_name   # default: pass path string to workers

    use_shm_for_cnmf = (dview is not None and isinstance(file_name, str))
    if use_shm_for_cnmf:
        # Guard: only copy the movie into SHM if there is enough headroom in
        # both RAM and /dev/shm AFTER accounting for:
        #   - the movie itself (1× movie_bytes in SHM)
        #   - n_processes worker heaps (~200 MB each)
        #   - the existing page cache (which the kernel will not evict until
        #     the last moment, so it must be treated as committed)
        # Using psutil.virtual_memory().available is not sufficient because
        # 'available' includes reclaimable page cache that the kernel keeps
        # warm as long as possible — it disappears only after RAM is already
        # under pressure, by which point the SHM allocation has already forced
        # anonymous pages to swap.
        #
        # Safer budget: total_ram - used_ram - movie_bytes (page cache)
        #               minus 2 GB headroom for worker/OS overhead.
        _movie_bytes  = int(d * T * np.dtype(np.float32).itemsize)
        try:
            import psutil as _psu
            _vm       = _psu.virtual_memory()
            _free_shm = _psu.disk_usage('/dev/shm').free
            # Workers reading from SHM access the movie as a shared
            # mapping — the 27 GB is counted once regardless of how many
            # workers are running.  Their private RSS is compute buffers
            # only (~2 GB each, not 3.5 GB).
            # Correct budget: physical - parent_rss - movie >= worker_compute
            import os as _os
            _parent_rss   = _psu.Process(_os.getpid()).memory_info().rss
            _n_proc       = n_processes or 1
            _worker_compute = int(_n_proc * 2.0 * 2**30)  # ~2 GB private/worker
            _overhead       = int(4 * 2**30)              # 4 GB OS headroom
            # SHM copy allocates movie_bytes NEW anonymous pages.
            # Must fit: parent_rss + existing_shm + new_shm + workers + overhead
            # Use vm.available (includes reclaimable cache) minus what we need.
            _ram_ok   = (_vm.available >=
                         _movie_bytes + _worker_compute + _overhead)
            _shm_ok   = _free_shm >= _movie_bytes
        except Exception:
            _ram_ok = _shm_ok = True
            _vm = type('_', (), {'available': -1})()
            _free_shm = -1
        if not (_ram_ok and _shm_ok):
            logger.info(
                f"run_CNMF_patches: skipping SHM — movie needs "
                f"{_movie_bytes / 2**30:.1f} GiB but "
                f"vm.available={_vm.available / 2**30:.1f} GiB, "
                f"free /dev/shm={_free_shm / 2**30:.1f} GiB; "
                f"using per-worker mmap (page cache already warm)"
            )
            use_shm_for_cnmf = False

    # SHM copy happens AFTER precompute (see below) so precompute's
    # page cache writes don't stack with the 27 GB SHM allocation.

    # ── corr_pnr precompute: filter full FOV once on GPU ──────────────────
    # init_neurons_corr_pnr runs a cv2.filter2D loop over all T frames for
    # every patch (~20 s/patch CPU).  Precomputing on GPU once (~16 s total)
    # and passing each patch a slice saves ~177 s wall time across 9 rounds.
    # _precomp_cleanup holds the temp file path for deletion after patching.
    _precomp_result   = None
    _precomp_cleanup  = None
    if (_method_init == 'corr_pnr'
            and isinstance(file_name, str)):
        # Reuse cached precomp from a previous fit() if available.
        _cached_precomp = params.get('init', 'precomp_cache')
        if (_cached_precomp is not None
                and _cached_precomp.get('filtered_path')
                and os.path.exists(_cached_precomp['filtered_path'])):
            logger.info(
                f'run_CNMF_patches: reusing cached precomp '
                f'({_cached_precomp["filtered_path"]})')
            _precomp_result = _cached_precomp
            # _precomp_cleanup left as None — caller owns the cache lifetime
        else:
            try:
                _precomp_result = precompute_corr_pnr_filtered_fov(
                    movie_path        = file_name,
                    dims              = (dims[0], dims[1], T),
                    gSig              = list(params.get('init', 'gSig')),
                    center_psf        = params.get('init', 'center_psf') or True,
                    chunk_frames      = 3000,
                    forder_movie_path = params.init.get('forder_movie_path'),
                )
                if _precomp_result is not None:
                    _precomp_cleanup = _precomp_result['filtered_path']
                    logger.info(
                        f"run_CNMF_patches: corr_pnr precompute done — "
                        f"filtered mmap at {_precomp_cleanup}")
                    # Store for caller (cnmf.fit) — bypass params.set to avoid logging
                    params.init['precomp_cache'] = _precomp_result
            except Exception as _pc_exc:
                logger.warning(
                    f"run_CNMF_patches: corr_pnr precompute failed ({_pc_exc}) — "
                    f"workers will filter per-patch")
                _precomp_result = None

    # ── SHM copy: AFTER precompute so filt_full pages are already evicted ──
    # At this point: movie is in page cache (warm), filt_full is evicted.
    # Copying movie → SHM just relabels cache pages → net zero new RAM.
    # Before precompute: movie cache + SHM copy + filt_full writes = 81 GB OOM.
    if use_shm_for_cnmf:
        try:
            logger.info("run_CNMF_patches: loading movie into shared memory …")
            from caiman.shared_memory_utils import SharedMovieBuffer
            _shm_buf = SharedMovieBuffer(file_name, order='C')
            file_name_or_handle = _shm_buf.worker_handle()
            logger.info(
                f"run_CNMF_patches: movie in SHM '{file_name_or_handle.name}'"
            )
        except Exception as shm_exc:
            logger.warning(
                f"run_CNMF_patches: shared-memory setup failed ({shm_exc}); "
                f"falling back to per-worker mmap"
            )
            file_name_or_handle = file_name

    args_in = []
    patch_centers = []
    for id_f, id_2d in zip(idx_flat, idx_2d):
        _p = deepcopy(params_copy)
        if _precomp_result is not None:
            # Derive bounding box from idx_flat (sorted F-order pixel indices).
            # extract_patch_coordinates returns shapes (not meshgrid) as idx_2d.
            # F-order: pixel p → row = p % d1, col = p // d1
            _rows = id_f % dims[0]
            _cols = id_f // dims[0]
            _x0, _x1 = int(_rows.min()), int(_rows.max()) + 1
            _y0, _y1 = int(_cols.min()), int(_cols.max()) + 1
            _patch_precomp = dict(_precomp_result)  # shallow copy of scalars
            _patch_precomp['x0'] = _x0; _patch_precomp['x1'] = _x1
            _patch_precomp['y0'] = _y0; _patch_precomp['y1'] = _y1
            # Slice precomputed sn and data_max to patch extent
            # sn_full/data_max_full/cn_full/pnr_full are (d1, d2) = (rows, cols)
            # _x0:_x1 = row range (dim0), _y0:_y1 = col range (dim1)
            _patch_precomp['sn']       = _precomp_result['sn_full'][_x0:_x1, _y0:_y1]
            _patch_precomp['data_max'] = _precomp_result['data_max_full'][_x0:_x1, _y0:_y1]
            if _precomp_result.get('cn_full') is not None:
                _patch_precomp['cn']  = _precomp_result['cn_full'][_x0:_x1, _y0:_y1]
                _patch_precomp['pnr'] = _precomp_result['pnr_full'][_x0:_x1, _y0:_y1]
            # Inject directly into underlying dict to bypass CaImAn's param-change logger
            _p.init['precomp'] = _patch_precomp
        # Estimate patch cost for longest-first scheduling.
        # Use count of pixels above min_pnr threshold from precomp if available;
        # otherwise use patch size (uniform cost assumption).
        if _precomp_result is not None and _precomp_result.get('pnr_full') is not None:
            _rows = id_f % dims[0]
            _cols = id_f // dims[0]
            _x0p, _x1p = int(_rows.min()), int(_rows.max()) + 1
            _y0p, _y1p = int(_cols.min()), int(_cols.max()) + 1
            _pnr_patch = _precomp_result['pnr_full'][_x0p:_x1p, _y0p:_y1p]
            _min_pnr   = params.get('init', 'min_pnr') or 1.0
            _cost      = float((_pnr_patch > _min_pnr).sum())
        else:
            _cost = float(len(id_f))  # uniform
        args_in.append((file_name_or_handle, id_f, id_2d, _p, _cost))
        if del_duplicates:
            foo = np.zeros(d, dtype=bool)
            foo[id_f] = 1
            patch_centers.append(scipy.ndimage.center_of_mass(
                foo.reshape(dims, order='F')))
    # Sort patches longest-first so workers finish at similar times.
    # patch_centers must stay aligned with args_in order.
    if len(args_in) > 1:
        _order = sorted(range(len(args_in)), key=lambda i: args_in[i][4], reverse=True)
        args_in       = [args_in[i][:4] for i in _order]  # strip cost tuple
        if patch_centers:
            patch_centers = [patch_centers[i] for i in _order]
    else:
        args_in = [a[:4] for a in args_in]
    logger.info(f'Patch size: {id_2d}')

    # ── RAM-safe worker cap ───────────────────────────────────────────────
    # Each worker allocates ~10× the patch data volume as anon private pages
    # (HALS intermediates, NMF buffers, scipy sparse ops).  The SHM/mmap
    # movie is shared, so its cost is paid once.  We compute a safe upper
    # bound on concurrent workers and silently reduce dview if needed.
    if dview is not None and 'multiprocessing' in str(type(dview)):
        try:
            import psutil as _psu
            _patch_pixels   = max(
                int(np.prod([2 * r for r in rfs])),          # 2*rf estimate
                max((len(f) for f in idx_flat), default=1),  # largest actual patch
            )
            K    = params.get("init", "K") or 4
            _ssub        = params.get("init", "ssub") or 1
            f32, c64 = 4, 8
            # Analytical peak RSS — all terms are deterministic given
            # (patch_pixels, T, K) for a fixed CaImAn version:
            #   patch_data : Yr loaded into worker
            #   hals_copy  : Yr copy inside HALS iterations
            #   nmf_bufs   : gradient + update buffers (~3× patch)
            #   noise_fft  : rfft output, complex64
            #   A_mat, C_mat: spatial/temporal components (small)
            # patch_data is accessed via mmap — shared page cache frames
            # already resident from the parent's open file.  Workers do
            # not allocate new physical pages for mmap reads; omitting
            # this term from the analytical estimate prevents the cap from
            # being set too conservatively (confirmed empirically: workers
            # show ~960 MB RSS of which ~560 MB is shared mmap, leaving
            # only ~400 MB private anonymous).
            _hals_copy   = _patch_pixels * T * f32
            _nmf_bufs    = 3 * _patch_pixels * T * f32
            _noise_fft   = _patch_pixels * (T // 2 + 1) * c64
            _A_mat       = _patch_pixels * K * f32
            _C_mat       = K * T * f32
            _analytical  = (_hals_copy + _nmf_bufs +
                            _noise_fft + _A_mat + _C_mat)
            # corr_pnr extra: greedyROI_corr writes a residual mmap
            # (_groi_B.mmap) of shape (patch_pixels/ssub², T).
            # parallel_dot_product passes the memmap directly to
            # SharedMovieBuffer (np.memmap is an ndarray subclass) so
            # only one copy into SHM occurs — no intermediate heap copy.
            if _method_init == "corr_pnr":
                _ds_pixels    = max(1, _patch_pixels // (_ssub ** 2))
                _groi_B_extra = 1 * _ds_pixels * T * f32
                _analytical  += _groi_B_extra
            # overhead_frac × analytical gives the per-worker RSS budget.
            # Default 1.6 was calibrated empirically; lower (e.g. 1.1) if
            # workers consistently use less RAM than estimated, raise if OOM.
            # Exposed in JSON as cluster.worker_overhead_frac.
            _overhead_frac = float(params.get("patch", "worker_overhead_frac") or 1.6)
            _worker_bytes = int(_analytical * _overhead_frac)
            _vm             = _psu.virtual_memory()
            # Budget: fraction of RAM that is genuinely free for workers.
            # cnmf.fit() releases the parent images and Yr mmaps before
            # calling run_CNMF_patches so the 27 GB movie is no longer
            # pinned by the parent.  Workers open the file independently
            # and share its pages via the OS page cache.
            # We no longer subtract movie_bytes from the budget — the
            # pages are reclaimable once the parent fd is closed.
            _movie_bytes = 0
            _ram_frac    = float(params.get("patch", "ram_budget_frac") or 0.75)
            _budget      = max(0, _vm.available - _movie_bytes) * _ram_frac
            _safe_workers   = max(1, int(_budget // _worker_bytes))
            _actual_workers = dview._processes
            logger.info(
                f"run_CNMF_patches RAM estimate: "
                f"patch={_patch_pixels}px  "
                f"analytical={_analytical/2**30:.2f} GB  "
                f"overhead={_overhead_frac:.1f}×  "
                f"worker_est={_worker_bytes/2**30:.2f} GB  "
                f"movie={_movie_bytes/2**30:.1f} GB  "
                f"vm.available={_vm.available/2**30:.1f} GB  "
                f"budget={_budget/2**30:.1f} GB  "
                f"→ {_safe_workers} workers (requested {_actual_workers})"
            )
            if _safe_workers < _actual_workers:
                logger.warning(
                    f"run_CNMF_patches: capping workers {_actual_workers} → "
                    f"{_safe_workers} to avoid OOM — lower worker_overhead_frac "
                    f"in JSON if workers consistently use less than "
                    f"{_worker_bytes/2**30:.2f} GB each"
                )
                # Replace pool with a smaller one for this run
                dview.terminate()
                dview.join()     # reap workers so they don't linger as orphans
                # Use spawn so workers start with a clean process —
                # no broken CUDA context inherited from the parent fork.
                # Pass log params via initargs (spawn workers do not
                # inherit module-level state from the parent).
                _lp = _collect_log_params()
                _lp['blas_threads'] = int(
                    params.get('patch', 'blas_threads_per_worker') or 1)
                _spawn_ctx = multiprocessing.get_context('spawn')
                dview = _spawn_ctx.Pool(
                    _safe_workers,
                    initializer = _worker_logging_init,
                    initargs    = (_lp,),
                )
        except Exception as _ram_exc:
            logger.debug(f"RAM cap check failed ({_ram_exc}); proceeding with original pool")
    st = time.time()
    try:
        if dview is not None:
            if 'multiprocessing' in str(type(dview)):
                # Terminate the pipeline dview pool before spawning the
                # dedicated patch pool.  Without this, both pools exist
                # simultaneously — n_proc idle pipeline workers + n_proc
                # active patch workers — doubling process count and wasting
                # the RAM those idle workers consume.
                n_proc = dview._processes
                dview.terminate()
                dview.join()
                logger.info(
                    f'run_CNMF_patches: spawning dedicated pool '
                    f'({n_proc} workers, maxtasksperchild=1)'
                )
                # spawn context: workers start fresh with no inherited
                # CUDA state — eliminates cudaErrorInitializationError
                # in get_noise_fft GPU path without needing a reset.
                # Pass log params via initargs (spawn does not inherit
                # module-level state from the parent process).
                _lp = _collect_log_params()
                _lp['blas_threads'] = int(
                    params.get('patch', 'blas_threads_per_worker') or 1)
                _spawn_ctx = multiprocessing.get_context('spawn')
                with _spawn_ctx.Pool(
                    processes       = n_proc,
                    maxtasksperchild= 1,
                    initializer     = _worker_logging_init,
                    initargs        = (_lp,),
                ) as _patch_pool:
                    file_res = list(
                        _patch_pool.imap_unordered(
                            cnmf_patches, args_in, chunksize=1
                        )
                    )
            else:
                try:
                    file_res = dview.map_sync(cnmf_patches, args_in)
                    dview.results.clear()
                except:
                    print('Something went wrong')
                    raise
                finally:
                    logger.info('Patch processing complete')
        else:
            # dview is None — happens when fit() terminated the pipeline
            # pool before precompute to save RAM.  Spawn a fresh dedicated
            # pool using n_processes from params rather than running serially.
            _n_proc_fallback = params.get('patch', 'n_processes') or 1
            if _n_proc_fallback > 1:
                logger.info(
                    f'run_CNMF_patches: spawning dedicated pool '
                    f'({_n_proc_fallback} workers, dview was None)')
                _lp = _collect_log_params()
                _lp['blas_threads'] = int(
                    params.get('patch', 'blas_threads_per_worker') or 1)
                _spawn_ctx = multiprocessing.get_context('spawn')
                with _spawn_ctx.Pool(
                    processes       = _n_proc_fallback,
                    maxtasksperchild= 1,
                    initializer     = _worker_logging_init,
                    initargs        = (_lp,),
                ) as _patch_pool:
                    file_res = list(
                        _patch_pool.imap_unordered(
                            cnmf_patches, args_in, chunksize=1
                        )
                    )
            else:
                file_res = list(map(cnmf_patches, args_in))
    finally:
        if _shm_buf is not None:
            _shm_buf.close()
        if _precomp_cleanup is not None:
            try:
                os.unlink(_precomp_cleanup)
                logger.debug(f"run_CNMF_patches: removed precomp mmap {_precomp_cleanup}")
            except OSError:
                pass

    logger.info('Elapsed time for processing patches: \
                 {0}s'.format(str(time.time() - st).split('.')[0]))
    # count components
    count = 0
    count_bgr = 0
    patch_id = 0
    num_patches = len(file_res)
    for jj, fff in enumerate(file_res):
        if fff is not None:
            idx_, shapes, A, b, C, f, S, bl, c1, neurons_sn, g, sn, _, YrA = fff
            for _ in range(b.shape[-1]):
                count_bgr += 1

            A = A.tocsc()
            if del_duplicates:
                keep = []
                for ii in range(A.shape[-1]):
                    neuron_center = (np.array(scipy.ndimage.center_of_mass(
                        A[:, ii].toarray().reshape(shapes, order='F'))) -
                        np.array(shapes) / 2. + np.array(patch_centers[jj]))
                    if np.argmin([np.linalg.norm(neuron_center - p) for p in
                                  np.array(patch_centers)]) == jj:
                        keep.append(ii)
                A = A[:, keep]
                file_res[jj][2] = A
                file_res[jj][4] = C[keep]
                if S is not None:
                    file_res[jj][6] = S[keep]
                    file_res[jj][7] = bl[keep]
                    file_res[jj][8] = c1[keep]
                    file_res[jj][9] = neurons_sn[keep]
                    file_res[jj][10] = g[keep]
                file_res[jj][-1] = YrA[keep]

            # for ii in range(A.shape[-1]):
            #     new_comp = A[:, ii] / np.sqrt(A[:, ii].power(2).sum())
            #     if new_comp.sum() > 0:
            #         count += 1
            count += np.sum(A.sum(0) > 0)

            patch_id += 1

    # INITIALIZING
    nb_patch = params.get('patch', 'nb_patch')
    C_tot = np.zeros((count, T), dtype=np.float32)
    if params.get('init', 'center_psf'):
        S_tot = np.zeros((count, T), dtype=np.float32)
    else:
        S_tot = None
    YrA_tot = np.zeros((count, T), dtype=np.float32)
    F_tot = np.zeros((max(0, num_patches * nb_patch), T), dtype=np.float32)
    mask = np.zeros(d, dtype=np.uint8)
    sn_tot = np.zeros((d))

    f_tot, bl_tot, c1_tot, neurons_sn_tot, g_tot, idx_tot, id_patch_tot, shapes_tot = [
    ], [], [], [], [], [], [], []
    patch_id, empty, count_bgr, count, f_bgr_count = 0, 0, 0, 0, 0
    idx_tot_B, idx_tot_A, a_tot, b_tot = [], [], [], []
    idx_ptr_B, idx_ptr_A = [0], [0]

    # instead of filling in the matrices, construct lists with their non-zero
    # entries and coordinates
    logger.info('Embedding patches results into whole FOV')
    for fff in file_res:
        if fff is not None:

            idx_, shapes, A, b, C, f, S, bl, c1, neurons_sn, g, sn, _, YrA = fff
            A = A.tocsc()

            # check A for nans, which result in corrupted outputs.  Better to fail here if any found
            nnan = np.isnan(A.data).sum()
            if nnan > 0:
                raise RuntimeError('found %d/%d nans in A, cannot continue' % (nnan, len(A.data)))

            sn_tot[idx_] = sn
            f_tot.append(f)
            bl_tot.append(bl)
            c1_tot.append(c1)
            neurons_sn_tot.append(neurons_sn)
            g_tot.append(g)
            idx_tot.append(idx_)
            shapes_tot.append(shapes)
            mask[idx_] += 1

            if scipy.sparse.issparse(b):
                b = scipy.sparse.csc_matrix(b)
                b_tot.append(b.data)
                idx_ptr_B += list(b.indptr[1:] - b.indptr[:-1])
                idx_tot_B.append(idx_[b.indices])
            else:
                for ii in range(b.shape[-1]):
                    b_tot.append(b[:, ii])
                    idx_tot_B.append(idx_)
                    idx_ptr_B.append(len(idx_))
                    # F_tot[patch_id, :] = f[ii, :]
            count_bgr += b.shape[-1]
            if nb_patch >= 0:
                # Use f_bgr_count (not patch_id*nb_patch) as the write offset.
                # patch_id advances for every patch including empty ones, so
                # patch_id*nb_patch drifts out of sync with count_bgr whenever
                # a patch returns fewer background components than nb_patch.
                # f_bgr_count tracks how many rows have actually been written.
                _f_rows = f.shape[0] if f is not None and hasattr(f, 'shape') else 0
                if _f_rows > 0:
                    F_tot[f_bgr_count:f_bgr_count + _f_rows] = f[:_f_rows]
                f_bgr_count += _f_rows
            else:  # full background per patch
                F_tot = np.concatenate([F_tot, f])

            for ii in range(A.shape[-1]):
                new_comp = A[:, ii]  # / np.sqrt(A[:, ii].power(2).sum())
                if new_comp.sum() > 0:
                    a_tot.append(new_comp.toarray().flatten())
                    idx_tot_A.append(idx_)
                    idx_ptr_A.append(len(idx_))
                    C_tot[count, :] = C[ii, :]
                    if params.get('init', 'center_psf'):
                        S_tot[count, :] = S[ii, :]
                    YrA_tot[count, :] = YrA[ii, :]
                    id_patch_tot.append(patch_id)
                    count += 1

            patch_id += 1
        else:
            empty += 1

    logger.debug(f'Skipped {empty} empty patches')
    if count_bgr > 0:
        idx_tot_B = np.concatenate(idx_tot_B)
        b_tot = np.concatenate(b_tot)
        idx_ptr_B = np.cumsum(np.array(idx_ptr_B))
        B_tot = scipy.sparse.csc_matrix(
            (b_tot, idx_tot_B, idx_ptr_B), shape=(d, count_bgr))
    else:
        B_tot = scipy.sparse.csc_matrix((d, count_bgr), dtype=np.float32)

    if len(idx_tot_A):
        idx_tot_A = np.concatenate(idx_tot_A)
        a_tot = np.concatenate(a_tot)
        idx_ptr_A = np.cumsum(np.array(idx_ptr_A))
    A_tot = scipy.sparse.csc_matrix(
        (a_tot, idx_tot_A, idx_ptr_A), shape=(d, count), dtype=np.float32)

    C_tot = C_tot[:count, :]
    YrA_tot = YrA_tot[:count, :]
    F_tot = F_tot[:count_bgr]

    optional_outputs = dict()
    optional_outputs['b_tot'] = b_tot
    optional_outputs['f_tot'] = f_tot
    optional_outputs['bl_tot'] = bl_tot
    optional_outputs['c1_tot'] = c1_tot
    optional_outputs['neurons_sn_tot'] = neurons_sn_tot
    optional_outputs['g_tot'] = g_tot
    optional_outputs['S_tot'] = S_tot
    optional_outputs['idx_tot'] = idx_tot
    optional_outputs['shapes_tot'] = shapes_tot
    optional_outputs['id_patch_tot'] = id_patch_tot
    optional_outputs['B'] = B_tot
    optional_outputs['F'] = F_tot
    optional_outputs['mask'] = mask

    logger.info("Constructing background")

    Im = scipy.sparse.csr_matrix(
        (1. / (mask + np.finfo(np.float32).eps), (np.arange(d), np.arange(d))), dtype=np.float32)

    if not del_duplicates:
        A_tot = Im.dot(A_tot)

    if count_bgr == 0:
        b = None
        f = None
    elif low_rank_background is None:
        b = Im.dot(B_tot)
        f = F_tot
        logger.info("Leaving background components intact")
    elif low_rank_background:
        logger.info("Compressing background components with a low rank NMF")
        B_tot = Im.dot(B_tot)
        Bm = (B_tot)
        #f = np.r_[np.atleast_2d(np.mean(F_tot, axis=0)),
        #          np.random.rand(gnb - 1, T)]
        # Filter out nan components before NMF
        nan_components = np.any(np.isnan(F_tot), axis=1)
        F_tot = F_tot[~nan_components, :]
        Bm = Bm[:, ~nan_components]
        # Guard: NMF requires n_components <= min(n_samples, n_features).
        # During refit some patches may return fewer background rows than gnb
        # (e.g. patches too small to support the requested nb).  Clamp
        # n_components to the number of available rows so NMF doesn't crash.
        _nmf_components = min(gnb, F_tot.shape[0])
        if _nmf_components < gnb:
            logger.warning(
                f'run_CNMF_patches: only {F_tot.shape[0]} background rows available '
                f'but gnb={gnb}; clamping NMF to {_nmf_components} components'
            )
        _nmf_init = 'nndsvdar' if _nmf_components <= min(F_tot.shape) else 'random'
        mdl = NMF(n_components=_nmf_components, verbose=False, init=_nmf_init,
                  tol=1e-10, max_iter=100, shuffle=False, random_state=1)
        mdl.fit(np.maximum(F_tot, 0))
        f = mdl.components_.squeeze()
        f = np.atleast_2d(f)
        for _ in range(100):
            f /= np.sqrt((f**2).sum(1)[:, None]) + np.finfo(np.float32).eps
            try:
                b = np.fmax(Bm.dot(F_tot.dot(f.T)).dot(
                    np.linalg.inv(f.dot(f.T))), 0)
            except np.linalg.LinAlgError:  # singular matrix
                b = np.fmax(Bm.dot(scipy.linalg.lstsq(f.T, F_tot.T)[0].T), 0)
            try:
                #f = np.linalg.inv(b.T.dot(b)).dot((Bm.T.dot(b)).T.dot(F_tot))
                f = np.linalg.solve(b.T.dot(b), (Bm.T.dot(b)).T.dot(F_tot))
            except np.linalg.LinAlgError:  # singular matrix
                f = scipy.linalg.lstsq(b, Bm.toarray())[0].dot(F_tot)

        nB = np.ravel(np.sqrt((b**2).sum(0)))
        b /= nB + np.finfo(np.float32).eps
        b = np.array(b, dtype=np.float32)
#        B_tot = scipy.sparse.coo_matrix(B_tot)
        f *= nB[:, None]
    else:
        logger.info('Removing overlapping background components \
                     from different patches')
        nA = np.ravel(np.sqrt(A_tot.power(2).sum(0)))
        A_tot /= nA
        A_tot = scipy.sparse.coo_matrix(A_tot)
        C_tot *= nA[:, None]
        YrA_tot *= nA[:, None]
        nB = np.ravel(np.sqrt(B_tot.power(2).sum(0)))
        B_tot /= nB
        B_tot = B_tot.toarray().astype(np.float32)
#        B_tot = scipy.sparse.coo_matrix(B_tot)
        F_tot *= nB[:, None]

        processed_idx:set = set([])
        # needed if a patch has more than 1 background component
        processed_idx_prev:set = set([])
        for _b in np.arange(B_tot.shape[-1]):
            idx_mask = np.where(B_tot[:, _b])[0]
            idx_mask_repeat = processed_idx.intersection(idx_mask)
            if len(idx_mask_repeat) < len(idx_mask):
                processed_idx_prev = processed_idx
            else:
                idx_mask_repeat = processed_idx_prev.intersection(idx_mask)
            processed_idx = processed_idx.union(idx_mask)
            if len(idx_mask_repeat) > 0:
                B_tot[np.array(list(idx_mask_repeat), dtype=int), _b] = 0

        b = B_tot
        f = F_tot

        logger.info('using one background component per patch')

    logger.info("Constructing background DONE")

    return A_tot, C_tot, YrA_tot, b, f, sn_tot, optional_outputs
