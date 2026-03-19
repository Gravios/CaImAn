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
        # Skip loading Yr when tile_path is set — Yr is not used in that path.
        _precomp_check = params.get('init', 'precomp') or {}
        if _precomp_check.get('tile_path') is None:
            Yr, dims, timesteps = load_memmap(file_name)
            images = np.reshape(Yr.T, [timesteps] + list(dims), order='F')
        else:
            Yr = None
            # dims = FULL movie dims (d1, d2) — idx_ are flat indices into
            # the full FOV, not the tile. tile_shape[1:] is WRONG here.
            _pc = params.get('init', 'precomp') or {}
            dims = (_pc['d1'], _pc['d2'])  # full movie spatial dims
            timesteps = _pc.get('T') or _pc.get('tile_shape', (1,))[0]

    # ── Spatial patch slicing ──────────────────────────────────────────────
    # Slice out the spatial patch for this worker (same logic as before).
    upper_left_corner = min(idx_)
    lower_right_corner = max(idx_)
    indices = np.unravel_index([upper_left_corner, lower_right_corner],
                               dims, order='F')  # indices as tuples
    slices = [slice(min_dim, max_dim + 1) for min_dim, max_dim in indices]
    # insert slice for timesteps, equivalent to :
    slices.insert(0, slice(timesteps))

    # Check for tile dispatcher coordinates.
    # Worker mmaps the /dev/shm tile file and copies its own slice.
    # No pickle of large arrays — only small coordinate tuples are sent.
    _precomp_inner = params.get('init', 'precomp') or {}
    _tile_path  = _precomp_inner.get('tile_path')
    _tile_shape = _precomp_inner.get('tile_shape')
    _tile_lx    = _precomp_inner.get('tile_lx')
    _tile_ly    = _precomp_inner.get('tile_ly')
    if _tile_path is not None and _tile_shape is not None:
        # Mmap tile from /dev/shm then copy slice as (d1p, d2p, T) F-order.
        # This layout makes Y=transpose(images,[1,2,0]) F-contiguous so that
        # Y.reshape(-1,T,order='F') and the subsequent Yr are zero-copy views.
        _tile_mm = np.memmap(_tile_path, dtype=np.float32, mode='r',
                             shape=_tile_shape, order='F')
        lx0, lx1 = _tile_lx
        ly0, ly1 = _tile_ly
        # asfortranarray of the transposed slice → (d1p, d2p, T) F-contiguous
        _images_f = np.asfortranarray(
            _tile_mm[:, lx0:lx1, ly0:ly1].transpose(1, 2, 0), dtype=np.float32)
        del _tile_mm
        # Wrap as (T, d1p, d2p) view for fit() — Y=transpose([1,2,0]) recovers
        # _images_f which IS F-contiguous → Y_ds_flat is a zero-copy view.
        images = _images_f.transpose(2, 0, 1)
    elif not isinstance(file_name, ShmHandle):
        images = np.reshape(Yr.T, [timesteps] + list(dims), order='F')
        images = np.asfortranarray(images[tuple(slices)], dtype=np.float32)
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

        logger.warning(f"[patch {idx_[0]}] starting fit")
        cnm.fit(images)
        _n_neurons = (cnm.estimates.A.shape[1]
                      if cnm.estimates.A is not None else 0)
        logger.warning(
            f"[patch {idx_[0]}] done — {_n_neurons} neurons"
        )
        # Extract result arrays BEFORE deleting cnm to avoid an extra copy.
        _result = [idx_, shapes, scipy.sparse.coo_matrix(cnm.estimates.A),
                   cnm.estimates.b, cnm.estimates.C, cnm.estimates.f,
                   cnm.estimates.S, cnm.estimates.bl, cnm.estimates.c1,
                   cnm.estimates.neurons_sn, cnm.estimates.g, cnm.estimates.sn,
                   cnm.params.to_dict(), cnm.estimates.YrA]
        # Explicitly free all large intermediate arrays so the worker's
        # RSS drops back before the next patch is assigned.
        # Python's allocator keeps freed blocks in its arenas by default;
        # malloc_trim(0) returns them to the OS immediately.
        del cnm, images
        try: del _images_f
        except NameError: pass
        try: del Yr
        except NameError: pass
        import gc as _gc; _gc.collect()
        try:
            import ctypes as _ct
            _ct.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception: pass
        from caiman.cluster import flush_worker_log
        flush_worker_log()
        return _result
    else:
        return None


def _tile_dispatch(pool, args_in, file_name, dims, T, _precomp_result, logger,
                   forder_movie_path=None):
    """Tile-based I/O dispatcher with rolling 2-tile prefetch.

    Two SHM slots (A/B) are used in rotation. While workers process patches
    from slot A, the dispatcher loads the next tile into slot B in a background
    thread. When a worker finishes a patch it immediately picks up the next
    available one — no intra-tile or inter-tile idle time.
    """
    import copy as _copy
    import numpy as _np_td
    import threading as _thr
    import os as _os_td

    d1, d2 = dims
    _shm_dir = _os_td.environ.get('CAIMAN_SHM', '/dev/shm')

    # ── Build patch spatial bounding boxes ──────────────────────────────────
    patch_boxes = []
    for _, id_f, id_2d, p in args_in:
        _rows = id_f % d1
        _cols = id_f // d1
        x0, x1 = int(_rows.min()), int(_rows.max()) + 1
        y0, y1 = int(_cols.min()), int(_cols.max()) + 1
        patch_boxes.append((x0, x1, y0, y1))

    xs = sorted(set(b[0] for b in patch_boxes))
    ys = sorted(set(b[2] for b in patch_boxes))
    stride_x = (xs[1] - xs[0]) if len(xs) > 1 else (patch_boxes[0][1] - patch_boxes[0][0])
    stride_y = (ys[1] - ys[0]) if len(ys) > 1 else (patch_boxes[0][3] - patch_boxes[0][2])
    tile_n   = 3
    tile_dx  = tile_n * stride_x
    tile_dy  = tile_n * stride_y

    tile_map = {}
    for i, (x0, x1, y0, y1) in enumerate(patch_boxes):
        tile_id = (x0 // tile_dx, y0 // tile_dy)
        tile_map.setdefault(tile_id, []).append(i)

    sorted_tiles = sorted(tile_map.items())  # raster order
    logger.info(
        f'TileDispatcher: {len(sorted_tiles)} tiles of ~{tile_n}×{tile_n} patches, '
        f'tile_dx={tile_dx} tile_dy={tile_dy}')

    # ── Open movie mmap ──────────────────────────────────────────────────────
    _use_forder = (forder_movie_path is not None
                   and _os_td.path.exists(forder_movie_path))
    if _use_forder:
        _movie_f = _np_td.memmap(forder_movie_path, dtype=_np_td.float32,
                                  mode='r', shape=(d1, d2, T), order='F')
        _Yr = None
    else:
        from caiman.mmapping import load_memmap as _load_mmap
        _Yr, _, _ = _load_mmap(file_name, mode='r')
        _movie_f = None

    _filt_full = None
    if _precomp_result is not None:
        _fp = _precomp_result.get('filtered_path')
        _fd1 = _precomp_result.get('d1', d1)
        _fd2 = _precomp_result.get('d2', d2)
        _fdtype = _precomp_result.get('filt_dtype', 'float32')
        if _fp and _os_td.path.exists(_fp):
            _filt_full = _np_td.memmap(
                _fp, dtype=_fdtype, mode='r',
                shape=(_fd1, _fd2, T), order='F')

    # ── Two SHM slot paths ───────────────────────────────────────────────────
    _n_slots = int(_os_td.environ.get('CAIMAN_TILE_SLOTS', '2'))
    _slot_paths = {
        0: (_os_td.path.join(_shm_dir, '_caiman_tile_A.mmap'),
            _os_td.path.join(_shm_dir, '_caiman_filt_A.mmap')),
        1: (_os_td.path.join(_shm_dir, '_caiman_tile_B.mmap'),
            _os_td.path.join(_shm_dir, '_caiman_filt_B.mmap')),
        2: (_os_td.path.join(_shm_dir, '_caiman_tile_C.mmap'),
            _os_td.path.join(_shm_dir, '_caiman_filt_C.mmap')),
    }
    _slot_paths = {k: v for k, v in _slot_paths.items() if k < _n_slots}

    def _write_tile(tile_id, patch_indices, slot):
        """Load one tile from FUSE into SHM slot. Returns tile metadata."""
        tile_xs = [patch_boxes[i][0] for i in patch_indices]
        tile_xe = [patch_boxes[i][1] for i in patch_indices]
        tile_ys = [patch_boxes[i][2] for i in patch_indices]
        tile_ye = [patch_boxes[i][3] for i in patch_indices]
        tx0, tx1 = min(tile_xs), max(tile_xe)
        ty0, ty1 = min(tile_ys), max(tile_ye)

        tile_path, filt_path = _slot_paths[slot]
        tile_shape = (T, tx1-tx0, ty1-ty0)

        # SHM headroom guard: estimate tile cost and check free space
        # before mmap'ing. Writing beyond /dev/shm capacity raises SIGBUS
        # (unrecoverable crash). Abort early with a clear error instead.
        # Only count images tile — filt tiles are skipped when filt_full
        # is on NVMe (workers read filt_full directly, no SHM copy).
        _tile_bytes = int(_np_td.prod(tile_shape)) * 4
        _filt_on_shm_hg = (
            _filt_full is not None
            and str(getattr(_filt_full, 'filename', '') or '').startswith(_shm_dir)
        )
        _filt_bytes = int(_np_td.prod(tile_shape)) * 2 if _filt_on_shm_hg else 0
        _needed_gb  = (_tile_bytes + _filt_bytes) / 2**30
        try:
            import shutil as _shutil_hg
            _shm_free_gb = _shutil_hg.disk_usage(_shm_dir).free / 2**30
            _headroom_gb = 2.0   # keep 2 GB in reserve for worker incp files
            if _shm_free_gb - _needed_gb < _headroom_gb:
                raise MemoryError(
                    f'TileDispatcher: insufficient SHM headroom for tile '
                    f'({tx0}:{tx1},{ty0}:{ty1}): need {_needed_gb:.1f} GB + '
                    f'{_headroom_gb:.1f} GB reserve, only {_shm_free_gb:.1f} GB free. '
                    f'Reduce CAIMAN_TILE_SLOTS (currently {_n_slots}) or '
                    f'increase /dev/shm: sudo mount -o remount,size=60G /dev/shm'
                )
        except (OSError, AttributeError):
            pass  # skip check if disk_usage fails (non-tmpfs /dev/shm)

        tm = _np_td.memmap(tile_path, dtype=_np_td.float32,
                           mode='w+', shape=tile_shape, order='F')
        if _use_forder:
            tm[:] = _np_td.transpose(
                _np_td.asarray(_movie_f[tx0:tx1, ty0:ty1, :], dtype=_np_td.float32),
                (2, 0, 1))
        else:
            _px_idx = _np_td.array(
                [(x * d2 + y) for x in range(tx0, tx1) for y in range(ty0, ty1)],
                dtype=_np_td.int64)
            tm[:] = _Yr[_px_idx, :].reshape(tx1-tx0, ty1-ty0, T).transpose(2, 0, 1)
        tm.flush(); del tm
        # Drop movie pages from kernel page cache after each tile read.
        # Without this, FUSE pages accumulate to 40+ GB of Cached RAM,
        # crowding out worker SHM pages → workers go D-state.
        _FADV_DONTNEED = 4
        try:
            import os as _os_fv
            if _use_forder:
                _os_fv.posix_fadvise(_movie_f._mmap.fileno(), 0, 0, _FADV_DONTNEED)
            elif _Yr is not None and hasattr(_Yr, '_mmap'):
                _os_fv.posix_fadvise(_Yr._mmap.fileno(), 0, 0, _FADV_DONTNEED)
        except (AttributeError, OSError): pass

        filt_shape = None
        if _filt_full is not None:
            # If filt_full is on NVMe (not /dev/shm), skip the SHM tile copy.
            # Pass the filt_full path + absolute FOV dims directly to workers.
            # Workers slice filt_full[x0:x1, y0:y1, :] using absolute patch
            # coords — the NVMe page cache is warm from the tile write above.
            # Saves 2 × ~3.7 GB = 7.4 GB /dev/shm per slot pair.
            _filt_filename = str(getattr(_filt_full, 'filename', '') or '')
            _filt_on_shm   = _filt_filename.startswith(_shm_dir)
            if _filt_on_shm:
                filt_shape = (tx1-tx0, ty1-ty0, T)
                fm = _np_td.memmap(filt_path, dtype=_np_td.float16,
                                   mode='w+', shape=filt_shape, order='F')
                fm[:] = _filt_full[tx0:tx1, ty0:ty1, :]
                fm.flush(); del fm
                filt_path = filt_path
            else:
                filt_shape = (_precomp_result['d1'], _precomp_result['d2'], T)
                filt_path  = _precomp_result['filtered_path']
            try:
                _os_fv.posix_fadvise(
                    _filt_full._mmap.fileno(), 0, 0, _FADV_DONTNEED)
            except (AttributeError, OSError, NameError): pass

        mb = int(_np_td.prod(tile_shape) * 4 // 2**20)
        try:
            import shutil as _sh_tl
            _du_tl = _sh_tl.disk_usage('/dev/shm')
            logger.info(
                f'TileDispatcher: tile {tile_id} ({tx0}:{tx1},{ty0}:{ty1}) '
                f'→ {len(patch_indices)} patches, {mb} MB written to SHM slot {slot} '
                f'| SHM {_du_tl.used/2**30:.2f}/{_du_tl.total/2**30:.1f}GB '
                f'free={_du_tl.free/2**30:.2f}GB')
        except Exception:
            logger.info(
                f'TileDispatcher: tile {tile_id} ({tx0}:{tx1},{ty0}:{ty1}) '
                f'→ {len(patch_indices)} patches, {mb} MB written to SHM slot {slot}')
        return tile_path, filt_path, tile_shape, filt_shape, (tx0, tx1, ty0, ty1)

    def _build_patch_args(patch_indices, tile_path, filt_path,
                          tile_shape, filt_shape, tile_extents):
        tx0, tx1, ty0, ty1 = tile_extents
        # Sort patches within this tile heaviest-first.
        # Workers pull from a shared queue so leading entries are grabbed first.
        # Cost was precomputed (seed pixel count) during args_in construction.
        # The global longest-first sort already ordered args_in; here we refine
        # within the tile group since the tile load re-bundles them.
        if (len(patch_indices) > 1
                and _precomp_result is not None
                and _precomp_result.get('pnr_full') is not None):
            # params is not in _tile_dispatch scope; read min_pnr from the patch params
            _first_p   = args_in[patch_indices[0]][3]
            _min_pnr_d = float(_first_p.get('init', 'min_pnr') or 1.0)
            def _patch_cost(i):
                x0, x1, y0, y1 = patch_boxes[i]
                _p = _precomp_result['pnr_full'][x0:x1, y0:y1]
                return float((_p > _min_pnr_d).sum())
            patch_indices = sorted(patch_indices, key=_patch_cost, reverse=True)
        patch_args = []
        for i in patch_indices:
            fn, id_f, id_2d, p = args_in[i]
            x0, x1, y0, y1 = patch_boxes[i]
            lx0, lx1 = x0 - tx0, x1 - tx0
            ly0, ly1 = y0 - ty0, y1 - ty0
            # Clip to tile extent — prevents images.shape from exceeding
            # tile_shape when a patch extends beyond the written tile region.
            lx1 = min(lx1, tile_shape[1])
            ly1 = min(ly1, tile_shape[2])
            _p = _copy.copy(p)
            _pc = dict(_p.init.get('precomp') or {})
            _pc['tile_path']  = tile_path
            _pc['tile_shape'] = tile_shape
            _pc['tile_lx']    = (lx0, lx1)
            _pc['tile_ly']    = (ly0, ly1)
            if filt_path and filt_shape:
                # If filt_shape is full-FOV (NVMe passthrough), skip filt_tile_path.
                # Workers then use the fallback path (direct filt_full read with
                # absolute patch coords x0:x1, y0:y1 — already implemented).
                _filt_is_full_fov = (filt_shape[0] == dims[0] and
                                     filt_shape[1] == dims[1])
                if not _filt_is_full_fov:
                    _pc['filt_tile_path']  = filt_path
                    _pc['filt_tile_shape'] = filt_shape
                # (else: filt_full path already in precomp['filtered_path'];
                #  workers read it directly via the fallback branch)
                # Clip filt tile coords to filt_shape extent.
                _flx1 = min(lx1, filt_shape[0])
                _fly1 = min(ly1, filt_shape[1])
                _pc['filt_tile_lx']    = (lx0, _flx1)
                _pc['filt_tile_ly']    = (ly0, _fly1)
            _p.init['precomp'] = _pc
            patch_args.append((fn, id_f, id_2d, _p))
        return patch_args

    # ── Callback-driven streaming dispatch ─────────────────────────────────
    # apply_async(callback=) fires in the pool's result-handler thread the
    # instant a worker result arrives. The callback decrements the slot count
    # and — if the slot is now free — immediately submits the next tile's
    # patches. Workers never see an empty queue as long as tiles remain.
    # The main thread only waits on a completion Event, never polls.
    _lock        = _thr.Lock()
    _done_event  = _thr.Event()
    _errors      = []
    file_res     = []
    total        = sum(len(v) for _, v in sorted_tiles)

    n             = len(sorted_tiles)
    _slot_meta    = [None] * _n_slots
    _slot_err     = [None] * _n_slots
    _slot_count   = [0]    * _n_slots
    _next_load    = [0]     # next tile index to load into SHM
    _next_submit  = [0]     # next tile index to submit to pool
    _load_threads = {}      # k_tile → Thread

    def _load_slot(k_tile, slot):
        tid, pidx = sorted_tiles[k_tile]
        try:
            _slot_meta[slot] = _write_tile(tid, pidx, slot)
        except Exception as e:
            _slot_err[slot] = e

    def _try_submit(slot):
        """Submit the next pending tile into `slot` if it is ready. Must hold _lock."""
        k = _next_submit[0]
        if k >= n:
            return
        if _slot_meta[slot] is None:
            return   # load not finished yet
        if _slot_err[slot] is not None:
            _errors.append(RuntimeError(f"tile load failed: {_slot_err[slot]}"))
            _done_event.set()
            return
        _, pidx = sorted_tiles[k]
        patch_args = _build_patch_args(pidx, *_slot_meta[slot])
        _slot_count[slot] = len(patch_args)
        _next_submit[0] += 1
        _slot_meta[slot] = None   # mark slot as in-use
        logger.info(f'TileDispatcher: submitted {len(patch_args)} patches '                    f'from tile {sorted_tiles[k][0]} slot {slot}')
        for pa in patch_args:
            pool.apply_async(cnmf_patches, (pa,),
                             callback=lambda r, s=slot: _on_result(r, s),
                             error_callback=lambda e, s=slot: _on_error(e, s))

    def _on_result(result, slot):
        """Fires in pool result-handler thread when a worker completes."""
        try:
            _start_load_thread = None
            with _lock:
                file_res.append(result)
                _slot_count[slot] -= 1
                _n_done = len(file_res)
                if _slot_count[slot] == 0:
                    # Slot freed — start loading next tile into it.
                    # All state reads/writes under _lock to prevent two
                    # simultaneous callbacks both seeing count==0.
                    nk = _next_load[0]
                    if nk < n:
                        _next_load[0] += 1
                        t = _thr.Thread(target=_load_and_submit,
                                        args=(nk, slot), daemon=True)
                        _load_threads[nk] = t
                        _start_load_thread = t   # start after releasing lock
                    elif _next_submit[0] >= n and len(file_res) >= total:
                        _done_event.set()
                if len(file_res) >= total:
                    _done_event.set()
            if _start_load_thread is not None:
                _start_load_thread.start()
            try:
                import shutil as _sh_cb
                _du_cb = _sh_cb.disk_usage('/dev/shm')
                logger.info(
                    f'TileDispatcher: result {len(file_res)}/{total} '
                    f'SHM={_du_cb.used/2**30:.2f}GB free={_du_cb.free/2**30:.2f}GB')
            except Exception: pass
        except Exception as _cb_exc:
            logger.error(f"TileDispatcher: _on_result callback failed: {_cb_exc}")
            with _lock:
                _errors.append(_cb_exc)
                _done_event.set()

    def _on_error(exc, slot):
        try:
            with _lock:
                _errors.append(exc)
                _done_event.set()
        except Exception as _cb_exc2:
            logger.error(f"TileDispatcher: _on_error callback failed: {_cb_exc2}")
            _done_event.set()

    def _load_and_submit(k_tile, slot):
        """Background: load tile k_tile into slot, then submit immediately."""
        _load_slot(k_tile, slot)
        with _lock:
            if _slot_err[slot] is not None:
                _errors.append(RuntimeError(f"tile load failed: {_slot_err[slot]}"))
                _done_event.set()
                return
            _try_submit(slot)

    # ── Clean stale tile files from previous runs ────────────────────────────
    for _s_idx in range(_n_slots):
        for _stale_f in _slot_paths[_s_idx]:
            try:
                if _os_td.path.exists(_stale_f): _os_td.unlink(_stale_f)
            except OSError: pass

    # ── Bootstrap: load first 2 tiles, submit first 2 ──────────────────────
    # Set _next_load BEFORE starting background thread to avoid race where
    # _on_result fires before main increments _next_load and tries to reload
    # tile 1 a second time.
    _load_slot(0, 0)
    with _lock:
        _next_load[0] = 1         # set under lock before bg thread can read it
        _try_submit(0)
        for _s in range(1, _n_slots):
            if n > _s:
                _next_load[0] = _s + 1
                _bgt = _thr.Thread(target=_load_and_submit, args=(_s, _s), daemon=True)
                _load_threads[_s] = _bgt
                _bgt.start()

    # Guard: if no patches at all, unblock immediately
    if total == 0:
        _done_event.set()

    # Main thread waits — workers run freely, callbacks drive all submissions.
    # Watchdog: if a worker is OOM-killed its result never arrives and
    # _done_event never fires. Poll every 60s and check pool worker pids.
    import os as _os_wd
    _timeout_s = 60    # seconds between liveness checks
    while not _done_event.wait(timeout=_timeout_s):
        # Check whether any pool workers have died unexpectedly
        try:
            _alive = []
            for _w in pool._pool:
                try:
                    _os_wd.kill(_w.pid, 0)  # signal 0 = liveness check
                    _alive.append(_w.pid)
                except (ProcessLookupError, PermissionError):
                    pass  # process gone
            _n_expected = pool._processes
            if len(_alive) < _n_expected and len(file_res) < total:
                _dead = _n_expected - len(_alive)
                try:
                    import shutil as _shu_wd
                    _shm_info = (f" SHM={_shu_wd.disk_usage(_shm_dir).used/2**30:.1f}"
                                 f"/{_shu_wd.disk_usage(_shm_dir).total/2**30:.0f}GB")
                except Exception:
                    _shm_info = ""
                _msg = (f"TileDispatcher watchdog: {_dead} of {_n_expected} "
                        f"workers died (OOM-kill?). "
                        f"Completed {len(file_res)}/{total} patches.{_shm_info} "
                        f"Reduce n_processes (currently {_n_expected}) "
                        f"or free RAM before running.")
                logger.error(_msg)
                with _lock:
                    _errors.append(RuntimeError(_msg))
                    _done_event.set()
        except Exception as _wd_exc:
            logger.warning(f"TileDispatcher watchdog check failed: {_wd_exc}")
    # ── Cleanup slot files (runs on both success and error paths) ──────────
    # Slot files linger on /dev/shm after dispatch — up to 22 GB for 2 slots.
    # Unlink them now; workers have closed their handles since _done_event
    # fires only after all results are collected.
    if _Yr is not None: del _Yr
    if _movie_f is not None: del _movie_f
    if _filt_full is not None: del _filt_full

    for _s_idx in range(_n_slots):
        for _slot_f in _slot_paths[_s_idx]:
            try:
                if _os_td.path.exists(_slot_f):
                    _os_td.unlink(_slot_f)
                    logger.debug(f'TileDispatcher: freed slot file {_slot_f}')
            except OSError:
                pass

    if _errors:
        raise _errors[0]

    return file_res


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

    # SHM causes OOM: 8 workers × 27 GB SHM mapped = 232 GB apparent RSS.
    # Linux OOM killer counts POSIX SHM in every process that maps it.
    # Use warm file-backed page cache instead — same DRAM speed, no OOM.
    use_shm_for_cnmf = False
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
        # Pre-populate from persistent file on disk if in-memory cache is absent.
        _cached_precomp = params.get('init', 'precomp_cache')
        if _cached_precomp is None or not os.path.exists(
                _cached_precomp.get('filtered_path', '')):
            # Derive persistent filename from movie path
            import re as _re_mb
            _mb = os.path.splitext(os.path.basename(file_name))[0]
            _mb = _re_mb.sub(r'[_-]d1_\d+.*$', '', _mb)
            _mb = _re_mb.sub(r'[_-](cnmf|raw_mp|rig).*$', '', _mb)
            from caiman.paths import get_tempdir as _get_td
            _fname_filt = (f"{_mb}_precomp_filt"
                           f"_d1_{dims[0]}_d2_{dims[1]}_d3_1"
                           f"_order_F_frames_{T}_f16.mmap")
            # Check /dev/shm first (written there when CAIMAN_SHM is set)
            _shm_dir = os.environ.get('CAIMAN_SHM', '')
            _pers_path = None
            for _td in [_shm_dir, _get_td()]:
                if _td and os.path.isdir(_td):
                    _cand = os.path.join(_td, _fname_filt)
                    if os.path.exists(_cand):
                        _pers_path = _cand
                        break
            if _pers_path is None:
                _pers_path = os.path.join(_get_td(), _fname_filt)  # fallback
            if os.path.exists(_pers_path):
                logger.info(
                    f'run_CNMF_patches: reusing persistent precomp '
                    f'({_pers_path})')
                # filt_full kept on NVMe (CAIMAN_TEMP) — not copied to /dev/shm.
                # Copying 14.5 GB to tmpfs adds swappable pressure that fills
                # the 27 GB swap partition. posix_fadvise(DONTNEED) after each
                # tile read keeps FUSE page cache clean instead.
                _cached_precomp = {'filtered_path': _pers_path,
                                   'filt_dtype':    'float16',
                                   'd1': dims[0], 'd2': dims[1], 'T': T,
                                   'filt_order': 'C'}
                # Load companion arrays (sn_full, pnr_full etc.) so
                # workers skip NaN scan and noise FFT.
                _npz_path = os.path.splitext(_pers_path)[0] + '_meta.npz'
                if os.path.exists(_npz_path):
                    try:
                        import numpy as _np_npz
                        _npz = _np_npz.load(_npz_path, allow_pickle=False)
                        _sn_candidate = _npz['sn_full']
                        if _sn_candidate.shape == (dims[0], dims[1]):
                            _cached_precomp['sn_full']       = _sn_candidate
                            _cached_precomp['data_max_full'] = _npz['data_max_full']
                            _cached_precomp['cn_full']  = (
                                _npz['cn_full'] if _npz['cn_full'].size > 0
                                else None)
                            _cached_precomp['pnr_full']      = _npz['pnr_full']
                            logger.info(
                                'run_CNMF_patches: loaded companion sn/Cn/PNR arrays')
                        else:
                            logger.warning(
                                f'run_CNMF_patches: companion npz shape '
                                f'{_sn_candidate.shape} != movie dims '
                                f'{(dims[0], dims[1])} — ignoring stale npz')
                    except Exception as _npz_exc:
                        logger.debug(f'companion npz load failed: {_npz_exc}')
                params.init['precomp_cache'] = _cached_precomp
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
                    chunk_frames      = int(
                        params.get('init', 'precompute_chunk_frames') or 1000),
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

    # Tile dispatcher reads data per-tile on demand.
    # No full C-order warm — that wastes 27 GB of RAM that workers need.

    # ── SHM copy: AFTER precompute so filt_full pages are already evicted ──
    # At this point: movie is in page cache (warm), filt_full is evicted.
    # Copying movie → SHM just relabels cache pages → net zero new RAM.
    # Before precompute: movie cache + SHM copy + filt_full writes = 81 GB OOM.
    if use_shm_for_cnmf:
        try:
            logger.info("run_CNMF_patches: loading movie into shared memory …")
            from caiman.shared_memory_utils import SharedMovieBuffer
            # Pass F-order mmap path so streaming uses contiguous frame reads
            # (~14 GB/s) instead of scattered C-order reads (~700 MB/s).
            _forder_path = params.init.get('forder_movie_path')
            _shm_src = _forder_path if _forder_path else file_name
            _shm_buf = SharedMovieBuffer(_shm_src, order='C')
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
            # sn_full/data_max_full may be absent if precomp was loaded from
            # a persistent file without a companion npz (old run or npz missing).
            if _precomp_result.get('sn_full') is not None:
                _patch_precomp['sn'] = _precomp_result['sn_full'][_x0:_x1, _y0:_y1]
            if _precomp_result.get('data_max_full') is not None:
                _patch_precomp['data_max'] = _precomp_result['data_max_full'][_x0:_x1, _y0:_y1]
            # cn/pnr from full-FOV npz are sliced by patch FOV coords, giving shape
            # (_x1-_x0, _y1-_y0).  In the tile path the worker's image data comes
            # from a tile whose column range may be clipped at the movie boundary,
            # so tile_cols < _y1-_y0 for boundary patches → ind_search.shape !=
            # ind_bd.shape inside init_neurons_corr_pnr → IndexError.
            # Only inject cn/pnr when the full filtered mmap exists (precompute
            # succeeded), because in that case workers read their slice directly
            # from the un-clipped filt_full mmap and dimensions always match.
            _filt_ok = bool(_precomp_result.get('filtered_path'))
            if _precomp_result.get('cn_full') is not None and _filt_ok:
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
            # incp_shm: the 5 per-worker /dev/shm mmap files (data_filtered,
            # data_raw, groi_B, groi_B0, computeW_X) are on tmpfs but still
            # consume physical RAM.  They must be added to _analytical before
            # computing the per-worker budget so the safe-worker cap accounts
            # for them.  Without this, the cap is too optimistic and workers
            # cause swap pressure or OOM kills.
            _tsub = params.get('init', 'tsub') or 1
            _incp_shm_b = 5 * _patch_pixels * max(1, T // _tsub) * f32
            _worker_bytes = int((_analytical + _incp_shm_b) * _overhead_frac)
            _vm             = _psu.virtual_memory()
            # Budget: tile dispatcher reads movie one tile at a time.
            # Workers never map the full movie — only the dispatcher holds
            # one tile (tile_d² × T × 4 bytes) + filt tile at a time.
            # Reserve dispatcher overhead + filt_full SHM + parent_rss.
            _tile_d        = 3 * int(params.get("patch", "stride") or 18) + \
                             2 * int(params.get("patch", "rf") or 36) + 1
            _tile_bytes    = int(_tile_d * _tile_d * T * 4)       # images tile
            _filt_tile_b   = int(_tile_d * _tile_d * T * 2)       # filt tile f16
            _filt_full_b   = int(d * T * np.dtype(np.float16).itemsize)  # filt_full SHM
            _movie_bytes   = _tile_bytes + _filt_tile_b + _filt_full_b
            _ram_frac      = float(params.get("patch", "ram_budget_frac") or 0.75)
            _parent_rss    = _psu.Process(_os.getpid()).memory_info().rss
            _budget        = max(0, (_vm.total - _movie_bytes - _parent_rss)
                                  * _ram_frac)
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
            # Warn if /dev/shm is too full for workers
            try:
                import shutil as _shu
                _shm_stat = _shu.disk_usage('/dev/shm')
                _shm_pct  = _shm_stat.used / _shm_stat.total * 100
                if _shm_pct > 70:
                    logger.warning(
                        f"/dev/shm {_shm_stat.used/2**30:.1f}/{_shm_stat.total/2**30:.1f} GB "
                        f"({_shm_pct:.0f}% used) — worker SHM files may cause SIGBUS. "
                        f"Run: rm -f /dev/shm/tmp*.mmap /dev/shm/_caiman_*.mmap")
            except Exception: pass
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
                    f'({n_proc} workers)'
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
                    maxtasksperchild= None,  # workers persist across tiles
                    initializer     = _worker_logging_init,
                    initargs        = (_lp,),
                ) as _patch_pool:
                    file_res = _tile_dispatch(
                        _patch_pool, args_in, file_name,
                        dims, T, _precomp_result, logger,
                        forder_movie_path=params.init.get('forder_movie_path'))
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
            # Apply same movie-cache budget cap as the dview path.
            try:
                import psutil as _psu_fb
                _vm_fb = _psu_fb.virtual_memory()
                # Same tile-aware budget as the dview path above
                _stride_fb = int(params.get('patch', 'stride') or 18)
                _rf_fb     = int(params.get('patch', 'rf') or 36)
                _td_fb     = 3 * _stride_fb + 2 * _rf_fb + 1
                _movie_fb  = int(_td_fb*_td_fb*T*4 + _td_fb*_td_fb*T*2 + d*T*2)
                import os as _os_fb2
                _parent_fb = _psu_fb.Process(_os_fb2.getpid()).memory_info().rss
                _per_worker_fb = int(2.5 * 2**30)  # analytical estimate at tsub=2
                _ram_frac_fb = float(params.get('patch', 'ram_budget_frac') or 0.75)
                _budget_fb = max(0, (_vm_fb.total - _movie_fb - _parent_fb)
                                   * _ram_frac_fb)
                _safe_fb = max(1, int(_budget_fb // _per_worker_fb))
                if _safe_fb < _n_proc_fallback:
                    logger.warning(
                        f'run_CNMF_patches: capping workers '
                        f'{_n_proc_fallback} → {_safe_fb} '
                        f'(movie cache reservation)')
                    _n_proc_fallback = _safe_fb
            except Exception:
                pass
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
                    maxtasksperchild= None,  # workers persist across tiles
                    initializer     = _worker_logging_init,
                    initargs        = (_lp,),
                ) as _patch_pool:
                    file_res = _tile_dispatch(
                        _patch_pool, args_in, file_name,
                        dims, T, _precomp_result, logger,
                        forder_movie_path=params.init.get('forder_movie_path'))
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

            if sn is not None and len(sn) == len(idx_):
                sn_tot[idx_] = sn
            elif sn is not None and len(sn) < len(idx_):
                # Tile-boundary shape mismatch: CNMF fit ran on fewer pixels
                # than idx_ covers (clipped tile data).  Assign what we have
                # to the first len(sn) elements of idx_ — the rest stay zero.
                # This only affects sn_tot (noise map) used for display; it
                # does not affect A, C, or the spatial/temporal estimates.
                logger.warning(
                    f"sn shape mismatch at patch {idx_[0]}: "
                    f"len(sn)={len(sn)} len(idx_)={len(idx_)} — "
                    f"tile boundary clip; partial sn assignment")
                sn_tot[idx_[:len(sn)]] = sn
            else:
                sn_tot[idx_] = sn
            f_tot.append(f)
            bl_tot.append(bl)
            c1_tot.append(c1)
            neurons_sn_tot.append(neurons_sn)
            g_tot.append(g)
            idx_tot.append(idx_)
            shapes_tot.append(shapes)
            mask[idx_] += 1

            # Boundary-tile guard: b.shape[0] may differ from len(idx_)
            # when the tile-clipped fit ran on fewer pixels than idx_ covers.
            # Rebuild a trimmed index that matches the actual fit dimensions.
            _b_npx = b.shape[0] if b is not None and hasattr(b, 'shape') else len(idx_)
            if _b_npx != len(idx_):
                logger.warning(
                    f'b shape mismatch at patch {idx_[0]}: '
                    f'b.shape[0]={_b_npx} len(idx_)={len(idx_)} '
                    f'— tile boundary clip; rebuilding idx_ from fit dims')
                # Use the first _b_npx flat indices from idx_ to stay consistent
                # with how the tile data was delivered (row-major within patch).
                _idx_b = idx_[:_b_npx]
            else:
                _idx_b = idx_
            if scipy.sparse.issparse(b):
                b = scipy.sparse.csc_matrix(b)
                b_tot.append(b.data)
                idx_ptr_B += list(b.indptr[1:] - b.indptr[:-1])
                idx_tot_B.append(_idx_b[b.indices])
            else:
                for ii in range(b.shape[-1]):
                    b_tot.append(b[:, ii])
                    idx_tot_B.append(_idx_b)
                    idx_ptr_B.append(len(_idx_b))
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

            _a_npx = A.shape[0] if A is not None else len(idx_)
            _idx_a = idx_[:_a_npx] if _a_npx != len(idx_) else idx_
            if _a_npx != len(idx_):
                logger.warning(
                    f'A shape mismatch at patch {idx_[0]}: '
                    f'A.shape[0]={_a_npx} len(idx_)={len(idx_)}')
            for ii in range(A.shape[-1]):
                new_comp = A[:, ii]  # / np.sqrt(A[:, ii].power(2).sum())
                if new_comp.sum() > 0:
                    a_tot.append(new_comp.toarray().flatten())
                    idx_tot_A.append(_idx_a)
                    idx_ptr_A.append(len(_idx_a))
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
