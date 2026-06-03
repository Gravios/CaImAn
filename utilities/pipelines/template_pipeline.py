"""
CaImAn 2-photon pipeline template: motion correction → CNMF → dF/F
====================================================================

Usage
-----
1. Copy this file and the companion ``_pipeline.json`` into the session folder,
   renaming both with the session stem::

       cp pipelines/template_pipeline.py  /data/src/<lab>/<exp>/<session>_pipeline.py
       cp pipelines/template_pipeline.json /data/src/<lab>/<exp>/<session>_pipeline.json

2. Edit the JSON — set ``session.data_root``, ``session.experiment``, and
   tune the ``cnmf`` / ``quality`` / ``gpu`` sections for the dataset.

3. Run::

       python <session>_pipeline.py

The session name is the script filename stem minus the ``_pipeline`` suffix.
All output files (log, QC images, mmaps, results, report) use this prefix
automatically — no edits to this script are needed.

See docs/ for the full parameter reference and troubleshooting guide.
"""

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Runs in EVERY process (parent and workers).  Keep it minimal and idempotent:
# only env-var application and stdlib imports belong here.  The guard
# `if __name__ == "__main__":` below ensures that all pipeline execution
# (logging, data loading, CNMF) runs only in the parent process.
#
# Why: multiprocessing `spawn`/`forkserver` starts workers by re-importing
# this script as __main__.  Any top-level code outside the guard re-executes
# in each worker — causing the pipeline body to run 9× (once per worker).
import os
import json as _j
from pathlib import Path

from caiman.utils.pipeline_setup import resolve_pipeline_path

_SCRIPT_PATH, _CONFIG_PATH, session = resolve_pipeline_path()
_SCRIPT_DIR = _SCRIPT_PATH.parent

# Apply env section before any caiman import
try:
    _env = _j.load(open(_CONFIG_PATH)).get("env", {})
except Exception as _e:
    import warnings
    warnings.warn(f"{_SCRIPT_PATH.name}: env load failed: {_e}")
    _env = {}

_FORCE = {"CAIMAN_DATA", "CAIMAN_TEMP", "CAIMAN_SHM", "CAIMAN_TILE_SLOTS"}
for _k, _v in _env.items():
    if _k in _FORCE:
        os.environ[_k] = str(_v)
    else:
        os.environ.setdefault(_k, str(_v))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

CAIMAN_DATA = os.environ.get("CAIMAN_DATA", "/data/caiman")
CAIMAN_TEMP = os.environ.get("CAIMAN_TEMP", "/data/caiman/temp")
CAIMAN_SHM  = os.environ.get("CAIMAN_SHM",  "/dev/shm")
for _d in [CAIMAN_DATA, os.path.join(CAIMAN_DATA, "model"), CAIMAN_TEMP]:
    os.makedirs(_d, exist_ok=True)

# ── Everything below runs in the PARENT PROCESS ONLY ─────────────────────────
if __name__ == "__main__":
    import gc
    import glob
    import warnings

    import numpy as np
    import cv2
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in remainder",
        category=RuntimeWarning,
        module=r"scipy\.sparse\._dia",
    )

    import caiman as cm
    import caiman.summary_images as csi
    from caiman.motion_correction import MotionCorrect

    from caiman.utils.tiff_io        import ensure_multipage_tiff, fc_convert_parallel
    from caiman.utils.xcorr_correction import correct_line_scan
    from caiman.utils.params_io      import load_pipeline_params
    from caiman.utils.param_summary  import log_params
    from caiman.utils.pipeline_setup import ensure_model_files, setup_logging, clean_stale_shm
    from caiman.utils.timing         import PipelineTimer, write_report
    from caiman.utils.memory         import malloc_trim, cupy_flush, cupy_register_cleanup
    from caiman.utils.cnmf_runner    import CNMFRunner
    from caiman.utils.qc             import QCRunner
    from caiman.utils.oscillation    import OscillationAnalyzer
    from caiman.utils.shm_movie      import load_to_shm, release_shm, check_shm_capacity

    import dill as _dill
    import multiprocessing.reduction as _mpr
    _mpr.ForkingPickler.dumps = _dill.dumps

    # ── Session identity ──────────────────────────────────────────────────────
    _P     = load_pipeline_params(_CONFIG_PATH)
    outdir = _SCRIPT_PATH.parent

    # ── Infrastructure ────────────────────────────────────────────────────────
    logger         = setup_logging(outdir / f"{session}.log")
    _cnn_available = ensure_model_files(os.path.join(CAIMAN_DATA, "model"))
    log_params(_P, logger, session=session)

    ADD_BASELINE = _P.data.add_baseline
    border_nan   = _P.motion_correction.border_nan

    timer = PipelineTimer(logger)
    qc    = QCRunner(_P, session, outdir)

    # ── 1. Input-file check ──────────────────────────────────────────────────
    # Prefer .tif when present, fall back to .msr (Imspector). cm.load_iter
    # dispatches on extension, so downstream code works for either input.
    _tif_in = outdir / f"{session}.tif"
    _msr_in = outdir / f"{session}.msr"
    if _tif_in.exists():
        fnames = str(_tif_in)
    elif _msr_in.exists():
        fnames = str(_msr_in)
    else:
        raise FileNotFoundError(
            f"Neither {_tif_in.name} nor {_msr_in.name} found in {outdir}"
        )

    @timer.step("Input format check")
    def check_tiff():
        global fnames
        fnames = ensure_multipage_tiff(fnames)

    @timer.step("QC: raw sample")
    def qc_raw():
        qc.raw_sample(fnames)

    check_tiff()
    qc_raw()

    # ── 1b. Noise correction ───────────────────────────────────────────────
    # Diagnose noise sources on the raw stack, build a correction recipe from
    # flagged sources at or above min_level, apply via streaming wrapper.
    # Writes <session>_Ncorrected.tif and uses it for all downstream
    # processing. Runs BEFORE xcorr_correction so that hot-pixel replacement
    # and row-pedestal subtraction operate on the raw data; if the noise
    # recipe includes sub-pixel correct_bidirectional, xcorr will see
    # residual ~0 shift below and no-op.
    #
    # Skipped entirely (no file rewrite) if the diagnostic finds nothing
    # actionable at min_level, so this block can stay enabled across many
    # sessions safely.
    _nc_cfg    = getattr(_P, "noise_correction", None)
    _nc_enable = bool(getattr(_nc_cfg, "enabled", False)) if _nc_cfg else False
    _nc_applied = False
    _nc_recipe  = []

    if _nc_enable:
        _nc_min_level = str(getattr(_nc_cfg, "min_level", "moderate"))
        _nc_chunk     = int(getattr(_nc_cfg, "chunk_frames", 500))
        _nc_dtype     = str(getattr(_nc_cfg, "out_dtype", "same"))
        _nc_n_frames  = int(getattr(_nc_cfg, "diagnostic_n_frames", 3000))
        _nc_smode     = str(getattr(_nc_cfg, "diagnostic_sampling_mode",
                                     "contiguous"))
        _nc_report = None

        @timer.step("Noise diagnostic")
        def run_noise_diagnostic():
            global _nc_report
            from utilities.noise.noise_diagnostics import run_diagnostics
            _nc_report = run_diagnostics(
                fnames,
                out_dir=str(outdir / "diag"),
                n_frames=_nc_n_frames,
                fs_hz=float(_P.data.fr),
                sampling_mode=_nc_smode,
            )

        run_noise_diagnostic()

        # Determine whether any flagged source warrants correction.
        _LEVELS = ["negligible", "low", "moderate", "high"]
        _cutoff = _LEVELS.index(_nc_min_level)
        # Skip sources without registered corrections or that are commonly
        # false-positive on cellular data.
        _skip = {"shot_noise_dominated",
                  "quantization_loss", "saturation_clipping",
                  "photobleaching", "illumination_drift_increase",
                  "frame_discontinuity",
                  "galvo_flyback_edge"}
        _flagged = {n: d for n, d in _nc_report.get("sources", {}).items()
                    if _LEVELS.index(d["level"]) >= _cutoff and n not in _skip}

        # fast_axis_periodic has an asymmetric trigger inside
        # recommend_corrections: any level >= 'low' triggers a column-
        # pedestal correction, regardless of min_level. Reflect that here
        # so the step isn't skipped when fast_axis_periodic is the only
        # actionable flag.
        _fap = _nc_report.get("sources", {}).get("fast_axis_periodic")
        if _fap and _LEVELS.index(_fap["level"]) >= _LEVELS.index("low"):
            _flagged.setdefault("fast_axis_periodic", _fap)

        if _flagged:
            logger.info(
                f"Noise diagnostic flagged at >= {_nc_min_level}: "
                f"{sorted(_flagged.keys())}")

            @timer.step("Noise correction")
            def run_noise_correction():
                global fnames, _nc_applied, _nc_recipe
                from utilities.noise.noise_correction import (
                    correct_stack_file, recommend_corrections)
                _nc_recipe = [fn.__name__ for fn, _ in
                               recommend_corrections(_nc_report,
                                                       min_level=_nc_min_level)]
                logger.info(f"Noise correction recipe: {_nc_recipe}")
                fnames = correct_stack_file(
                    fnames,
                    report=_nc_report,
                    chunk_frames=_nc_chunk,
                    out_dtype=_nc_dtype,
                    logger=logger,
                )
                _nc_applied = True
                logger.info(f"Noise correction done: {Path(fnames).name}")

            run_noise_correction()
        else:
            logger.info(
                f"Noise diagnostic clean at >= {_nc_min_level}; "
                f"skipping correction.")

    # ── 1c. Line-scan phase correction ──────────────────────────────────────
    # Resonant and galvo-resonant scanners acquire alternating rows in opposite
    # directions.  A mechanical phase delay causes a consistent column shift
    # between even and odd rows (the "comb" artefact).  Correct before MC so
    # that registration targets a clean reference frame.
    # Save original TIFF path so the xcorr QC figure can show before/after
    fnames_orig = fnames

    _xcorr_cfg    = getattr(_P, "xcorr_correction", None)
    _xcorr_enable = bool(getattr(_xcorr_cfg, "enabled", False)) if _xcorr_cfg else False

    if _xcorr_enable:
        _xcorr_max_shift = int(getattr(_xcorr_cfg, "max_shift", 16))
        _xcorr_n_frames  = int(getattr(_xcorr_cfg, "n_frames",  500))

        @timer.step("Line-scan X correction")
        def run_xcorr():
            global fnames
            fnames = correct_line_scan(
                fnames,
                max_shift = _xcorr_max_shift,
                n_frames  = _xcorr_n_frames,
                use_gpu   = bool(getattr(_xcorr_cfg, "use_gpu", True)),
                logger    = logger,
            )
            logger.info(f"Line-scan correction done: {Path(fnames).name}")

        run_xcorr()

        @timer.step("QC: line-scan correction")
        def qc_xcorr():
            qc.xcorr_correction(
                fnames_orig, fnames,
                max_shift=_xcorr_max_shift,
                n_frames=_xcorr_n_frames,
            )

        qc_xcorr()

    # mc_stem: stem of the TIFF actually fed into motion correction.
    # When xcorr correction is enabled this is e.g. "session_Xcorrected";
    # otherwise it is the bare session name.  All downstream glob patterns
    # use mc_stem so they match the correct mmap files.
    mc_stem = Path(fnames).stem

    # ── 2. Motion correction ──────────────────────────────────────────────────
    _mc_existing = sorted(glob.glob(
        os.path.join(CAIMAN_TEMP, f"{mc_stem}_rig*order_F*.mmap")))

    if _mc_existing:
        fname_mc, shifts_rig = _mc_existing[-1], [(0, 0)]
        logger.info(f"Reusing MC mmap: {fname_mc}")
    else:
        mc = MotionCorrect(fnames, dview=None, use_gpu=True, nonneg_movie=True,
                           **{k: v for k, v in _P.motion_correction.items()
                              if not k.startswith("_")})
        # Pass GPU batch size from JSON gpu section (None = auto from VRAM)
        mc.gpu_batch_size = (
            int(getattr(getattr(_P, "gpu", None), "mc_batch_size", None) or 0)
            or None
        )

        @timer.step("Motion correction")
        def run_mc():
            mc.motion_correct(save_movie=True)

        @timer.step("QC: motion correction")
        def qc_mc():
            qc.motion_correction(mc)

        run_mc()
        fname_mc, shifts_rig = mc.mmap_file[0], mc.shifts_rig
        logger.info(f"MC done: {fname_mc}")
        qc_mc()
        del mc

    bord_px = 0 if border_nan == "copy" else int(np.ceil(np.max(np.abs(shifts_rig))))

    # ── 3. F→C mmap conversion ────────────────────────────────────────────────
    _cnmf_existing = sorted(glob.glob(
        os.path.join(CAIMAN_TEMP, f"{mc_stem}_cnmf*order_C*.mmap")))

    if _cnmf_existing and os.path.getmtime(_cnmf_existing[-1]) >= os.path.getmtime(fname_mc):
        fname_cnmf = _cnmf_existing[-1]
        _, dims, T = cm.mmapping.load_memmap(fname_mc)
        logger.info(f"Reusing C-order mmap: {fname_cnmf}")
    else:
        Yr_F, dims, T = cm.mmapping.load_memmap(fname_mc)
        n_px = int(np.prod(dims))
        fname_cnmf = cm.paths.fn_relocated(
            cm.paths.memmap_frames_filename(mc_stem + "_cnmf", dims, T, "C"),
            force_temp=True,
        )
        Yr_C = np.memmap(fname_cnmf, mode="w+", dtype=np.float32,
                         shape=cm.mmapping.prepare_shape((n_px, T)), order="C")

        @timer.step("F→C mmap conversion")
        def convert():
            fc_convert_parallel(Yr_F, Yr_C, n_px, T, ADD_BASELINE, logger)

        convert()
        del Yr_C, Yr_F

    # ── 4. Correlation image ──────────────────────────────────────────────────
    Yr, dims, T = cm.mmapping.load_memmap(fname_cnmf)
    images = np.reshape(Yr.T, [T] + list(dims), order="F")
    images.filename = Yr.filename
    logger.info(f"Data: {images.shape}  dtype={images.dtype}")

    # ── 4b. Temporal detrend ───────────────────────────────────────────────
    # Remove a slow per-pixel polynomial trend (default linear) before Cn /
    # CNMF.  SUPPORT preserves the brightness-decay ramp; left in, every pixel
    # shares one dominant slow component and the local-correlation image
    # saturates (Cn -> ~1, min_corr gate dead).  Detrending removes the ramp
    # and restores Cn contrast.  Written back to the C-order mmap so both the
    # seeded (single-FOV) and patch paths read detrended data.
    _dt_cfg = getattr(_P, "detrend", None)
    if bool(getattr(_dt_cfg, "enabled", False)):
        @timer.step("Temporal detrend")
        def run_detrend():
            global images, Yr
            from utilities.noise.noise_correction import detrend_temporal
            _order = int(getattr(_dt_cfg, "order", 1))
            _pm    = bool(getattr(_dt_cfg, "preserve_mean", True))
            _gpu   = bool(getattr(_dt_cfg, "use_gpu", True))
            _npix  = Yr.shape[0]
            Yr_rw  = np.memmap(fname_cnmf, mode="r+", dtype=np.float32,
                               shape=Yr.shape, order="C")
            _CH = 8192
            for _i in range(0, _npix, _CH):
                _blk = np.asarray(Yr_rw[_i:_i + _CH])               # (n, T)
                _d = detrend_temporal(_blk.T[:, :, None],
                                      order=_order, preserve_mean=_pm,
                                      use_gpu=_gpu)
                Yr_rw[_i:_i + _CH] = _d[:, :, 0].T                  # (n, T)
            Yr_rw.flush(); del Yr_rw
            Yr, _dims_dt, _T_dt = cm.mmapping.load_memmap(fname_cnmf)
            images = np.reshape(Yr.T, [T] + list(dims), order="F")
            images.filename = Yr.filename
            logger.info(f"Temporal detrend applied: order={_order} "
                        f"preserve_mean={_pm} use_gpu={_gpu}")
        run_detrend()

    cupy_flush(logger, label="before summary-image step")

    @timer.step("Correlation image (Cn)")
    def compute_cn():
        global Cn
        Cn = csi.local_correlations_fft(images[::5], swap_dim=False)
        Cn[np.isnan(Cn)] = 0
        np.save(str(outdir / f"{session}_Cn.npy"), Cn)

    @timer.step("QC: correlation image")
    def qc_cn():
        qc.correlation_image(Cn)

    compute_cn()
    qc_cn()

    # Release Cn-step RSS so filt_full writes don't compete for page cache
    del images, Yr
    gc.collect()
    malloc_trim(logger)

    # ── 4b. Load movie into shared memory ─────────────────────────────────────
    # Always attempt to copy the C-order mmap into /dev/shm so patch workers
    # read from RAM instead of NVMe.  Falls back to disk transparently if SHM
    # lacks sufficient free space (logs a warning).
    import psutil as _psutil
    _shm_dir   = os.environ.get("CAIMAN_SHM", "/dev/shm")
    _shm_path  = None
    _avail, _fits = check_shm_capacity(os.path.getsize(fname_cnmf), _shm_dir)
    if _fits:
        @timer.step("SHM: copy movie to shared memory")
        def _load_shm():
            global _shm_path, fname_cnmf
            _shm_path  = load_to_shm(fname_cnmf, session, _shm_dir, logger)
            fname_cnmf = _shm_path
        _load_shm()
    else:
        logger.warning(
            f"SHM: need {os.path.getsize(fname_cnmf)/1024**3:.1f} GB, "
            f"only {_avail/1024**3:.1f} GB free in {_shm_dir} — "
            f"using disk-backed mmap."
        )

    # Use explicit cluster.n_processes when set; fall back to all logical cores
    # (logical=True — hyperthreads help on NNLS workloads) when in SHM mode,
    # or the JSON value when falling back to disk.
    # Two-step getattr: tolerates missing `cluster` section AND missing
    # `n_processes` key inside it.  ParamBag.__getattr__ raises
    # AttributeError on missing keys, so the previous one-liner
    #   _json_n = getattr(_P, "cluster", None) and _P.cluster.n_processes or None
    # crashed when the section existed but the key did not.
    _cluster_cfg = getattr(_P, "cluster", None)
    _json_n      = getattr(_cluster_cfg, "n_processes", None) if _cluster_cfg else None
    _cluster_n = (
        _json_n or (_psutil.cpu_count(logical=True) or os.cpu_count())
        if _shm_path else
        _json_n
    )

    Yr, dims, T = cm.mmapping.load_memmap(fname_cnmf)
    images = np.reshape(Yr.T, [T] + list(dims), order="F")
    images.filename = Yr.filename

    if _shm_path:
        # Collapse all patches into a single tile so workers receive the full
        # patch list at once with no inter-tile waiting.
        # tile_n × stride must cover max(d1, d2); 1 slot = no prefetch needed.
        import math as _math
        _stride = int(_P.cnmf.stride)
        _tile_n = _math.ceil(max(dims) / _stride)
        os.environ["CAIMAN_TILE_N"]     = str(_tile_n)
        os.environ["CAIMAN_TILE_SLOTS"] = "1"
        logger.info(
            f"SHM: tile_n={_tile_n}, stride={_stride} — "
            f"all patches in one {dims[0]}×{dims[1]} tile, 1 slot"
        )

    # Register CuPy atexit cleanup to prevent CUDA_ERROR_ILLEGAL_ADDRESS
    # at process exit when torch (CUDA 13) and cupy-cuda12x coexist.
    cupy_register_cleanup()
    clean_stale_shm(CAIMAN_SHM, CAIMAN_TEMP, logger)
    logger.info("Starting CNMF cluster")
    _, dview, n_processes = cm.cluster.setup_cluster(
        backend="multiprocessing",
        n_processes=_cluster_n,
        single_thread=False,
    )

    # ── Anatomical seeding (census-complete extraction) ────────────────────
    # When enabled, detect cells on a summary projection (or load an external
    # label image) and seed CNMF with them.  Bypasses corr_pnr/min_corr
    # detection (degenerate on SUPPORT-denoised data) and keeps silent cells
    # for an inactive-cell count.  Seeding forces single-FOV extraction
    # (run_CNMF_patches ignores Ain) — handled inside CNMFRunner.fit.
    _seed = None
    _seed_cfg = getattr(_P, "seeding", None)
    if bool(getattr(_seed_cfg, "enabled", False)):
        @timer.step("Anatomical seeding")
        def build_seed():
            global _seed
            from caiman.utils import seeding as _sd
            _src = str(getattr(_seed_cfg, "source", "projection"))
            if _src == "label_file":
                _lbl = _sd.load_label_image(str(_seed_cfg.label_file))
            else:
                _proj = _sd.summary_projection(
                    images,
                    kind=str(getattr(_seed_cfg, "projection", "max_div_mean")),
                    q=getattr(_seed_cfg, "percentile", 90.0))
                np.save(str(outdir / f"{session}_seed_projection.npy"), _proj)
                _segmenter = str(getattr(_seed_cfg, "segmenter", "cellpose"))
                if _segmenter == "watershed":
                    _lbl = _sd.segment_watershed(
                        _proj,
                        min_distance=int(getattr(_seed_cfg, "min_distance", 5)),
                        threshold_rel=float(getattr(_seed_cfg,
                                                    "threshold_rel", 0.2)),
                        use_otsu=bool(getattr(_seed_cfg, "use_otsu", False)),
                        smooth_sigma=float(getattr(_seed_cfg,
                                                   "smooth_sigma", 1.0)),
                        min_pixels=int(getattr(_seed_cfg, "min_pixels", 8)))
                elif _segmenter == "peaks":
                    _lbl = _sd.segment_peaks(
                        _proj,
                        min_distance=int(getattr(_seed_cfg, "min_distance", 5)),
                        radius=int(getattr(_seed_cfg, "radius", 4)),
                        threshold_rel=float(getattr(_seed_cfg,
                                                    "threshold_rel", 0.2)),
                        use_otsu=bool(getattr(_seed_cfg, "use_otsu", False)),
                        smooth_sigma=float(getattr(_seed_cfg,
                                                   "smooth_sigma", 1.0)),
                        min_pixels=int(getattr(_seed_cfg, "min_pixels", 8)))
                else:
                    _lbl = _sd.segment_anatomical(
                        _proj,
                        diameter=getattr(_seed_cfg, "diameter", None),
                        flow_threshold=float(getattr(_seed_cfg,
                                                     "flow_threshold", 0.4)),
                        cellprob_threshold=float(getattr(_seed_cfg,
                                                         "cellprob_threshold", 0.0)),
                        gpu=bool(getattr(_seed_cfg, "use_gpu", True)))
                np.save(str(outdir / f"{session}_seed_labels.npy"), _lbl)
            _Ain = _sd.masks_to_Ain(
                _lbl, dims=dims,
                min_pixels=int(getattr(_seed_cfg, "min_pixels", 8)))
            _Cin, _b, _f = _sd.complete_seed(
                _Ain, Yr, nb=int(getattr(_seed_cfg, "nb", 2)))
            _seed = (_Ain, _Cin, _b, _f)
            logger.info(f"Anatomical seeding: {_Ain.shape[1]} ROIs seeded "
                        f"(single-FOV; corr_pnr/min_corr bypassed)")
        build_seed()

    runner = CNMFRunner(
        _P, session, outdir,
        fname_mc      = fname_mc,
        fname_cnmf    = fname_cnmf,
        dims          = dims,
        bord_px       = bord_px,
        dview         = dview,
        n_processes   = n_processes,
        cnn_available = _cnn_available,
    )

    try:
        cnm2 = runner.run_all(images, Yr=Yr, qc=qc, Cn=Cn, timer=timer, seed=_seed)
        del images, Yr, Cn

    finally:
        logger.info("Stopping cluster")
        cm.stop_server(dview=dview)
        if _shm_path:
            release_shm(_shm_path, logger)

    # ── 6. Report ─────────────────────────────────────────────────────────────
    # Component counts for the report
    _n_accepted = _n_rejected = _n_total = None
    try:
        _n_total    = int(cnm2.estimates.A.shape[1])
        _idx_acc    = cnm2.estimates.idx_components
        _idx_rej    = cnm2.estimates.idx_components_bad
        _n_accepted = int(len(_idx_acc)) if _idx_acc is not None else _n_total
        _n_rejected = int(len(_idx_rej)) if _idx_rej is not None else 0
    except Exception:
        pass

    _extra = {}
    if _n_total    is not None: _extra["Components (total)"]    = _n_total
    if _n_accepted is not None: _extra["Components (accepted)"] = _n_accepted
    if _n_rejected is not None: _extra["Components (rejected)"] = _n_rejected
    try:
        _census = getattr(cnm2.estimates, "census", None)
        if _census:
            _extra["Census (total ROIs)"] = _census["n_total"]
            _extra["Census (active)"]     = _census["n_active"]
            _extra["Census (inactive)"]   = _census["n_inactive"]
    except Exception:
        pass
    if _nc_enable:
        if _nc_applied:
            _extra["Noise correction"] = (
                f"applied  ({', '.join(_nc_recipe)})")
        else:
            _extra["Noise correction"] = (
                f"enabled, no flags at >= {_nc_min_level}")
    if _xcorr_enable:
        _extra["Line-scan X shift"] = (
            f"enabled  (max_shift={_xcorr_max_shift} px)")

    # ── 7. Oscillation analysis ───────────────────────────────────────────
    _osc_npz = None
    _osc_cfg = getattr(_P, "oscillation", None)
    _osc_enable = bool(getattr(_osc_cfg, "enabled", True))

    if _osc_enable:
        _osc_fs       = float(_P.data.fr)
        _osc_NW       = float(getattr(_osc_cfg, "NW",       4.0))
        _osc_win_s    = float(getattr(_osc_cfg, "win_s",    4.0))
        _osc_overlap  = getattr(_osc_cfg, "overlap_s", None)
        _osc_overlap  = float(_osc_overlap) if _osc_overlap is not None else None
        _osc_gpu      = bool(getattr(_osc_cfg, "use_gpu",   True))
        _osc_adaptive = bool(getattr(_osc_cfg, "adaptive",  True))

        @timer.step("Oscillation analysis")
        def run_oscillation():
            global _osc_npz
            osc = OscillationAnalyzer(
                cnm2.estimates,
                fs       = _osc_fs,
                NW       = _osc_NW,
                adaptive = _osc_adaptive,
                use_gpu  = _osc_gpu,
            )
            summary = osc.run_all(
                output_dir = str(outdir),
                session_id = session,
                win_s      = _osc_win_s,
                overlap_s  = _osc_overlap,
            )
            _osc_npz = str(outdir / f"{session}_oscillations.npz")
            logger.info(f"Oscillation analysis done: {_osc_npz}")
            return summary

        @timer.step("QC: oscillation")
        def qc_oscillation_step():
            if _osc_npz:
                qc.oscillation(_osc_npz)

        run_oscillation()
        qc_oscillation_step()

    # ── 8. Report ─────────────────────────────────────────────────────────
    if _osc_enable and _osc_npz:
        _extra["Oscillation NPZ"] = Path(_osc_npz).name

    write_report(timer, session, outdir, logger, extra=_extra)
