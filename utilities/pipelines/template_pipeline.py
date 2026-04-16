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
import sys
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
    import caiman.mmapping
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

    # ── 1. TIFF check ─────────────────────────────────────────────────────────
    fnames = str(outdir / f"{session}.tif")

    @timer.step("TIFF format check")
    def check_tiff():
        global fnames
        fnames = ensure_multipage_tiff(fnames)

    @timer.step("QC: raw sample")
    def qc_raw():
        qc.raw_sample(fnames)

    check_tiff()
    qc_raw()

    # ── 1b. Line-scan phase correction ──────────────────────────────────────
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
    _json_n = getattr(_P, "cluster", None) and _P.cluster.n_processes or None
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
        cnm2 = runner.run_all(images, Yr=Yr, qc=qc, Cn=Cn, timer=timer)
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
