#!/usr/bin/env python
"""
Two-photon batch CNMF pipeline demo
=====================================
Updated for Gravios/CaImAn fork.  Replaces ``demos/general/demo_pipeline.py``.

Changes from the upstream demo
-------------------------------
- ``if __name__ == "__main__":`` guard prevents worker re-execution under
  multiprocessing ``spawn``
- GPU motion correction (``use_gpu=True``, no cluster)
- ``fc_convert_parallel`` for memory-safe F→C conversion (single-pass slabs)
- ``cupy_flush`` + ``madvise_dontneed`` + ``malloc_trim`` before CNMF
- ``CNMFRunner`` replaces manual fit → refit → evaluate → select → dF/F
- ``QCRunner`` produces headless PNG figures at each step
- ``PipelineTimer`` records timing and resource usage per step
- ``write_report`` writes ``<session>_report.txt`` and ``_report.json``

Usage
-----
    python demos/demo_pipeline_2p.py

Downloads the Sue_2x_3000_40_-46.tif demo dataset automatically via
``caiman.utils.utils.download_demo`` if not already present.
"""

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Must run in every process (parent + workers).  Keep stdlib-only.
import os
import sys

os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("MPLBACKEND",           "Agg")

# ── Main pipeline ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import gc
    import json
    import warnings
    from pathlib import Path

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
    from caiman.utils.utils          import download_demo

    from caiman.utils.tiff_io        import fc_convert_parallel
    from caiman.utils.params_io      import ParamBag
    from caiman.utils.pipeline_setup import setup_logging, ensure_model_files, clean_stale_shm
    from caiman.utils.timing         import PipelineTimer, write_report
    from caiman.utils.memory         import malloc_trim, madvise_dontneed, cupy_flush
    from caiman.utils.cnmf_runner    import CNMFRunner
    from caiman.utils.qc             import QCRunner

    import dill as _dill
    import multiprocessing.reduction as _mpr
    _mpr.ForkingPickler.dumps = _dill.dumps

    # ── Paths and session ─────────────────────────────────────────────────
    CAIMAN_DATA = os.environ.get("CAIMAN_DATA", cm.paths.caiman_datadir())
    CAIMAN_TEMP = os.environ.get("CAIMAN_TEMP", os.path.join(CAIMAN_DATA, "temp"))
    CAIMAN_SHM  = os.environ.get("CAIMAN_SHM",  "/dev/shm")

    for _d in [CAIMAN_DATA, os.path.join(CAIMAN_DATA, "model"), CAIMAN_TEMP]:
        os.makedirs(_d, exist_ok=True)

    outdir  = Path(CAIMAN_DATA) / "demo_2p_output"
    outdir.mkdir(exist_ok=True)
    session = "demo_2p"

    logger         = setup_logging(outdir / f"{session}.log")
    _cnn_available = ensure_model_files(os.path.join(CAIMAN_DATA, "model"))
    timer          = PipelineTimer(logger)

    # ── Download demo data ────────────────────────────────────────────────
    fname = download_demo("Sue_2x_3000_40_-46.tif")
    logger.info(f"Input: {fname}")

    # ── Parameters ───────────────────────────────────────────────────────
    # Build a minimal ParamBag directly from a dict.
    # In a real session this comes from load_pipeline_params("session_pipeline.json").
    _params_dict = {
        "data": {
            "fr": 15,
            "decay_time": 0.4,
            "add_baseline": 100.0,
        },
        "motion_correction": {
            "max_shifts": [6, 6],
            "strides": [48, 48],
            "overlaps": [24, 24],
            "max_deviation_rigid": 3,
            "pw_rigid": False,
            "shifts_opencv": True,
            "border_nan": "copy",
        },
        "cnmf": {
            "p": 1,
            "gnb": 2,
            "merge_thr": 0.8,
            "rf": 15,
            "stride": 6,
            "K": 4,
            "gSig": [4, 4],
            "gSiz": [17, 17],
            "ring_size_factor": 1.5,
            "min_corr": 0.8,
            "min_pnr": 10.0,
            "ssub": 1,
            "tsub": 1,
            "method_init": "greedy_roi",
            "method_deconv": "oasis",
            "method_ls": "lasso_lars",
            "dff_quantile_min": 8,
            "dff_frames_window": 250,
        },
        "quality": {
            "min_SNR": 2.0,
            "rval_thr": 0.85,
            "use_cnn": True,
            "min_cnn_thr": 0.5,
            "cnn_lowest": 0.1,
        },
        "cluster": {
            "n_processes": None,
            "ram_budget_frac": 0.75,
            "worker_overhead_frac": 1.6,
            "blas_threads_per_worker": 1,
        },
        "gpu": {
            "precompute_chunk_frames": 500,
        },
    }
    _P = ParamBag(_params_dict)
    qc = QCRunner(_P, session, outdir)

    # ── Motion correction ─────────────────────────────────────────────────
    mc = MotionCorrect(
        [fname], dview=None, use_gpu=True, nonneg_movie=True,
        max_shifts          = _P.motion_correction.max_shifts,
        strides             = _P.motion_correction.strides,
        overlaps            = _P.motion_correction.overlaps,
        max_deviation_rigid = _P.motion_correction.max_deviation_rigid,
        shifts_opencv       = _P.motion_correction.shifts_opencv,
        border_nan          = _P.motion_correction.border_nan,
        pw_rigid            = _P.motion_correction.pw_rigid,
    )
    with timer("Motion correction"):
        mc.motion_correct(save_movie=True)
    fname_mc   = mc.mmap_file[0]
    shifts_rig = mc.shifts_rig
    logger.info(f"MC done: {fname_mc}")

    with timer("QC: motion correction"):
        qc.motion_correction(mc)
    del mc

    bord_px = 0  # border_nan="copy"

    # ── F→C mmap conversion ───────────────────────────────────────────────
    Yr_F, dims, T = cm.mmapping.load_memmap(fname_mc)
    n_px       = int(np.prod(dims))
    fname_cnmf = cm.paths.fn_relocated(
        cm.paths.memmap_frames_filename(session + "_cnmf", dims, T, "C"),
        force_temp=True,
    )
    Yr_C = np.memmap(fname_cnmf, mode="w+", dtype=np.float32,
                     shape=cm.mmapping.prepare_shape((n_px, T)), order="C")
    with timer("F→C mmap conversion"):
        fc_convert_parallel(Yr_F, Yr_C, n_px, T,
                            _P.data.add_baseline, logger)
    del Yr_C, Yr_F
    logger.info(f"C-order mmap: {fname_cnmf}")

    # ── Correlation image ─────────────────────────────────────────────────
    Yr, dims, T = cm.mmapping.load_memmap(fname_cnmf)
    images = np.reshape(Yr.T, [T] + list(dims), order="F")
    images.filename = Yr.filename
    logger.info(f"Data: {images.shape}  dtype={images.dtype}")

    cupy_flush(logger, label="before summary-image step")
    with timer("Correlation image (Cn)"):
        Cn = csi.local_correlations_fft(images[::5], swap_dim=False)
        Cn[np.isnan(Cn)] = 0
        np.save(str(outdir / f"{session}_Cn.npy"), Cn)

    with timer("QC: correlation image"):
        qc.correlation_image(Cn)

    # Release Cn-step RSS before CNMF
    del images, Yr
    gc.collect()
    malloc_trim(logger)
    Yr, dims, T = cm.mmapping.load_memmap(fname_cnmf)
    images = np.reshape(Yr.T, [T] + list(dims), order="F")
    images.filename = Yr.filename

    # ── CNMF ──────────────────────────────────────────────────────────────
    clean_stale_shm(CAIMAN_SHM, CAIMAN_TEMP, logger)
    logger.info("Starting CNMF cluster")
    _, dview, n_processes = cm.cluster.setup_cluster(
        backend="multiprocessing",
        n_processes=None,
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

    # ── Report ────────────────────────────────────────────────────────────
    write_report(timer, session, outdir, logger)

    print(f"\nResults saved to: {outdir}/{session}_results.hdf5")
    print(f"Components:       {cnm2.estimates.A.shape[1]}")
    print(f"Report:           {outdir}/{session}_report.txt")
