#!/usr/bin/env python
"""
One-photon / CNMF-E microendoscope pipeline demo
==================================================
Updated for Gravios/CaImAn fork.  Replaces ``demos/general/demo_pipeline_cnmfE.py``.

Key differences from 2P
------------------------
- ``method_init = "corr_pnr"`` with ring neuropil background model
- ``gnb = -1``: ring model replaces global background components
- ``ring_size_factor``: must satisfy ``ring_size_factor × gSiz < rf``
- Smaller ``gSig`` (1–3 px for microendoscope / Miniscope data)
- ``ssub_B = 2``: background spatial downsampling

Ring size constraint check
---------------------------
    ring_size_factor × gSiz[0] < rf

With the defaults below: 1.4 × 13 = 18.2 < 40 ✓

Usage
-----
    python demos/demo_pipeline_cnmfe.py

Downloads ``data_endoscope.tif`` automatically if not present.
"""

# ── Bootstrap ─────────────────────────────────────────────────────────────────
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

    outdir  = Path(CAIMAN_DATA) / "demo_cnmfe_output"
    outdir.mkdir(exist_ok=True)
    session = "demo_cnmfe"

    logger         = setup_logging(outdir / f"{session}.log")
    _cnn_available = ensure_model_files(os.path.join(CAIMAN_DATA, "model"))
    timer          = PipelineTimer(logger)

    # ── Download demo data ────────────────────────────────────────────────
    fname = download_demo("data_endoscope.tif")
    logger.info(f"Input: {fname}")

    # ── Parameters ───────────────────────────────────────────────────────
    # 1P / microendoscope parameters.
    # Critical differences from 2P:
    #   gnb = -1          → ring neuropil model (no global background)
    #   method_init       → corr_pnr (required for ring model)
    #   ring_size_factor  → must satisfy: ring_size_factor × gSiz < rf
    #   ssub_B            → background spatial downsampling
    _params_dict = {
        "data": {
            "fr": 10,
            "decay_time": 0.4,
            "add_baseline": 0.0,   # endoscope data is already baseline-corrected
        },
        "motion_correction": {
            "max_shifts": [5, 5],
            "strides": [48, 48],
            "overlaps": [24, 24],
            "max_deviation_rigid": 3,
            "pw_rigid": False,
            "shifts_opencv": True,
            "border_nan": "copy",
        },
        "cnmf": {
            "p": 1,
            "gnb": -1,               # ring model: -1 = exact ring background
            "ssub_B": 2,             # background spatial downsampling factor
            "merge_thr": 0.7,
            "rf": 40,
            "stride": 20,
            "K": 10,
            "gSig": [3, 3],          # smaller than 2P: ~1–4 px for microendoscope
            "gSiz": [13, 13],
            "ring_size_factor": 1.4, # ring outer radius / gSiz; must be < rf/gSiz
            "min_corr": 0.8,
            "min_pnr": 10.0,
            "ssub": 1,
            "tsub": 2,
            "method_init": "corr_pnr",   # required for ring model
            "method_deconv": "oasis",
            "method_ls": "lasso_lars",
            "dff_quantile_min": 8,
            "dff_frames_window": 500,
        },
        "quality": {
            "min_SNR": 2.5,
            "rval_thr": 0.85,
            "use_cnn": False,        # CNN not trained on 1P data by default
            "min_cnn_thr": 0.9,
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

    # Ring size constraint check
    _gsiz = _P.cnmf.gSiz[0]
    _rsf  = _P.cnmf.ring_size_factor
    _rf   = _P.cnmf.rf
    assert _rsf * _gsiz < _rf, (
        f"Ring constraint violated: ring_size_factor({_rsf}) × gSiz({_gsiz}) "
        f"= {_rsf*_gsiz:.1f} ≥ rf({_rf}). "
        f"Increase rf or decrease ring_size_factor."
    )
    logger.info(f"Ring check: {_rsf} × {_gsiz} = {_rsf*_gsiz:.1f} < {_rf} ✓")

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
    bord_px = 0

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

    # ── Correlation + PNR images ──────────────────────────────────────────
    Yr, dims, T = cm.mmapping.load_memmap(fname_cnmf)
    images = np.reshape(Yr.T, [T] + list(dims), order="F")
    images.filename = Yr.filename
    logger.info(f"Data: {images.shape}  dtype={images.dtype}")

    cupy_flush(logger, label="before summary-image step")
    with timer("Correlation image (Cn) + PNR"):
        from caiman.summary_images import correlation_pnr
        Cn, pnr = correlation_pnr(
            images[::2],            # every 2nd frame for speed
            gSig  = _P.cnmf.gSig[0],
            center_psf = True,
            swap_dim   = False,
        )
        Cn[np.isnan(Cn)] = 0
        np.save(str(outdir / f"{session}_Cn.npy"),  Cn)
        np.save(str(outdir / f"{session}_PNR.npy"), pnr)

    with timer("QC: Cn + PNR"):
        qc.correlation_image(Cn)
        qc.pnr_image(Cn, pnr)       # shows threshold lines from JSON

    # Release RSS before CNMF
    del images, Yr
    gc.collect()
    malloc_trim(logger)
    Yr, dims, T = cm.mmapping.load_memmap(fname_cnmf)
    images = np.reshape(Yr.T, [T] + list(dims), order="F")
    images.filename = Yr.filename

    # ── CNMF-E ───────────────────────────────────────────────────────────
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
