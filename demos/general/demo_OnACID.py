#!/usr/bin/env python
"""
Online calcium imaging analysis via OnACID
==========================================
Updated for Gravios/CaImAn fork.  Replaces ``demos/general/demo_OnACID.py``.

OnACID processes calcium imaging data frame-by-frame without storing the full
movie in memory.  It is suitable for:
- Real-time closed-loop experiments
- Very long recordings (hours) that would not fit in RAM as a single mmap
- Mesoscope wide-field data where batch CNMF is impractical

Architecture note
-----------------
OnACID does not use the ``CNMFRunner`` or the tile-based batch pipeline — it
uses ``cnmf.online_cnmf.OnACID.fit_online()``.  However it still benefits
from:
- ``setup_logging`` for structured log output
- ``PipelineTimer`` for step timing
- ``QCRunner`` for final figure generation

Usage
-----
    python demos/demo_onacid.py
    python demos/demo_onacid.py --input /path/to/movie.tif
"""

# ── Bootstrap ─────────────────────────────────────────────────────────────────
import os
import sys
import argparse

os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("MPLBACKEND",           "Agg")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import numpy as np

    import caiman as cm
    from caiman.source_extraction import cnmf
    from caiman.utils.utils          import download_demo
    from caiman.utils.pipeline_setup import setup_logging, ensure_model_files
    from caiman.utils.timing         import PipelineTimer, write_report
    from caiman.utils.qc             import QCRunner
    from caiman.utils.params_io      import ParamBag

    # ── CLI ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="OnACID online demo")
    parser.add_argument("--input", default=None,
                        help="Path to input TIFF (downloads demo if not given)")
    parser.add_argument("--n-processes", type=int, default=None,
                        help="Worker count for batch initialization (default: auto)")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    CAIMAN_DATA = os.environ.get("CAIMAN_DATA", cm.paths.caiman_datadir())
    outdir      = Path(CAIMAN_DATA) / "demo_onacid_output"
    outdir.mkdir(exist_ok=True)
    session     = "demo_onacid"

    logger         = setup_logging(outdir / f"{session}.log")
    _cnn_available = ensure_model_files(os.path.join(CAIMAN_DATA, "model"))
    timer          = PipelineTimer(logger)

    # ── Input data ────────────────────────────────────────────────────────
    fname = args.input or download_demo("demoMovie.tif")
    logger.info(f"Input: {fname}")

    # ── Parameters ───────────────────────────────────────────────────────
    # OnACID params use the same CNMFParams structure as batch CNMF, with
    # additional keys in the "online" group.
    _params_dict = {
        "data": {
            "fr": 30,
            "decay_time": 0.5,
            "fnames": [fname],
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
        },
        "quality": {
            "min_SNR": 2.0,
            "rval_thr": 0.85,
            "use_cnn": True,
            "min_cnn_thr": 0.5,
            "cnn_lowest": 0.1,
        },
        "online": {
            "init_batch": 200,       # frames used for batch initialization
            "expected_comps": 500,   # pre-allocate space for this many components
            "update_num_comps": True,# allow new components to be detected online
            "sniper_mode": False,    # use correlation image for seeding
            "test_both": False,
            "motion_correct": True,  # run motion correction online
            "min_num_trial": 5,      # minimum number of seeds per batch
        },
    }
    opts = cnmf.params.CNMFParams(params_dict=_params_dict)

    # ── Online fit ────────────────────────────────────────────────────────
    online_cnm = cnmf.online_cnmf.OnACID(params=opts)
    with timer("OnACID online fit"):
        online_cnm.fit_online()

    n_comp = online_cnm.estimates.A.shape[-1]
    logger.info(f"Components found: {n_comp}")

    # ── Post-processing: evaluate components ──────────────────────────────
    # Load the motion-corrected mmap that OnACID wrote
    if hasattr(online_cnm, 'mmap_file') and online_cnm.mmap_file:
        fname_mc = online_cnm.mmap_file
        Yr, dims, T = cm.mmapping.load_memmap(fname_mc)
        images = np.reshape(Yr.T, [T] + list(dims), order="F")
        images.filename = Yr.filename

        with timer("Component evaluation"):
            online_cnm.estimates.evaluate_components(
                images, online_cnm.params, dview=None)

        n_acc = len(online_cnm.estimates.idx_components)
        n_rej = len(online_cnm.estimates.idx_components_bad)
        logger.info(f"Components: {n_acc} accepted / {n_rej} rejected")

        # ── QC figures ────────────────────────────────────────────────────
        import caiman.summary_images as csi
        Cn = csi.local_correlations_fft(images[::5], swap_dim=False)
        Cn[np.isnan(Cn)] = 0
        np.save(str(outdir / f"{session}_Cn.npy"), Cn)

        _P = ParamBag(_params_dict)
        qc = QCRunner(_P, session, outdir)
        with timer("QC figures"):
            qc.correlation_image(Cn)
            qc.component_evaluation(online_cnm, Cn)
            qc.traces(online_cnm)

        del images, Yr

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = str(outdir / f"{session}_results.hdf5")
    with timer("Save results"):
        online_cnm.save(save_path)
    logger.info(f"Results saved: {save_path}")

    # ── Report ────────────────────────────────────────────────────────────
    write_report(timer, session, outdir, logger)

    print(f"\nResults saved to: {save_path}")
    print(f"Components:       {n_comp}")
