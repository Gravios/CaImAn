#!/usr/bin/env python
"""
Parameter estimation demo
==========================
Demonstrates ``caiman.utils.params_estimator.estimate_params`` as a
standalone tool.  Runs motion correction, estimates CNMF parameters from the
corrected movie, and prints the suggested JSON values.

This is what ``new_session.py --run-mc --estimate-params`` does internally —
exposed here for interactive use and debugging.

Usage
-----
    # Estimate from an existing MC mmap
    python demos/demo_param_estimation.py \
        --mc-mmap /data/caiman/temp/session_rig__d1_512_d2_512_*.mmap

    # Run MC first, then estimate
    python demos/demo_param_estimation.py \
        --tif /data/src/session/session.tif \
        --run-mc

    # Save inspection figure
    python demos/demo_param_estimation.py \
        --mc-mmap /data/caiman/temp/session_*.mmap \
        --out-fig /tmp/param_estimate.png \
        --n-frames 1000
"""

import os
import sys
import argparse
import logging
from pathlib import Path

os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("MPLBACKEND",           "Agg")


def main():
    parser = argparse.ArgumentParser(description="Estimate CNMF parameters from a movie")
    parser.add_argument("--mc-mmap", default=None,
                        help="Path to an existing F-order MC mmap")
    parser.add_argument("--tif", default=None,
                        help="Path to input TIF (requires --run-mc)")
    parser.add_argument("--run-mc", action="store_true",
                        help="Run GPU MC on --tif before estimation")
    parser.add_argument("--n-frames", type=int, default=500,
                        help="Frames to subsample for estimation (default 500)")
    parser.add_argument("--fr", type=float, default=30.0,
                        help="Frame rate in Hz (default 30)")
    parser.add_argument("--gSig", type=int, default=None,
                        help="Override gSig hint (skip blob detection)")
    parser.add_argument("--out-fig", default=None,
                        help="Save inspection figure to this path")
    parser.add_argument("--apply-to", default=None,
                        help="Apply suggestions to this JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --apply-to: print changes without writing")
    args = parser.parse_args()

    if args.mc_mmap is None and (args.tif is None or not args.run_mc):
        parser.error("Provide either --mc-mmap or (--tif + --run-mc)")

    # ── Logging ───────────────────────────────────────────────────────────
    logger = logging.getLogger("caiman")
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)

    # ── Optional MC ───────────────────────────────────────────────────────
    mc_path = args.mc_mmap
    _mc_obj = None
    if args.run_mc and args.tif:
        from caiman.motion_correction import MotionCorrect
        import numpy as np

        CAIMAN_TEMP = os.environ.get("CAIMAN_TEMP", "/tmp")
        os.makedirs(CAIMAN_TEMP, exist_ok=True)

        logger.info(f"Running MC on {args.tif}...")
        mc = MotionCorrect(
            [args.tif], dview=None, use_gpu=True, nonneg_movie=True,
            max_shifts=[6, 6], strides=[64, 64], overlaps=[32, 32],
            max_deviation_rigid=3, shifts_opencv=True,
            border_nan="copy", pw_rigid=False,
        )
        mc.motion_correct(save_movie=True)
        mc_path = mc.mmap_file[0]
        _mc_obj = mc

        # Analyse shifts
        shifts = np.array(mc.shifts_rig)
        mag    = np.hypot(shifts[:, 0], shifts[:, 1])
        p99_r  = float(np.percentile(np.abs(shifts[:, 0]), 99))
        p99_c  = float(np.percentile(np.abs(shifts[:, 1]), 99))
        logger.info(f"MC done: median={np.median(mag):.2f} px  "
                    f"p99=[{p99_r:.2f}, {p99_c:.2f}] px")
        logger.info(f"Suggested max_shifts = "
                    f"[{max(4, int(p99_r/2+1)*2)}, {max(4, int(p99_c/2+1)*2)}]")

    # ── Parameter estimation ──────────────────────────────────────────────
    from caiman.utils.params_estimator import estimate_params, apply_suggestions

    suggestions = estimate_params(
        mc_path,
        gSig_hint  = args.gSig,
        n_frames   = args.n_frames,
        fr         = args.fr,
        out_path   = args.out_fig,
        logger     = logger,
    )

    print("\n" + "=" * 60)
    print("  Suggested JSON cnmf section values:")
    print("=" * 60)
    for k, v in suggestions.items():
        print(f"  {k!r:20s}: {v}")
    print("=" * 60)

    if args.apply_to:
        apply_suggestions(args.apply_to, suggestions, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"\n  JSON updated: {args.apply_to}")

    # Clean up temp MC if we created it
    if _mc_obj is not None and args.tif is not None:
        try:
            import os as _os
            _os.unlink(mc_path)
            logger.info(f"Deleted temporary MC mmap: {mc_path}")
        except OSError:
            pass


if __name__ == "__main__":
    main()
