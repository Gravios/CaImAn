#!/usr/bin/env python3
"""
Stationary vs frame-varying lattice — distinguish two diagnoses.

If a periodic pattern is the SAME in every frame (stationary FPN), it
appears as sharp peaks in the temporal-mean FFT. ``subtract_fixed_pattern``
handles this cleanly.

If a periodic pattern is present in EACH FRAME but with different phase
or amplitude (frame-varying structured noise), the temporal mean
averages it down toward zero. Per-frame FFTs still see it, but the
temporal-mean FFT does not — and ``subtract_fixed_pattern`` cannot fix
it because there's nothing in the mean to subtract.

This script computes both:
  1. Temporal-mean FFT — the sharp-peak landscape subtract_fixed_pattern sees
  2. Mean of per-frame FFT magnitudes — the spectrum present in each frame
     regardless of phase coherence

Comparing the two tells you which kind of noise dominates and therefore
which corrective method (if any) applies.

Usage:
    python diagnose_stationary_vs_perframe.py \
        --input /path/to/session.msr \
        --output ./diag_stat_vs_pf \
        --n-frames 2000
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path,
                    default=Path("./diag_stat_vs_pf"))
    p.add_argument("--n-frames", type=int, default=2000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load via the same reader the pipeline uses
    sys.path.insert(0, "/data/dev/CaImAn")
    from utilities.support.io import read_stack_to_array

    print(f"Loading {args.input} (up to {args.n_frames} frames)...")
    stack = read_stack_to_array(args.input, dtype=np.float32,
                                  progress=True)[:args.n_frames]
    T, H, W = stack.shape
    print(f"  shape={stack.shape}")

    # 1. Temporal-mean FFT — what subtract_fixed_pattern sees
    print("\n1. Computing temporal mean and its FFT...")
    M = stack.mean(axis=0)
    M_demean = M - M.mean()
    F_mean = np.fft.fftshift(np.fft.fft2(M_demean))
    A_mean = np.abs(F_mean)
    A_mean_log = np.log10(A_mean + 1)

    # 2. Mean of per-frame FFT magnitudes — what each frame contains
    print("2. Computing per-frame FFTs (mean of magnitudes)...")
    A_perframe_acc = np.zeros((H, W), dtype=np.float32)
    for i in range(T):
        f = np.fft.fftshift(np.fft.fft2(stack[i] - stack[i].mean()))
        A_perframe_acc += np.abs(f)
        if (i + 1) % 200 == 0:
            print(f"     {i+1}/{T}")
    A_perframe = A_perframe_acc / T
    A_perframe_log = np.log10(A_perframe + 1)

    # 3. Ratio: how much of the per-frame energy survives in the mean?
    # Where this is near 1, the noise is stationary (phase-coherent across frames)
    # Where this is near 0, the noise is frame-varying (averages down in the mean)
    print("3. Computing coherence ratio...")
    coherence = A_mean / (A_perframe + 1e-6)
    coherence_log = np.log10(coherence + 1e-3)

    # 4. Probe specific lattice coords
    cy, cx = H // 2, W // 2
    targets = [
        ("56×fr, 16×lr   (Dec-6 lattice)",  56, 16),
        ("23×fr, 0       (Dec-16 dominant Y axis)", 23, 0),
        ("0,    25×lr    (Dec-16 dominant X axis)", 0,  25),
        ("baseline      (non-lattice point)",       73, 41),
    ]
    print("\n4. Probing specific bins in BOTH spectra:")
    print(f"   {'target':<40s}  {'A_mean':>10s}  {'A_perframe':>10s}  {'ratio':>8s}")
    print(f"   {'-' * 40}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    for label, dy, dx in targets:
        y, x = cy + dy, cx + dx
        am = A_mean[y, x]
        ap = A_perframe[y, x]
        r = am / (ap + 1e-9)
        print(f"   {label:<40s}  {am:>10.1f}  {ap:>10.1f}  {r:>8.3f}")

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Top: temporal mean, its FFT, and a per-frame example
    axes[0, 0].imshow(M, cmap="gray",
                       vmin=np.percentile(M, 1), vmax=np.percentile(M, 99))
    axes[0, 0].set_title("temporal mean (real space)")

    axes[0, 1].imshow(A_mean_log, cmap="viridis",
                       vmin=np.percentile(A_mean_log, 80),
                       vmax=np.percentile(A_mean_log, 99.5))
    axes[0, 1].set_title("|FFT(temporal mean)| log-scale\n"
                          "↑ stationary lattice peaks live here")

    axes[0, 2].imshow(stack[0], cmap="gray",
                       vmin=np.percentile(stack[0], 1),
                       vmax=np.percentile(stack[0], 99))
    axes[0, 2].set_title("example single frame (#0)")

    # Bottom: per-frame avg FFT, coherence map, ratio bars
    axes[1, 0].imshow(A_perframe_log, cmap="viridis",
                       vmin=np.percentile(A_perframe_log, 80),
                       vmax=np.percentile(A_perframe_log, 99.5))
    axes[1, 0].set_title("mean of |FFT(each frame)| log-scale\n"
                          "↑ any per-frame structure lives here")

    # Coherence: stationary noise → near 1; varying → near 0
    # Note: clip to [-3, 0] in log scale = [0.001, 1] for legibility
    im = axes[1, 1].imshow(np.clip(coherence_log, -3, 0),
                            cmap="RdBu_r", vmin=-3, vmax=0)
    axes[1, 1].set_title("log10(A_mean / A_perframe)\n"
                          "red = stationary, blue = frame-varying")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # Bar comparison at the probed points
    labels = [t[0].split("(")[0].strip() for t in targets]
    A_m_vals = [A_mean[cy + dy, cx + dx] for _, dy, dx in targets]
    A_p_vals = [A_perframe[cy + dy, cx + dx] for _, dy, dx in targets]
    ratios = [m / (p + 1e-9) for m, p in zip(A_m_vals, A_p_vals)]
    y_pos = np.arange(len(labels))
    axes[1, 2].barh(y_pos, ratios, color=["green" if r > 0.5 else
                                            "red" if r < 0.1 else
                                            "gray" for r in ratios])
    axes[1, 2].axvline(1.0, color="k", linestyle="--", alpha=0.5,
                        label="purely stationary")
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(labels, fontsize=8)
    axes[1, 2].set_xlabel("A_mean / A_perframe\n"
                           "(1 = stationary; 0 = frame-varying)")
    axes[1, 2].set_xlim(0, 1.2)
    axes[1, 2].set_title("coherence at probed bins")
    axes[1, 2].legend()

    fig.tight_layout()
    out_path = args.output / "stationary_vs_perframe.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nWrote {out_path}")

    # Save the spectra for inspection
    np.save(args.output / "A_mean.npy", A_mean)
    np.save(args.output / "A_perframe.npy", A_perframe)


if __name__ == "__main__":
    main()
