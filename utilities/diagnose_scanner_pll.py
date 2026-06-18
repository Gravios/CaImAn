#!/usr/bin/env python3
"""
Diagnostic for the scanner PLL: extracts per-bin trajectories,
computes phase coherence statistics, plots the trajectories before and
after the demodulate/smooth/remodulate cycle. Use this BEFORE committing
to a full-pipeline PLL run on a session, to verify the contamination
is actually trackable by PLL (vs needing notching).

Use:
    python diagnose_scanner_pll.py \\
        --input <stack.tif/.msr> \\
        --output ./diag_pll \\
        --use-gpu

Outputs:
    <output>/pll_trajectories.png  — multi-panel view of top-K bins
    <output>/trajectories.npy      — raw per-frame complex values, (T, K)
    <output>/corrections.npy       — estimated contamination, (T, K)
    <output>/summary.txt           — per-bin phase statistics
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, default=Path("./diag_pll"))
    p.add_argument("--smooth-window-frames", type=int, default=1000)
    p.add_argument("--omega-track-window-frames", type=int, default=None,
                    help="enable time-varying omega tracking with the given "
                         "window (default: scalar omega per bin). Recommended: "
                         "smooth_window_frames // 5.")
    p.add_argument("--max-bins-to-plot", type=int, default=6)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--max-frames", type=int, default=None,
                    help="cap frames processed (for quick tests)")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, "/data/dev/CaImAn")
    from caiman.utils.stack_io import StackReader
    from utilities.noise.scanner_pll import subtract_scanner_pll

    # Load
    print(f"Loading {args.input}...")
    with StackReader(args.input) as r:
        T = int(r.n_frames)
        if args.max_frames is not None:
            T = min(T, args.max_frames)
        H = int(r.h); W = int(r.w)
        stack = np.empty((T, H, W), dtype=np.float32)
        for i in range(T):
            stack[i] = r.read_frame(i).astype(np.float32, copy=False)
    print(f"  shape={stack.shape}")

    # Run PLL with diagnostics
    cleaned, diag, pattern = subtract_scanner_pll(
        stack,
        smooth_window_frames=args.smooth_window_frames,
        omega_track_window_frames=args.omega_track_window_frames,
        use_gpu=args.use_gpu,
        return_diagnostics=True,
    )

    # Save raw arrays
    np.save(args.output / "trajectories.npy", diag["trajectories"])
    np.save(args.output / "corrections.npy", diag["corrections"])
    np.save(args.output / "omega_per_bin.npy", diag["omega_per_bin"])
    np.save(args.output / "demod_phase.npy", diag["demod_phase"])
    if diag.get("omega_t") is not None:
        np.save(args.output / "omega_t.npy", diag["omega_t"])

    # Per-bin summary text
    cy, cx = H // 2, W // 2
    with open(args.output / "summary.txt", "w") as f:
        f.write("# scanner PLL diagnostic summary\n")
        f.write(f"input: {args.input}\n")
        f.write(f"T={T}, H={H}, W={W}\n")
        f.write(f"smooth_window_frames={args.smooth_window_frames}\n")
        f.write(f"omega_track_window_frames={args.omega_track_window_frames}\n")
        f.write(f"n_bins: {diag['n_bins']}\n")
        f.write(f"median_coherence_lag1:  {diag['median_coherence_lag1']:.4f}\n")
        f.write(f"median_coherence_lag10: {diag['median_coherence_lag10']:.4f}\n")
        f.write(f"max_coherence_lag1:     {diag['max_coherence_lag1']:.4f}\n")
        f.write(f"p25_coherence_lag1:     {diag['p25_coherence_lag1']:.4f}\n")
        f.write(f"p75_coherence_lag1:     {diag['p75_coherence_lag1']:.4f}\n\n")
        if diag["median_coherence_lag1"] >= 0.3:
            verdict = "PLL is providing real value (high phase coherence)"
        elif diag["median_coherence_lag1"] >= 0.1:
            verdict = "INTERMEDIATE; try both PLL and notching, compare CNMF"
        else:
            verdict = ("PLL near no-op; phase is random per frame. "
                       "Use subtract_per_frame_pattern instead.")
        f.write(f"# Verdict: {verdict}\n\n")
        f.write("# per-bin (sorted by coherence_lag1, descending):\n")
        f.write(f"{'#':>4s}  {'dy':>5s}  {'dx':>5s}  {'mag_mean':>9s}  "
                f"{'mag_std':>9s}  {'coh_lag1':>8s}  {'coh_lag10':>9s}  "
                f"{'omega':>8s}\n")
        sorted_bins = sorted(diag["per_bin"],
                              key=lambda b: -b["phase_coherence_lag1"])
        for i, b in enumerate(sorted_bins):
            omega = diag["omega_per_bin"][diag["per_bin"].index(b)]
            f.write(f"{i:>4d}  {b['dy']:+5d}  {b['dx']:+5d}  "
                    f"{b['magnitude_mean']:9.0f}  {b['magnitude_std']:9.0f}  "
                    f"{b['phase_coherence_lag1']:8.3f}  "
                    f"{b['phase_coherence_lag10']:9.3f}  "
                    f"{omega:+8.4f}\n")

    # Multi-panel plot: pick top-K bins by coherence
    K_plot = min(args.max_bins_to_plot, diag["n_bins"])
    if K_plot == 0:
        print("No bins to plot; exiting.")
        return
    coh_idx = np.argsort([-b["phase_coherence_lag1"] for b in diag["per_bin"]])
    plot_bins = coh_idx[:K_plot]

    fig, axes = plt.subplots(K_plot, 3, figsize=(16, 2.6 * K_plot),
                              squeeze=False)
    trajectories = diag["trajectories"]
    corrections = diag["corrections"]
    omegas = diag["omega_per_bin"]
    t_axis = np.arange(T)

    for i, k in enumerate(plot_bins):
        b = diag["per_bin"][k]
        traj = trajectories[:, k]
        corr = corrections[:, k]

        # Panel 1: Re/Im of raw trajectory and estimated correction
        ax = axes[i, 0]
        ax.plot(t_axis, traj.real, color='C0', alpha=0.25, linewidth=0.4,
                 label='Re(F)')
        ax.plot(t_axis, traj.imag, color='C1', alpha=0.25, linewidth=0.4,
                 label='Im(F)')
        ax.plot(t_axis, corr.real, color='C0', linewidth=1.5,
                 label='Re(PLL est.)')
        ax.plot(t_axis, corr.imag, color='C1', linewidth=1.5,
                 label='Im(PLL est.)')
        ax.set_title(f"bin ({b['dy']:+d}, {b['dx']:+d})  "
                     f"coh-lag1 = {b['phase_coherence_lag1']:.3f}  "
                     f"ω = {omegas[k]:+.4f} rad/fr",
                     fontsize=9)
        ax.set_xlabel("frame")
        ax.legend(fontsize=7, ncol=2)

        # Panel 2: magnitude over time
        ax = axes[i, 1]
        ax.plot(t_axis, np.abs(traj), color='gray', alpha=0.3, linewidth=0.4,
                 label='|F|')
        ax.plot(t_axis, np.abs(corr), color='C2', linewidth=1.5,
                 label='|PLL estimate|')
        ax.set_title("magnitude", fontsize=9)
        ax.set_xlabel("frame")
        ax.legend(fontsize=7)

        # Panel 3: phase over time (unwrapped), with the PLL's actual
        # model phase overlaid. For scalar omega this is a straight
        # line; for time-varying omega this curves with the drift.
        ax = axes[i, 2]
        phase_raw = np.unwrap(np.angle(traj))
        model_phase = diag["demod_phase"][:, k]
        # Offset model so it starts at phase_raw[0] (the model has
        # arbitrary integration constant; we anchor to the raw)
        model_phase_anchored = model_phase + (phase_raw[0] - model_phase[0])
        ax.plot(t_axis, phase_raw, color='gray', alpha=0.3, linewidth=0.4,
                 label='phase(F)')
        ax.plot(t_axis, model_phase_anchored, color='C3', linewidth=1.2,
                 linestyle='--', label='PLL model phase')
        # If time-varying omega, also indicate the omega(t) trajectory
        # as a secondary y-axis to show frequency drift visually.
        if diag.get("omega_t") is not None:
            ax2 = ax.twinx()
            ax2.plot(t_axis[1:], diag["omega_t"][:, k], color='C4',
                      alpha=0.6, linewidth=0.6, label='ω(t) [rad/fr]')
            ax2.set_ylabel("ω(t)", color='C4', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='C4', labelsize=7)
        ax.set_title("phase (unwrapped) + PLL model", fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("radians")
        ax.legend(fontsize=7, loc='upper left')

    omega_track_note = (
        f"  |  ω-track window={args.omega_track_window_frames}"
        if args.omega_track_window_frames is not None
        else "  |  scalar ω")
    title = (f"PLL trajectory diagnostic — {args.input.name}\n"
             f"T={T}, K={diag['n_bins']}, smooth window={args.smooth_window_frames}"
             f"{omega_track_note}"
             f"  |  median coh-lag1 = {diag['median_coherence_lag1']:.3f}")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_png = args.output / "pll_trajectories.png"
    fig.savefig(out_png, dpi=110)
    print(f"\nWrote {out_png}")
    print(f"Wrote {args.output / 'summary.txt'}")
    print(f"median_coherence_lag1: {diag['median_coherence_lag1']:.3f}")


if __name__ == "__main__":
    main()
