#!/usr/bin/env python3
"""
Diagnose 50 Hz monitor / mains contamination in a 2P recording.

Run from the CaImAn repo root:
    python utilities/noise/diagnose_50hz.py \
        --input /data/source/.../strohA-sa-000082-...fc018540.msr \
        --output /data/diagnose_50hz \
        --fr 30.79

What this checks (three independent tests, all from one recording):

1. Temporal spectrum of background mean intensity. A 50 Hz contaminant
   sampled at the frame rate aliases predictably. With frame rate
   fr Hz, 50 Hz aliases to f_alias = |50 - round(50/fr) * fr|. For
   fr=30.79 this gives ~19.2 Hz. We compute the periodogram of the
   per-frame mean intensity over a small background patch and look
   for a peak at the predicted alias frequency.

2. Spatial modulation along Y, averaged over many frames. Line-rate
   contamination shows up as a horizontal band that drifts vertically
   frame-to-frame. Averaging the line-mean (over X) across all frames
   should show ~1.6 full cycles per frame for 50 Hz at 30.79 fps with
   512 lines (line rate 15.76 kHz).

3. Phase walk between consecutive frames. The phase of the 50 Hz hum
   at the start of each frame walks by ~225° per frame at fr=30.79.
   We extract the dominant spatial frequency from each frame's Y-mean
   profile and confirm its phase increments by the predicted amount.

If all three agree, 50 Hz monitor pickup is the cause. If only test 1
fires, it could be a different aliased source. If only test 2 fires
(stationary horizontal stripes), it's geometric not electronic. None
firing → look elsewhere.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def predict_alias(f_contam: float, fr: float) -> float:
    """Aliased frequency observed at sampling rate fr."""
    return abs(f_contam - round(f_contam / fr) * fr)


def background_mean_series(stack: np.ndarray, roi=None) -> np.ndarray:
    """Per-frame mean over a background ROI.

    If no ROI given, picks a 32x32 region in the corner most likely to
    be background (lowest temporal variance among 4 corners)."""
    T, H, W = stack.shape
    if roi is None:
        size = 32
        candidates = {
            "TL": (slice(0, size), slice(0, size)),
            "TR": (slice(0, size), slice(W - size, W)),
            "BL": (slice(H - size, H), slice(0, size)),
            "BR": (slice(H - size, H), slice(W - size, W)),
        }
        vars_by_corner = {
            k: float(np.var(stack[:, ry, rx].mean(axis=(1, 2))))
            for k, (ry, rx) in candidates.items()
        }
        best = min(vars_by_corner, key=vars_by_corner.get)
        roi = candidates[best]
        print(f"  background ROI: {best} corner "
              f"(temporal var {vars_by_corner[best]:.2f})")
    ry, rx = roi
    return stack[:, ry, rx].mean(axis=(1, 2))


def temporal_spectrum_test(stack: np.ndarray, fr: float,
                             f_contam: float = 50.0) -> dict:
    """Test 1: peak at predicted alias frequency in background mean."""
    sig = background_mean_series(stack)
    sig = sig - sig.mean()
    T = len(sig)
    freqs = np.fft.rfftfreq(T, d=1.0 / fr)
    spec = np.abs(np.fft.rfft(sig)) / T

    f_alias = predict_alias(f_contam, fr)
    # Search ±0.5 Hz window around prediction
    in_win = (freqs > f_alias - 0.5) & (freqs < f_alias + 0.5)
    peak_idx = np.argmax(spec[in_win])
    peak_f = freqs[in_win][peak_idx]
    peak_amp = spec[in_win][peak_idx]
    median_amp = float(np.median(spec[freqs > 1.0]))
    snr_db = 20 * np.log10(peak_amp / median_amp) if median_amp > 0 else float("inf")
    return {
        "freqs": freqs,
        "spec": spec,
        "predicted_alias": f_alias,
        "observed_peak": peak_f,
        "snr_db": snr_db,
        "verdict": "POSITIVE" if snr_db > 15 else "negative",
    }


def line_modulation_test(stack: np.ndarray, fr: float,
                          line_rate_hz: float | None = None,
                          f_contam: float = 50.0) -> dict:
    """Test 2: spatial period along Y matches 50 Hz at the line rate.

    Y-mean profile averaged over time should show ~1.6 cycles per frame
    for 50 Hz at fr=30.79 with 512 lines.
    """
    T, H, W = stack.shape
    if line_rate_hz is None:
        line_rate_hz = fr * H
    # Time-averaged Y profile, with the temporal mean removed per row
    Y_profile = stack.mean(axis=(0, 2))            # (H,)
    Y_profile = Y_profile - Y_profile.mean()
    # Spatial FFT — looking for the dominant Y-period
    Y_freqs = np.fft.rfftfreq(H, d=1.0 / line_rate_hz)
    Y_spec = np.abs(np.fft.rfft(Y_profile)) / H
    # Predicted spatial frequency: 50 Hz appears as 50 Hz at the line
    # rate → spatial period H * 50/line_rate lines per cycle
    pred_freq_hz = f_contam
    in_win = (Y_freqs > pred_freq_hz - 5) & (Y_freqs < pred_freq_hz + 5)
    if not np.any(in_win):
        observed_freq = float("nan")
        snr_db = float("nan")
    else:
        peak_idx = np.argmax(Y_spec[in_win])
        observed_freq = Y_freqs[in_win][peak_idx]
        peak_amp = Y_spec[in_win][peak_idx]
        median_amp = float(np.median(Y_spec[Y_freqs > 10]))
        snr_db = 20 * np.log10(peak_amp / median_amp) if median_amp > 0 else float("inf")
    return {
        "y_profile": Y_profile,
        "Y_freqs": Y_freqs,
        "Y_spec": Y_spec,
        "predicted_freq": pred_freq_hz,
        "observed_freq": observed_freq,
        "snr_db": snr_db,
        "line_rate_hz": line_rate_hz,
        "verdict": "POSITIVE" if snr_db > 15 else "negative",
    }


def phase_walk_test(stack: np.ndarray, fr: float,
                      f_contam: float = 50.0,
                      n_frames: int = 200) -> dict:
    """Test 3: per-frame phase of dominant Y-modulation walks linearly.

    For 50 Hz at fr=30.79 the phase walk per frame is
    2*pi * (50/fr - round(50/fr)) ≈ 2*pi * 0.624 rad/frame
    ≈ 3.92 rad/frame (≈ 225° per frame).
    """
    T, H, W = stack.shape
    n = min(n_frames, T)
    # Extract Y-mean of each frame, remove the per-frame DC
    profiles = stack[:n].mean(axis=2)            # (n, H)
    profiles = profiles - profiles.mean(axis=1, keepdims=True)
    # FFT along Y for each frame
    F = np.fft.rfft(profiles, axis=1)            # (n, H//2+1)
    # Identify dominant spatial bin from the time-averaged amplitude
    avg_amp = np.abs(F).mean(axis=0)
    avg_amp[0:3] = 0  # ignore DC + lowest few bins
    k_dom = int(np.argmax(avg_amp))
    phases = np.angle(F[:, k_dom])               # (n,)
    # Unwrap and fit a line
    phases_u = np.unwrap(phases)
    frame_idx = np.arange(n)
    slope, intercept = np.polyfit(frame_idx, phases_u, 1)
    pred_slope = 2 * np.pi * (f_contam / fr - round(f_contam / fr))
    return {
        "phases": phases,
        "phases_unwrapped": phases_u,
        "dominant_y_bin": k_dom,
        "observed_slope_rad_per_frame": slope,
        "predicted_slope_rad_per_frame": pred_slope,
        "verdict": "POSITIVE" if abs(slope - pred_slope) < 0.5 else "negative",
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Diagnose 50 Hz line/monitor contamination in a 2P stack")
    p.add_argument("--input", "-i", type=Path, required=True,
                    help="path to .tif/.tiff/.msr stack")
    p.add_argument("--output", "-o", type=Path, default=Path("./diagnose_50hz"),
                    help="output directory for diagnostic plots")
    p.add_argument("--fr", type=float, default=30.79,
                    help="frame rate in Hz (default 30.79)")
    p.add_argument("--f-contam", type=float, default=50.0,
                    help="contamination frequency hypothesis (Hz). "
                         "Try 60 in North America, 50 in EU.")
    p.add_argument("--n-frames", type=int, default=2000,
                    help="max frames to load (default 2000 — enough for "
                         "all three tests, fast on any storage)")
    args = p.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} (up to {args.n_frames} frames)...")
    # Use the same loader the pipeline uses, so this script works on MSR too
    try:
        from utilities.support.io import read_stack_to_array
        stack = read_stack_to_array(args.input, dtype=np.float32,
                                     progress=True)
        stack = stack[:args.n_frames]
    except ImportError:
        # Fallback: tifffile direct (TIFF only)
        import tifffile
        stack = tifffile.imread(str(args.input), key=range(args.n_frames))
        stack = stack.astype(np.float32)
    T, H, W = stack.shape
    print(f"  loaded: shape={stack.shape} dtype={stack.dtype}")
    print(f"  frame rate: {args.fr} Hz")
    print(f"  line rate:  {args.fr * H:.1f} Hz "
          f"(= {args.fr} fps × {H} lines)")
    print(f"  testing for contamination at {args.f_contam} Hz")

    # Run the three tests
    print("\n=== Test 1: temporal spectrum of background mean ===")
    t1 = temporal_spectrum_test(stack, args.fr, args.f_contam)
    print(f"  predicted alias: {t1['predicted_alias']:.2f} Hz")
    print(f"  observed peak:   {t1['observed_peak']:.2f} Hz")
    print(f"  SNR vs median:   {t1['snr_db']:.1f} dB")
    print(f"  verdict:         {t1['verdict']}")

    print("\n=== Test 2: spatial modulation along Y ===")
    t2 = line_modulation_test(stack, args.fr, f_contam=args.f_contam)
    print(f"  line rate:       {t2['line_rate_hz']:.1f} Hz")
    print(f"  predicted freq:  {t2['predicted_freq']:.1f} Hz")
    print(f"  observed freq:   {t2['observed_freq']:.2f} Hz")
    print(f"  SNR vs median:   {t2['snr_db']:.1f} dB")
    print(f"  verdict:         {t2['verdict']}")

    print("\n=== Test 3: frame-to-frame phase walk ===")
    t3 = phase_walk_test(stack, args.fr, args.f_contam)
    print(f"  dominant Y bin:   {t3['dominant_y_bin']}")
    print(f"  predicted slope:  {t3['predicted_slope_rad_per_frame']:.3f} rad/frame")
    print(f"  observed slope:   {t3['observed_slope_rad_per_frame']:.3f} rad/frame")
    print(f"  verdict:          {t3['verdict']}")

    verdicts = [t1["verdict"], t2["verdict"], t3["verdict"]]
    n_pos = verdicts.count("POSITIVE")
    print(f"\n=== Overall: {n_pos}/3 tests positive ===")
    if n_pos == 3:
        print(f"  → STRONG evidence for {args.f_contam} Hz line contamination")
    elif n_pos >= 2:
        print(f"  → MODERATE evidence; check the plots and consider running")
        print(f"     a control recording with monitors off")
    elif n_pos == 1:
        print(f"  → WEAK evidence; the source may be different from {args.f_contam} Hz")
        print(f"     Try --f-contam 60 (North America mains) or 25/100 (harmonics)")
    else:
        print(f"  → NO evidence for {args.f_contam} Hz contamination")

    # Plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    ax = axes[0]
    ax.semilogy(t1["freqs"], t1["spec"])
    ax.axvline(t1["predicted_alias"], color="r", linestyle="--",
                label=f"predicted alias = {t1['predicted_alias']:.2f} Hz")
    ax.axvline(t1["observed_peak"], color="g", linestyle=":",
                label=f"observed peak = {t1['observed_peak']:.2f} Hz")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Test 1: temporal spectrum of background mean "
                  f"(SNR={t1['snr_db']:.1f} dB → {t1['verdict']})")
    ax.legend()
    ax.set_xlim(0, args.fr / 2)

    ax = axes[1]
    ax.plot(t2["y_profile"])
    ax.set_xlabel("line (Y)")
    ax.set_ylabel("residual intensity")
    ax.set_title(f"Test 2: time-averaged Y modulation "
                  f"(SNR={t2['snr_db']:.1f} dB → {t2['verdict']})")
    ax.set_xlim(0, H)

    ax = axes[2]
    n = len(t3["phases_unwrapped"])
    pred_line = (t3["predicted_slope_rad_per_frame"] * np.arange(n)
                  + t3["phases_unwrapped"][0])
    ax.plot(t3["phases_unwrapped"], label="observed (unwrapped)")
    ax.plot(pred_line, "r--",
             label=f"predicted slope = {t3['predicted_slope_rad_per_frame']:.2f} rad/frame")
    ax.set_xlabel("frame index")
    ax.set_ylabel("phase (rad)")
    ax.set_title(f"Test 3: phase walk in dominant Y bin "
                  f"(observed slope {t3['observed_slope_rad_per_frame']:.2f} "
                  f"→ {t3['verdict']})")
    ax.legend()

    fig.tight_layout()
    out_path = args.output / f"diagnose_{args.f_contam:.0f}Hz.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nWrote {out_path}")

    # Also save the per-frame background series for manual inspection
    sig_path = args.output / f"bg_series_{args.f_contam:.0f}Hz.npy"
    np.save(sig_path, background_mean_series(stack))
    print(f"Wrote {sig_path}")


if __name__ == "__main__":
    main()
