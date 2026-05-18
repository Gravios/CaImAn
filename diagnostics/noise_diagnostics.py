"""
noise_diagnostics.py
=====================

Diagnostic suite for characterizing noise in two-photon (2P) calcium imaging
data. Runs a battery of independent tests on a sampled subset of frames and
maps the metrics to a calibrated likelihood per known noise source.

Tests
-----
  1. photon_transfer         gain (DN/e-), read noise (DN), shot-vs-read regime
  2. bidirectional           even/odd row phase offset, sub-pixel shift
  3. spectral                narrowband peaks on fast & slow axes; stationarity
  4. edge_artifacts          galvo-flyback dead columns/rows
  5. hot_pixels              fixed bright/dead pixels by local-z + low temp var
  6. drift                   photobleaching, PMT warm-up, illumination drift
  7. fixed_pattern           static spatial pattern vs shot noise floor
  8. saturation              ADC ceiling/floor clipping & dynamic-range usage
  9. frame_discontinuity     frame drops, sync glitches

Mapped sources (with `negligible / low / moderate / high` likelihood)
---------------------------------------------------------------------
  shot_noise_dominated, bidirectional_phase_offset,
  horizontal_banding_fixed, horizontal_banding_drifting,
  fast_axis_periodic, galvo_flyback_edge, hot_dead_pixels,
  photobleaching, illumination_drift_increase, fixed_pattern_noise,
  saturation_clipping, quantization_loss, frame_discontinuity

Usage
-----
  from noise_diagnostics import run_diagnostics
  report = run_diagnostics("rec.tif", out_dir="diag_out", n_frames=500)

  # or CLI
  python noise_diagnostics.py rec.tif --out diag_out --n_frames 500

Supported input formats
-----------------------
  .tif / .tiff / .btf       via tifffile (is_ome=False — bypasses OME companion linking)
  .npy                      memory-mapped read
  .msr                      Leica LAS X via caiman.utils.stack_io (IMSpectorReader)
  .h5 / .hdf5 / .nwb        via caiman.utils.stack_io (cm.load dispatch)
  np.ndarray                direct (T, H, W) input

Outputs
-------
  diag_out/diagnostic_report.json   full numeric report
  diag_out/diagnostic_panel.png     9-panel visual summary
  diag_out/summary.txt              human-readable top-issue ranking

Dependencies: numpy, scipy, matplotlib, tifffile (optional, for .tif input).
caiman.utils.stack_io is required only for .msr / .h5 / .nwb inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage, signal as sps, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("noise_diag")


# ============================================================================
# Frame loading
# ============================================================================

ArrayLike = Union[np.ndarray, str, Path]


def _load_subset(src: ArrayLike,
                 n_frames: int = 500,
                 rng_seed: int = 0) -> Tuple[np.ndarray, dict]:
    """Sample n_frames from src and return (frames, info).

    For tif inputs, only the sampled pages are decoded — safe on multi-GB stacks.
    For msr/h5/nwb inputs, dispatches via caiman.utils.stack_io.

    Parameters
    ----------
    src : ndarray or path
        (T, H, W) array or path to .tif/.tiff/.btf/.npy/.msr/.h5/.nwb.
    n_frames : int
        Number of frames to sample (random without replacement).
    rng_seed : int
        Reproducible sampling.

    Returns
    -------
    (frames, info)
        frames: float32 ndarray (n, H, W)
        info  : dict with n_total, n_sampled, source, dtype_orig, fmax_orig
    """
    rng = np.random.default_rng(rng_seed)

    if isinstance(src, np.ndarray):
        if src.ndim != 3:
            raise ValueError(f"expected (T, H, W); got {src.shape}")
        T = src.shape[0]
        n = min(n_frames, T)
        idx = np.sort(rng.choice(T, size=n, replace=False))
        return (src[idx].astype(np.float32),
                dict(n_total=int(T), n_sampled=int(n), source="ndarray",
                     dtype_orig=str(src.dtype), fmax_orig=float(src.max())))

    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()

    if suf == ".npy":
        a = np.load(path, mmap_mode="r")
        if a.ndim != 3:
            raise ValueError(f"expected (T, H, W); got {a.shape}")
        T = a.shape[0]
        n = min(n_frames, T)
        idx = np.sort(rng.choice(T, size=n, replace=False))
        sub = np.asarray(a[idx], dtype=np.float32)
        return (sub, dict(n_total=int(T), n_sampled=int(n), source=str(path),
                          dtype_orig=str(a.dtype),
                          fmax_orig=float(sub.max())))

    if suf in (".tif", ".tiff", ".btf"):
        try:
            import tifffile
        except ImportError as e:
            raise RuntimeError("tifffile required for .tif input") from e
        # is_ome=False avoids loading sibling companion files for OME-TIFF
        with tifffile.TiffFile(path, is_ome=False) as tf:
            n_pages = len(tf.pages)
            n = min(n_frames, n_pages)
            idx = np.sort(rng.choice(n_pages, size=n, replace=False))
            stacks = [tf.pages[int(i)].asarray() for i in idx]
        a = np.stack(stacks).astype(np.float32)
        return (a, dict(n_total=int(n_pages), n_sampled=int(n), source=str(path),
                        dtype_orig=str(stacks[0].dtype),
                        fmax_orig=float(a.max())))

    if suf in (".msr", ".h5", ".hdf5", ".nwb"):
        # Format-agnostic dispatch via caiman.utils.stack_io.  Picks up MSR
        # (Leica LAS X resonant-scanner raw files via IMSpectorReader),
        # HDF5, and NWB.  Random sampling matches the .tif branch above so
        # test statistics are comparable across formats.
        try:
            from caiman.utils.stack_io import stack_size, stack_sample
        except ImportError as e:
            raise RuntimeError(
                f"{suf} input requires caiman.utils.stack_io"
            ) from e
        dims, T_total = stack_size(path)
        n = min(n_frames, T_total)
        idx = np.sort(rng.choice(T_total, size=n, replace=False))
        a = stack_sample(path, idx, dtype=np.float32)
        # Defensive squeeze for backends that yield (T, 1, H, W) or
        # (T, H, W, 1) — IMSpectorReader returns (T, H, W) but cm.load may
        # add singleton axes for some HDF5/NWB layouts.
        if a.ndim == 4 and a.shape[1] == 1:
            a = a[:, 0]
        elif a.ndim == 4 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim != 3:
            raise ValueError(
                f"unexpected frame shape from stack_sample: {a.shape}"
            )
        return (a, dict(n_total=int(T_total), n_sampled=int(n),
                        source=str(path),
                        dtype_orig="via stack_sample (cast to float32)",
                        fmax_orig=float(a.max())))

    raise ValueError(f"unsupported source: {src!r}")


# ============================================================================
# Individual diagnostic tests
# ============================================================================

def test_photon_transfer(stack: np.ndarray) -> dict:
    """Estimate gain (DN/e-) and read noise from temporal-difference PTC.

    For Poisson-dominated signal,  Var(f) = gain * mean + read_var.
    Var(f[t+1] - f[t]) / 2 estimates per-pixel noise variance with slow
    signal cancellation.  We bin pixels by intensity and fit a line.

    Median-of-squared-diffs makes the variance robust to occasional bright
    transients (calcium events).

    Returns
    -------
    dict with:
        gain_dn_per_e        slope estimate
        read_noise_dn        sqrt(intercept) if positive, else NaN
        fit_r2               R² of the variance/mean line
        shot_dominant_frac   fraction of pixels where shot >> read at typical DN
        n_bins_used          number of intensity bins with enough pixels
    """
    if stack.shape[0] < 2:
        return dict(gain_dn_per_e=np.nan, read_noise_dn=np.nan,
                    fit_r2=np.nan, shot_dominant_frac=np.nan, n_bins_used=0)

    # Per-pixel temporal mean and robust noise variance from diffs
    d = np.diff(stack, axis=0)
    # Robust variance: median(d**2) / 2 estimates the noise variance
    # (median of chi-squared(1) ≈ 0.4549; rescale by 1/0.4549 ≈ 2.198)
    # but for the line fit only the relative scale matters, so we just use
    # median(d**2) / 2 which is consistent across bins.
    noise_var = np.median(d ** 2, axis=0) / 2.0
    mean_dn   = stack.mean(axis=0)

    # Flatten and bin by intensity
    mean_flat = mean_dn.ravel()
    var_flat  = noise_var.ravel()

    # Drop pixels with zero or NaN variance (dead pixels, edges) — they bias
    # the intercept toward zero.
    valid = np.isfinite(var_flat) & (var_flat > 0) & np.isfinite(mean_flat)
    mean_flat = mean_flat[valid]
    var_flat  = var_flat[valid]

    if mean_flat.size < 100:
        return dict(gain_dn_per_e=np.nan, read_noise_dn=np.nan,
                    fit_r2=np.nan, shot_dominant_frac=np.nan, n_bins_used=0)

    # Use log-spaced bins between the 1st and 99th percentile of intensity
    lo, hi = np.percentile(mean_flat, [1, 99])
    if hi <= lo or lo <= 0:
        # Linear fall-back if the percentile range is degenerate
        edges = np.linspace(mean_flat.min(), mean_flat.max(), 33)
    else:
        edges = np.geomspace(max(lo, 1.0), hi, 33)
    bin_idx = np.digitize(mean_flat, edges)

    bin_x, bin_y = [], []
    for b in range(1, len(edges)):
        sel = bin_idx == b
        if sel.sum() < 30:
            continue
        bin_x.append(np.median(mean_flat[sel]))
        bin_y.append(np.median(var_flat[sel]))
    bin_x = np.asarray(bin_x)
    bin_y = np.asarray(bin_y)

    if bin_x.size < 4:
        return dict(gain_dn_per_e=np.nan, read_noise_dn=np.nan,
                    fit_r2=np.nan, shot_dominant_frac=np.nan,
                    n_bins_used=int(bin_x.size))

    # Linear regression var = gain * mean + read_var
    slope, intercept, r, _, _ = stats.linregress(bin_x, bin_y)
    read_noise_dn = float(np.sqrt(intercept)) if intercept > 0 else float("nan")

    # Shot dominance at typical intensity
    typ_dn   = float(np.median(mean_flat))
    shot_var = max(slope, 1e-9) * typ_dn
    shot_dom = shot_var / max(shot_var + max(intercept, 0), 1e-9)

    return dict(
        gain_dn_per_e=float(slope),
        read_noise_dn=read_noise_dn,
        fit_r2=float(r ** 2),
        shot_dominant_frac=float(shot_dom),
        n_bins_used=int(bin_x.size),
        bin_means=bin_x.tolist(),
        bin_vars=bin_y.tolist(),
    )


def test_bidirectional(stack: np.ndarray) -> dict:
    """Detect bidirectional resonant-scanner phase offset.

    Even and odd rows are scanned in opposite directions; phase error causes
    a sub-pixel column shift between them.  Compute mean projection, then
    cross-correlate even-row mean against odd-row mean to find the shift.

    Also reports the row-mean alternation: a sawtooth-like difference
    between adjacent rows of the temporal mean image — diagnostic of
    intensity-level (not position) bidirectional artefact.

    Returns
    -------
    dict with:
        bidir_shift_px       sub-pixel even-row column shift
        alternation_db       odd-vs-even row mean alternation strength (dB)
        even_mean_dn         mean DN on even rows
        odd_mean_dn          mean DN on odd rows
    """
    if stack.shape[1] < 4:
        return dict(bidir_shift_px=0.0, alternation_db=-np.inf,
                    even_mean_dn=0.0, odd_mean_dn=0.0)

    proj = stack.mean(axis=0)              # (H, W)
    even = proj[0::2].mean(axis=0)         # row-axis mean -> (W,)
    odd  = proj[1::2].mean(axis=0)

    # Cross-correlate even vs odd row profiles (1D)
    n = even.size
    f1 = (even - even.mean()) / (even.std() + 1e-9)
    f2 = (odd  - odd.mean())  / (odd.std()  + 1e-9)
    # Limit search to ±8 px lag
    max_lag = min(8, n // 4)
    lags = np.arange(-max_lag, max_lag + 1)
    xcorr = np.array([np.sum(f1[max(0, k):n + min(0, k)]
                              * f2[max(0, -k):n + min(0, -k)])
                       for k in lags])
    k_peak = int(np.argmax(xcorr))
    # Parabolic refinement for sub-pixel peak
    if 0 < k_peak < xcorr.size - 1:
        y0, y1, y2 = xcorr[k_peak - 1], xcorr[k_peak], xcorr[k_peak + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
    else:
        delta = 0.0
    shift_px = float(lags[k_peak] + delta)

    # Alternation: every-other-row mean difference in the temporal mean
    row_means = proj.mean(axis=1)
    alt = row_means[1:-1:2].mean() - row_means[0:-1:2].mean()
    base = row_means.mean()
    alt_db = 20.0 * np.log10(abs(alt) / max(abs(base), 1e-9) + 1e-12)

    return dict(
        bidir_shift_px=shift_px,
        alternation_db=float(alt_db),
        even_mean_dn=float(even.mean()),
        odd_mean_dn=float(odd.mean()),
    )


def test_spectral(stack: np.ndarray) -> dict:
    """Detect narrowband peaks along fast (column) and slow (row) axes.

    For each axis, compute the 1D power spectrum of the temporal mean image
    along that axis (averaged across the other axis), then look for peaks
    that stand out from the local baseline by more than `peak_z_thresh`
    standard deviations on log-power.

    Stationarity is checked by comparing the power spectrum of the
    temporal-mean over the first half of frames vs the second half.

    Returns
    -------
    dict with:
        peak_freq_fast_cyc_px   normalized frequency of dominant fast-axis peak
        peak_power_fast_db      its prominence in dB above local median
        peak_freq_slow_cyc_px   ditto, slow axis
        peak_power_slow_db
        stationarity_corr       Pearson r between first-half and second-half spectra
    """
    T = stack.shape[0]

    def _peaks_1d(profile: np.ndarray) -> Tuple[float, float]:
        # Centered, mean-removed FFT
        p = profile - profile.mean()
        # Avoid edge ringing via Hann window
        w = np.hanning(p.size)
        spec = np.abs(np.fft.rfft(p * w)) ** 2
        freqs = np.fft.rfftfreq(p.size, d=1.0)   # cycles per pixel
        # Skip DC and the very lowest bins (slow trend)
        spec[:2] = 0
        if spec.max() <= 0:
            return 0.0, -np.inf
        log_spec = np.log10(spec + 1e-12)
        # Local baseline = median-filtered log-spectrum
        baseline = ndimage.median_filter(log_spec, size=max(5, len(log_spec) // 16))
        prom = log_spec - baseline
        k = int(np.argmax(prom))
        peak_freq  = float(freqs[k])
        peak_db    = float(10.0 * prom[k])
        return peak_freq, peak_db

    # Fast axis: column-wise FFT of each row, average across rows
    mean_img = stack.mean(axis=0)
    col_fft = np.array([_peaks_1d(mean_img[r]) for r in range(mean_img.shape[0])])
    # Use the median peak across rows to avoid one outlier row dominating
    fast_freq = float(np.median(col_fft[:, 0]))
    fast_db   = float(np.median(col_fft[:, 1]))

    # Slow axis: row-wise FFT of each column
    row_fft = np.array([_peaks_1d(mean_img[:, c]) for c in range(mean_img.shape[1])])
    slow_freq = float(np.median(row_fft[:, 0]))
    slow_db   = float(np.median(row_fft[:, 1]))

    # Stationarity: compare mean spectrum over first half vs second half
    if T >= 4:
        m1 = stack[: T // 2].mean(axis=0)
        m2 = stack[T // 2:].mean(axis=0)
        s1 = np.abs(np.fft.rfft(m1.mean(axis=0))) ** 2
        s2 = np.abs(np.fft.rfft(m2.mean(axis=0))) ** 2
        s1 = np.log10(s1[2:] + 1e-12)
        s2 = np.log10(s2[2:] + 1e-12)
        if s1.size > 4:
            stationarity = float(np.corrcoef(s1, s2)[0, 1])
        else:
            stationarity = 1.0
    else:
        stationarity = 1.0

    return dict(
        peak_freq_fast_cyc_px=fast_freq,
        peak_power_fast_db=fast_db,
        peak_freq_slow_cyc_px=slow_freq,
        peak_power_slow_db=slow_db,
        stationarity_corr=stationarity,
    )


def test_edge_artifacts(stack: np.ndarray) -> dict:
    """Detect galvo-flyback dead columns/rows at FOV edges.

    Compute the temporal mean image, then walk from each edge inward
    counting rows/columns whose mean intensity is below 10% of the global
    mean — these are the dead-stripe galvo-flyback artefacts.

    Returns
    -------
    dict with dead_{top,bottom,left,right} counts.
    """
    proj = stack.mean(axis=0)
    g = proj.mean()
    thr = 0.10 * g

    rows_mean = proj.mean(axis=1)
    cols_mean = proj.mean(axis=0)

    def _walk(arr):
        n = 0
        for v in arr:
            if v < thr:
                n += 1
            else:
                break
        return n

    return dict(
        dead_top=_walk(rows_mean),
        dead_bottom=_walk(rows_mean[::-1]),
        dead_left=_walk(cols_mean),
        dead_right=_walk(cols_mean[::-1]),
        global_mean=float(g),
        threshold=float(thr),
    )


def test_hot_pixels(stack: np.ndarray) -> dict:
    """Detect fixed hot/dead pixels via spatial outlier z-score plus low
    temporal variance.

    A hot pixel sits orders of magnitude above its 3×3 neighbourhood in
    the temporal mean *and* has low temporal variance (it's stuck high).
    A dead pixel is the same but stuck low.

    Returns
    -------
    dict with hot_fraction, dead_fraction, hot_count, dead_count.
    """
    mean_img = stack.mean(axis=0)
    var_img  = stack.var(axis=0)

    # Local-z: subtract a 3x3 median, divide by 3x3 MAD-like spread
    local_med = ndimage.median_filter(mean_img, size=3)
    local_mad = ndimage.median_filter(np.abs(mean_img - local_med), size=3)
    local_z   = (mean_img - local_med) / (local_mad + 1e-3)

    # Temporal variance percentile (low var = stuck pixel)
    var_p10 = np.percentile(var_img, 10)
    low_var = var_img < (0.1 * var_p10 + 1e-9)

    hot  = (local_z >  10) & low_var
    dead = (local_z < -10) & low_var

    total = mean_img.size
    return dict(
        hot_count=int(hot.sum()),
        dead_count=int(dead.sum()),
        hot_fraction=float(hot.sum()) / total,
        dead_fraction=float(dead.sum()) / total,
        hot_coords=list(map(tuple, np.argwhere(hot).tolist()[:20])),
    )


def test_drift(stack: np.ndarray) -> dict:
    """Detect photobleaching, PMT warm-up, or illumination drift.

    Per-frame mean intensity should be approximately stationary for a
    well-conditioned recording.  Fit:

      - linear trend (slope as % of mean per recording)
      - single-exponential decay A * exp(-t/tau) + B (photobleaching signature)

    Returns
    -------
    dict with drift_pct, decay_tau_frames, decay_amp_frac, direction.
    """
    f_mean = stack.mean(axis=(1, 2))
    T = f_mean.size
    t = np.arange(T)

    if T < 3:
        return dict(drift_pct=0.0, decay_tau_frames=np.inf,
                    decay_amp_frac=0.0, direction="none")

    slope, intercept, _, _, _ = stats.linregress(t, f_mean)
    drift_pct = float(slope * T / max(intercept, 1e-9) * 100.0)
    direction = ("decrease" if slope < -1e-6 else
                 "increase" if slope >  1e-6 else "none")

    # Exponential fit: f(t) = A * exp(-t/tau) + B
    try:
        from scipy.optimize import curve_fit
        def _exp(x, A, tau, B):
            return A * np.exp(-x / max(tau, 1e-3)) + B
        # Initial guesses
        A0   = float(f_mean[0] - f_mean[-1])
        tau0 = T / 3.0
        B0   = float(f_mean[-1])
        popt, _ = curve_fit(_exp, t, f_mean, p0=[A0, tau0, B0],
                            maxfev=2000)
        A, tau, B = popt
        # Amplitude as fraction of total mean
        amp_frac = float(abs(A) / max(abs(B) + abs(A), 1e-9))
        tau_frames = float(max(tau, 1.0))
    except Exception:
        amp_frac = 0.0
        tau_frames = float("inf")

    return dict(
        drift_pct=drift_pct,
        decay_tau_frames=tau_frames,
        decay_amp_frac=amp_frac,
        direction=direction,
        frame_means=f_mean.astype(float).tolist(),
    )


def test_fixed_pattern(stack: np.ndarray) -> dict:
    """Detect fixed pattern noise (FPN): static spatial structure that
    persists across frames beyond what shot noise would produce.

    Approach: temporal mean shows both true scene structure and any FPN
    that doesn't average out.  High-pass the temporal mean (subtract a
    Gaussian-blurred copy) to isolate fine-scale spatial pattern, then
    compare its variance to the expected shot-noise floor.

    Returns
    -------
    dict with fpn_strength (high-pass variance / shot floor),
             hp_variance (raw), shot_floor (raw).
    """
    proj = stack.mean(axis=0)
    blurred = ndimage.gaussian_filter(proj, sigma=2.0)
    hp = proj - blurred
    hp_var = float(hp.var())

    # Expected shot-noise variance on the mean = mean(image) / N_frames
    # under unit-gain Poisson assumption.  Multiply by ~empirical gain
    # estimate if available, else use 1.
    N = stack.shape[0]
    shot_floor = float(proj.mean() / max(N, 1))
    fpn_strength = float(hp_var / max(shot_floor, 1e-9))

    return dict(
        fpn_strength=fpn_strength,
        hp_variance=hp_var,
        shot_floor=shot_floor,
    )


def test_saturation_quantization(stack: np.ndarray, info: dict) -> dict:
    """Detect ADC ceiling/floor clipping and effective bit-depth issues.

    Fraction of pixels at the original max value indicates saturation;
    fraction at zero indicates floor clipping.  Effective bit-depth is
    estimated from the histogram of unique values vs the recording length.

    Returns
    -------
    dict with sat_fraction, floor_fraction, effective_bits, dynamic_range_usage.
    """
    fmax = info.get("fmax_orig", stack.max())
    dtype_orig = info.get("dtype_orig", "")
    # ADC ceiling estimate from dtype
    if "uint16" in dtype_orig:
        adc_max = 65535
    elif "uint8" in dtype_orig:
        adc_max = 255
    elif "int16" in dtype_orig:
        adc_max = 32767
    else:
        adc_max = fmax

    sat_frac = float((stack >= 0.99 * adc_max).mean())
    floor_frac = float((stack <= 0.5).mean())

    # Dynamic range usage = p99/adc_max
    p99 = float(np.percentile(stack, 99))
    dr_usage = float(p99 / max(adc_max, 1))

    # Effective bits: log2 of (p99 - p1) / typical noise level
    p01 = float(np.percentile(stack, 1))
    noise_est = max(np.median(np.std(stack[::max(1, stack.shape[0] // 8)], axis=0)),
                    1.0)
    eff_bits = float(np.log2(max(p99 - p01, 1.0) / noise_est))

    return dict(
        sat_fraction=sat_frac,
        floor_fraction=floor_frac,
        adc_max_est=float(adc_max),
        p99=p99,
        dynamic_range_usage=dr_usage,
        effective_bits=eff_bits,
    )


def test_frame_discontinuity(stack: np.ndarray) -> dict:
    """Detect frame drops / sync glitches via outlier jumps in frame mean.

    Compute diff of per-frame mean.  Z-score the diff against its MAD-based
    spread.  Frames whose entering jump exceeds 6σ are flagged.

    Returns
    -------
    dict with discontinuity_count, max_jump_z, mean_jump_z, glitch_frames.
    """
    f_mean = stack.mean(axis=(1, 2))
    if f_mean.size < 3:
        return dict(discontinuity_count=0, max_jump_z=0.0,
                    mean_jump_z=0.0, glitch_frames=[])

    d = np.diff(f_mean)
    mad = np.median(np.abs(d - np.median(d)))
    sigma = max(1.4826 * mad, 1e-9)
    z = (d - np.median(d)) / sigma
    glitches = np.where(np.abs(z) > 6.0)[0].tolist()

    return dict(
        discontinuity_count=int(len(glitches)),
        max_jump_z=float(np.max(np.abs(z))) if z.size else 0.0,
        mean_jump_z=float(np.mean(np.abs(z))) if z.size else 0.0,
        glitch_frames=glitches[:50],
    )


# ============================================================================
# Source scoring
# ============================================================================

LEVELS = ("negligible", "low", "moderate", "high")


def _level(score: float) -> str:
    if   score < 0.2: return "negligible"
    elif score < 0.4: return "low"
    elif score < 0.7: return "moderate"
    else:             return "high"


def score_sources(report: dict) -> dict:
    """Map raw test metrics to per-source likelihood + recommendation.

    Each source gets:
        level         negligible / low / moderate / high
        score         continuous 0–1
        evidence      contributing metrics
        recommendation one-line action
    """
    pt   = report["photon_transfer"]
    bi   = report["bidirectional"]
    sp   = report["spectral"]
    ed   = report["edge_artifacts"]
    hp   = report["hot_pixels"]
    dr   = report["drift"]
    fp   = report["fixed_pattern"]
    st   = report["saturation"]
    fd   = report["frame_discontinuity"]

    out: Dict[str, dict] = {}

    # --- shot_noise_dominated (informational — high is healthy) -------------
    shot = pt.get("shot_dominant_frac", 0.0) or 0.0
    out["shot_noise_dominated"] = dict(
        level=_level(shot), score=float(shot),
        evidence=dict(shot_dominant_frac=shot, fit_r2=pt.get("fit_r2")),
        recommendation="No action — high shot dominance indicates well-conditioned acquisition.",
    )

    # --- bidirectional_phase_offset ----------------------------------------
    shift = abs(bi.get("bidir_shift_px", 0.0) or 0.0)
    # The row-averaged xcorr method under-reports true sub-pixel shifts by
    # roughly a factor of 3 (validated against injected 0.6 px shifts that
    # come back as ~0.2 px). Calibrate the score divisor accordingly: a
    # detector reading of 0.4 px corresponds to ~1.2 px true shift, which
    # is unambiguously bad. The xcorr value is still reported as evidence
    # so the operator sees the actual reading.
    s_bi = float(min(shift / 0.4, 1.0))
    out["bidirectional_phase_offset"] = dict(
        level=_level(s_bi), score=s_bi,
        evidence=dict(bidir_shift_px=shift, alternation_db=bi.get("alternation_db")),
        recommendation="Enable xcorr_correction.enabled in pipeline JSON.",
    )

    # --- horizontal_banding_fixed (slow-axis peak, stationary) -------------
    slow_db   = sp.get("peak_power_slow_db", -np.inf) or -np.inf
    stationary = sp.get("stationarity_corr", 1.0) or 1.0
    # Calibration: clean Poisson stack produces spurious peaks at 5–10 dB
    # against the per-pixel baseline (varies a bit by seed). Inject-grade
    # real bands hit 15–25 dB. Floor at 8 dB makes the score 0 for natural
    # variation and saturates at 20 dB above the floor. The stationarity
    # multiplier then partitions the score between *_fixed and *_drifting.
    band_fixed_score = (
        float(min(max((slow_db - 8.0) / 12.0, 0.0), 1.0))
        * float(max(stationary, 0))
    )
    out["horizontal_banding_fixed"] = dict(
        level=_level(band_fixed_score), score=band_fixed_score,
        evidence=dict(peak_power_slow_db=slow_db,
                      peak_freq_slow_cyc_px=sp.get("peak_freq_slow_cyc_px"),
                      stationarity_corr=stationary),
        recommendation="Add row-pedestal subtraction (rank-1 per-frame row offset).",
    )

    # --- horizontal_banding_drifting (slow-axis peak, low stationarity) ----
    band_drift_score = (
        float(min(max((slow_db - 8.0) / 12.0, 0.0), 1.0))
        * float(max(1 - stationary, 0))
    )
    out["horizontal_banding_drifting"] = dict(
        level=_level(band_drift_score), score=band_drift_score,
        evidence=dict(peak_power_slow_db=slow_db,
                      stationarity_corr=stationary),
        recommendation="Apply per-frame 1D notch filter on the column-FFT.",
    )

    # --- fast_axis_periodic (column-FFT peak) ------------------------------
    fast_db = sp.get("peak_power_fast_db", -np.inf) or -np.inf
    fast_score = float(min(max((fast_db - 10.0) / 15.0, 0.0), 1.0))
    out["fast_axis_periodic"] = dict(
        level=_level(fast_score), score=fast_score,
        evidence=dict(peak_power_fast_db=fast_db,
                      peak_freq_fast_cyc_px=sp.get("peak_freq_fast_cyc_px")),
        recommendation="Check pixel-clock / sample-hold timing on the acquisition card.",
    )

    # --- galvo_flyback_edge -----------------------------------------------
    dead_total = (ed.get("dead_top", 0) + ed.get("dead_bottom", 0)
                  + ed.get("dead_left", 0) + ed.get("dead_right", 0))
    # 8 dead cols/rows total → moderate (0.5); 16 → high (1.0). The
    # CaImAn motion-correction border_pix is typically 8–16, so anything
    # ≥8 is worth flagging.
    edge_score = float(min(dead_total / 16.0, 1.0))
    out["galvo_flyback_edge"] = dict(
        level=_level(edge_score), score=edge_score,
        evidence=dict(dead_top=ed.get("dead_top"),
                      dead_bottom=ed.get("dead_bottom"),
                      dead_left=ed.get("dead_left"),
                      dead_right=ed.get("dead_right")),
        recommendation=f"Raise motion_correction.border_pix to at least "
                       f"{max(ed.get('dead_top',0), ed.get('dead_bottom',0), ed.get('dead_left',0), ed.get('dead_right',0)) + 1}.",
    )

    # --- hot_dead_pixels --------------------------------------------------
    hd = (hp.get("hot_fraction", 0.0) + hp.get("dead_fraction", 0.0))
    # 1e-5 → low; 1e-4 → moderate; 1e-3 → high
    hp_score = float(min(hd * 1e4, 1.0))
    out["hot_dead_pixels"] = dict(
        level=_level(hp_score), score=hp_score,
        evidence=dict(hot_count=hp.get("hot_count"),
                      dead_count=hp.get("dead_count"),
                      hot_fraction=hp.get("hot_fraction"),
                      dead_fraction=hp.get("dead_fraction")),
        recommendation="Median-replace hot/dead pixels before motion correction.",
    )

    # --- photobleaching ---------------------------------------------------
    tau = dr.get("decay_tau_frames", float("inf")) or float("inf")
    amp = dr.get("decay_amp_frac",   0.0) or 0.0
    T = len(dr.get("frame_means", []))
    # photobleaching detectable when tau < T (decay completes within
    # recording) AND amplitude is meaningful
    if T > 0 and np.isfinite(tau):
        decay_completion = float(min(T / max(tau, 1e-3) / 3.0, 1.0))  # ≥3τ ⇒ 1.0
        photobleach_score = float(decay_completion * min(amp / 0.3, 1.0))
    else:
        photobleach_score = 0.0
    out["photobleaching"] = dict(
        level=_level(photobleach_score), score=photobleach_score,
        evidence=dict(decay_tau_frames=tau, decay_amp_frac=amp,
                      direction=dr.get("direction")),
        recommendation="Apply pixel-wise detrending pre-CNMF, or accept and rely on detrend_df_f after.",
    )

    # --- illumination_drift_increase --------------------------------------
    drift_pct = dr.get("drift_pct", 0.0) or 0.0
    drift_score = float(min(max(drift_pct / 30.0, 0.0), 1.0)) if drift_pct > 0 else 0.0
    out["illumination_drift_increase"] = dict(
        level=_level(drift_score), score=drift_score,
        evidence=dict(drift_pct=drift_pct, direction=dr.get("direction")),
        recommendation="Drop the warm-up frames (typically first 100–500) or run with PMT pre-warmed.",
    )

    # --- fixed_pattern_noise ----------------------------------------------
    fpn = fp.get("fpn_strength", 0.0) or 0.0
    # The ratio is unitless; calibrate roughly: <100 negligible, ≥3000 high
    fpn_score = float(min(max((np.log10(max(fpn, 1.0)) - 2.0) / 2.0, 0.0), 1.0))
    out["fixed_pattern_noise"] = dict(
        level=_level(fpn_score), score=fpn_score,
        evidence=dict(fpn_strength=fpn,
                      hp_variance=fp.get("hp_variance"),
                      shot_floor=fp.get("shot_floor")),
        recommendation="Acquire a dark frame and subtract before processing.",
    )

    # --- saturation_clipping ----------------------------------------------
    sat_frac = st.get("sat_fraction", 0.0) or 0.0
    sat_score = float(min(sat_frac * 200.0, 1.0))   # 0.5% → 1.0
    out["saturation_clipping"] = dict(
        level=_level(sat_score), score=sat_score,
        evidence=dict(sat_fraction=sat_frac, adc_max_est=st.get("adc_max_est"),
                      p99=st.get("p99")),
        recommendation="Reduce PMT gain at acquisition — software cannot recover clipped photons.",
    )

    # --- quantization_loss -------------------------------------------------
    # Low effective_bits (<6) OR very low dynamic-range usage (<5%) is bad
    eff_bits = st.get("effective_bits", 16.0) or 16.0
    dr_use   = st.get("dynamic_range_usage", 1.0) or 1.0
    qz_score = float(max(min((8.0 - eff_bits) / 4.0, 1.0), 0.0))
    qz_score = max(qz_score, float(max(min((0.05 - dr_use) / 0.05, 1.0), 0.0)))
    out["quantization_loss"] = dict(
        level=_level(qz_score), score=qz_score,
        evidence=dict(effective_bits=eff_bits, dynamic_range_usage=dr_use),
        recommendation="Increase PMT gain at acquisition — signal is too small relative to ADC step.",
    )

    # --- frame_discontinuity ----------------------------------------------
    n_glitch = fd.get("discontinuity_count", 0) or 0
    T = len(dr.get("frame_means", [])) or 1
    g_score = float(min(n_glitch / max(T * 0.01, 1.0), 1.0))
    out["frame_discontinuity"] = dict(
        level=_level(g_score), score=g_score,
        evidence=dict(discontinuity_count=n_glitch,
                      max_jump_z=fd.get("max_jump_z"),
                      glitch_frames=fd.get("glitch_frames", [])[:10]),
        recommendation="Drop / interpolate flagged glitch frames before processing.",
    )

    return out


# ============================================================================
# Reporting
# ============================================================================

def print_summary(report: dict, fh=None) -> None:
    """Print a ranked summary of detected noise sources."""
    sources = report.get("sources", {})
    if not sources:
        return
    # Sort by level then score desc
    order = sorted(
        sources.items(),
        key=lambda kv: (LEVELS.index(kv[1]["level"]), kv[1]["score"]),
        reverse=True,
    )
    lines = [
        "",
        "=" * 72,
        "noise_diagnostics — source ranking",
        "=" * 72,
    ]
    info = report.get("info", {})
    if info:
        lines.append(f"  source     : {info.get('source')}")
        lines.append(f"  n_total    : {info.get('n_total')}")
        lines.append(f"  n_sampled  : {info.get('n_sampled')}")
        lines.append(f"  dtype_orig : {info.get('dtype_orig')}")
        lines.append(f"  fmax_orig  : {info.get('fmax_orig')}")
        lines.append("-" * 72)
    for name, d in order:
        line = f"  [{d['level']:>10}]  {name:<32}  score={d['score']:.3f}"
        lines.append(line)
    lines.append("-" * 72)
    # And the action items for any non-negligible source
    actions = [(n, d) for n, d in order if d["level"] != "negligible"]
    if actions:
        lines.append("Recommended actions:")
        for n, d in actions:
            lines.append(f"  * {n:<32}  {d['recommendation']}")
    lines.append("=" * 72)
    text = "\n".join(lines)
    print(text, file=fh if fh is not None else sys.stdout)


def plot_panel(stack: np.ndarray, report: dict, out_path: Path) -> None:
    """Render a 9-panel visual summary into out_path (PNG)."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes = axes.ravel()
    proj = stack.mean(axis=0)

    pt = report["photon_transfer"]
    bi = report["bidirectional"]
    sp = report["spectral"]
    ed = report["edge_artifacts"]
    hp = report["hot_pixels"]
    dr = report["drift"]
    fp = report["fixed_pattern"]
    st = report["saturation"]
    fd = report["frame_discontinuity"]

    # 1. Temporal mean image
    ax = axes[0]
    im = ax.imshow(proj, cmap="gray")
    ax.set_title("Temporal mean")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2. PTC fit
    ax = axes[1]
    if pt.get("bin_means"):
        bx = np.asarray(pt["bin_means"])
        by = np.asarray(pt["bin_vars"])
        ax.loglog(bx, by, "o", markersize=4)
        ax.set_xlabel("Bin mean DN")
        ax.set_ylabel("Bin noise var")
        x_fit = np.linspace(bx.min(), bx.max(), 50)
        gain = pt.get("gain_dn_per_e", 0)
        rn   = pt.get("read_noise_dn", 0) or 0
        rn_var = rn ** 2 if np.isfinite(rn) else 0
        ax.loglog(x_fit, gain * x_fit + rn_var, "r-",
                  label=f"gain={gain:.2f} DN/e-, rn≈{rn:.2f} DN")
        ax.legend(fontsize=8)
    ax.set_title("Photon-transfer curve")

    # 3. 2D FFT magnitude (log)
    ax = axes[2]
    fft = np.fft.fftshift(np.abs(np.fft.fft2(proj - proj.mean())))
    ax.imshow(np.log1p(fft), cmap="magma")
    ax.set_title(
        f"2D FFT  |  slow {sp['peak_power_slow_db']:.1f} dB, "
        f"fast {sp['peak_power_fast_db']:.1f} dB"
    )

    # 4. Bidirectional row alternation
    ax = axes[3]
    row_means = proj.mean(axis=1)
    ax.plot(row_means)
    ax.set_title(
        f"Row-mean trace  |  bidir shift {bi['bidir_shift_px']:+.2f} px"
    )
    ax.set_xlabel("Row")

    # 5. Edge profiles
    ax = axes[4]
    ax.plot(proj.mean(axis=1), label="row-mean (top→bottom)")
    ax.plot(proj.mean(axis=0), label="col-mean (left→right)")
    ax.axhline(ed["threshold"], color="r", linestyle=":", label="10% threshold")
    ax.legend(fontsize=8, loc="lower center")
    ax.set_title(
        f"Edge profile  |  dead T/B/L/R = "
        f"{ed['dead_top']}/{ed['dead_bottom']}/{ed['dead_left']}/{ed['dead_right']}"
    )

    # 6. Hot-pixel overlay
    ax = axes[5]
    ax.imshow(proj, cmap="gray")
    for (y, x) in hp.get("hot_coords", []):
        ax.plot(x, y, "r+", markersize=8)
    ax.set_title(
        f"Hot/dead pixels  |  hot={hp['hot_count']}  dead={hp['dead_count']}"
    )

    # 7. Frame-mean trace
    ax = axes[6]
    f_mean = dr.get("frame_means", [])
    if f_mean:
        ax.plot(f_mean)
        ax.set_title(
            f"Frame-mean drift  |  Δ={dr['drift_pct']:.1f}%  "
            f"τ={dr['decay_tau_frames']:.0f} fr"
        )
        ax.set_xlabel("Frame")

    # 8. Fixed-pattern high-pass
    ax = axes[7]
    blurred = ndimage.gaussian_filter(proj, sigma=2.0)
    hp_img  = proj - blurred
    im = ax.imshow(hp_img, cmap="RdBu_r",
                   vmin=-3 * hp_img.std(), vmax=3 * hp_img.std())
    ax.set_title(f"Fixed-pattern (HP)  |  strength {fp['fpn_strength']:.0f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 9. Saturation + summary text
    ax = axes[8]
    ax.axis("off")
    txt = (
        f"Saturation\n"
        f"  sat_fraction       = {st['sat_fraction']:.4%}\n"
        f"  floor_fraction     = {st['floor_fraction']:.4%}\n"
        f"  dynamic-range use  = {st['dynamic_range_usage']:.1%}\n"
        f"  effective_bits     = {st['effective_bits']:.1f}\n\n"
        f"Frame discontinuity\n"
        f"  glitch_count       = {fd['discontinuity_count']}\n"
        f"  max_jump_z         = {fd['max_jump_z']:.1f}"
    )
    ax.text(0.05, 0.95, txt, family="monospace", fontsize=10,
            verticalalignment="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Orchestrator
# ============================================================================

def run_diagnostics(src: ArrayLike,
                    out_dir: Union[str, Path] = "diag_out",
                    n_frames: int = 500,
                    rng_seed: int = 0,
                    save_json: bool = True,
                    save_panel: bool = True,
                    write_summary: bool = True) -> dict:
    """Run the full diagnostic battery and write report/panel/summary.

    Returns the full report dict.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("loading frames from %s ...", src)
    stack, info = _load_subset(src, n_frames=n_frames, rng_seed=rng_seed)
    log.info("loaded %d frames of shape %s in %.1f s",
             info["n_sampled"], stack.shape, time.time() - t0)

    report: dict = dict(info=info)
    report["photon_transfer"]     = test_photon_transfer(stack)
    report["bidirectional"]       = test_bidirectional(stack)
    report["spectral"]            = test_spectral(stack)
    report["edge_artifacts"]      = test_edge_artifacts(stack)
    report["hot_pixels"]          = test_hot_pixels(stack)
    report["drift"]               = test_drift(stack)
    report["fixed_pattern"]       = test_fixed_pattern(stack)
    report["saturation"]          = test_saturation_quantization(stack, info)
    report["frame_discontinuity"] = test_frame_discontinuity(stack)
    report["sources"]             = score_sources(report)

    if save_json:
        with open(out / "diagnostic_report.json", "w") as fh:
            json.dump(report, fh, indent=2, default=str)
    if save_panel:
        plot_panel(stack, report, out / "diagnostic_panel.png")
    if write_summary:
        print_summary(report)
        with open(out / "summary.txt", "w") as fh:
            print_summary(report, fh)

    return report


# ============================================================================
# CLI
# ============================================================================

def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source",
                    help=".tif/.tiff/.btf/.npy/.msr/.h5/.nwb path")
    ap.add_argument("--out", default="diag_out",
                    help="output directory (default: diag_out)")
    ap.add_argument("--n_frames", type=int, default=500,
                    help="frames to sample (default: 500)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the random subset selection")
    ap.add_argument("--verbose", action="store_true",
                    help="info-level logging during the run")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_diagnostics(args.source, out_dir=args.out,
                    n_frames=args.n_frames, rng_seed=args.seed)


if __name__ == "__main__":
    _main()
