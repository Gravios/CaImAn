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

Outputs
-------
  diag_out/diagnostic_report.json   full numeric report
  diag_out/diagnostic_panel.png     9-panel visual summary
  diag_out/summary.txt              human-readable top-issue ranking

Dependencies: numpy, scipy, matplotlib, tifffile (optional, for .tif input).
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
    """
    rng = np.random.default_rng(rng_seed)

    if isinstance(src, np.ndarray):
        if src.ndim != 3:
            raise ValueError(f"expected (T, H, W); got {src.shape}")
        T = src.shape[0]
        n = min(n_frames, T)
        idx = np.sort(rng.choice(T, size=n, replace=False))
        return (src[idx].astype(np.float32),
                dict(n_total=T, n_sampled=n, source="ndarray",
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
        return (sub, dict(n_total=T, n_sampled=n, source=str(path),
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
        return (a, dict(n_total=n_pages, n_sampled=n, source=str(path),
                        dtype_orig=str(stacks[0].dtype),
                        fmax_orig=float(a.max())))

    raise ValueError(f"unsupported source: {src!r}")


# ============================================================================
# Individual diagnostic tests
# ============================================================================

def test_photon_transfer(stack: np.ndarray) -> dict:
    """Estimate gain (DN/e-) and read noise from temporal-difference PTC.

    Uses Var(f[t+1] - f[t]) / 2 as a per-pixel noise variance estimator
    (signal cancels for slow signal change). Bins pixels by intensity, fits
    var = gain * mean + read_var. Reports shot-vs-read dominance at typical
    intensity. Median-of-squared-diffs makes it robust to occasional calcium
    transients.
    """
    T = stack.shape[0]
    if T < 4:
        return dict(error="too few frames for PTC", n_frames=T)

    diffs = np.diff(stack, axis=0)                       # (T-1, H, W)
    # Robust per-pixel noise variance. median(d²) for d = N(0, 2σ²) equals
    # 2σ² × median(χ²₁) = 2σ² × 0.4549. So divide by (2 × 0.4549) to recover σ².
    # This is robust to occasional calcium transients.
    noise_var = np.median(diffs * diffs, axis=0) / (2.0 * 0.4549)  # (H, W)
    pixel_mean = stack.mean(axis=0)                      # (H, W)

    flat_m = pixel_mean.ravel()
    flat_v = noise_var.ravel()

    # Quantile-bin by intensity, median variance per bin (robust to outliers)
    nbins = 40
    edges = np.quantile(flat_m, np.linspace(0.02, 0.98, nbins + 1))
    centers, varbin = [], []
    for i in range(nbins):
        mask = (flat_m >= edges[i]) & (flat_m < edges[i + 1])
        if mask.sum() < 32:
            continue
        centers.append(float(flat_m[mask].mean()))
        varbin.append(float(np.median(flat_v[mask])))

    centers = np.array(centers)
    varbin = np.array(varbin)
    if centers.size < 3:
        return dict(error="not enough usable bins", n_bins=int(centers.size))

    slope, intercept, r, _, _ = stats.linregress(centers, varbin)
    gain = float(slope)
    read_var = float(intercept)
    read_dn = float(np.sqrt(read_var)) if read_var > 0 else float("nan")
    typ_m = float(np.median(flat_m))
    shot_var = gain * typ_m
    ratio = float(shot_var / max(read_var, 1e-6))

    return dict(
        gain_dn_per_e=gain,
        read_noise_dn=read_dn,
        intercept_var=read_var,
        fit_r2=float(r * r),
        typical_mean=typ_m,
        shot_to_read_ratio=ratio,
        pt_centers=centers.tolist(),
        pt_vars=varbin.tolist(),
    )


def test_bidirectional(stack: np.ndarray) -> dict:
    """Detect bidirectional resonant-scanner phase offset.

    Phase mismatch shifts even and odd rows in opposite directions along x.
    The best detector is to find the Δx that minimises adjacent-row
    discontinuity in the temporal-mean image. This is what suite2p uses.

    Algorithm:
      1. M = temporal-mean image, band-pass to retain mid-frequency structure
         (cell-scale, ~3-20 px) where bidirectional ringing is visible.
      2. For a grid of candidate shifts Δx in [-2, 2] px, shift the even rows
         by Δx (linear-interp) and compute Σ(row[2k+1] - row[2k]_shifted)² over
         all adjacent pairs. The shift that minimises this is the offset.
      3. Parabolic-interpolate around the minimum for sub-pixel.

    Also reports the every-other-row alternation power in the row-mean signal
    as an independent signature.
    """
    from scipy.ndimage import shift as nd_shift

    T, H, W = stack.shape
    n_pairs = H // 2

    # (a) Mean image, band-pass to suppress smooth FOV gradient
    M = stack.mean(axis=0)
    M_hp = M - ndimage.gaussian_filter(M, sigma=12)

    # (b) Adjacent-row residual as a function of even-row x-shift
    even = M_hp[0:2 * n_pairs:2, :]              # (n_pairs, W)
    odd  = M_hp[1:2 * n_pairs:2, :]
    # Edge mask to suppress contamination from the few flyback pixels
    edge = max(8, int(0.03 * W))
    candidates = np.linspace(-2.0, 2.0, 41)
    errs = np.empty_like(candidates)
    for i, s in enumerate(candidates):
        shifted = nd_shift(even, shift=(0, s), order=1, mode="nearest")
        d = (shifted - odd)[:, edge:W - edge]
        errs[i] = float(np.sum(d * d))
    k = int(np.argmin(errs))
    if 0 < k < errs.size - 1:
        y0, y1, y2 = errs[k - 1], errs[k], errs[k + 1]
        denom = (y0 - 2 * y1 + y2)
        dx = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        shift_px = float(candidates[k] + dx * (candidates[1] - candidates[0]))
    else:
        shift_px = float(candidates[k])

    # (c) Every-other-row alternation power (cheap independent check)
    row_mean = stack.mean(axis=2)                        # (T, H)
    even_rm = row_mean[:, 0:2 * n_pairs:2]
    odd_rm  = row_mean[:, 1:2 * n_pairs:2]
    bidir = (even_rm - odd_rm).mean(axis=0)
    bw = bidir * np.hanning(bidir.size)
    P = np.abs(np.fft.rfft(bw)) ** 2
    P[0] = 0
    nyq_power_ratio = float(P[-1] / max(P[1:].sum(), 1e-12))
    resid_std = float(np.std(bidir))

    return dict(
        bidir_shift_px=shift_px,
        nyquist_power_ratio=nyq_power_ratio,
        bidir_residual_std_dn=resid_std,
        bidir_residual_y_profile=bidir.tolist(),
        shift_search_grid=candidates.tolist(),
        shift_search_errs=errs.tolist(),
    )


def _peak_prominence_db(spectrum: np.ndarray,
                        smooth_sigma: float = 6.0,
                        skip_dc: int = 5,
                        prominence_db: float = 1.5) -> List[Tuple[int, float]]:
    """Peaks above a smoothed 1/f baseline, returned as [(bin_idx, prominence_dB)]."""
    s = spectrum.copy().astype(np.float64)
    s[:skip_dc] = 0.0
    baseline = ndimage.gaussian_filter1d(s, smooth_sigma, mode="nearest")
    eps = max(baseline.max(), 1.0) * 1e-9
    excess_db = 10.0 * np.log10((s + eps) / (baseline + eps))
    peaks, _ = sps.find_peaks(excess_db, prominence=prominence_db)
    return [(int(p), float(excess_db[p])) for p in peaks]


def _axis_spectrum(stack: np.ndarray, axis: int) -> np.ndarray:
    """Time-and-orthogonal-axis-averaged rFFT power along `axis` (1=row, 2=col)."""
    T, H, W = stack.shape
    L = stack.shape[axis]
    win_shape = [1, 1, 1]; win_shape[axis] = L
    win = np.hanning(L).reshape(win_shape)
    bg = ndimage.gaussian_filter(stack, sigma=(0, 20, 20))
    ac = (stack - bg) * win
    F = np.fft.rfft(ac, axis=axis)
    other = 1 if axis == 2 else 2
    return (np.abs(F) ** 2).mean(axis=(0, other))


def test_spectral(stack: np.ndarray) -> dict:
    """Narrowband peaks on fast (x) and slow (y) axes; split-half stationarity.

    Stationary peaks (present in both halves at the same frequency) -> fixed
    spatial banding requiring row/column pedestal subtraction. Non-stationary
    peaks -> drifting electrical interference; address with a notch.
    """
    T, H, W = stack.shape
    Px = _axis_spectrum(stack, axis=2)
    Py = _axis_spectrum(stack, axis=1)
    freqs_x = np.fft.rfftfreq(W, d=1.0)
    freqs_y = np.fft.rfftfreq(H, d=1.0)

    peaks_x = _peak_prominence_db(Px)
    peaks_y = _peak_prominence_db(Py)

    # Split-half stationarity
    half = T // 2
    if half >= 2:
        Px1 = _axis_spectrum(stack[:half], axis=2)
        Px2 = _axis_spectrum(stack[half:], axis=2)
        Py1 = _axis_spectrum(stack[:half], axis=1)
        Py2 = _axis_spectrum(stack[half:], axis=1)
        px1, px2 = _peak_prominence_db(Px1), _peak_prominence_db(Px2)
        py1, py2 = _peak_prominence_db(Py1), _peak_prominence_db(Py2)

        def _match(a, b, tol=2):
            out = []
            for ia, da in a:
                for ib, db in b:
                    if abs(ia - ib) <= tol:
                        out.append((ia, min(da, db)))
                        break
            return out
        stat_x = _match(px1, px2)
        stat_y = _match(py1, py2)
    else:
        stat_x, stat_y = [], []

    def _fmt(peaks, freqs):
        out = []
        for idx, db in peaks:
            f = float(freqs[idx])
            out.append(dict(freq_cyc_per_px=f,
                            period_px=float(1.0 / f) if f > 0 else float("inf"),
                            prominence_db=float(db),
                            bin=int(idx)))
        return out

    return dict(
        fast_axis_peaks=_fmt(peaks_x, freqs_x),
        slow_axis_peaks=_fmt(peaks_y, freqs_y),
        fast_axis_stationary_peaks=_fmt(stat_x, freqs_x),
        slow_axis_stationary_peaks=_fmt(stat_y, freqs_y),
        freqs_x=freqs_x.tolist(),
        freqs_y=freqs_y.tolist(),
        spectrum_x=Px.tolist(),
        spectrum_y=Py.tolist(),
    )


def test_edge_artifacts(stack: np.ndarray) -> dict:
    """Galvo flyback dead columns/rows at the four image edges."""
    M = stack.mean(axis=0)
    col_mean = M.mean(axis=0)
    row_mean = M.mean(axis=1)
    cmed = float(np.median(col_mean))
    rmed = float(np.median(row_mean))
    thr_col = 0.30 * cmed
    thr_row = 0.30 * rmed

    def _count_edge(profile, thr):
        left = 0
        for v in profile:
            if v < thr: left += 1
            else: break
        right = 0
        for v in profile[::-1]:
            if v < thr: right += 1
            else: break
        return int(left), int(right)

    cL, cR = _count_edge(col_mean, thr_col)
    rT, rB = _count_edge(row_mean, thr_row)

    return dict(
        dead_cols_left=cL, dead_cols_right=cR,
        dead_rows_top=rT, dead_rows_bottom=rB,
        col_mean_median_dn=cmed, row_mean_median_dn=rmed,
        col_profile=col_mean.tolist(), row_profile=row_mean.tolist(),
    )


def test_hot_pixels(stack: np.ndarray) -> dict:
    """Hot/dead pixels via local z-score on temporal mean with low temp variance."""
    M = stack.mean(axis=0)
    V = stack.var(axis=0)

    local_med = ndimage.median_filter(M, size=9)
    local_mad = ndimage.median_filter(np.abs(M - local_med), size=9) + 1e-6
    z = (M - local_med) / (1.4826 * local_mad)

    hot = (z > 8) & (V < 0.25 * float(V.mean()))
    dead = (M < 0.05 * float(np.median(M)))
    n_hot = int(hot.sum())
    n_dead = int(dead.sum())
    n_pix = int(M.size)

    return dict(
        n_hot=n_hot, n_dead=n_dead, n_pixels=n_pix,
        hot_fraction=float(n_hot / n_pix),
        dead_fraction=float(n_dead / n_pix),
    )


def test_drift(stack: np.ndarray) -> dict:
    """Photobleaching, PMT warm-up, illumination drift via per-frame mean trend."""
    T = stack.shape[0]
    frame_mean = stack.mean(axis=(1, 2))
    t = np.arange(T, dtype=np.float64)
    slope, intercept, r, _, _ = stats.linregress(t, frame_mean)
    start = float(intercept)
    end = float(intercept + slope * (T - 1))
    pct = float((end - start) / max(start, 1e-6) * 100.0)

    # Decay-tau estimate (only meaningful if a clear monotonic decay)
    tau = float("inf")
    if slope < 0:
        c0 = float(frame_mean.min())
        a0 = float(frame_mean[0] - c0)
        af = float(frame_mean[-1] - c0)
        if a0 > 0 and af > 0:
            tau = float(T / max(np.log(a0 / af), 1e-6))

    return dict(
        frame_mean_first=float(frame_mean[0]),
        frame_mean_last=float(frame_mean[-1]),
        slope_dn_per_frame=float(slope),
        linear_pct_change=pct,
        decay_tau_frames_est=tau,
        fit_r2=float(r * r),
        frame_mean_series=frame_mean.tolist(),
    )


def test_fixed_pattern(stack: np.ndarray) -> dict:
    """Fixed-pattern noise = high-pass of temporal mean, normalised by shot floor."""
    M = stack.mean(axis=0)
    lp = ndimage.gaussian_filter(M, sigma=8)
    hp = M - lp
    fpn_var = float(np.var(hp))
    # Variance of the temporal-mean estimator under shot noise:
    # ~ <pixel noise var> / T  (use median noise var with the χ²₁-correction)
    diffs = np.diff(stack, axis=0)
    pix_noise_var = float(np.median(diffs * diffs) / (2.0 * 0.4549))
    expected = pix_noise_var / stack.shape[0]
    ratio = fpn_var / max(expected, 1e-9)
    return dict(fpn_variance=fpn_var,
                expected_mean_var=expected,
                fpn_to_shot_ratio=float(ratio))


def test_saturation_quantization(stack: np.ndarray, info: dict) -> dict:
    """ADC ceiling/floor clipping & dynamic-range usage."""
    dtype_orig = info.get("dtype_orig", "uint16")
    if "uint16" in dtype_orig:
        ceiling = 65535
    elif "uint8" in dtype_orig:
        ceiling = 255
    elif "uint12" in dtype_orig:
        ceiling = 4095
    else:
        ceiling = int(max(stack.max(), 1))

    sat = float((stack >= ceiling - 1).mean())
    floor = float((stack <= 1).mean())
    dyn_used = float((np.percentile(stack, 99.9) - np.percentile(stack, 0.1))
                     / max(ceiling, 1))

    return dict(
        saturated_fraction=sat,
        floor_fraction=floor,
        dynamic_range_used=dyn_used,
        adc_ceiling=int(ceiling),
    )


def test_frame_discontinuity(stack: np.ndarray) -> dict:
    """Frame drops / sync glitches via robust outlier z on |Δframe_mean|."""
    fm = stack.mean(axis=(1, 2))
    diff = np.abs(np.diff(fm))
    if diff.size == 0:
        return dict(max_frame_jump_dn=0, median_frame_jump_dn=0,
                    n_outlier_jumps=0)
    mad = float(np.median(np.abs(diff - np.median(diff))) + 1e-9)
    z = (diff - np.median(diff)) / (1.4826 * mad)
    outliers = int((z > 6).sum())
    return dict(
        max_frame_jump_dn=float(diff.max()),
        median_frame_jump_dn=float(np.median(diff)),
        n_outlier_jumps=outliers,
    )


# ============================================================================
# Source-likelihood scoring
# ============================================================================

LEVELS = ["negligible", "low", "moderate", "high"]


def _level(score: float, thresholds: Tuple[float, float, float]) -> str:
    """Map a score to one of 4 levels via 3 cut points (low, moderate, high)."""
    lo, md, hi = thresholds
    if score >= hi: return "high"
    if score >= md: return "moderate"
    if score >= lo: return "low"
    return "negligible"


def score_sources(m: dict) -> Dict[str, dict]:
    """Map raw metrics → per-source {level, score, evidence, recommendation}."""
    src: Dict[str, dict] = {}

    # 1. Shot-noise-dominated regime (informational; "high" is healthy)
    ratio = m["photon_transfer"].get("shot_to_read_ratio", 0.0) or 0.0
    src["shot_noise_dominated"] = dict(
        level=_level(ratio, (1.0, 3.0, 10.0)),
        score=float(ratio),
        evidence=dict(shot_to_read_ratio=ratio,
                      typical_mean=m["photon_transfer"].get("typical_mean")),
        recommendation=("Healthy photon-limited regime; no read-noise fix needed."
                        if ratio > 3 else
                        "Read noise comparable to shot noise — consider higher "
                        "PMT gain or longer dwell time."),
    )

    # 2. Bidirectional scan phase offset
    bd = m["bidirectional"]
    ny = bd["nyquist_power_ratio"]
    sh = abs(bd["bidir_shift_px"])
    score = max(ny / 0.10, sh / 0.5)
    src["bidirectional_phase_offset"] = dict(
        level=_level(score, (0.2, 0.5, 1.0)),
        score=float(score),
        evidence=dict(nyquist_power_ratio=ny,
                      bidir_shift_px=bd["bidir_shift_px"]),
        recommendation=("Apply sub-pixel even-row shift correction "
                        "(suite2p `do_bidi_correct` or NoRMCorre bidirectional "
                        "option) before motion correction."),
    )

    # 3a/b. Horizontal banding — split by stationarity
    sp = m["spectral"]
    sl_peaks = sp["slow_axis_peaks"]
    sl_stat  = sp["slow_axis_stationary_peaks"]
    sl_max = max((p["prominence_db"] for p in sl_peaks), default=0.0)
    is_stationary = len(sl_stat) > 0
    src["horizontal_banding_fixed"] = dict(
        level=_level(sl_max if is_stationary else 0.0, (2.0, 4.0, 7.0)),
        score=float(sl_max if is_stationary else 0.0),
        evidence=dict(peaks=sl_stat),
        recommendation=("Subtract a row pedestal computed as the temporal "
                        "median over the recording (rank-1 across frames). "
                        "Alternative: SVD on (frames × row_mean) and drop "
                        "the first 1–2 components."),
    )
    src["horizontal_banding_drifting"] = dict(
        level=_level(sl_max if not is_stationary else 0.0, (2.0, 4.0, 7.0)),
        score=float(sl_max if not is_stationary else 0.0),
        evidence=dict(peaks=[p for p in sl_peaks if p not in sl_stat]),
        recommendation=("Per-frame 1D notch on the column FFT at the detected "
                        "frequencies; or rolling-window row pedestal with "
                        "window << drift timescale."),
    )

    # 4. Fast-axis periodic
    fa_peaks = sp["fast_axis_peaks"]
    fa_max = max((p["prominence_db"] for p in fa_peaks), default=0.0)
    src["fast_axis_periodic"] = dict(
        level=_level(fa_max, (2.0, 4.0, 7.0)),
        score=float(fa_max),
        evidence=dict(peaks=fa_peaks),
        recommendation=("Rare. Check pixel-clock / sample-and-hold timing on "
                        "the DAQ. Row-by-row 1D notch is the software fallback."),
    )

    # 5. Galvo flyback edge
    ed = m["edge_artifacts"]
    n_edge = ed["dead_cols_left"] + ed["dead_cols_right"]
    src["galvo_flyback_edge"] = dict(
        level=_level(n_edge, (1, 5, 15)),
        score=float(n_edge),
        evidence=dict(dead_cols_left=ed["dead_cols_left"],
                      dead_cols_right=ed["dead_cols_right"]),
        recommendation=("Crop the flagged edge columns before NoRMCorre and "
                        "CNMF (set `bord_px` or a spatial mask in params)."),
    )

    # 6. Hot / dead pixels
    hp = m["hot_pixels"]
    hot_score = hp["hot_fraction"] * 1e4
    src["hot_dead_pixels"] = dict(
        level=_level(hot_score, (1.0, 5.0, 20.0)),
        score=float(hot_score),
        evidence=dict(n_hot=hp["n_hot"], n_dead=hp["n_dead"],
                      hot_fraction=hp["hot_fraction"]),
        recommendation=("Replace flagged pixels with 3×3 spatial median before "
                        "motion correction (CaImAn `removeBadPixels`)."),
    )

    # 7. Photobleaching (negative trend)
    dr = m["drift"]
    decay_pct = max(-dr["linear_pct_change"], 0.0)
    src["photobleaching"] = dict(
        level=_level(decay_pct, (5.0, 15.0, 30.0)),
        score=float(decay_pct),
        evidence=dict(linear_pct_change=dr["linear_pct_change"],
                      decay_tau_frames=dr["decay_tau_frames_est"]),
        recommendation=("Detrend each pixel's trace via double-exponential "
                        "or percentile baseline before CNMF "
                        "(`detrend_df_f`)."),
    )

    # 8. Illumination drift (positive trend — warm-up)
    pos_drift = max(dr["linear_pct_change"], 0.0)
    src["illumination_drift_increase"] = dict(
        level=_level(pos_drift, (5.0, 15.0, 30.0)),
        score=float(pos_drift),
        evidence=dict(linear_pct_change=dr["linear_pct_change"]),
        recommendation=("PMT/laser warm-up. Discard the first ~N frames "
                        "until baseline plateaus, or fit and subtract a "
                        "saturating-exponential warm-up curve."),
    )

    # 9. Fixed-pattern noise
    fpn = m["fixed_pattern"]["fpn_to_shot_ratio"]
    src["fixed_pattern_noise"] = dict(
        level=_level(fpn, (1.05, 1.5, 3.0)),
        score=float(fpn),
        evidence=dict(fpn_to_shot_ratio=fpn),
        recommendation=("Subtract a dark frame (shutter closed, same dwell). "
                        "If unavailable, subtract a high-pass of the temporal "
                        "mean from every frame."),
    )

    # 10. Saturation
    sq = m["saturation"]
    sat = sq["saturated_fraction"]
    src["saturation_clipping"] = dict(
        level=_level(sat * 1e3, (0.01, 0.1, 1.0)),
        score=float(sat * 1e3),
        evidence=dict(saturated_fraction=sat,
                      adc_ceiling=sq["adc_ceiling"]),
        recommendation=("Reduce PMT gain or laser power so saturated fraction "
                        "drops below 1e-4."),
    )

    # 11. Quantization loss (when dynamic range usage is tiny)
    dyn = sq["dynamic_range_used"]
    qscore = max(0.0, 0.05 - dyn) / 0.05
    src["quantization_loss"] = dict(
        level=_level(qscore, (0.2, 0.5, 0.8)),
        score=float(qscore),
        evidence=dict(dynamic_range_used=dyn),
        recommendation=("Less than 5% of the ADC range is used. Increase PMT "
                        "gain so bright pixels reach ~50% of full scale."),
    )

    # 12. Frame drops
    fd = m["frame_discontinuity"]
    src["frame_discontinuity"] = dict(
        level=_level(fd["n_outlier_jumps"], (1, 5, 20)),
        score=float(fd["n_outlier_jumps"]),
        evidence=dict(n_outlier_jumps=fd["n_outlier_jumps"],
                      max_jump_dn=fd["max_frame_jump_dn"]),
        recommendation=("Identify the frame indices via the discontinuity "
                        "z-score; drop or interpolate. Check DAQ → acquisition "
                        "sync timestamps."),
    )

    return src


# ============================================================================
# Plotting + reports
# ============================================================================

def plot_panel(stack: np.ndarray, m: dict, out_path: Path):
    """Single 3×3 PNG summary."""
    fig, axs = plt.subplots(3, 3, figsize=(15, 13))

    M = stack.mean(axis=0)
    axs[0, 0].imshow(M, cmap="gray"); axs[0, 0].axis("off")
    axs[0, 0].set_title("Temporal-mean image")

    V = stack.var(axis=0)
    axs[0, 1].imshow(np.log10(V + 1e-3), cmap="magma"); axs[0, 1].axis("off")
    axs[0, 1].set_title("log Variance image")

    pt = m["photon_transfer"]
    if "pt_centers" in pt:
        c = np.array(pt["pt_centers"])
        v = np.array(pt["pt_vars"])
        axs[0, 2].plot(c, v, ".", ms=4)
        axs[0, 2].plot(c, pt["gain_dn_per_e"] * c + pt["intercept_var"],
                       "r-", lw=1,
                       label=f"g={pt['gain_dn_per_e']:.2f}, "
                             f"σ_r={pt['read_noise_dn']:.1f} DN")
        axs[0, 2].legend(fontsize=8)
        axs[0, 2].set_xlabel("mean (DN)"); axs[0, 2].set_ylabel("var (DN²)")
        axs[0, 2].set_title("Photon-transfer curve")
    else:
        axs[0, 2].axis("off")

    sp = m["spectral"]
    fx = np.array(sp["freqs_x"]); Sx = np.array(sp["spectrum_x"])
    fy = np.array(sp["freqs_y"]); Sy = np.array(sp["spectrum_y"])
    axs[1, 0].semilogy(fx[1:], Sx[1:])
    for p in sp["fast_axis_peaks"]:
        axs[1, 0].axvline(p["freq_cyc_per_px"], color="r", alpha=0.3)
    axs[1, 0].set_title("Fast-axis spectrum"); axs[1, 0].set_xlabel("cyc/px")

    axs[1, 1].semilogy(fy[1:], Sy[1:])
    for p in sp["slow_axis_peaks"]:
        is_stat = any(abs(sp_["bin"] - p["bin"]) <= 2
                      for sp_ in sp["slow_axis_stationary_peaks"])
        axs[1, 1].axvline(p["freq_cyc_per_px"],
                          color="r" if is_stat else "orange",
                          ls="-" if is_stat else "--", alpha=0.4)
    axs[1, 1].set_title("Slow-axis spectrum (red=stationary, orange=drifting)")
    axs[1, 1].set_xlabel("cyc/px")

    bd = m["bidirectional"]
    axs[1, 2].plot(bd["bidir_residual_y_profile"])
    axs[1, 2].set_title(f"Bidir residual: Nyq={bd['nyquist_power_ratio']:.3f}, "
                        f"shift={bd['bidir_shift_px']:+.2f} px")
    axs[1, 2].set_xlabel("pair index")

    ed = m["edge_artifacts"]
    axs[2, 0].plot(ed["col_profile"], label="col mean")
    axs[2, 0].plot(ed["row_profile"], label="row mean", alpha=0.7)
    axs[2, 0].axhline(0.30 * ed["col_mean_median_dn"], color="r",
                      ls=":", alpha=0.5, lw=0.8)
    axs[2, 0].set_title(f"Edge profiles (deadL={ed['dead_cols_left']}, "
                        f"deadR={ed['dead_cols_right']})")
    axs[2, 0].legend(fontsize=8)

    dr = m["drift"]
    axs[2, 1].plot(dr["frame_mean_series"])
    axs[2, 1].set_title(f"Frame mean: {dr['linear_pct_change']:+.1f}%")
    axs[2, 1].set_xlabel("sampled frame")

    axs[2, 2].axis("off")
    srcs = m["sources"]
    order = sorted(srcs.items(),
                   key=lambda kv: (LEVELS.index(kv[1]["level"]), kv[1]["score"]),
                   reverse=True)
    txt = "Top issues by level\n" + "─" * 32 + "\n"
    for name, info in order[:12]:
        marker = {"high": "●●●", "moderate": "●●○", "low": "●○○",
                  "negligible": "○○○"}[info["level"]]
        txt += f"{marker} {info['level']:<10} {name}\n"
    axs[2, 2].text(0, 1, txt, family="monospace",
                   transform=axs[2, 2].transAxes, va="top", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def print_summary(report: dict, fh=sys.stdout):
    srcs = report["sources"]; info = report["info"]
    print(f"\nNoise diagnostics for {info.get('source')}", file=fh)
    print(f"  sampled {info['n_sampled']}/{info['n_total']} frames, "
          f"dtype={info['dtype_orig']}", file=fh)
    print("─" * 72, file=fh)
    order = sorted(srcs.items(),
                   key=lambda kv: (LEVELS.index(kv[1]["level"]), kv[1]["score"]),
                   reverse=True)
    any_issue = False
    for name, d in order:
        if d["level"] == "negligible":
            break
        any_issue = True
        print(f"  [{d['level']:>8}]  {name:<32}  score={d['score']:9.3f}",
              file=fh)
        if d["level"] in ("moderate", "high"):
            print(f"             → {d['recommendation']}", file=fh)
    if not any_issue:
        print("  No issues above 'negligible' detected.", file=fh)
    print("─" * 72, file=fh)


# ============================================================================
# Public entry point
# ============================================================================

def run_diagnostics(src: ArrayLike,
                    out_dir: Union[str, Path] = "diag_out",
                    n_frames: int = 500,
                    rng_seed: int = 0,
                    save_panel: bool = True,
                    save_json: bool = True,
                    write_summary: bool = True) -> dict:
    """Run the full diagnostic suite on a 2P recording.

    Parameters
    ----------
    src : ndarray or path
        (T, H, W) array or path to .tif/.tiff/.btf/.npy.
    out_dir : path
        Where to write report/panel/summary. Created if missing.
    n_frames : int
        Number of frames to sample (default 500).
    rng_seed : int
        Reproducible sampling.

    Returns
    -------
    dict
        Full report — keys: info, photon_transfer, bidirectional, spectral,
        edge_artifacts, hot_pixels, drift, fixed_pattern, saturation,
        frame_discontinuity, sources.
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
    ap.add_argument("source", help=".tif/.tiff/.btf/.npy path")
    ap.add_argument("--out", default="diag_out", help="output directory")
    ap.add_argument("--n_frames", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_diagnostics(args.source, out_dir=args.out,
                    n_frames=args.n_frames, rng_seed=args.seed)


if __name__ == "__main__":
    _main()
