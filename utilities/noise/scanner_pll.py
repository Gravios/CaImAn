"""
Software phase-locked-loop (PLL) for resonant-scanner contamination.

This module implements the post-hoc software analogue of an analog PLL
inserted between the resonant scanner's sync output and the DAQ's
line-trigger input (see Rupprecht's 2023 hardware design at
https://gcamp6f.com/2023/10/19/improving-the-resonant-scanners-sync-signal-using-a-phase-locked-loop-pll/
for the original analog implementation; Ching-Roa et al. 2024 for the
physical motivation).

What it does
============
Estimates the slowly-varying complex amplitude of scanner contamination
at each detected lattice bin of the per-frame FFT and subtracts it
coherently, preserving the bin's static cell-content baseline and any
fast cell-activity transients.

Conceptual difference from spectral notching
============================================
:func:`subtract_per_frame_pattern` (the notch approach):
    For each detected lattice bin k and each frame t:
        F_clean(t, k) = 0      # zero the bin entirely
    Wins: simple, robust, no temporal assumptions.
    Costs: removes any cell content sitting at the lattice bin.

:func:`subtract_scanner_pll` (this module):
    For each detected lattice bin k and each frame t:
        F_clean(t, k) = F(t, k) - smooth_low_pass(F(:, k))[t] + mean(F(:, k))
    The smoothed value minus the temporal mean is the estimated
    *slowly-varying* contamination component at that bin and time.
    Subtracting it removes the slow drift while preserving:
        - the bin's static cell baseline (the temporal mean), and
        - any fast variations (cell transients) that survive the
          low-pass filter.

When the PLL helps and when notching is better
==============================================
The PLL approach assumes the scanner contamination has *temporal
structure* — i.e. the complex amplitude at lattice bins drifts smoothly
across consecutive frames, rather than jumping randomly. The phase
coherence between consecutive frames quantifies this:

  median phase-coherence at lag 1 > ~0.3  → PLL is providing real value
                                          (drift is trackable)
  median phase-coherence at lag 1 < ~0.1  → PLL ≈ no-op; the
                                          contamination is essentially
                                          random per frame. Notching
                                          (subtract_per_frame_pattern)
                                          is the right tool instead.
  0.1 ≤ coherence ≤ 0.3                   → intermediate; both can be
                                          tried and outputs compared.

The diagnostic this module always emits (median + per-bin coherence
metrics) makes this assessment quantitative and reproducible. The
default return is `cleaned` only; with `return_diagnostics=True` the
full diagnostic dict is also returned for downstream analysis.

Composition with the rest of the pipeline
=========================================
PLL and notching target the same lattice bins but with different
filters. They should be used as ALTERNATIVES, not in sequence — running
notch then PLL on the notched data gives the PLL no signal to work
with (the bin is already zeroed); running PLL then notch undoes the
PLL's signal-preservation work.

The two stages this module *does* compose with are :func:`subtract_row_pedestal`,
:func:`subtract_column_pedestal`, and :func:`subtract_fixed_pattern`,
which handle orthogonal noise mechanisms (per-row uniform offsets,
per-column uniform offsets, and stationary spatial structure
respectively). Pipeline order:
    1. subtract_row_pedestal
    2. subtract_column_pedestal
    3. subtract_fixed_pattern
    4. ONE OF (subtract_per_frame_pattern, subtract_scanner_pll) but not both
"""

import logging
import time as _time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from .noise_correction import (
    _detrend_xp,
    _pool_magnitude_xp,
    detect_fpn_peaks,
)

log = logging.getLogger(__name__)


# ============================================================================
# Private helpers
# ============================================================================

def _extract_trajectories_xp(stack: np.ndarray,
                              ys_unshifted: np.ndarray,
                              xs_unshifted: np.ndarray,
                              xp,
                              batch_size: int = 512) -> np.ndarray:
    """Per-frame complex FFT values at specified unshifted-coord bins.

    Returns a host-side (T, K) complex64 array. For each frame the FFT
    is computed and the K specified bins are extracted; the FFT itself
    is discarded to keep memory bounded. Total work: one fft2 per frame
    (~PCIe-bandwidth-limited on GPU).
    """
    T, _, _ = stack.shape
    K = len(ys_unshifted)
    trajectories = np.empty((T, K), dtype=np.complex64)
    if xp is not np:
        ys_xp = xp.asarray(ys_unshifted)
        xs_xp = xp.asarray(xs_unshifted)
    else:
        ys_xp = ys_unshifted
        xs_xp = xs_unshifted
    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        batch = stack[s:e].astype(np.float32, copy=False)
        batch_xp = xp.asarray(batch) if xp is not np else batch
        F = xp.fft.fft2(batch_xp)
        # Index extraction: F[:, ys, xs] -> shape (batch_size, K)
        bin_values = F[:, ys_xp, xs_xp]
        host = (np.asarray(bin_values) if xp is np
                else xp.asnumpy(bin_values))
        trajectories[s:e] = host.astype(np.complex64)
    return trajectories


def _apply_pll_corrections_xp(stack: np.ndarray,
                               ys: np.ndarray, xs: np.ndarray,
                               ys_conj: np.ndarray, xs_conj: np.ndarray,
                               self_conj: np.ndarray,
                               corrections: np.ndarray,
                               xp,
                               batch_size: int,
                               return_pattern: bool,
                               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Subtract per-frame per-bin complex corrections from the stack's FFT.

    For each lattice bin k:
        F_t(ys[k], xs[k])           -= corrections[t, k]
        F_t(ys_conj[k], xs_conj[k]) -= conj(corrections[t, k])
    For self-conjugate bins (Nyquist row/col) the correction is forced
    real by taking corrections[:, k].real before subtraction, so that
    the IFFT is exactly real-valued.

    Returns cleaned (T, H, W) float32, and pattern (T, H, W) float32 if
    return_pattern is True (the actual subtracted contamination).
    """
    T, H, W = stack.shape
    not_self = ~self_conj
    ys_p = ys[not_self]
    xs_p = xs[not_self]
    ys_p_conj = ys_conj[not_self]
    xs_p_conj = xs_conj[not_self]
    ys_s = ys[self_conj]
    xs_s = xs[self_conj]

    if xp is not np:
        ys_p_xp = xp.asarray(ys_p)
        xs_p_xp = xp.asarray(xs_p)
        ys_p_conj_xp = xp.asarray(ys_p_conj)
        xs_p_conj_xp = xp.asarray(xs_p_conj)
        ys_s_xp = xp.asarray(ys_s)
        xs_s_xp = xp.asarray(xs_s)
    else:
        ys_p_xp = ys_p; xs_p_xp = xs_p
        ys_p_conj_xp = ys_p_conj; xs_p_conj_xp = xs_p_conj
        ys_s_xp = ys_s; xs_s_xp = xs_s

    cleaned = np.empty_like(stack, dtype=np.float32)
    pattern = np.empty_like(stack, dtype=np.float32) if return_pattern else None
    has_pairs = bool(not_self.any())
    has_self = bool(self_conj.any())

    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        batch = stack[s:e].astype(np.float32, copy=False)
        batch_xp = xp.asarray(batch) if xp is not np else batch
        F = xp.fft.fft2(batch_xp)

        # Build correction delta (initially zero everywhere)
        delta_F = xp.zeros_like(F)

        # Move this batch's corrections to GPU
        corr_batch = corrections[s:e]   # (batch, K) complex64 host
        corr_xp = (xp.asarray(corr_batch) if xp is not np
                    else corr_batch)
        corr_xp = corr_xp.astype(F.dtype)

        if has_pairs:
            # Subtract corr at (ys, xs), and conj(corr) at conjugate pair
            delta_F[:, ys_p_xp, xs_p_xp] = corr_xp[:, not_self]
            delta_F[:, ys_p_conj_xp, xs_p_conj_xp] = \
                xp.conj(corr_xp[:, not_self])
        if has_self:
            # Self-conjugate bins must take real part only
            delta_F[:, ys_s_xp, xs_s_xp] = \
                corr_xp[:, self_conj].real.astype(F.dtype)

        F_clean = F - delta_F
        cleaned_batch = xp.real(xp.fft.ifft2(F_clean))
        cleaned[s:e] = (np.asarray(cleaned_batch) if xp is np
                        else xp.asnumpy(cleaned_batch))

        if return_pattern:
            pat = xp.real(xp.fft.ifft2(delta_F))
            pattern[s:e] = (np.asarray(pat) if xp is np
                            else xp.asnumpy(pat))

    return cleaned, pattern


def _detect_lattice_bins(stack: np.ndarray,
                          xp,
                          batch_size: int,
                          n_chunks: int,
                          cell_scale_px: float,
                          prominence_db: float,
                          max_peaks: int,
                          min_frames_per_chunk: int,
                          ) -> List[Tuple[int, int]]:
    """Detect lattice bin coordinates via per-chunk detect_fpn_peaks.

    Takes the union of detected peak centres across all chunks, returning
    each unique (y_shifted, x_shifted) bin centre. Matches the per-chunk
    detection logic from :func:`subtract_per_frame_pattern` so the two
    routines see the same set of bins.

    Centres are the per-blob argmax in each chunk's pooled magnitude
    (rather than every notched bin) — gives one coordinate per detected
    peak, suitable for use as a single representative point at which to
    fit the PLL trajectory.
    """
    T, H, W = stack.shape
    actual_chunks = min(max(1, n_chunks),
                         max(1, T // max(1, min_frames_per_chunk)))
    chunk_size = T // actual_chunks
    chunk_bounds = [(k * chunk_size,
                      (k + 1) * chunk_size if k < actual_chunks - 1 else T)
                     for k in range(actual_chunks)]
    seen = set()
    bin_coords = []  # list of (y_shifted, x_shifted)
    for k, (s, e) in enumerate(chunk_bounds):
        A_chunk = _pool_magnitude_xp(stack, s, e, xp, batch_size)
        notch_mask, _ = detect_fpn_peaks(
            A_chunk,
            cell_scale_px=cell_scale_px,
            prominence_db=prominence_db,
            max_peaks=max_peaks,
            magnitude_in=True,
        )
        labeled, n_blobs = ndimage.label(notch_mask)
        for blob_id in range(1, n_blobs + 1):
            ys_blob, xs_blob = np.where(labeled == blob_id)
            magnitudes = A_chunk[ys_blob, xs_blob]
            max_idx = int(np.argmax(magnitudes))
            y_shifted = int(ys_blob[max_idx])
            x_shifted = int(xs_blob[max_idx])
            if (y_shifted, x_shifted) not in seen:
                seen.add((y_shifted, x_shifted))
                bin_coords.append((y_shifted, x_shifted))
    log.info("scanner_pll detected %d unique lattice bins across %d chunks",
              len(bin_coords), actual_chunks)
    return bin_coords


def _diagnose_phase_dynamics(trajectories: np.ndarray,
                              bin_coords_shifted: List[Tuple[int, int]],
                              cy: int, cx: int,
                              ) -> Dict:
    """Per-bin and aggregate phase-coherence statistics on the trajectories.

    Coherence metric is the resultant length of unit-vector phase
    differences:  coherence = |<exp(i (φ_{t+lag} - φ_t))>|.
    1 = deterministic phase increment;  0 = uniform-random per frame.
    PLL gains scale with this metric; near-zero means PLL is a no-op
    and notching is the right tool instead.
    """
    T, K = trajectories.shape
    per_bin = []
    for k in range(K):
        traj = trajectories[:, k]
        mag = np.abs(traj)
        # Wrapped phase differences via complex ratio (no unwrap needed)
        with np.errstate(invalid='ignore', divide='ignore'):
            dphase1 = np.angle(traj[1:] * np.conj(traj[:-1]))
            dphase10 = np.angle(traj[10:] * np.conj(traj[:-10]))
        coh1 = float(np.abs(np.mean(np.exp(1j * dphase1))))
        coh10 = float(np.abs(np.mean(np.exp(1j * dphase10))))
        # Drift = circular mean of dphase1 (rad/frame)
        drift = float(np.angle(np.mean(np.exp(1j * dphase1))))
        y_shifted, x_shifted = bin_coords_shifted[k]
        per_bin.append({
            "bin_shifted": (int(y_shifted), int(x_shifted)),
            "dy": int(y_shifted - cy),
            "dx": int(x_shifted - cx),
            "magnitude_mean": float(mag.mean()),
            "magnitude_std": float(mag.std()),
            "phase_drift_rad_per_frame": drift,
            "phase_coherence_lag1": coh1,
            "phase_coherence_lag10": coh10,
        })
    coh1_arr = np.array([b["phase_coherence_lag1"] for b in per_bin])
    coh10_arr = np.array([b["phase_coherence_lag10"] for b in per_bin])
    return {
        "n_bins": K,
        "median_coherence_lag1": float(np.median(coh1_arr)),
        "median_coherence_lag10": float(np.median(coh10_arr)),
        "max_coherence_lag1": float(np.max(coh1_arr)) if K > 0 else 0.0,
        "p25_coherence_lag1": float(np.percentile(coh1_arr, 25)),
        "p75_coherence_lag1": float(np.percentile(coh1_arr, 75)),
        "per_bin": per_bin,
    }


# ============================================================================
# Public API
# ============================================================================

def subtract_scanner_pll(stack: np.ndarray,
                          lattice_bins: Optional[List[Tuple[int, int]]] = None,
                          smooth_window_frames: int = 1000,
                          detect_n_chunks: int = 20,
                          detect_min_frames_per_chunk: int = 500,
                          detect_cell_scale_px: float = 12.0,
                          detect_prominence_db: float = 15.0,
                          detect_max_peaks: int = 32,
                          batch_size: int = 512,
                          return_diagnostics: bool = False,
                          use_gpu: bool = False,
                          ) -> Union[np.ndarray, Tuple[np.ndarray, Dict, np.ndarray]]:
    """Subtract scanner contamination using per-bin temporal PLL filter.

    Algorithm
    =========
    1. Detect lattice bins (per-chunk union from detect_fpn_peaks). Caller
       may supply lattice_bins directly to bypass detection.
    2. Extract per-frame complex FFT values at each lattice bin → (T, K).
    3. Apply Gaussian temporal low-pass filter to each bin's trajectory
       (sigma = smooth_window_frames / 6). The smoothed value is the
       slowly-varying complex amplitude at that bin and time.
    4. Build correction = smoothed_trajectory - temporal_mean(trajectory).
       This is the slow drift component, isolated from the static cell
       baseline (which we keep) and from fast variations (cell activity,
       shot noise — also keep).
    5. Subtract corrections coherently from F_t at each lattice bin and
       its conjugate, then IFFT to recover cleaned frames.

    Why this preserves cell content
    ===============================
    At lattice bin k, the per-frame complex value F(t, k) decomposes as

        F(t, k) = C_cell(k)                    [static, all frames]
                + δ_cell(t, k)                 [cell activity, fast]
                + A_scan(t) e^{i θ_scan(t)}    [scanner, slow]
                + noise

    The temporal mean of F(:, k) is dominated by C_cell(k) — the scanner
    term phase-averages partly toward zero, and δ_cell averages exactly
    to zero (zero-mean activity).

    The smoothed trajectory captures C_cell + the slow scanner term
    (because the cell activity is faster than the smoothing scale and
    averages out). Subtracting (smoothed - mean) thus removes only the
    slow scanner term, leaving the bin's static cell content and any
    fast variations intact.

    When the PLL helps
    ==================
    When phase has temporal structure (drift, not random per-frame).
    The diagnostic returned (median phase_coherence_lag1) quantifies
    this. Use the diagnostic to decide whether PLL or notching is right
    for a given session before committing the cleaned output downstream.

    Parameters
    ----------
    stack : (T, H, W) ndarray
    lattice_bins : list of (y_shifted, x_shifted), optional
        Bin coordinates in fftshifted convention (DC at (H//2, W//2)).
        If None, run detect_fpn_peaks per chunk and take the union.
    smooth_window_frames : int
        Temporal smoothing window for the PLL low-pass filter. Default
        1000 frames. The Gaussian sigma used internally is window/6 so
        that the ±3σ width spans the window. For sa-000093 at 30.79 Hz
        this corresponds to ~33 seconds of smoothing — well above cell
        transient timescales (1-3 s) and well below session-evolution
        timescales (5-60 min).
    detect_n_chunks, detect_min_frames_per_chunk, detect_cell_scale_px,
    detect_prominence_db, detect_max_peaks :
        Forwarded to the lattice-bin detection step. Defaults match
        :func:`subtract_per_frame_pattern`.
    batch_size : int
        Frames per GPU/CPU batch in the extraction and application passes.
    return_diagnostics : bool
        If True, also returns a dict of trajectories + per-bin phase
        statistics, and a (T, H, W) pattern array (subtracted content).
    use_gpu : bool
        Use CuPy if available for FFTs.

    Returns
    -------
    cleaned : (T, H, W) float32 ndarray
    diag : dict, optional   (only if return_diagnostics=True)
    pattern : (T, H, W) float32 ndarray, optional   (only if return_diagnostics=True)
    """
    t_start = _time.perf_counter()
    xp = _detrend_xp(use_gpu)
    stack = np.asarray(stack)
    T, H, W = stack.shape
    cy, cx = H // 2, W // 2

    log.info("subtract_scanner_pll: T=%d, H=%d, W=%d, smooth_window=%d, "
              "backend=%s",
              T, H, W, smooth_window_frames,
              "cupy" if xp is not np else "numpy")

    # --- Detect lattice bins ---
    if lattice_bins is None:
        bin_coords_shifted = _detect_lattice_bins(
            stack, xp, batch_size, detect_n_chunks,
            detect_cell_scale_px, detect_prominence_db,
            detect_max_peaks, detect_min_frames_per_chunk)
    else:
        bin_coords_shifted = list(lattice_bins)

    if len(bin_coords_shifted) == 0:
        log.info("subtract_scanner_pll: no lattice bins detected; "
                  "returning input unchanged")
        if return_diagnostics:
            return (stack.astype(np.float32, copy=False),
                    {"n_bins": 0, "per_bin": []}, None)
        return stack.astype(np.float32, copy=False)

    K = len(bin_coords_shifted)

    # Convert shifted to unshifted FFT coords. Shifted (y, x) with DC at
    # (cy, cx) corresponds to unshifted ((y - cy) mod H, (x - cx) mod W).
    ys_shifted = np.array([b[0] for b in bin_coords_shifted], dtype=np.int64)
    xs_shifted = np.array([b[1] for b in bin_coords_shifted], dtype=np.int64)
    ys_unshifted = (ys_shifted - cy) % H
    xs_unshifted = (xs_shifted - cx) % W
    ys_conj_unshifted = (-ys_unshifted) % H
    xs_conj_unshifted = (-xs_unshifted) % W
    self_conj = ((ys_unshifted == ys_conj_unshifted)
                  & (xs_unshifted == xs_conj_unshifted))

    # --- Extract per-frame trajectories ---
    t_extract = _time.perf_counter()
    trajectories = _extract_trajectories_xp(
        stack, ys_unshifted, xs_unshifted, xp, batch_size)
    log.info("subtract_scanner_pll: extracted %d trajectories in %.1fs",
              K, _time.perf_counter() - t_extract)

    # --- Diagnostic on phase dynamics ---
    diag = _diagnose_phase_dynamics(
        trajectories, bin_coords_shifted, cy, cx)
    log.info("subtract_scanner_pll: phase coherence lag1: "
              "median=%.3f, p25=%.3f, p75=%.3f, max=%.3f "
              "(1 = predictable phase, 0 = random)",
              diag["median_coherence_lag1"],
              diag["p25_coherence_lag1"],
              diag["p75_coherence_lag1"],
              diag["max_coherence_lag1"])
    if diag["median_coherence_lag1"] < 0.1:
        log.warning("subtract_scanner_pll: phase coherence is very low; "
                     "scanner contamination appears random per-frame, "
                     "PLL will be near no-op. Consider "
                     "subtract_per_frame_pattern instead.")

    # --- Smooth trajectories with Gaussian filter ---
    # Naive smoothing of the raw trajectories doesn't work: when the
    # contamination at a bin has a rotating phase (any non-zero per-
    # frame phase increment), the signal averages to ~0 across the
    # smoothing window. The smoothed value would carry no contamination
    # information.
    #
    # Standard PLL solution: demodulate before smoothing. Estimate the
    # per-bin per-frame phase increment ω_k (the "carrier rate"), then
    # multiply the trajectory by exp(-i·ω_k·t) to bring the rotating
    # contamination into a static frame. Smooth in the static frame.
    # Re-modulate by exp(+i·ω_k·t) to put the smoothed estimate back into
    # the original rotating frame. The result is the contamination
    # signal *with* the carrier rotation, suitable for direct subtraction
    # from F.
    #
    # This works for both coherent phase (drift rate ω ≠ 0; PLL recovers
    # the contamination) and incoherent phase (drift rate ω ≈ random;
    # the smoothed signal in unrotated frame is ~0 and the correction is
    # ~0, gracefully no-op).
    #
    # Mean handling: we subtract the temporal mean *before* unrotation,
    # so the cell-content baseline (which is static, not rotating) sits
    # in F_mean and is not subject to the demod/smooth/remod cycle. After
    # remod, the correction we apply is ONLY the rotating scanner term,
    # not the static baseline.

    F_mean = trajectories.mean(axis=0, keepdims=True)  # (1, K) complex
    deviations = (trajectories - F_mean).astype(np.complex64)

    # Per-bin rate estimate: circular mean of consecutive phase differences,
    # measured on the deviations. With the static baseline removed, the
    # dominant term in `deviations` is the rotating contamination, so its
    # mean per-frame phase shift IS ω_k.
    # Implementation: ratio of consecutive complex values, then circular
    # mean. Robust to phase wrapping (no unwrap needed).
    with np.errstate(invalid='ignore'):
        ratios = deviations[1:] * np.conj(deviations[:-1])      # (T-1, K)
        # Weight by amplitude to suppress noise-dominated samples
        rate_vector = ratios.mean(axis=0)                       # (K,) complex
        omega_per_bin = np.angle(rate_vector).astype(np.float32) # (K,) rad/frame

    log.info("subtract_scanner_pll: estimated per-bin phase rates: "
              "min=%+.4f, median=%+.4f, max=%+.4f rad/frame",
              float(omega_per_bin.min()),
              float(np.median(omega_per_bin)),
              float(omega_per_bin.max()))

    sigma = max(1.0, smooth_window_frames / 6.0)
    t_smooth = _time.perf_counter()

    # Demodulate: multiply trajectory by exp(-i·ω_k·t) so the rotating
    # contamination becomes static (in the rotating-frame coordinates).
    t_axis = np.arange(T, dtype=np.float32)[:, None]            # (T, 1)
    demod_phases = -t_axis * omega_per_bin[None, :]             # (T, K)
    demod_factor = np.exp(1j * demod_phases).astype(np.complex64)
    deviations_demod = deviations * demod_factor

    # Smooth in the rotating frame: now the slowly-varying amplitude is
    # truly slowly-varying, so a low-pass filter preserves it.
    real_smooth = ndimage.gaussian_filter1d(
        deviations_demod.real.astype(np.float32),
        sigma=sigma, axis=0, mode='nearest')
    imag_smooth = ndimage.gaussian_filter1d(
        deviations_demod.imag.astype(np.float32),
        sigma=sigma, axis=0, mode='nearest')
    deviations_demod_smooth = (real_smooth + 1j * imag_smooth).astype(np.complex64)

    # Re-modulate: multiply by exp(+i·ω_k·t) to restore the carrier
    # rotation. The result is the estimated contamination in the original
    # (non-rotating) frame.
    remod_factor = np.exp(-1j * demod_phases).astype(np.complex64)  # = exp(+i·ω_k·t)
    corrections = (deviations_demod_smooth * remod_factor).astype(np.complex64)

    log.info("subtract_scanner_pll: smoothed trajectories in %.1fs "
              "(sigma=%.1f frames, mean |correction|=%.2f DN)",
              _time.perf_counter() - t_smooth, sigma,
              float(np.abs(corrections).mean()))

    # --- Apply corrections (per-frame coherent subtraction) ---
    t_apply = _time.perf_counter()
    cleaned, pattern = _apply_pll_corrections_xp(
        stack, ys_unshifted, xs_unshifted,
        ys_conj_unshifted, xs_conj_unshifted, self_conj,
        corrections, xp, batch_size,
        return_pattern=return_diagnostics)
    log.info("subtract_scanner_pll: applied corrections in %.1fs",
              _time.perf_counter() - t_apply)
    log.info("subtract_scanner_pll: total %.1fs",
              _time.perf_counter() - t_start)

    if return_diagnostics:
        diag["trajectories"] = trajectories
        diag["deviations_demod_smooth"] = deviations_demod_smooth
        diag["omega_per_bin"] = omega_per_bin
        diag["corrections"] = corrections
        diag["bin_coords_shifted"] = bin_coords_shifted
        diag["smooth_window_frames"] = smooth_window_frames
        return cleaned, diag, pattern
    return cleaned


__all__ = ["subtract_scanner_pll"]
