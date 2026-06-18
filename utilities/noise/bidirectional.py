"""
Bidirectional scan correction for resonant-scanner microscopy.

The fast (X) axis is driven by a resonant mirror that oscillates
sinusoidally. Photons are collected during BOTH the forward and reverse
sweeps to double the line rate. The forward sweep produces every other
line; the reverse sweep produces the rest. After the acquisition
software flips the reverse-sweep lines (so all lines read
left-to-right), the timing offset between the forward and reverse line
triggers shows up as a horizontal shift between adjacent lines —
typically sub-pixel to a few pixels.

In Fourier space this produces a peak at the Nyquist row (fy = H/2 in
fftshifted coords) — a different signature from the scanner-rate
lattice peaks at (fy, fx) like (52, 12). The two artifacts are
orthogonal and require different corrections:

  Lattice contamination (scanner-rate harmonics): handled by
    subtract_per_frame_pattern (notching) or subtract_scanner_pll
    (phase-coherent subtraction).

  Bidirectional misalignment (Nyquist-row, this module): handled by
    horizontal shift of even-indexed lines.

Algorithm
---------
1. Compute the temporal mean image (averages out shot noise, preserves
   structure including the bidirectional misalignment which is constant
   per-frame).
2. Cross-correlate odd lines vs even lines along X. The peak position
   indicates the misalignment offset δ in pixels.
3. Refine to sub-pixel precision via parabolic interpolation around
   the integer peak.
4. Apply Fourier-domain horizontal shift of all even-indexed lines by
   -δ to align with odd lines. Uses exp(-2πi·δ·k/W) phase factor on
   the per-line FFT (circular boundary in X; for typical δ < 2 px the
   wrap-around at the right edge corrupts < 0.5% of the image).

Optionally, the offset can be estimated per-chunk for sessions where
thermal drift causes δ to vary slowly; the per-chunk offsets are then
linearly interpolated to per-frame and applied frame-by-frame.

Composition with the rest of the pipeline
-----------------------------------------
Bidirectional correction is a pure geometric pre-processing step. It
should run BEFORE the spectral cleanup stages, in order:

    1. correct_bidirectional_scan       (this module)
    2. subtract_row_pedestal
    3. subtract_column_pedestal
    4. subtract_fixed_pattern
    5. ONE OF (subtract_per_frame_pattern, subtract_scanner_pll)
"""

import logging
import time as _time
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .noise_correction import _detrend_xp

log = logging.getLogger(__name__)


# ============================================================================
# Private helpers
# ============================================================================

def _estimate_offset_from_image(M: np.ndarray,
                                 search_range_px: int,
                                 xp) -> float:
    """Cross-correlate odd vs even lines to find sub-pixel horizontal offset.

    Given a 2D image M of shape (H, W), partitions it into odd-indexed
    rows (M[1::2]) and even-indexed rows (M[0::2]), computes their
    horizontal cross-correlation summed over lines, and locates the
    sub-pixel peak via parabolic interpolation.

    Returns
    -------
    offset_px : float
        The amount by which even lines need to be shifted (to the right
        if positive) to align with odd lines.
    """
    M_xp = xp.asarray(M) if xp is not np else M
    H, W = M_xp.shape

    n = H // 2
    even = M_xp[0:2 * n:2, :].astype(xp.float32)   # n rows
    odd = M_xp[1:2 * n:2, :].astype(xp.float32)    # n rows

    # Subtract per-row mean: focus on horizontal pattern, ignore overall
    # row brightness differences that would dominate the cross-correlation.
    even = even - even.mean(axis=1, keepdims=True)
    odd = odd - odd.mean(axis=1, keepdims=True)

    # Cross-correlation via FFT, summed over y:
    #   R(k) = sum_y of ifft(fft(odd[y]) * conj(fft(even[y])))[k]
    # Peak at lag k indicates: shifting even by k pixels aligns it with odd.
    F_even = xp.fft.fft(even, axis=1)
    F_odd = xp.fft.fft(odd, axis=1)
    cross_spectrum = xp.sum(F_odd * xp.conj(F_even), axis=0)   # (W,) complex
    cross_corr = xp.real(xp.fft.ifft(cross_spectrum))           # (W,) real

    # Move to host for the peak search + parabolic interpolation
    cross_corr_host = (np.asarray(cross_corr) if xp is np
                        else xp.asnumpy(cross_corr))
    cross_corr_shifted = np.fft.fftshift(cross_corr_host)

    # fftshift places lag=0 at index W//2
    center = W // 2
    sr = int(search_range_px)
    lo = max(0, center - sr)
    hi = min(W, center + sr + 1)
    region = cross_corr_shifted[lo:hi]

    peak_local = int(np.argmax(region))
    peak_global = lo + peak_local

    # Sub-pixel refinement via parabolic interpolation through three points
    # around the integer peak. For a parabola fit y = a(x - x*)^2 + b,
    # x* = peak_global + (y_minus - y_plus) / (2 * (y_minus - 2*y_zero + y_plus))
    if 0 < peak_global < len(cross_corr_shifted) - 1:
        ym = float(cross_corr_shifted[peak_global - 1])
        y0 = float(cross_corr_shifted[peak_global])
        yp = float(cross_corr_shifted[peak_global + 1])
        denom = 2.0 * (ym - 2.0 * y0 + yp)
        if abs(denom) > 1e-10:
            subpx = (ym - yp) / denom
        else:
            subpx = 0.0
    else:
        subpx = 0.0

    offset_px = float(peak_global - center + subpx)
    return offset_px


def _compute_temporal_mean(stack: np.ndarray,
                            start: int, end: int,
                            batch_size: int) -> np.ndarray:
    """Streaming temporal mean of stack[start:end] without holding the
    full subset in memory at once. Returns float64 (H, W)."""
    _, H, W = stack.shape
    M = np.zeros((H, W), dtype=np.float64)
    for s in range(start, end, batch_size):
        e = min(s + batch_size, end)
        batch = stack[s:e].astype(np.float32, copy=False)
        M += batch.sum(axis=0, dtype=np.float64)
    M /= (end - start)
    return M


def _apply_constant_shift_xp(stack: np.ndarray,
                              offset_px: float,
                              batch_size: int,
                              xp) -> np.ndarray:
    """FFT-domain horizontal shift of even-indexed lines by offset_px.

    Uses exp(-2πi · offset_px · k / W) phase factor on per-line FFTs.
    Circular boundary in X — for sub-pixel offsets this corrupts < 1
    pixel column at the right edge (where left-edge content wraps in).
    """
    T, H, W = stack.shape
    k = xp.fft.fftfreq(W).astype(xp.float32) * W            # (W,) cycles/sample
    phase = xp.exp(-2j * xp.pi * float(offset_px) * k / W).astype(xp.complex64)

    corrected = np.empty_like(stack, dtype=np.float32)
    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        batch = stack[s:e].astype(np.float32, copy=False)
        batch_xp = (xp.asarray(batch) if xp is not np else batch.copy())

        # Extract even-indexed lines, FFT along X, apply phase, IFFT
        even_lines = batch_xp[:, ::2, :]                    # (batch, H/2, W)
        F = xp.fft.fft(even_lines.astype(xp.complex64), axis=2)
        F = F * phase[None, None, :]
        even_shifted = xp.real(xp.fft.ifft(F, axis=2)).astype(xp.float32)
        batch_xp[:, ::2, :] = even_shifted

        corrected[s:e] = (np.asarray(batch_xp) if xp is np
                          else xp.asnumpy(batch_xp))
    return corrected


def _apply_per_frame_shift_xp(stack: np.ndarray,
                               offsets_per_frame: np.ndarray,
                               batch_size: int,
                               xp) -> np.ndarray:
    """FFT-domain horizontal shift with per-frame offset.

    offsets_per_frame : (T,) float32 array, one offset per frame.
    """
    T, H, W = stack.shape
    k = xp.fft.fftfreq(W).astype(xp.float32) * W            # (W,)

    corrected = np.empty_like(stack, dtype=np.float32)
    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        batch = stack[s:e].astype(np.float32, copy=False)
        batch_xp = (xp.asarray(batch) if xp is not np else batch.copy())

        # Per-frame phase factors: shape (batch, W)
        offsets_batch = offsets_per_frame[s:e].astype(np.float32)
        offsets_xp = (xp.asarray(offsets_batch) if xp is not np
                      else offsets_batch)
        # phase[i, kk] = exp(-2πi · offsets[i] · k[kk] / W)
        phase = xp.exp(-2j * xp.pi *
                       offsets_xp[:, None] * k[None, :] / W).astype(xp.complex64)

        even_lines = batch_xp[:, ::2, :]                    # (batch, H/2, W)
        F = xp.fft.fft(even_lines.astype(xp.complex64), axis=2)
        F = F * phase[:, None, :]                           # broadcast over rows
        even_shifted = xp.real(xp.fft.ifft(F, axis=2)).astype(xp.float32)
        batch_xp[:, ::2, :] = even_shifted

        corrected[s:e] = (np.asarray(batch_xp) if xp is np
                          else xp.asnumpy(batch_xp))
    return corrected


# ============================================================================
# Public API
# ============================================================================

def correct_bidirectional_scan(
    stack: np.ndarray,
    offset_px: Optional[float] = None,
    search_range_px: int = 4,
    estimate_from: str = 'session_mean',
    n_chunks: int = 20,
    min_frames_per_chunk: int = 500,
    batch_size: int = 512,
    return_diagnostics: bool = False,
    use_gpu: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """Correct bidirectional resonant-scanner line misalignment.

    Estimates the horizontal sub-pixel offset between odd-indexed and
    even-indexed lines via cross-correlation, then applies a Fourier-
    domain shift to align the even lines with the odd lines.

    Parameters
    ----------
    stack : (T, H, W) ndarray
    offset_px : float, optional
        If provided, use this offset directly (skip estimation). The
        offset is the amount to SHIFT even-indexed lines (positive =
        right) to align them with odd-indexed lines. This matches what
        :func:`_estimate_offset_from_image` returns: the cross-correlation
        peak directly gives the shift needed for alignment.
    search_range_px : int
        ±search range for the cross-correlation peak (default ±4 px).
        Set larger if you expect very large misalignment; smaller for
        slightly more robust subpixel estimation.
    estimate_from : {'session_mean', 'per_chunk'}
        Estimation strategy:
          'session_mean' (default): estimate offset once from the
              session-wide temporal mean and apply uniformly.
          'per_chunk': estimate offset per chunk and linearly
              interpolate per-frame offsets between chunk centers.
              Useful when thermal drift causes the bidirectional
              offset to vary slowly across the session.
    n_chunks : int
        Number of chunks if estimate_from='per_chunk'. Auto-clamped to
        T // min_frames_per_chunk.
    min_frames_per_chunk : int
        Floor for per-chunk frame count when estimate_from='per_chunk'.
    batch_size : int
        Frames per GPU/CPU batch in the apply pass.
    return_diagnostics : bool
        If True, also returns a dict containing the offset(s) used.
    use_gpu : bool
        Use CuPy if available.

    Returns
    -------
    corrected : (T, H, W) float32 ndarray
    diag : dict, optional (when return_diagnostics=True)
        Contains 'offset_px' (the estimated/used scalar offset), and if
        estimate_from='per_chunk', also 'chunk_offsets' (per-chunk
        offsets) and 'chunk_centers' (frame indices at chunk midpoints).

    Notes on subpixel precision
    ---------------------------
    Parabolic interpolation around the integer cross-correlation peak
    typically gives < 0.05 px error for offsets in the range ±2 px on
    images with broadband horizontal structure (cells). Performance
    degrades for offsets close to integer values (where the parabolic
    fit's curvature is poorly defined) and for images with very narrow
    spatial frequency content. For sa-000093 with its mix of cell
    structure and noise, subpixel error should be well below 0.1 px.

    Edge effects
    ------------
    The FFT-based shift is circular, so content from the right edge
    wraps to the left during a leftward shift (and vice versa). For
    typical bidirectional offsets (sub-pixel to a few pixels), this
    corrupts a few columns at the boundary. CNMF tolerates this if the
    cells are not pressed against the edge; for sessions where edge
    cells matter, consider trimming the boundary columns post-correction.
    """
    t_start = _time.perf_counter()
    xp = _detrend_xp(use_gpu)
    stack = np.asarray(stack)
    T, H, W = stack.shape

    log.info("correct_bidirectional_scan: T=%d, H=%d, W=%d, "
              "search=±%d px, estimate_from=%s, backend=%s",
              T, H, W, search_range_px, estimate_from,
              "cupy" if xp is not np else "numpy")

    diag: Dict = {}
    offsets_per_frame: Optional[np.ndarray] = None

    if offset_px is not None:
        log.info("correct_bidirectional_scan: using user-provided "
                  "offset_px=%+.3f", offset_px)
        chosen_offset = float(offset_px)

    elif estimate_from == 'session_mean':
        t_mean = _time.perf_counter()
        M = _compute_temporal_mean(stack, 0, T, batch_size)
        log.info("correct_bidirectional_scan: temporal mean computed in %.1fs",
                  _time.perf_counter() - t_mean)
        chosen_offset = _estimate_offset_from_image(M, search_range_px, xp)
        log.info("correct_bidirectional_scan: estimated offset = %+.4f px "
                  "(from session mean)", chosen_offset)

    elif estimate_from == 'per_chunk':
        actual_chunks = min(max(1, n_chunks),
                             max(1, T // max(1, min_frames_per_chunk)))
        chunk_size = T // actual_chunks
        bounds = [(k * chunk_size,
                    (k + 1) * chunk_size if k < actual_chunks - 1 else T)
                   for k in range(actual_chunks)]
        chunk_offsets = np.empty(actual_chunks, dtype=np.float32)
        chunk_centers = np.empty(actual_chunks, dtype=np.float32)
        t_chunks = _time.perf_counter()
        for k, (s, e) in enumerate(bounds):
            M_k = _compute_temporal_mean(stack, s, e, batch_size)
            chunk_offsets[k] = _estimate_offset_from_image(
                M_k, search_range_px, xp)
            chunk_centers[k] = (s + e) / 2.0
        log.info("correct_bidirectional_scan: per-chunk estimation: "
                  "%d chunks in %.1fs; offsets min=%+.4f, "
                  "median=%+.4f, max=%+.4f px",
                  actual_chunks, _time.perf_counter() - t_chunks,
                  float(chunk_offsets.min()),
                  float(np.median(chunk_offsets)),
                  float(chunk_offsets.max()))
        # Interpolate per-frame offsets between chunk centers; clip at
        # boundaries via np.interp's default behaviour (constant outside).
        offsets_per_frame = np.interp(
            np.arange(T, dtype=np.float32),
            chunk_centers, chunk_offsets).astype(np.float32)
        chosen_offset = float(np.median(chunk_offsets))
        diag["chunk_offsets"] = chunk_offsets
        diag["chunk_centers"] = chunk_centers

    else:
        raise ValueError(f"unknown estimate_from: {estimate_from!r}; "
                          f"expected 'session_mean' or 'per_chunk'")

    diag["offset_px"] = chosen_offset
    diag["offsets_per_frame"] = offsets_per_frame

    # Skip the apply pass if the offset is sub-tenth-of-a-pixel
    # (correction would be a tiny dithering of no practical value).
    if (offsets_per_frame is None and abs(chosen_offset) < 0.05):
        log.info("correct_bidirectional_scan: |offset| < 0.05 px; "
                  "skipping apply pass (returning input as float32)")
        corrected = stack.astype(np.float32, copy=False)
    else:
        t_apply = _time.perf_counter()
        # `chosen_offset` is the amount we need to shift even lines to
        # align them with odd lines (the cross-correlation peak directly
        # gives this — positive means even lines need to move right).
        if offsets_per_frame is None:
            corrected = _apply_constant_shift_xp(
                stack, chosen_offset, batch_size, xp)
        else:
            corrected = _apply_per_frame_shift_xp(
                stack, offsets_per_frame, batch_size, xp)
        log.info("correct_bidirectional_scan: applied shift in %.1fs",
                  _time.perf_counter() - t_apply)

    log.info("correct_bidirectional_scan: total %.1fs",
              _time.perf_counter() - t_start)

    if return_diagnostics:
        return corrected, diag
    return corrected


__all__ = ["correct_bidirectional_scan"]
