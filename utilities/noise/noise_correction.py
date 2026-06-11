"""
noise_correction.py — 2P noise-removal primitives keyed to the diagnostic suite
================================================================================

Companion to ``noise_diagnostics.py``. Each function targets one source
flagged by the diagnostic suite and operates on an in-memory ``(T, H, W)``
float32 stack. The primitives are composable; ``apply_corrections`` chains a
list of (callable, kwargs) pairs in order, and ``recommend_corrections``
turns a diagnostic-report dict into such a list automatically.

Pre-existing correction in the package
--------------------------------------
``caiman.utils.xcorr_correction.correct_line_scan`` already does
**integer-pixel** bidirectional correction at file granularity, with GPU
acceleration. The ``correct_bidirectional`` here is the **sub-pixel**
complement for in-memory stacks (typical shifts on resonant scanners are
0.2–0.8 px, which integer-pixel rounding handles poorly).

Algorithmic references
----------------------
- Bidirectional sub-pixel shift: cross-correlation of even/odd row column
  profiles followed by parabolic interpolation around the discrete peak.
  Same algorithm family used by suite2p (Pachitariu et al. 2017) and
  NoRMCorre (Pnevmatikakis & Giovannucci 2017). Implemented from scratch
  here to avoid GPL-3-vs-GPL-2 license mixing with suite2p; the algorithm
  itself is not novel to either project.
- Common-mode regression: linear projection out of the leading temporal
  mode. Standard in mesoscale calcium imaging (Allen et al. 2017) and
  fMRI denoising (CompCor; Behzadi et al. 2007).
- Row-pedestal subtraction: per-row temporal-median offset removal.
  Standard rank-1 fixed-banding removal; used routinely in raster-scan
  electron microscopy and ScanImage post-processing.
- Notch filtering: textbook scipy.signal.iirnotch + filtfilt.

Memory note
-----------
All primitives operate in-memory. For multi-GB stacks, drive them from
``caiman.utils.stack_io.stack_iter`` in chunks (the chunked variants live
in a separate follow-up; this module keeps the core algorithms simple
and unit-testable).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage, signal as sps

log = logging.getLogger("noise_correction")


# ============================================================================
# 1. Bidirectional sub-pixel correction
# ============================================================================

def estimate_bidirectional_shift(stack: np.ndarray,
                                 max_shift_px: float = 2.0,
                                 grid_step_px: float = 0.1) -> float:
    """Estimate the sub-pixel even-row x-shift that minimises adjacent-row
    discontinuity on the band-pass-filtered temporal mean.

    Same algorithm the diagnostic test uses, exposed standalone so callers
    can run estimate + apply as separate steps.

    Returns
    -------
    float
        The shift to apply to even rows to ALIGN them with odd rows. Pass
        directly to ``correct_bidirectional(stack, shift_px=s)`` — no sign
        flip needed.
    """
    H, W = stack.shape[1:]
    n_pairs = H // 2
    M = stack.mean(axis=0)
    M_hp = M - ndimage.gaussian_filter(M, sigma=12)
    even = M_hp[0:2 * n_pairs:2, :]
    odd = M_hp[1:2 * n_pairs:2, :]
    edge = max(8, int(0.03 * W))

    grid = np.arange(-max_shift_px, max_shift_px + grid_step_px, grid_step_px)
    errs = np.empty_like(grid)
    for i, s in enumerate(grid):
        shifted = ndimage.shift(even, shift=(0, s), order=1, mode="nearest")
        d = (shifted - odd)[:, edge:W - edge]
        errs[i] = float(np.sum(d * d))
    k = int(np.argmin(errs))
    if 0 < k < errs.size - 1:
        y0, y1, y2 = errs[k - 1], errs[k], errs[k + 1]
        denom = (y0 - 2 * y1 + y2)
        dx = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        return float(grid[k] + dx * grid_step_px)
    return float(grid[k])


def correct_bidirectional(stack: np.ndarray,
                          shift_px: Optional[float] = None,
                          max_shift_px: float = 2.0) -> np.ndarray:
    """Apply a sub-pixel bidirectional phase correction.

    Shifts even rows by ``shift_px`` along x using 1D linear interpolation
    (scipy.ndimage.shift, order=1). If ``shift_px`` is None the shift is
    estimated from the stack itself.

    Notes
    -----
    Edge mode is ``nearest`` to avoid introducing dark borders at the FOV
    edges; the existing flyback columns (typically ~8 px on the left) will
    still need cropping. Use the pipeline's ``bord_px`` if those columns
    matter for downstream analysis.

    Returns a new (T, H, W) float32 array; does not modify ``stack`` in place.
    """
    if shift_px is None:
        shift_px = estimate_bidirectional_shift(stack, max_shift_px=max_shift_px)
        log.info("correct_bidirectional: estimated shift = %.3f px", shift_px)
    out = stack.astype(np.float32, copy=True)
    out[:, 0::2, :] = ndimage.shift(
        out[:, 0::2, :], shift=(0, 0, shift_px), order=1, mode="nearest")
    return out


# ============================================================================
# 2. Row pedestal subtraction (stationary horizontal banding)
# ============================================================================

def subtract_row_pedestal(stack: np.ndarray,
                          mode: str = "temporal_median") -> np.ndarray:
    """Remove a per-row offset that is fixed across time (stationary banding).

    Modes
    -----
    "temporal_median" (default): for each row r, compute the temporal median
        of ``stack[:, r, :].mean(axis=1)`` (one number per row), then subtract
        ``pedestal[r] - global_median`` from row r of every frame. Removes a
        rank-1 fixed pattern without disturbing relative pixel intensities
        within a row.

    "per_frame_median": for each (frame, row), compute the median of that
        row and subtract its deviation from the frame median. More aggressive;
        also removes drift in the banding amplitude over time. Use this when
        the diagnostic flags ``horizontal_banding_drifting`` rather than
        ``horizontal_banding_fixed``.

    Returns a new (T, H, W) float32 array.
    """
    if mode not in ("temporal_median", "per_frame_median"):
        raise ValueError(f"mode must be one of "
                         f"'temporal_median'/'per_frame_median'; got {mode!r}")
    out = stack.astype(np.float32, copy=True)
    if mode == "temporal_median":
        row_traces = stack.mean(axis=2)                 # (T, H)
        pedestal = np.median(row_traces, axis=0)        # (H,)
        offset = pedestal - np.median(pedestal)         # (H,)
        out -= offset[None, :, None]
    else:  # per_frame_median
        # Per-frame row median, minus per-frame global median.
        row_med = np.median(stack, axis=2)              # (T, H)
        frame_med = np.median(row_med, axis=1, keepdims=True)  # (T, 1)
        offset = row_med - frame_med                    # (T, H)
        out -= offset[:, :, None]
    return out


def subtract_column_pedestal(stack: np.ndarray,
                             mode: str = "temporal_median") -> np.ndarray:
    """Remove a per-column offset that is fixed across time (vertical stripes).

    Symmetric twin of ``subtract_row_pedestal`` for the fast-scan axis. The
    common motivating artifact on resonant-scanner 2P systems is per-column
    brightness variation caused by scanner-velocity nonlinearity or detector
    column-clocked structure — produces the dense vertical-stripe pattern
    that contaminates correlation images and makes CNMF hallucinate
    column-aligned "cells".

    Modes
    -----
    "temporal_median" (default): for each column c, compute the temporal
        median of ``stack[:, :, c].mean(axis=1)`` (one number per column),
        then subtract ``pedestal[c] - global_median`` from column c of every
        frame. FOV-uniform fixed pattern.

    "per_frame_median": for each (frame, column), compute the median of that
        column and subtract its deviation from the frame median. Use this
        when the stripe amplitude drifts over time (rare on stable systems).

    Caveat
    ------
    Column-pedestal subtraction removes any signal that is FOV-uniform
    along the slow (y) axis. Real cells are not column-aligned and span
    only a few rows, so they contribute negligibly to a column's temporal-
    mean intensity — but if the FOV happens to contain a vertical blood
    vessel or a vertical columnar structure that is genuinely present in
    the biology, this primitive will partially flatten it. Inspect the
    correction's effect on the temporal-mean image before committing to it
    as a default for a given preparation.

    Returns a new (T, H, W) float32 array.
    """
    if mode not in ("temporal_median", "per_frame_median"):
        raise ValueError(f"mode must be one of "
                         f"'temporal_median'/'per_frame_median'; got {mode!r}")
    out = stack.astype(np.float32, copy=True)
    if mode == "temporal_median":
        col_traces = stack.mean(axis=1)                 # (T, W)
        pedestal = np.median(col_traces, axis=0)        # (W,)
        offset = pedestal - np.median(pedestal)         # (W,)
        out -= offset[None, None, :]
    else:  # per_frame_median
        col_med = np.median(stack, axis=1)              # (T, W)
        frame_med = np.median(col_med, axis=1, keepdims=True)  # (T, 1)
        offset = col_med - frame_med                    # (T, W)
        out -= offset[:, None, :]
    return out


# ============================================================================
# 3. Fixed-pattern noise — 2D FFT notch on the temporal mean
# ============================================================================

def detect_fpn_peaks(M: np.ndarray,
                     cell_scale_px: float = 12.0,
                     prominence_db: float = 15.0,
                     max_peaks: int = 32,
                     magnitude_in: bool = False,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Locate isolated 2D FFT peaks in the temporal-mean image that
    correspond to fixed periodic spatial structure (FPN).

    Returns
    -------
    notch_mask : (H, W) bool
        Bins to zero in the fftshift'd 2D FFT of M. Includes a 1-bin
        radius around each detected peak and its conjugate-symmetric
        counterpart (required to keep ``ifft2`` real-valued).
    F_shifted : (H, W) complex or float
        When ``magnitude_in=False`` (default), the fftshift'd 2D FFT of
        ``M - M.mean()`` — returned so callers that already paid the FFT
        cost don't have to repeat it. When ``magnitude_in=True``, the
        input ``M`` itself (treated as a precomputed magnitude
        spectrum), returned unchanged for API symmetry.

    Parameters
    ----------
    cell_scale_px : float
        Spatial scale (radius, in pixels) below which Fourier content is
        protected by the central-disk mask. Defaults to 12 (covering
        cells up to ~24 px diameter).
    prominence_db : float
        Minimum height in dB above the median spectral floor for a bin
        to qualify as a candidate peak. Default 15.
    max_peaks : int
        Hard cap on the number of peaks detected, after sorting all
        candidates by prominence (highest first). This is essential for
        full-FOV temporal means: the shot-noise floor falls as 1/√T, so
        on long recordings (T ≥ 10³) the dB-prominence test catches many
        small peaks induced by random cell positioning rather than real
        FPN. Real lattice peaks consistently rise 40-70 dB above the
        floor, far above any cell-induced peak (typically 15-30 dB), so
        keeping the top ``max_peaks`` by prominence reliably selects the
        FPN. Default 32 — generous enough for a 2D lattice with several
        harmonics and their 4-fold conjugate mirrors, restrictive enough
        to prevent catastrophic over-notching of cellular content.
    magnitude_in : bool
        When False (default), ``M`` is interpreted as a real image and
        the detector computes its FFT internally. When True, ``M`` is
        interpreted as an already-shifted magnitude spectrum (i.e.
        ``|fftshift(fft2(image))|``, possibly averaged over many frames
        as in ``subtract_per_frame_pattern``). The peak-detection logic
        is otherwise identical: same disk mask, same prominence test,
        same conjugate-symmetric notch construction. Use the True path
        when peaks need detecting in a pooled-magnitude spectrum that
        captures frame-varying contamination invisible to the temporal
        mean.

    Algorithm
    ---------
    1. 2D FFT of mean-subtracted M, fftshift so DC sits at the centre.
       (Skipped when ``magnitude_in=True`` — the input is already the
       magnitude.)
    2. Mask out a central disk of radius ``1/(2·cell_scale_px)`` cyc/px
       so cell-scale and slower spatial structure is preserved (the
       broad spectral lobes from cells, vignetting, blood vessels never
       form isolated peaks — they're protected anyway, but the disk
       guards against unusually compact cells getting clipped).
    3. Estimate a noise floor as ``median(P_db)`` over the non-central
       bins. The median is robust to a handful of FPN peaks.
    4. Detect local maxima with 3×3 neighbourhood that exceed
       ``floor + prominence_db``. These are the FPN peaks.
    5. For each peak (y, x), mark the 3×3 bin neighbourhood AND the
       conjugate-symmetric peak ``((H - y) % H, (W - x) % W)`` and
       its neighbourhood. Conjugate marking is needed because the
       inverse FFT of a non-conjugate-symmetric spectrum is complex;
       any single-peak notch would otherwise leak an imaginary part
       into the inverse and corrupt the output.

    Robustness to cellular content
    ------------------------------
    Cells produce broad, smooth spectral content (their footprint is
    spatially localised → their FFT is spread). They do NOT produce
    isolated sharp peaks. The local-maxima + prominence test specifically
    rejects them. This is what makes the routine safe to fire on the
    ``fixed_pattern_noise`` flag even when that flag has a known
    false-positive tendency on cellular FOVs.
    """
    M = np.asarray(M, dtype=np.float32)
    H, W = M.shape

    if magnitude_in:
        # ``M`` is already a magnitude spectrum (shifted, DC at centre).
        # Convert to power for the dB-prominence test; F_shifted is the
        # input itself, returned for API symmetry (callers that already
        # have it).
        P = M.astype(np.float32) ** 2
        F_shifted = M
    else:
        # FFT of mean-subtracted M, shifted so DC is at center
        F_shifted = np.fft.fftshift(np.fft.fft2(M - M.mean()))
        P = np.abs(F_shifted) ** 2
    P_db = 10.0 * np.log10(P + 1e-12)
    cy, cx = H // 2, W // 2

    # Central-disk mask — preserve cell-scale spatial content
    yy, xx = np.ogrid[:H, :W]
    r_norm = np.sqrt(((yy - cy) / H) ** 2 + ((xx - cx) / W) ** 2)
    cell_freq = 1.0 / (2.0 * max(cell_scale_px, 1.0))
    central_mask = r_norm < cell_freq

    # Noise floor estimate (median of non-central bins)
    floor_db = float(np.median(P_db[~central_mask]))
    threshold_db = floor_db + prominence_db

    # Local maxima above threshold (3x3 neighbourhood)
    is_lmax = (P_db == ndimage.maximum_filter(P_db, size=3))
    candidates = is_lmax & (P_db > threshold_db) & (~central_mask)

    # Cap at top-K by prominence (sorted highest first). Essential for
    # full-FOV temporal means where cell-induced spectral noise produces
    # many small candidates that all pass a fixed dB threshold; the real
    # FPN peaks consistently rise much higher above the floor, so taking
    # the top-K reliably selects them. See max_peaks docstring.
    candidate_coords = np.argwhere(candidates)
    if len(candidate_coords) > max_peaks:
        candidate_proms = P_db[candidate_coords[:, 0], candidate_coords[:, 1]]
        keep_idx = np.argsort(candidate_proms)[::-1][:max_peaks]
        candidate_coords = candidate_coords[keep_idx]
        log.debug("detect_fpn_peaks: %d candidates → capped at top %d by prominence",
                   int(candidates.sum()), max_peaks)

    # Build notch mask: 1-bin radius around each peak + conjugate pair
    notch_mask = np.zeros_like(candidates, dtype=bool)
    for y, x in candidate_coords:
        for sy, sx in [(y, x), ((H - y) % H, (W - x) % W)]:
            y0, y1 = max(0, sy - 1), min(H, sy + 2)
            x0, x1 = max(0, sx - 1), min(W, sx + 2)
            notch_mask[y0:y1, x0:x1] = True

    log.debug("detect_fpn_peaks: floor=%.1f dB, %d candidates, %d notched bins",
              floor_db, int(candidates.sum()), int(notch_mask.sum()))
    return notch_mask, F_shifted


def subtract_fixed_pattern(stack: np.ndarray,
                            cell_scale_px: float = 12.0,
                            prominence_db: float = 15.0,
                            max_peaks: int = 32,
                            return_fpn: bool = False
                            ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Subtract a 2D-periodic fixed-pattern noise component from every frame.

    Detects oriented or lattice spatial structure that persists across
    time (the kind that survives bidirectional correction, common-mode
    regression, and row/column pedestal subtraction because it's neither
    axis-aligned nor temporally coherent). The standard example on
    resonant-scanner 2P data is a 2D lattice from scanner-clock pickup or
    detector-electronics structure — appears as a faint diamond/mesh
    pattern in the temporal mean and a forest of sharp peaks in its
    2D FFT.

    Algorithm: compute temporal mean M, FFT, mask the central disk,
    detect isolated peaks at ≥ ``prominence_db`` above the median spectral
    floor, zero those peaks (with conjugate-symmetric pairs), inverse
    FFT, and subtract ``(M - M_clean)`` from every frame.

    Caveats
    -------
    - The FPN must be both **spatially periodic** (concentrated at
      discrete FFT peaks) and **temporally stable** (visible in the
      temporal mean). Drifting FPN, broad spectral pickup, and non-
      periodic fixed structure are not addressed.
    - If the data is shot-noise dominated and the FPN sits below the
      shot-noise spectral floor, no peaks will pass the prominence
      threshold and the function is a no-op. To check what was actually
      removed, pass ``return_fpn=True``.
    - Cells produce broad spectral content and do not usually form
      isolated sharp peaks; the routine is robust against detecting
      cellular content as FPN in the common case. However, if cells
      happen to be approximately periodically distributed (e.g. neat
      cortical columns or a regularly-spaced injection grid), the
      between-cell spacing CAN produce isolated low-frequency peaks
      that the notch will detect and zero. This is benign in practice
      — the subtracted FPN amplitude in such cases is typically <1 DN
      and per-pixel time series (and therefore CNMF outputs) are
      unaffected, because the notch only modifies the spatial
      time-average, not the temporal variation. If you suspect this is
      happening, inspect ``return_fpn=True`` output: spurious peaks
      tend to be at low frequency (just outside the central mask) and
      their summed energy is small.

    Returns
    -------
    out : (T, H, W) float32
        The corrected stack (always returned).
    fpn : (H, W) float32, optional
        The fixed-pattern map subtracted from every frame (only when
        ``return_fpn=True``).
    """
    stack = np.asarray(stack, dtype=np.float32)
    M = stack.mean(axis=0)
    notch_mask, F_shifted = detect_fpn_peaks(
        M, cell_scale_px=cell_scale_px,
        prominence_db=prominence_db, max_peaks=max_peaks)

    if not notch_mask.any():
        log.info("subtract_fixed_pattern: no peaks above %.1f dB prominence; "
                  "no-op", prominence_db)
        out = stack.copy()
        fpn = np.zeros_like(M)
        return (out, fpn) if return_fpn else out

    F_clean = F_shifted.copy()
    F_clean[notch_mask] = 0
    M_clean = np.real(np.fft.ifft2(np.fft.ifftshift(F_clean))) + M.mean()
    fpn = (M - M_clean).astype(np.float32)

    log.info("subtract_fixed_pattern: notched %d FFT bins, "
              "fpn std=%.3f max|fpn|=%.2f DN",
              int(notch_mask.sum()), float(fpn.std()), float(np.abs(fpn).max()))

    out = stack - fpn[None, :, :]
    return (out, fpn) if return_fpn else out


# ============================================================================
# 3b. Per-frame spectral notch (frame-varying lattice contamination)
# ============================================================================

def subtract_per_frame_pattern(stack: np.ndarray,
                                cell_scale_px: float = 12.0,
                                prominence_db: float = 15.0,
                                max_peaks: int = 32,
                                batch_size: int = 512,
                                return_pattern: bool = False,
                                use_gpu: bool = False
                                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Remove a frame-varying periodic spatial pattern by spectral notching.

    Targets scanner-synchronous or other periodic contamination whose
    *spatial* lattice is stable across frames but whose *phase* drifts
    frame-to-frame. The Dec-6 strohA cross-hatch at (fy=±0.109, fx=±0.031)
    cyc/px is the motivating example: pooled-magnitude coherence
    ratio ≈ 0.04 in fc046200 (almost entirely frame-varying), so the
    temporal-mean approach of :func:`subtract_fixed_pattern` recovers
    only ~4% of the contamination amplitude. This routine recovers the
    rest.

    Unlike :func:`subtract_fixed_pattern`, which subtracts one stationary
    pattern from every frame, this routine:

    1. **Pass 1 — pool magnitude spectra.** Compute ``|FFT(frame_i)|``
       for every frame and average across the time axis. This captures
       any spatially-stable spectral content regardless of per-frame
       phase, because magnitude is phase-invariant.
    2. **Peak detection.** Identify peak bins in the pooled magnitude
       using the same local-max + prominence + conjugate-symmetric
       logic as :func:`detect_fpn_peaks`. The pooled magnitude has
       T-times more SNR than any individual frame's spectrum, so peaks
       that are dim in each frame still stand out clearly here.
    3. **Pass 2 — per-frame notch.** Apply the same notch mask to each
       frame's individual FFT, zeroing the offending bins, then IFFT
       back. Each frame loses *its own* phase contribution at those
       bins — different per frame, but at fixed spatial coordinates.

    Compose with :func:`subtract_fixed_pattern` for mixed-stationarity
    cases (e.g. Apr-22 strohA with coherence ratio 0.28 at the lattice):
    run :func:`subtract_fixed_pattern` first to remove the stationary
    fraction, then this routine on the residual to clean up the
    frame-varying remainder. The two filters target orthogonal
    components of the same spectral peaks.

    Parameters
    ----------
    stack : (T, H, W) ndarray
        Input movie, host RAM. Float32 is fastest; other dtypes are cast.
    cell_scale_px : float
        Central-disk radius for cell-content protection in
        :func:`detect_fpn_peaks`. Default 12.0 — same as
        :func:`subtract_fixed_pattern`. The pooled-magnitude geometry
        is identical to the temporal-mean geometry: cells produce
        broad smooth content centred on DC, lattice peaks sit cleanly
        off-axis, the same disk size protects the same content.
    prominence_db : float
        Minimum dB above the spectral floor for a candidate peak.
        Default 15.0 — same as :func:`subtract_fixed_pattern`. An
        a-priori argument that the pooled spectrum has T-times higher
        SNR than the temporal mean (so the threshold could be tighter)
        is wrong: pooling magnitudes averages |signal + noise| ≈
        |signal| at each bin, so the absolute prominence of a real
        peak is comparable to a single-frame value, not T-times
        higher. The gain over per-frame analysis is in floor
        *stability* (variance drops as 1/√T), not floor *level*. The
        same 15 dB threshold therefore catches genuine lattice peaks
        in both cases. Calibrated against sa-000093 (T=110880):
        lattice peaks at (±52, ±12) cyc/px register at +15-20 dB
        prominence, well-separated from cell-content baseline at +2-5 dB.
    max_peaks : int
        Hard cap on detected peaks. Default 32, matching
        :func:`subtract_fixed_pattern`. Larger than naively necessary
        because a 2D lattice with multiple harmonics produces 4-fold
        conjugate quartets — sa-000093 fills 9 of the 32 slots
        (1 Nyquist + 4 fundamental lattice + 4 first-harmonic
        lattice), other sessions may fill more.
    batch_size : int
        Frames per GPU/CPU batch in both pass 1 and pass 2. Default 512.
        With H=W=512 float32, one batch is ``batch_size`` MB on host
        and roughly 2-4× that on GPU (input + FFT workspace + output).
        For a 96 GB GPU, ``batch_size=1024`` is comfortable; for 16 GB
        cards, stay at 256-512.
    return_pattern : bool
        If True, return ``(cleaned, pattern)`` where ``pattern`` is the
        per-frame ``(T, H, W)`` contamination that was subtracted.
        Memory cost: doubles peak host RAM. Useful for QC plots.
    use_gpu : bool
        When True, the per-frame FFTs run on CuPy if available, falling
        back to NumPy if not. Falsy default for safety; for sessions
        with T > ~10⁴ frames at 512×512, enabling GPU drops wall time
        from minutes to seconds.

    Returns
    -------
    cleaned : (T, H, W) ndarray, float32
        Stack with the detected lattice removed per-frame.
    pattern : (T, H, W) ndarray, optional
        Only when ``return_pattern=True``. The per-frame contamination
        that was subtracted (i.e. ``stack - cleaned``).

    Memory
    ------
    Pass 1 streams batches and accumulates a single (H, W) magnitude
    buffer — peak host RAM is ``batch_size × H × W × 4 bytes``.

    Pass 2 streams batches but writes into a (T, H, W) output array —
    peak host RAM is ``T × H × W × 4 bytes`` for the output plus one
    batch in flight. For T=110880, H=W=512, that's 110 GB. Comfortable
    on lab workstations with ≥256 GB RAM; for memory-tight environments,
    use ``return_pattern=False`` (avoids the second large allocation)
    and consider raising the input dtype-check to require float32 input
    (avoiding a host-side astype copy).

    No-op gate
    ----------
    If the top peak in the pooled magnitude is less than
    ``prominence_db`` above the spectral floor, the routine concludes
    no real lattice is present, logs an info message, and returns the
    input stack unchanged (no Pass 2, no allocation). This makes the
    routine safe to enable session-wide: sessions without a frame-
    varying lattice incur only the Pass 1 cost (~20 s on GPU for
    T=10⁵).

    Diagnostics
    -----------
    Logs at INFO:
      - pass 1: time, top peak prominence in pooled magnitude
      - peak count, notch mask coverage as % of spectrum
      - pass 2: time, mean and max |pattern| amplitude
      - no-op: peak prominence below threshold, returning unchanged
    """
    xp = _detrend_xp(use_gpu)  # reuse the existing GPU dispatcher

    stack = np.asarray(stack)
    T, H, W = stack.shape
    log.info("subtract_per_frame_pattern: T=%d, H=%d, W=%d, batch=%d, backend=%s",
              T, H, W, batch_size, "cupy" if xp is not np else "numpy")

    # ---- PASS 1: pool magnitude spectra across all frames ----
    import time as _time
    t0 = _time.perf_counter()
    A_pooled = xp.zeros((H, W), dtype=xp.float32)
    n_done = 0
    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        batch = stack[s:e].astype(np.float32, copy=False)
        # Move batch onto whichever backend we're using
        batch_xp = xp.asarray(batch) if xp is not np else batch
        # Per-frame demean → batched fft2 over the time axis →
        # accumulate magnitudes. fftshift is folded in after the sum
        # to save one shift per batch (fftshift is a permutation, so
        # sum-then-shift = shift-then-sum).
        batch_xp = batch_xp - batch_xp.mean(axis=(1, 2), keepdims=True)
        F = xp.fft.fft2(batch_xp)
        A_pooled += xp.abs(F).sum(axis=0)
        n_done = e
        if (e // batch_size) % 20 == 0 or e == T:
            log.debug("subtract_per_frame_pattern pass 1: %d/%d frames",
                       n_done, T)
    A_pooled = xp.fft.fftshift(A_pooled / T)

    # Bring pooled magnitude to host for peak detection (small, (H,W))
    A_pooled_host = (np.asarray(A_pooled) if xp is np
                      else xp.asnumpy(A_pooled))
    t_pass1 = _time.perf_counter() - t0

    # ---- Peak detection on the pooled magnitude ----
    notch_mask, _ = detect_fpn_peaks(
        A_pooled_host,
        cell_scale_px=cell_scale_px,
        prominence_db=prominence_db,
        max_peaks=max_peaks,
        magnitude_in=True,
    )
    n_notched = int(notch_mask.sum())

    # Probe the top peak's prominence above floor for the no-op gate.
    # If we couldn't find any peak above the threshold, detect_fpn_peaks
    # returned an empty notch_mask and we should return early.
    if n_notched == 0:
        log.info("subtract_per_frame_pattern: pass 1 done in %.1fs, "
                  "no peaks above %.1f dB prominence; returning unchanged",
                  t_pass1, prominence_db)
        if return_pattern:
            return stack.astype(np.float32, copy=True), \
                   np.zeros_like(stack, dtype=np.float32)
        return stack.astype(np.float32, copy=False)

    log.info("subtract_per_frame_pattern: pass 1 done in %.1fs, "
              "%d FFT bins notched (%.3f%% of spectrum)",
              t_pass1, n_notched, 100.0 * n_notched / (H * W))

    # The notch mask was built in fftshifted coordinates (DC at centre).
    # For pass-2 application we need un-shifted coordinates to match
    # what cp.fft.fft2 produces directly. ifftshift inverts fftshift.
    notch_mask_unshifted = np.fft.ifftshift(notch_mask)
    notch_mask_xp = (xp.asarray(notch_mask_unshifted) if xp is not np
                      else notch_mask_unshifted)

    # ---- PASS 2: apply per-frame notch ----
    t0 = _time.perf_counter()
    cleaned = np.empty_like(stack, dtype=np.float32)
    pattern_acc = np.empty_like(stack, dtype=np.float32) if return_pattern else None
    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        batch = stack[s:e].astype(np.float32, copy=False)
        batch_xp = xp.asarray(batch) if xp is not np else batch
        means = batch_xp.mean(axis=(1, 2), keepdims=True)
        F = xp.fft.fft2(batch_xp - means)
        # Compute the contamination contribution (notched bins only)
        # by zeroing the *complement* of the mask, then ifft. The
        # cleaned output is then the input minus this contamination.
        if return_pattern:
            F_pattern = F * notch_mask_xp[None, :, :]
            pat = xp.real(xp.fft.ifft2(F_pattern))
            cleaned_batch = batch_xp - pat
            cleaned[s:e] = (np.asarray(cleaned_batch) if xp is np
                            else xp.asnumpy(cleaned_batch))
            pattern_acc[s:e] = (np.asarray(pat) if xp is np
                                else xp.asnumpy(pat))
        else:
            # Zero notched bins, ifft. Faster than the return_pattern
            # path because we don't materialise pattern.
            F = F * (~notch_mask_xp)[None, :, :]
            cleaned_batch = xp.real(xp.fft.ifft2(F)) + means
            cleaned[s:e] = (np.asarray(cleaned_batch) if xp is np
                            else xp.asnumpy(cleaned_batch))
    t_pass2 = _time.perf_counter() - t0

    if return_pattern:
        pat_max = float(np.abs(pattern_acc).max())
        pat_std = float(pattern_acc.std())
        log.info("subtract_per_frame_pattern: pass 2 done in %.1fs, "
                  "pattern std=%.3f max|pattern|=%.2f DN",
                  t_pass2, pat_std, pat_max)
        return cleaned, pattern_acc

    log.info("subtract_per_frame_pattern: pass 2 done in %.1fs", t_pass2)
    return cleaned


# ============================================================================
# 4. Common-mode regression
# ============================================================================

def regress_common_mode(stack: np.ndarray,
                        trace: Optional[np.ndarray] = None,
                        return_trace: bool = False
                        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Regress a coherent temporal trace out of every pixel (OLS).

    For each pixel ``p`` and centred common trace ``c``::

        beta_p = (x_p · c) / (c · c)
        x_p   <- x_p - beta_p * c

    Only the residual common-mode component is removed; the per-pixel mean
    is preserved (we centre the trace, not the pixel). Catches globally-
    coherent oscillations flagged by ``periodic_temporal_global`` —
    physiology, aliased mains, room-light flicker — in one shot, without
    needing to know the offending frequency.

    Parameters
    ----------
    trace : ndarray of shape (T,), optional
        The common-mode signal to regress out. If None, uses the frame mean
        ``stack.mean(axis=(1, 2))``. Pass a custom trace (e.g. heartbeat-
        gated trigger, mains reference, pupil signal) to target a known
        source.
    return_trace : bool
        If True, also return the centred trace used.

    Returns the corrected (T, H, W) float32 stack (and the trace if requested).
    """
    T = stack.shape[0]
    if trace is None:
        trace = stack.mean(axis=(1, 2))
    trace = np.asarray(trace, dtype=np.float32).reshape(T)
    if trace.size != T:
        raise ValueError(f"trace length {trace.size} != stack T={T}")

    c = trace - trace.mean()
    denom = float(c @ c)
    if denom <= 0:
        log.warning("regress_common_mode: trace has zero variance; no-op")
        return (stack.astype(np.float32, copy=True), c) if return_trace \
            else stack.astype(np.float32, copy=True)

    # Vectorised: betas = (frames.T @ c) / denom, where frames is (T, H*W).
    H, W = stack.shape[1:]
    flat = stack.reshape(T, H * W).astype(np.float32)
    betas = (flat.T @ c) / denom                        # (H*W,)
    flat = flat - np.outer(c, betas)                    # (T, H*W)
    out = flat.reshape(T, H, W)
    return (out, c) if return_trace else out


_NOISE_GPU_XP = None  # cached array module for the GPU path (resolved once)


def _detrend_xp(use_gpu: bool):
    """Return cupy (if requested and usable) else numpy. Resolved once per
    process so chunked callers don't re-smoke-test the device each call."""
    global _NOISE_GPU_XP
    if not use_gpu:
        return np
    if _NOISE_GPU_XP is not None:
        return _NOISE_GPU_XP
    try:
        import cupy as cp
        _ = cp.array([1.0])                       # smoke-test allocation
        log.info("detrend_temporal: CuPy GPU backend active")
        _NOISE_GPU_XP = cp
    except Exception as e:                          # pragma: no cover
        log.warning(f"detrend_temporal: CuPy unavailable ({e}); using NumPy")
        _NOISE_GPU_XP = np
    return _NOISE_GPU_XP


def _detrend_to_numpy(arr, xp) -> np.ndarray:
    """Move an array to host numpy regardless of backend."""
    return np.asarray(arr) if xp is np else xp.asnumpy(arr)


def detrend_temporal(stack: np.ndarray,
                     order: int = 1,
                     preserve_mean: bool = True,
                     return_trend: bool = False,
                     use_gpu: bool = False
                     ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Remove a slow per-pixel polynomial trend over time (default: linear).

    Fits and subtracts an order-``order`` polynomial in time *independently
    per pixel*.  The polynomial basis columns are mean-centred over time, so
    the removal has zero temporal mean per pixel and the per-pixel baseline
    is preserved (``preserve_mean=True``) — only the slope (and higher terms)
    are taken out.  This targets the global brightness decay / photobleaching
    ramp that survives denoising (SUPPORT preserves slow dynamics by design),
    and which, left in, makes every pixel share one dominant slow component
    and saturates the local-correlation image.

    For ``order=1`` this is a straight linear detrend; raise to 2-3 to remove
    a curved bleach decay while leaving fast calcium transients untouched.

    Parameters
    ----------
    order : int
        Polynomial order to remove (1 = linear).  Must be >= 1.
    preserve_mean : bool
        Keep each pixel's temporal mean (recommended — protects the F0 that
        dF/F divides by).  If False, also remove the per-pixel mean.
    return_trend : bool
        If True, also return the removed ``(T, H, W)`` trend.
    use_gpu : bool
        Run the regression on the GPU via CuPy when available (the two
        matmuls and the tiny ``(order, order)`` solve map cleanly to the
        device).  Falls back to NumPy if CuPy is missing.  Output is always
        returned as host NumPy (callers write it back to a host mmap).

    Returns the corrected ``(T, H, W)`` float32 stack (and trend if asked).
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    T, H, W = stack.shape
    xp = _detrend_xp(use_gpu)

    t = xp.linspace(-1.0, 1.0, T, dtype=xp.float64)
    cols = []
    for k in range(1, order + 1):
        ck = t ** k
        cols.append(ck - ck.mean())          # mean-centre → removal keeps F0
    B = xp.stack(cols, axis=1).astype(xp.float32)        # (T, order)

    flat = xp.asarray(stack.reshape(T, H * W), dtype=xp.float32)  # upload if GPU
    betas = xp.linalg.solve(B.T @ B, B.T @ flat)         # (order, H*W)
    trend = B @ betas                                    # (T, H*W), 0 mean / pixel
    out = flat - trend
    if not preserve_mean:
        out = out - out.mean(axis=0, keepdims=True)

    out_np = _detrend_to_numpy(out, xp).reshape(T, H, W)
    if return_trend:
        return out_np, _detrend_to_numpy(trend, xp).reshape(T, H, W)
    return out_np


# ============================================================================
# 4. Temporal notch filter (per-pixel, chunked)
# ============================================================================

def notch_temporal(stack: np.ndarray,
                   fs_hz: float,
                   freq_hz: Union[float, List[float]],
                   Q: float = 30.0,
                   chunk_pixels: int = 8192) -> np.ndarray:
    """Per-pixel IIR notch filter at one or more frequencies.

    Uses ``scipy.signal.iirnotch`` to design a 2nd-order notch with quality
    factor ``Q`` (notch bandwidth = freq_hz / Q at -3 dB), applied
    bidirectionally via ``filtfilt`` for zero phase shift.

    Use when the diagnostic flags ``periodic_temporal_global`` at a known
    frequency that you don't want to remove by regression (e.g. you want
    to keep the slow drift but kill the 10 Hz mains-alias line). Otherwise
    ``regress_common_mode`` is simpler and faster.

    Parameters
    ----------
    freq_hz : float or list of floats
        Notch frequency, or list of frequencies to notch in sequence.
    Q : float
        Quality factor. Higher = narrower notch. 30 is a reasonable default
        for line-noise (~0.3 Hz bandwidth at 10 Hz).
    chunk_pixels : int
        Process the stack in groups of this many pixels at a time to bound
        peak memory at ~ chunk_pixels * T * 4 bytes * 4 (filtfilt scratch).
    """
    if fs_hz <= 0:
        raise ValueError(f"fs_hz must be positive; got {fs_hz}")
    freqs = [float(freq_hz)] if np.isscalar(freq_hz) else [float(f) for f in freq_hz]
    nyq = fs_hz / 2.0
    for f in freqs:
        if not (0 < f < nyq):
            raise ValueError(f"notch freq {f} Hz outside (0, Nyquist={nyq})")

    T, H, W = stack.shape
    out = stack.astype(np.float32, copy=True).reshape(T, H * W)

    # Cascade notch filters
    sos_list = []
    for f in freqs:
        b, a = sps.iirnotch(w0=f / nyq, Q=Q)
        sos_list.append((b, a))

    npix = H * W
    for start in range(0, npix, chunk_pixels):
        end = min(start + chunk_pixels, npix)
        block = out[:, start:end]
        for b, a in sos_list:
            block = sps.filtfilt(b, a, block, axis=0)
        out[:, start:end] = block.astype(np.float32)

    log.info("notch_temporal: applied %d notches at %s Hz",
             len(freqs), [f"{f:.2f}" for f in freqs])
    return out.reshape(T, H, W)


# ============================================================================
# 5. Hot-pixel replacement
# ============================================================================

def detect_hot_pixels(stack: np.ndarray,
                      z_threshold: float = 6.0,
                      vm_ratio_threshold: float = 1.5) -> np.ndarray:
    """Return a (H, W) bool mask of hot pixels.

    A pixel is "hot" when:
      (a) its temporal mean is a high local-z outlier (above the 9×9
          neighbourhood median by ``z_threshold`` MADs), AND
      (b) its variance-to-mean ratio is below ``vm_ratio_threshold`` ×
          the local-median var/mean — i.e. it has only shot-noise-level
          variance (consistent with a dead-but-bright transistor) rather
          than the transient-driven high variance of a real cell.

    The variance/mean ratio (≈ detector gain under Poisson) makes (b)
    independent of the absolute brightness, unlike a global variance
    threshold which fails for bright FOVs.
    """
    M = stack.mean(axis=0)
    V = stack.var(axis=0)
    local_med = ndimage.median_filter(M, size=9)
    local_mad = ndimage.median_filter(np.abs(M - local_med), size=9) + 1e-6
    z = (M - local_med) / (1.4826 * local_mad)

    # Variance-over-mean: ~ detector gain for shot-noise-only pixels
    vm = V / np.maximum(M, 1.0)
    vm_local_med = ndimage.median_filter(vm, size=9)

    return (z > z_threshold) & (vm < vm_ratio_threshold * vm_local_med)


def replace_hot_pixels(stack: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       z_threshold: float = 6.0,
                       vm_ratio_threshold: float = 1.5) -> np.ndarray:
    """Replace flagged pixels with a 3×3 spatial median of their neighbours.

    The replacement is done independently per frame so that *transient*
    structure in neighbours is preserved (in contrast to replacing with a
    single neighbourhood value computed on the temporal mean).

    If ``mask`` is None, ``detect_hot_pixels`` is called with the given
    thresholds.
    """
    mask_was_provided = mask is not None
    if mask is None:
        mask = detect_hot_pixels(stack, z_threshold=z_threshold,
                                 vm_ratio_threshold=vm_ratio_threshold)
    if not mask.any():
        return stack.astype(np.float32, copy=True)

    out = stack.astype(np.float32, copy=True)
    coords = np.argwhere(mask)                  # (n_hot, 2)
    H, W = stack.shape[1:]
    # Only log when we detected the mask ourselves. The streaming wrapper
    # loops over chunks with a pre-supplied mask; logging per chunk would
    # spam the log.
    if not mask_was_provided:
        log.info("replace_hot_pixels: %d pixels", coords.shape[0])
    for y, x in coords:
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        neighbourhood = out[:, y0:y1, x0:x1].copy()
        cy, cx = y - y0, x - x0
        neighbourhood[:, cy, cx] = np.nan
        out[:, y, x] = np.nanmedian(
            neighbourhood.reshape(stack.shape[0], -1), axis=1)
    return out


# ============================================================================
# Recipe builder + chain runner
# ============================================================================

# Ordered such that earlier corrections don't bias the estimators of later
# ones. Hot pixels first (outliers bias everything). Bidirectional next
# (row-based ops downstream rely on aligned rows). Row pedestal before
# temporal ops (so a fixed pattern doesn't pollute the common-mode trace).
_CORRECTION_PRIORITY = [
    "replace_hot_pixels",
    "correct_bidirectional",
    "subtract_row_pedestal",
    "subtract_column_pedestal",
    "subtract_fixed_pattern",
    "regress_common_mode",
    "notch_temporal",
]


def recommend_corrections(report: Dict[str, Any],
                          min_level: str = "moderate",
                          fs_hz: Optional[float] = None,
                          stack: Optional[np.ndarray] = None
                          ) -> List[Tuple[Callable, dict]]:
    """Map a diagnostic report dict to an ordered correction list.

    Reads ``report["sources"]`` and emits ``[(fn, kwargs), ...]`` for every
    flagged source at or above ``min_level`` that has a registered fix.

    Sources without a registered correction (saturation_clipping,
    quantization_loss, photobleaching, illumination_drift_increase, frame
    drops, FPN beyond hot pixels) are skipped — they need acquisition-side
    fixes or post-CNMF detrending, not in-place denoising.

    Parameters
    ----------
    min_level : {"low", "moderate", "high"}
        Lowest severity to act on. Default "moderate".
    fs_hz : float, optional
        Reserved for future routing (e.g. selecting notch over regression
        when the user has nominated specific known-bad frequencies).
        Currently unused: ``periodic_temporal_global`` always maps to
        ``regress_common_mode`` because the diagnostic's peak list will
        often include biological frequencies the user does not want
        notched out (calcium oscillations, breathing). Apply
        ``notch_temporal`` manually when you have explicit confidence in
        a frequency.
    stack : ndarray, optional
        If provided, hot-pixel detection is run on the stack directly with
        the correction module's variance/mean criterion, which is stricter
        than the diagnostic's ``test_hot_pixels`` and catches cases the
        diagnostic misses on bright FOVs. Recommended.
    """
    levels = ["negligible", "low", "moderate", "high"]
    if min_level not in levels:
        raise ValueError(f"min_level must be one of {levels[1:]}; got {min_level!r}")
    cutoff = levels.index(min_level)

    srcs = report.get("sources", {})
    def _flagged(name):
        d = srcs.get(name)
        return d is not None and levels.index(d["level"]) >= cutoff

    ops: List[Tuple[Callable, dict]] = []

    # Hot pixel detection: prefer the correction module's stricter
    # variance/mean criterion if a stack is provided, falling back to
    # the diagnostic's flag otherwise.
    if stack is not None:
        mask = detect_hot_pixels(stack)
        if mask.any():
            log.info("recommend_corrections: detected %d hot pixels on stack",
                     int(mask.sum()))
            ops.append((replace_hot_pixels, {"mask": mask}))
    elif _flagged("hot_dead_pixels"):
        ops.append((replace_hot_pixels, {}))

    if _flagged("bidirectional_phase_offset"):
        shift = report.get("bidirectional", {}).get("bidir_shift_px")
        ops.append((correct_bidirectional,
                    {"shift_px": float(shift)} if shift is not None else {}))

    if _flagged("horizontal_banding_fixed"):
        ops.append((subtract_row_pedestal, {"mode": "temporal_median"}))
    elif _flagged("horizontal_banding_drifting"):
        ops.append((subtract_row_pedestal, {"mode": "per_frame_median"}))

    # Column pedestal — triggered by fast_axis_periodic. We use an
    # **asymmetric trigger threshold** here: any level at or above "low"
    # qualifies, regardless of the global ``min_level``. Rationale: the
    # ``fast_axis_periodic`` test reports peak prominence in dB on the
    # fast-axis spatial spectrum, and even a modest ~3 dB peak corresponds
    # to vertical stripes that severely degrade CNMF's correlation image
    # and produce floods of false-positive column-aligned components
    # (observed empirically on resonant-scanner data). The fix is cheap
    # and FOV-uniform structure is rarely real biology, so we trigger
    # whenever the test sees anything at all.
    fap = srcs.get("fast_axis_periodic")
    if fap is not None and levels.index(fap["level"]) >= levels.index("low"):
        ops.append((subtract_column_pedestal, {"mode": "temporal_median"}))

    # 2D fixed-pattern noise — oriented/lattice structure in the temporal
    # mean that survives axis-aligned corrections (the column-aligned
    # 32-px lattice on the strohA recording is the motivating example).
    # The detector is robust against false positives on cellular FOVs
    # because cells produce broad spectral content, not isolated peaks
    # — so the routine is a no-op when no peaks pass the prominence
    # test. Triggers on the diagnostic's ``fixed_pattern_noise`` flag at
    # the user's chosen ``min_level``.
    if _flagged("fixed_pattern_noise"):
        ops.append((subtract_fixed_pattern, {}))

    # Frame-varying lattice contamination — same family of scanner-
    # synchronous noise as fixed_pattern_noise but with phase that walks
    # between frames. The temporal mean averages it down (typical
    # coherence ratio 0.04-0.3), so subtract_fixed_pattern misses most
    # of it. The per-frame method pools magnitude spectra to find the
    # peak coordinates with high SNR, then notches them in each frame
    # independently. Composes correctly with fixed_pattern_noise: run
    # the stationary subtraction first to remove the small coherent
    # fraction, then this routine for the rest. Gated by the
    # ``frame_varying_pattern`` flag in the JSON; runs on GPU via
    # CuPy when available (gracefully falls back to NumPy otherwise).
    # Other thresholds (cell_scale_px, prominence_db, max_peaks,
    # batch_size) use the function defaults; tune by editing the
    # function signature or pass kwargs from the calling pipeline if
    # session-specific tuning is needed.
    if _flagged("frame_varying_pattern"):
        ops.append((subtract_per_frame_pattern, {"use_gpu": True}))

    if _flagged("periodic_temporal_global"):
        # Default: common-mode regression. It removes whatever is globally
        # coherent in frame mean, regardless of frequency, in one shot. The
        # notch_temporal alternative needs the user to nominate frequencies
        # explicitly — applying it to every peak the diagnostic reports
        # would risk notching biology (calcium oscillations, breathing) that
        # happens to be globally coherent. Prefer regression as the default.
        ops.append((regress_common_mode, {}))

    return ops


def apply_corrections(stack: np.ndarray,
                      ops: List[Tuple[Callable, dict]],
                      verbose: bool = True) -> np.ndarray:
    """Run a list of ``(callable, kwargs)`` corrections in order.

    Each callable must accept the stack as its first positional argument
    and return a corrected stack of the same shape.
    """
    out = stack.astype(np.float32, copy=True)
    for i, (fn, kwargs) in enumerate(ops):
        if verbose:
            log.info("[%d/%d] %s(%s)", i + 1, len(ops), fn.__name__,
                     ", ".join(f"{k}={v!r}" for k, v in kwargs.items()))
        out = fn(out, **kwargs)
    return out


# ============================================================================
# Streaming file-based wrapper
# ============================================================================
#
# The in-memory primitives above are convenient but require holding the full
# (T, H, W) float32 stack in RAM — 38 GB for the canonical 512×512×37,100
# dataset. ``correct_stack_file`` runs the same corrections in a streaming
# fashion via tifffile so peak memory is bounded by a small chunk size
# (default 500 frames ≈ 0.5 GB) regardless of stack size.
#
# Pass structure
# --------------
#  Pass 1 — collect statistics:
#      frame_means    : (T,)       — for common-mode trace
#      row_means      : (T, H)     — for temporal-median row pedestal
#      temporal_sum   : (H, W)     — accumulates → temporal mean
#      temporal_sumsq : (H, W)     — accumulates → temporal variance
#  Then derive: bidir shift_px, row pedestal offsets, hot-pixel mask,
#               centred common-mode trace c, c·c.
#
#  Pass 2 (only if regress_common_mode in recipe): per-pixel x·c
#      Pre-multiply x_p(t) · c(t) and accumulate over t to get the OLS
#      numerator. Divide by c·c → beta_p (per-pixel regression coefficient).
#
#  Pass 3: apply corrections chunk-by-chunk, write to BigTIFF.
#
# Memory for the stat arrays: a 512×37100 stack uses 76 MB for row_means
# plus a few MB for the (H, W) accumulators — bounded and trivial against
# the chunk size.

def _open_stack_for_read(src: "Path"):
    """Open a stack file via caiman.utils.stack_io.StackReader for sequential
    sequential page reads. Returns (reader, n_pages, rows, cols, dtype)."""
    from caiman.utils.stack_io import StackReader
    reader = StackReader(src)
    if reader.n_frames == 0:
        reader.close()
        raise ValueError(f"{src}: no frames found")
    return reader, reader.n_frames, reader.h, reader.w, reader.dtype


def _iter_chunks(reader, n_pages: int, chunk_frames: int):
    """Yield (start, end, block) tuples of contiguous frames as float32.

    ``reader`` must be an open ``StackReader`` (or any object exposing
    ``read_frame(idx) -> np.ndarray``). Frames are stacked along axis 0
    and squeezed if the backend returned extra leading dimensions.
    """
    for start in range(0, n_pages, chunk_frames):
        end = min(start + chunk_frames, n_pages)
        frames = []
        for i in range(start, end):
            fr = reader.read_frame(i)
            if fr.ndim > 2:
                fr = fr.reshape(fr.shape[-2], fr.shape[-1])
            frames.append(fr)
        block = np.stack(frames, axis=0).astype(np.float32, copy=False)
        yield start, end, block


def correct_stack_file(src_tif: "Union[str, os.PathLike]",
                       ops: Optional[List[Tuple[Callable, dict]]] = None,
                       report: Optional[Dict[str, Any]] = None,
                       out_tif: "Optional[Union[str, os.PathLike]]" = None,
                       chunk_frames: int = 500,
                       out_dtype: str = "same",
                       overwrite: bool = False,
                       logger: Optional[logging.Logger] = None,
                       ) -> str:
    """Apply a noise-correction recipe to a stack file, streaming.

    Reads ``src_tif`` in chunks via ``caiman.utils.stack_io.StackReader``
    (supports ``.tif``/``.tiff``/``.msr`` — same format coverage as the
    diagnostic), computes per-correction statistics in a streaming first
    pass, then writes the corrected stack to ``<stem>_Ncorrected.tif``
    (or ``out_tif`` if given). Output is always TIFF (BigTIFF if needed)
    regardless of input format. Peak memory is bounded by the chunk size
    — ~0.5 GB at the default 500 frames.

    Recipe selection: pass ``ops`` for explicit control, or ``report``
    (a diagnostic JSON dict) to derive the recipe via
    ``recommend_corrections``. Passing both is an error.

    Parameters
    ----------
    src_tif : path
        Input stack — TIFF, BigTIFF, or MSR. (Argument name retained for
        backward compatibility; it does not have to be a TIFF.)
    ops : list of (callable, kwargs) tuples, optional
        Explicit correction recipe. Each callable must be one of the
        module's primitives (the streaming dispatcher knows them by name).
    report : dict, optional
        Diagnostic report from ``noise_diagnostics.run_diagnostics``. The
        recipe is built via ``recommend_corrections(report, stack=...)``
        using the temporal mean for hot-pixel detection.
    out_tif : path, optional
        Output path. Default: ``<src_stem>_Ncorrected.tif`` alongside src.
    chunk_frames : int
        Frames per chunk for the streaming passes. Default 500.
    out_dtype : {"same", "float32", "uint16"}
        Output dtype. "same" preserves the input dtype (with clipping
        warning if corrections push values out of range). "float32"
        preserves all precision at 2× file size for uint16 sources.
    overwrite : bool
        If False and the output exists, skip and return its path.

    Returns
    -------
    str
        Absolute path of the written output.

    Notes
    -----
    For efficiency, all statistics needed by the recipe (bidirectional shift,
    row-pedestal offsets, common-mode trace, hot-pixel mask, per-pixel
    regression coefficients) are computed from the raw input stack in 1–2
    streaming passes, then applied chunk-by-chunk in the final pass. This
    differs slightly from the in-memory ``apply_corrections`` chain, where
    each correction re-derives its statistics from the partially-corrected
    stack handed to it by the previous step. In practice the two outputs
    agree to within ~1 % relative on synthetic stacks (median |diff| ≈ 0.04
    DN on a uint16 source), well below shot-noise. The streamed version is
    strictly more deterministic — its output depends only on the raw input
    and the recipe, not on the application order of intermediate
    corrections.

    ``notch_temporal`` is not supported in the streamed path (``filtfilt``
    has acausal lookback over the full temporal axis); pre-apply it
    in-memory if needed, or use ``regress_common_mode`` instead (which
    handles all globally-coherent oscillations regardless of frequency).
    """
    import os
    import tifffile
    from pathlib import Path

    if (ops is None) == (report is None):
        raise ValueError("pass exactly one of `ops` or `report`")
    if out_dtype not in ("same", "float32", "uint16"):
        raise ValueError(f"out_dtype must be same/float32/uint16; got {out_dtype!r}")

    log_ = logger or log
    src = Path(src_tif).resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    out = Path(out_tif).resolve() if out_tif else src.with_name(
        src.stem + "_Ncorrected.tif")
    if out.exists() and not overwrite:
        log_.info("correct_stack_file: output exists, skipping — %s", out.name)
        return str(out)

    # ── Pass 1: per-frame stats + per-pixel temporal stats ─────────────────
    reader, n_pages, rows, cols, src_dtype = _open_stack_for_read(src)
    log_.info("correct_stack_file: %d frames %d×%d dtype=%s",
              n_pages, rows, cols, src_dtype)

    # Build recipe upfront if we got a report; we need to know which stats
    # to gather. (The mask is filled in below once we have the temporal
    # mean/variance; for now we know whether hot-pixel replacement is
    # wanted at all.)
    if ops is None:
        ops = recommend_corrections(report, min_level="moderate")
        # NOTE: we can't pass stack= here because we don't have it. Hot-
        # pixel detection happens below using the temporal stats we
        # collect in this same pass — equivalent result, no extra I/O.
        ops_names = [fn.__name__ for fn, _ in ops]
        # Insert replace_hot_pixels at front if not already there; we'll
        # fill its mask below. We do this unconditionally because the
        # diagnostic's hot-pixel test is less reliable than ours.
        if "replace_hot_pixels" not in ops_names:
            ops.insert(0, (replace_hot_pixels, {"mask": None}))

    ops_names = [fn.__name__ for fn, _ in ops]
    log_.info("correct_stack_file: recipe = %s", ops_names)
    needs_row_means = "subtract_row_pedestal" in ops_names and any(
        kw.get("mode", "temporal_median") == "temporal_median"
        for fn, kw in ops if fn.__name__ == "subtract_row_pedestal")
    needs_col_means = "subtract_column_pedestal" in ops_names and any(
        kw.get("mode", "temporal_median") == "temporal_median"
        for fn, kw in ops if fn.__name__ == "subtract_column_pedestal")
    needs_common_mode = "regress_common_mode" in ops_names
    needs_temporal_stats = ("correct_bidirectional" in ops_names or
                             "replace_hot_pixels" in ops_names or
                             "subtract_fixed_pattern" in ops_names)

    frame_means = np.empty(n_pages, dtype=np.float32) if needs_common_mode else None
    row_means = np.empty((n_pages, rows), dtype=np.float32) if needs_row_means else None
    col_means = np.empty((n_pages, cols), dtype=np.float32) if needs_col_means else None
    temporal_sum = np.zeros((rows, cols), dtype=np.float64) if needs_temporal_stats else None
    temporal_sumsq = np.zeros((rows, cols), dtype=np.float64) if needs_temporal_stats else None

    log_.info("correct_stack_file: pass 1/3 — collecting statistics")
    try:
        for start, end, block in _iter_chunks(reader, n_pages, chunk_frames):
            if frame_means is not None:
                frame_means[start:end] = block.mean(axis=(1, 2))
            if row_means is not None:
                row_means[start:end] = block.mean(axis=2)
            if col_means is not None:
                col_means[start:end] = block.mean(axis=1)
            if temporal_sum is not None:
                temporal_sum += block.sum(axis=0)
                temporal_sumsq += (block.astype(np.float64) ** 2).sum(axis=0)
    finally:
        reader.close()

    # Derive per-correction state from the gathered stats.
    state: Dict[str, Any] = {}
    if temporal_sum is not None:
        temporal_mean = (temporal_sum / n_pages).astype(np.float32)
        temporal_var = (temporal_sumsq / n_pages - temporal_mean.astype(np.float64) ** 2
                        ).astype(np.float32)
        # bidir shift uses the temporal mean directly via the standalone
        # estimator path — we stuff a (1, H, W) into estimate_*
        if "correct_bidirectional" in ops_names:
            est_shift = estimate_bidirectional_shift(temporal_mean[None, :, :])
            state["correct_bidirectional"] = {"shift_px": est_shift}
            log_.info("  bidirectional shift = %+.3f px", est_shift)
        if "replace_hot_pixels" in ops_names:
            # Replicate detect_hot_pixels logic using cached stats (avoids
            # a second pass over the file).
            local_med = ndimage.median_filter(temporal_mean, size=9)
            local_mad = ndimage.median_filter(np.abs(temporal_mean - local_med),
                                                size=9) + 1e-6
            z = (temporal_mean - local_med) / (1.4826 * local_mad)
            vm = temporal_var / np.maximum(temporal_mean, 1.0)
            vm_local_med = ndimage.median_filter(vm, size=9)
            hot_mask = (z > 6.0) & (vm < 1.5 * vm_local_med)
            state["replace_hot_pixels"] = {"mask": hot_mask}
            log_.info("  hot pixels detected = %d", int(hot_mask.sum()))
            if not hot_mask.any():
                # Drop the no-op from the recipe to save chunk-time
                ops = [(fn, kw) for fn, kw in ops if fn.__name__ != "replace_hot_pixels"]
                ops_names = [fn.__name__ for fn, _ in ops]

    if row_means is not None:
        # temporal_median row pedestal — single offset per row
        pedestal = np.median(row_means, axis=0).astype(np.float32)
        row_offsets = (pedestal - np.median(pedestal)).astype(np.float32)
        state["subtract_row_pedestal"] = {"offsets": row_offsets}
        log_.info("  row pedestal: |offsets| max = %.2f, median %.2f",
                  float(np.max(np.abs(row_offsets))), float(np.median(np.abs(row_offsets))))

    if col_means is not None:
        # temporal_median column pedestal — single offset per column
        pedestal = np.median(col_means, axis=0).astype(np.float32)
        col_offsets = (pedestal - np.median(pedestal)).astype(np.float32)
        state["subtract_column_pedestal"] = {"offsets": col_offsets}
        log_.info("  column pedestal: |offsets| max = %.2f, median %.2f",
                  float(np.max(np.abs(col_offsets))), float(np.median(np.abs(col_offsets))))

    if "subtract_fixed_pattern" in ops_names and temporal_sum is not None:
        # 2D FFT notch on the temporal mean. Streaming detects FPN from
        # the RAW temporal mean (pass-1 stats), not from the temporal
        # mean of the bidir/pedestal-corrected stack. For typical bidir
        # shifts (sub-pixel) and pedestal magnitudes (a few DN), the
        # FPN peak locations in M are unchanged by those earlier
        # corrections — the lattice frequencies are an intrinsic
        # property of the acquisition. If the recipe didn't include
        # any earlier in-place modification of M, the result is exactly
        # equivalent to the in-memory chain.
        fpn_kwargs = next((kw for fn, kw in ops
                            if fn.__name__ == "subtract_fixed_pattern"), {})
        cell_scale = float(fpn_kwargs.get("cell_scale_px", 12.0))
        prom = float(fpn_kwargs.get("prominence_db", 15.0))
        mx_peaks = int(fpn_kwargs.get("max_peaks", 32))
        notch_mask, F_shifted = detect_fpn_peaks(
            temporal_mean, cell_scale_px=cell_scale,
            prominence_db=prom, max_peaks=mx_peaks)
        if notch_mask.any():
            F_clean = F_shifted.copy()
            F_clean[notch_mask] = 0
            M_clean = (np.real(np.fft.ifft2(np.fft.ifftshift(F_clean)))
                       + temporal_mean.mean()).astype(np.float32)
            fpn_map = (temporal_mean - M_clean).astype(np.float32)
            state["subtract_fixed_pattern"] = {"fpn": fpn_map}
            log_.info("  fixed pattern: notched %d FFT bins, "
                      "fpn std=%.3f, max|fpn|=%.2f DN",
                      int(notch_mask.sum()), float(fpn_map.std()),
                      float(np.abs(fpn_map).max()))
        else:
            log_.info("  fixed pattern: no peaks above %.1f dB prominence; "
                      "dropping op from recipe", prom)
            ops = [(fn, kw) for fn, kw in ops
                    if fn.__name__ != "subtract_fixed_pattern"]
            ops_names = [fn.__name__ for fn, _ in ops]

    if frame_means is not None:
        c = (frame_means - frame_means.mean()).astype(np.float32)
        c_dot_c = float(c @ c)
        state["regress_common_mode"] = {"c": c, "c_dot_c": c_dot_c}
        log_.info("  common-mode trace: std=%.3f", float(c.std()))

    # ── Pass 2 (conditional): per-pixel x·c for common-mode regression ─────
    if needs_common_mode and state["regress_common_mode"]["c_dot_c"] > 0:
        log_.info("correct_stack_file: pass 2/3 — per-pixel x·c accumulator")
        c = state["regress_common_mode"]["c"]
        per_pixel_xTc = np.zeros((rows, cols), dtype=np.float64)
        reader, *_ = _open_stack_for_read(src)
        try:
            for start, end, block in _iter_chunks(reader, n_pages, chunk_frames):
                c_chunk = c[start:end]   # (n,)
                # block (n, H, W) · c_chunk (n,) → (H, W)
                per_pixel_xTc += np.tensordot(c_chunk, block, axes=([0], [0]))
        finally:
            reader.close()
        beta = (per_pixel_xTc / state["regress_common_mode"]["c_dot_c"]
                ).astype(np.float32)
        state["regress_common_mode"]["beta"] = beta
        log_.info("  beta: max=%.3f std=%.3f", float(beta.max()), float(beta.std()))

    # ── Pass 3: apply corrections chunk-by-chunk, stream to output ─────────
    log_.info("correct_stack_file: pass %d/%d — writing → %s",
              3 if needs_common_mode else 2, 3 if needs_common_mode else 2, out.name)
    target_dtype = np.dtype(src_dtype if out_dtype == "same"
                              else np.float32 if out_dtype == "float32"
                              else np.uint16)
    bytes_per_frame = rows * cols * target_dtype.itemsize
    bigtiff = (n_pages * bytes_per_frame) > 2 ** 31
    out_tmp = out.with_suffix(out.suffix + ".tmp")
    if out_tmp.exists():
        out_tmp.unlink()

    clipped_total = 0
    reader, *_ = _open_stack_for_read(src)
    try:
        with tifffile.TiffWriter(str(out_tmp), bigtiff=bigtiff) as writer:
            for start, end, block in _iter_chunks(reader, n_pages, chunk_frames):
                for fn, kw in ops:
                    name = fn.__name__
                    if name == "replace_hot_pixels":
                        block = replace_hot_pixels(block, mask=state[name]["mask"])
                    elif name == "correct_bidirectional":
                        block = correct_bidirectional(block,
                                                       shift_px=state[name]["shift_px"])
                    elif name == "subtract_row_pedestal":
                        block = block - state[name]["offsets"][None, :, None]
                    elif name == "subtract_column_pedestal":
                        block = block - state[name]["offsets"][None, None, :]
                    elif name == "subtract_fixed_pattern":
                        block = block - state[name]["fpn"][None, :, :]
                    elif name == "regress_common_mode":
                        beta = state[name]["beta"]
                        c_chunk = state[name]["c"][start:end]
                        # block -= beta(H,W) * c(t)[:, None, None]
                        block = block - beta[None, :, :] * c_chunk[:, None, None]
                    elif name == "notch_temporal":
                        # Notch needs full-temporal context — cannot stream
                        # naively (filtfilt has acausal lookback). Reject
                        # at recipe-build time.
                        raise NotImplementedError(
                            "notch_temporal cannot be streamed; run apply_corrections "
                            "in-memory or apply notch_temporal as a separate pre-pass.")
                    else:
                        raise NotImplementedError(f"unknown streaming op: {name}")

                # Cast to target dtype with clip-tracking
                if target_dtype != np.float32:
                    info = np.iinfo(target_dtype) if np.issubdtype(target_dtype, np.integer) else None
                    if info is not None:
                        clipped_total += int(np.sum((block < info.min) | (block > info.max)))
                        block = np.clip(block, info.min, info.max)
                    block_out = block.astype(target_dtype)
                else:
                    block_out = block.astype(np.float32)

                for fi in range(block_out.shape[0]):
                    writer.write(block_out[fi], contiguous=True)
    finally:
        reader.close()

    os.replace(str(out_tmp), str(out))
    if clipped_total > 0:
        log_.warning("correct_stack_file: %d pixels clipped to %s range "
                      "(out_dtype=%s); consider out_dtype='float32'",
                      clipped_total, target_dtype, out_dtype)
    log_.info("correct_stack_file: done — %s", out)
    return str(out)


__all__ = [
    "estimate_bidirectional_shift",
    "correct_bidirectional",
    "subtract_row_pedestal",
    "subtract_column_pedestal",
    "detect_fpn_peaks",
    "subtract_fixed_pattern",
    "subtract_per_frame_pattern",
    "regress_common_mode",
    "notch_temporal",
    "detect_hot_pixels",
    "replace_hot_pixels",
    "recommend_corrections",
    "apply_corrections",
    "correct_stack_file",
]
