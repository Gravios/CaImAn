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
# 3. Common-mode regression
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
                             "replace_hot_pixels" in ops_names)

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
    "regress_common_mode",
    "notch_temporal",
    "detect_hot_pixels",
    "replace_hot_pixels",
    "recommend_corrections",
    "apply_corrections",
    "correct_stack_file",
]
