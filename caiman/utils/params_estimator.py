"""
caiman/utils/params_estimator.py
=================================
Data-driven estimation of CaImAn pipeline parameters from a movie sample.

Given a small temporal subsample of the movie, ``estimate_params`` computes the
Cn/PNR images and from them estimates:

- **gSig** — Gaussian half-width of the PSF (soma radius proxy).  Measured by
  brightness-weighted LoG blob detection on the PNR image.  Only blobs whose
  peak PNR exceeds the image 85th percentile are retained — this excludes thin
  dendrites and processes (moderate PNR) and keeps soma bodies (high PNR).
- **gSiz** — Spatial support, set to ``4 × gSig + 1``.
- **min_corr** — Cn threshold derived by finding the *valley* between the
  background and signal modes in the Cn histogram (Otsu-like valley detection),
  rather than the inflection of the falling background edge.
- **min_pnr** — PNR threshold derived from the signal pixel population
  (pixels where Cn > min_corr), not the full distribution.  Anchored to the
  10th percentile of PNR at those pixels so weak neurons are not excluded.
- **rf** — Patch half-size computed as ``ceil(gSiz × 1.5)`` rounded to a clean
  multiple, with the ring constraint ``ring_size_factor × gSiz < rf`` verified.

All estimates are *starting points* — inspect the QC figures and tune from
there.

Usage
-----
    from caiman.utils.params_estimator import estimate_params

    suggestions = estimate_params(
        fname_mc   = fname_mc,
        n_frames   = 500,
        out_path   = outdir / f"{session}_qc_00_param_estimate.png",
        logger     = logger,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("caiman")


# ── Public entry point ────────────────────────────────────────────────────────

def estimate_params(
    fname_mc: Union[str, Path],
    *,
    gSig_hint:        Optional[int]            = None,
    n_frames:         int                       = 500,
    fr:               float                     = 30.0,
    ring_size_factor: float                     = 0.9,
    out_path:         Optional[Union[str, Path]] = None,
    logger:           logging.Logger            = logger,
) -> dict:
    """Estimate CNMF parameters from a subsampled movie.

    Computes Cn and PNR images from *n_frames* evenly-spaced frames of the
    motion-corrected movie, then derives ``gSig``, ``gSiz``, ``rf``,
    ``stride``, ``min_corr``, and ``min_pnr`` from those images.

    Parameters
    ----------
    fname_mc
        Path to the F-order motion-corrected mmap (``*rig*order_F*.mmap``).
    gSig_hint
        Prior estimate of the soma half-width in pixels.  If ``None``, gSig
        is estimated automatically.
    n_frames
        Number of frames to subsample.  500 is fast (~2 s); use 1000+ for
        noisy or sparse datasets.
    fr
        Acquisition frame rate in Hz (used only for log messages).
    ring_size_factor
        Must match the JSON ``cnmf`` section.  Used to verify the ring
        constraint on the suggested ``rf``.
    out_path
        If given, save a Cn/PNR inspection figure to this path.
    logger
        Logger for progress messages.

    Returns
    -------
    dict
        Suggested values ready to copy into the JSON ``cnmf`` section::

            {
                "gSig":     [N, N],
                "gSiz":     [M, M],
                "rf":       R,
                "stride":   R // 2,
                "min_corr": float,
                "min_pnr":  float,
            }
    """
    import caiman.mmapping as _mmap

    logger.info(f"Parameter estimation: loading {n_frames} frames from {fname_mc}")

    # ── Load subsampled movie ─────────────────────────────────────────────────
    Yr, dims, T = _mmap.load_memmap(str(fname_mc))
    step   = max(1, T // n_frames)
    frames = np.reshape(Yr.T, [T] + list(dims), order="F")[::step]
    del Yr
    logger.info(f"  Loaded {len(frames)} frames  dims={dims}  fr={fr} Hz")

    # ── Compute Cn and PNR ────────────────────────────────────────────────────
    # First pass: unfiltered (gSig=None) to get the true PNR image for gSig
    # estimation.  The filter kernel depends on gSig, so we need the soma size
    # before we can filter — chicken-and-egg resolved by estimating gSig from
    # the unfiltered PNR then recomputing with the correct filter.
    from caiman.summary_images import correlation_pnr as _corr_pnr

    if gSig_hint is not None:
        # Single pass with the provided hint
        gSig = int(round(gSig_hint))
        logger.info(f"  gSig: using hint = {gSig} px (single-pass)")
        cn, pnr = _corr_pnr(frames, gSig=[gSig, gSig], center_psf=True,
                             swap_dim=False)
    else:
        # Two-pass: estimate gSig from unfiltered PNR, then recompute
        logger.info("  Pass 1: unfiltered PNR for gSig estimation")
        cn_raw, pnr_raw = _corr_pnr(frames, gSig=None, center_psf=False,
                                     swap_dim=False)
        gSig = _estimate_gsig_from_pnr(pnr_raw, logger=logger)

        logger.info(f"  Pass 2: filtered Cn/PNR with gSig={gSig}")
        cn, pnr = _corr_pnr(frames, gSig=[gSig, gSig], center_psf=True,
                             swap_dim=False)
        del cn_raw, pnr_raw

    cn[np.isnan(cn)] = 0

    logger.info(
        f"  Cn:  mean={cn.mean():.3f}  p75={np.percentile(cn, 75):.3f}  "
        f"p99={np.percentile(cn, 99):.3f}  max={cn.max():.3f}"
    )
    logger.info(
        f"  PNR: mean={pnr.mean():.1f}  p75={np.percentile(pnr, 75):.1f}  "
        f"p99={np.percentile(pnr, 99):.1f}  max={np.percentile(pnr, 99.9):.1f}"
    )

    # ── Geometry ──────────────────────────────────────────────────────────────
    gSiz   = gSig * 4 + 1
    # rf must satisfy ring constraint AND be large enough to capture the full
    # soma + local background annulus.  ceil(gSiz * 1.5) rounded to next even
    # number, minimum ring_constraint + 2.
    rf_ring = int(np.ceil(gSiz * ring_size_factor)) + 2
    rf_soma = int(np.ceil(gSiz * 1.5 / 2)) * 2      # even number ≥ 1.5×gSiz
    rf      = max(rf_ring, rf_soma)
    stride  = rf // 2

    assert rf > gSiz * ring_size_factor, (
        f"Ring constraint violated: rf={rf} ≤ ring_size_factor×gSiz="
        f"{ring_size_factor*gSiz:.1f}"
    )
    logger.info(
        f"  Geometry: gSig={gSig}  gSiz={gSiz}  rf={rf}  stride={stride}  "
        f"(ring check: {ring_size_factor:.2f}×{gSiz}={ring_size_factor*gSiz:.1f} < {rf} ✓)"
    )

    # ── Thresholds ────────────────────────────────────────────────────────────
    # min_corr: valley between background and signal modes in the Cn histogram
    min_corr = _corr_threshold_from_valley(cn, logger=logger)

    # min_pnr: lower-tail PNR of signal pixels (Cn > min_corr).
    # This anchors min_pnr to the actual signal population rather than the
    # background PNR distribution which dominates the full histogram.
    min_pnr = _pnr_threshold_from_signal_pixels(cn, pnr, min_corr,
                                                 logger=logger)

    min_corr = round(float(min_corr), 2)
    min_pnr  = round(float(min_pnr),  1)

    suggestions = {
        "gSig":     [gSig, gSig],
        "gSiz":     [gSiz, gSiz],
        "rf":        rf,
        "stride":    stride,
        "min_corr":  min_corr,
        "min_pnr":   min_pnr,
    }
    logger.info(f"  Suggestions: {suggestions}")

    if out_path is not None:
        _save_inspection_figure(cn, pnr, suggestions, gSig, out_path, logger)

    return suggestions


# ── gSig estimation ───────────────────────────────────────────────────────────

def _estimate_gsig_from_pnr(
    pnr:        np.ndarray,
    *,
    min_sigma:  float = 3.5,      # skip thin processes (< 7 px diameter)
    max_sigma:  float = 25.0,
    num_sigma:  int   = 15,
    threshold:  float = 0.10,     # fraction of max LoG response
    pnr_pct:    float = 87.0,     # only keep blobs brighter than this PNR percentile
    fallback:   int   = 7,
    logger:     logging.Logger = logger,
) -> int:
    """Estimate gSig by brightness-weighted LoG blob detection on the PNR image.

    Two-stage filtering:

    1. LoG blob detection with ``min_sigma=3.5`` to skip responses from thin
       dendrites and axons (< 7 px diameter).
    2. Brightness filter: retain only blobs whose peak PNR exceeds the image
       ``pnr_pct`` percentile.  Bright somas have high PNR; dim dendrites do not.

    The estimated gSig is the 25th percentile of the retained blob radii
    (conservative — avoids bias from oversized merged-soma blobs).

    Falls back to ``fallback`` (default 7) if fewer than 5 blobs survive
    filtering.

    Parameters
    ----------
    pnr
        2-D PNR image.
    min_sigma
        Minimum LoG sigma.  Default 3.5 corresponds to ~7 px diameter —
        thin processes are effectively excluded.
    max_sigma
        Maximum LoG sigma.
    num_sigma
        Number of sigma steps.
    threshold
        LoG response threshold (fraction of maximum response).
    pnr_pct
        Percentile of the PNR image used as the brightness cutoff for blob
        retention.  Blobs with peak PNR below this are rejected.  Default 87
        keeps only the brightest soma bodies while excluding dendrites.
    fallback
        gSig returned when blob detection yields too few results.
    """
    try:
        from skimage.feature import blob_log as _blob_log
    except ImportError:
        logger.warning("  scikit-image not available — using fallback gSig")
        return fallback

    # Percentile-normalise PNR for the LoG detector
    p_bright = float(np.percentile(pnr, pnr_pct))
    pnr_norm = np.clip(pnr, 0, np.percentile(pnr, 99.9))
    p_max    = pnr_norm.max()
    if p_max < 1e-10:
        logger.warning("  PNR image is flat — using fallback gSig")
        return fallback
    pnr_norm = pnr_norm / p_max

    # LoG detection
    blobs = _blob_log(
        pnr_norm,
        min_sigma  = min_sigma,
        max_sigma  = max_sigma,
        num_sigma  = num_sigma,
        threshold  = threshold,
    )

    if len(blobs) == 0:
        logger.warning(f"  No blobs found — using fallback gSig={fallback}")
        return fallback

    # Brightness filter: keep blobs whose peak PNR exceeds pnr_pct percentile
    bright_blobs = []
    h, w = pnr.shape
    for r, c, sigma in blobs:
        r_px, c_px = int(round(r)), int(round(c))
        if 0 <= r_px < h and 0 <= c_px < w:
            peak_pnr = float(pnr[r_px, c_px])
            if peak_pnr >= p_bright:
                bright_blobs.append((r, c, sigma, peak_pnr))

    n_total  = len(blobs)
    n_bright = len(bright_blobs)
    logger.info(
        f"  Blob detection: {n_total} total blobs → "
        f"{n_bright} bright (PNR ≥ {p_bright:.1f})"
    )

    if n_bright < 5:
        logger.warning(
            f"  Only {n_bright} bright blobs — using fallback gSig={fallback}"
        )
        return fallback

    # Convert LoG sigma to radius: radius = sigma × sqrt(2)
    # Use 25th percentile (conservative) to avoid bias from large merged blobs
    sigmas = np.array([b[2] for b in bright_blobs])
    radii  = sigmas * np.sqrt(2)
    gSig   = max(2, int(round(float(np.percentile(radii, 25)))))

    logger.info(
        f"  Bright blob radii: p25={np.percentile(radii,25):.1f}  "
        f"median={np.median(radii):.1f}  p75={np.percentile(radii,75):.1f} px"
        f"  → gSig={gSig}"
    )
    return gSig


# ── Threshold estimation ──────────────────────────────────────────────────────

def _corr_threshold_from_valley(
    cn:     np.ndarray,
    *,
    n_bins: int = 300,
    logger: logging.Logger = logger,
) -> float:
    """Estimate min_corr by finding the valley between background and signal.

    The Cn histogram of a 2P movie is bimodal: a large background peak at
    low Cn (noise pixels, ~0.05–0.20) and a smaller signal tail at higher Cn
    (active neurons, dendrites).  The threshold should sit in the valley
    between these two modes, not on the falling edge of the background peak.

    Algorithm:

    1. Histogram Cn > 0.05 pixels, smoothed with a Gaussian (sigma=3 bins).
    2. Identify the background mode (highest peak in the lower 50% of range).
    3. Search rightward from the background mode for a local minimum — the
       valley bottom.  Clip between [p60, p92] of the input values as a
       sanity bound.
    4. If no clear valley is found (flat tail), fall back to the 80th percentile
       of Cn > 0.05 pixels.
    """
    from scipy.ndimage import gaussian_filter1d as _gf1d

    vals = cn[cn > 0.05].ravel()
    if len(vals) == 0:
        return 0.3

    vlo = float(np.percentile(vals, 1))
    vhi = float(np.percentile(vals, 99))
    hist, edges = np.histogram(np.clip(vals, vlo, vhi), bins=n_bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    smooth = _gf1d(hist.astype(float), sigma=3)

    # Background mode: highest peak in the lower 50% of the histogram range
    mid_idx   = n_bins // 2
    bg_mode   = int(np.argmax(smooth[:mid_idx]))

    # Search right of bg_mode for the first local minimum (valley)
    valley_idx = None
    for i in range(bg_mode + 1, len(smooth) - 1):
        if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]:
            valley_idx = i
            break

    if valley_idx is not None:
        threshold = float(centres[valley_idx])
        logger.info(
            f"  min_corr: valley at Cn={threshold:.3f} "
            f"(bg_mode={centres[bg_mode]:.3f})"
        )
    else:
        # No clear valley — fall back to 80th percentile of signal pixels
        threshold = float(np.percentile(vals, 80))
        logger.info(
            f"  min_corr: no clear valley — using p80={threshold:.3f}"
        )

    # Sanity clip: prevent extreme values but allow the valley to sit high.
    # [p65, p97] — wide enough that a true valley in the signal range isn't
    # clipped down into the background distribution.
    lo = float(np.percentile(vals, 65))
    hi = float(np.percentile(vals, 97))
    threshold = float(np.clip(threshold, lo, hi))

    logger.info(f"  min_corr → {threshold:.3f}  (clipped to [{lo:.3f}, {hi:.3f}])")
    return threshold


def _pnr_threshold_from_signal_pixels(
    cn:       np.ndarray,
    pnr:      np.ndarray,
    min_corr: float,
    *,
    signal_pnr_pct:  float = 10.0,   # use the 10th percentile PNR at signal pixels
    logger:   logging.Logger = logger,
) -> float:
    """Estimate min_pnr from the signal pixel population.

    Rather than finding the inflection of the full PNR histogram (which is
    dominated by background pixels), this function:

    1. Masks to pixels where Cn > min_corr (the signal population).
    2. Returns the ``signal_pnr_pct`` percentile of PNR at those pixels.

    Using the 10th percentile captures even the weakest active neurons in
    the signal population.  A hard lower bound of 2.5 is applied.

    If fewer than 100 signal pixels are found (sparse FOV or aggressive
    min_corr), falls back to the 85th percentile of all PNR > 1 pixels.
    """
    signal_mask = cn > min_corr
    n_signal    = int(signal_mask.sum())

    if n_signal >= 100:
        signal_pnr  = pnr[signal_mask]
        # Clip extreme values before computing percentile
        pnr_99      = float(np.percentile(signal_pnr, 99))
        signal_pnr  = signal_pnr[signal_pnr < pnr_99]
        threshold   = float(np.percentile(signal_pnr, signal_pnr_pct))
        logger.info(
            f"  min_pnr: {n_signal} signal pixels (Cn>{min_corr:.2f})  "
            f"p10={threshold:.1f}  median={np.median(signal_pnr):.1f}"
        )
    else:
        # Fallback: 85th percentile of all PNR > 1
        all_pnr   = pnr[pnr > 1].ravel()
        threshold = float(np.percentile(all_pnr, 85))
        logger.warning(
            f"  min_pnr: only {n_signal} signal pixels — "
            f"fallback to p85 of all PNR: {threshold:.1f}"
        )

    threshold = max(2.5, threshold)
    logger.info(f"  min_pnr → {threshold:.1f}")
    return threshold


# ── Inspection figure ─────────────────────────────────────────────────────────

def _save_inspection_figure(
    cn:          np.ndarray,
    pnr:         np.ndarray,
    suggestions: dict,
    gSig:        int,
    out_path:    Union[str, Path],
    logger:      logging.Logger,
) -> None:
    """Save a four-panel Cn/PNR inspection figure with estimated thresholds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        min_corr = suggestions["min_corr"]
        min_pnr  = suggestions["min_pnr"]

        fig = plt.figure(figsize=(16, 8), facecolor="k")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                                left=0.05, right=0.97, top=0.93, bottom=0.06)

        def _dax(spec):
            ax = fig.add_subplot(spec)
            ax.set_facecolor("k")
            ax.tick_params(colors="0.6", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("0.3")
            return ax

        # Images — top row
        ax_cn  = _dax(gs[0, 0]); ax_cn.axis("off")
        ax_pnr = _dax(gs[0, 1]); ax_pnr.axis("off")

        im_cn  = ax_cn.imshow(cn, cmap="inferno", aspect="equal",
                               interpolation="nearest")
        ax_cn.set_title(
            f"Correlation (Cn)  |  min_corr={min_corr:.2f}",
            color="0.85", fontsize=8)
        div = make_axes_locatable(ax_cn)
        cb  = plt.colorbar(im_cn, cax=div.append_axes("right", "3%", pad=0.05))
        cb.ax.tick_params(colors="0.6", labelsize=6)

        pnr_clip = np.clip(pnr, 0, np.percentile(pnr, 99.5))
        im_pnr   = ax_pnr.imshow(pnr_clip, cmap="inferno", aspect="equal",
                                  interpolation="nearest")
        ax_pnr.set_title(
            f"PNR  |  min_pnr={min_pnr:.1f}  |  gSig≈{gSig} px",
            color="0.85", fontsize=8)
        div2 = make_axes_locatable(ax_pnr)
        cb2  = plt.colorbar(im_pnr, cax=div2.append_axes("right", "3%", pad=0.05))
        cb2.ax.tick_params(colors="0.6", labelsize=6)

        # Histograms — bottom row
        ax_ch = _dax(gs[1, 0])
        cn_vals = cn[cn > 0.05].ravel()
        ax_ch.hist(cn_vals, bins=200, color="#4fc3f7", edgecolor="none", alpha=0.8,
                   range=(float(np.percentile(cn_vals, 1)),
                          float(np.percentile(cn_vals, 99))))
        ax_ch.axvline(min_corr, color="white", lw=1.5, ls="--",
                      label=f"min_corr={min_corr:.2f}")
        ax_ch.set_xlabel("Cn value", color="0.6", fontsize=7)
        ax_ch.set_ylabel("pixels",   color="0.6", fontsize=7)
        ax_ch.set_title("Cn distribution  (threshold = valley between modes)",
                        color="0.85", fontsize=8)
        ax_ch.legend(fontsize=7, facecolor="0.15", labelcolor="0.85")

        ax_ph = _dax(gs[1, 1])
        # Show PNR for signal pixels (Cn > min_corr) overlaid on full distribution
        pnr_all    = pnr[(pnr > 1) & (pnr < np.percentile(pnr, 99.5))].ravel()
        pnr_signal = pnr[cn > min_corr].ravel()
        pnr_signal = pnr_signal[(pnr_signal > 1) &
                                 (pnr_signal < np.percentile(pnr_signal, 99.5))]
        ax_ph.hist(pnr_all, bins=150, color="#444", edgecolor="none", alpha=0.6,
                   label="all pixels")
        ax_ph.hist(pnr_signal, bins=100, color="#f48fb1", edgecolor="none",
                   alpha=0.85, label=f"Cn>{min_corr:.2f} pixels")
        ax_ph.axvline(min_pnr, color="white", lw=1.5, ls="--",
                      label=f"min_pnr={min_pnr:.1f}")
        ax_ph.set_xlabel("PNR value", color="0.6", fontsize=7)
        ax_ph.set_ylabel("pixels",    color="0.6", fontsize=7)
        ax_ph.set_title("PNR distribution  (pink = signal pixels)",
                        color="0.85", fontsize=8)
        ax_ph.legend(fontsize=6, facecolor="0.15", labelcolor="0.85")

        suggestions_str = (
            f"gSig={suggestions['gSig'][0]}  gSiz={suggestions['gSiz'][0]}  "
            f"rf={suggestions['rf']}  stride={suggestions['stride']}  "
            f"min_corr={min_corr}  min_pnr={min_pnr}"
        )
        fig.suptitle(
            f"Parameter Estimation  |  {suggestions_str}",
            color="0.85", fontsize=9)

        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="k")
        plt.close(fig)
        logger.info(f"  Inspection figure saved: {out_path}")

    except Exception as exc:
        logger.warning(f"  Parameter estimation figure failed: {exc}")


# ── JSON helper ───────────────────────────────────────────────────────────────

def apply_suggestions(
    json_path: Union[str, Path],
    suggestions: dict,
    *,
    section: str = "cnmf",
    dry_run: bool = False,
) -> None:
    """Write parameter suggestions into an existing pipeline JSON file.

    Keys in *suggestions* may be plain strings (written into *section*) or
    dotted ``"section.key"`` strings (written into the named section).
    All other values in the JSON are preserved.

    Parameters
    ----------
    json_path
        Path to ``<session>_pipeline.json``.
    suggestions
        Dict of ``{key: value}`` or ``{"section.key": value}`` pairs.
        Plain keys are written into *section* (default ``"cnmf"``).
        Dotted keys override the section: ``"motion_correction.max_shifts"``
        writes ``max_shifts`` into the ``motion_correction`` section.
    section
        Default section for plain (non-dotted) keys (default ``"cnmf"``).
    dry_run
        If ``True``, log what would be written without modifying the file.
    """
    import json as _json

    path = Path(json_path)
    raw  = _json.loads(path.read_text())

    changed = {}
    for k, v in suggestions.items():
        if "." in k:
            sec, leaf = k.split(".", 1)
        else:
            sec, leaf = section, k
        target = raw.setdefault(sec, {})
        if target.get(leaf) != v:
            changed[f"{sec}.{leaf}"] = (target.get(leaf), v)
        target[leaf] = v

    if not changed:
        logger.info("apply_suggestions: no changes needed")
        return

    if dry_run:
        logger.info("apply_suggestions (dry run):")
        for k, (old_v, new_v) in changed.items():
            logger.info(f"  {k}: {old_v!r} → {new_v!r}")
        return

    path.write_text(_json.dumps(raw, indent=4) + "\n")
    logger.info(f"apply_suggestions: updated {path.name}")
    for k, (old_v, new_v) in changed.items():
        logger.info(f"  {k}: {old_v!r} → {new_v!r}")
