"""
caiman/utils/params_estimator.py
=================================
Data-driven estimation of CaImAn pipeline parameters from a movie sample.

Given a small temporal subsample of the movie, ``estimate_params`` computes the
Cn/PNR images and from them estimates:

- **gSig** — Gaussian half-width of the PSF (soma radius proxy).  Measured by
  brightness-weighted LoG blob detection on the PNR image.  Only blobs whose
  peak PNR ≥ the image 87th percentile are retained (soma bodies have high PNR;
  dendrites have moderate PNR).  Uses the 25th percentile of retained blob radii
  — conservative against merged-soma outliers.  Hard-capped at 10 px (20 px
  diameter) since cortical neurons in vivo rarely exceed this.
- **gSiz** — Spatial support, set to ``4 × gSig + 1``.
- **min_corr** — Cn threshold derived from the upper tail of the Cn distribution.
  For sparse 2P data the Cn histogram is dominated by a single background peak
  with a long tail; a valley between modes may not exist.  We use the p92 of all
  Cn pixels (including zero, not just Cn>0.05) which reliably lands in the
  active-neuron tail without requiring bimodality.
- **min_pnr** — PNR threshold derived from signal pixels (Cn ≥ min_corr), using
  the 10th percentile of their PNR values.  Hard floor of 3.0.
- **rf** — Patch half-size computed as ``ceil(gSiz × 1.5)`` rounded to an even
  number, verified against the ring constraint.

All estimates are *starting points* — inspect the QC figures and tune from there.

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

    Parameters
    ----------
    fname_mc
        Path to the F-order motion-corrected mmap (``*rig*order_F*.mmap``).
    gSig_hint
        Prior estimate of the soma half-width in pixels.  Skips blob detection.
    n_frames
        Number of frames to subsample.  500 is fast (~2 s).
    fr
        Acquisition frame rate in Hz (for log messages only).
    ring_size_factor
        Must match the JSON ``cnmf`` section.  Verified against suggested ``rf``.
    out_path
        If given, save a Cn/PNR inspection figure to this path.
    logger
        Logger for progress messages.

    Returns
    -------
    dict
        ``{"gSig", "gSiz", "rf", "stride", "min_corr", "min_pnr"}``
    """
    import caiman.mmapping as _mmap

    logger.info(f"Parameter estimation: loading {n_frames} frames from {fname_mc}")

    Yr, dims, T = _mmap.load_memmap(str(fname_mc))
    step   = max(1, T // n_frames)
    frames = np.reshape(Yr.T, [T] + list(dims), order="F")[::step]
    del Yr
    logger.info(f"  Loaded {len(frames)} frames  dims={dims}  fr={fr} Hz")

    # ── Compute Cn and PNR (two-pass if gSig unknown) ─────────────────────────
    from caiman.summary_images import correlation_pnr as _corr_pnr

    if gSig_hint is not None:
        gSig = int(round(gSig_hint))
        logger.info(f"  gSig: using hint = {gSig} px")
        cn, pnr = _corr_pnr(frames, gSig=[gSig, gSig], center_psf=True,
                             swap_dim=False)
    else:
        # Pass 1: unfiltered PNR to estimate gSig from blob sizes
        logger.info("  Pass 1: unfiltered PNR for gSig estimation")
        _cn_raw, pnr_raw = _corr_pnr(frames, gSig=None, center_psf=False,
                                      swap_dim=False)
        gSig = _estimate_gsig_from_pnr(pnr_raw, logger=logger)
        del _cn_raw, pnr_raw

        logger.info(f"  Pass 2: filtered Cn/PNR with gSig={gSig}")
        cn, pnr = _corr_pnr(frames, gSig=[gSig, gSig], center_psf=True,
                             swap_dim=False)

    cn[np.isnan(cn)] = 0

    logger.info(
        f"  Cn:  mean={cn.mean():.3f}  p90={np.percentile(cn, 90):.3f}  "
        f"p99={np.percentile(cn, 99):.3f}  max={cn.max():.3f}"
    )
    logger.info(
        f"  PNR: mean={pnr.mean():.1f}  p90={np.percentile(pnr, 90):.1f}  "
        f"p99={np.percentile(pnr, 99):.1f}"
    )

    # ── Geometry ──────────────────────────────────────────────────────────────
    gSiz   = gSig * 4 + 1
    rf_ring = int(np.ceil(gSiz * ring_size_factor)) + 2
    rf_soma = int(np.ceil(gSiz * 1.5 / 2)) * 2
    rf      = max(rf_ring, rf_soma)
    stride  = rf // 2

    assert rf > gSiz * ring_size_factor, \
        f"Ring constraint violated: rf={rf} ≤ {ring_size_factor}×{gSiz}"
    logger.info(
        f"  Geometry: gSig={gSig}  gSiz={gSiz}  rf={rf}  stride={stride}  "
        f"(ring check: {ring_size_factor:.2f}×{gSiz}={ring_size_factor*gSiz:.1f} < {rf} ✓)"
    )

    # ── Thresholds ────────────────────────────────────────────────────────────
    min_corr = _corr_threshold_upper_tail(cn, logger=logger)
    min_pnr  = _pnr_threshold_from_signal_pixels(cn, pnr, min_corr, logger=logger)

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
    min_sigma:  float = 3.5,
    max_sigma:  float = 12.0,   # cap: 12 px sigma → ~17px diameter max
    num_sigma:  int   = 12,
    threshold:  float = 0.10,
    pnr_pct:    float = 87.0,
    gsig_min:   int   = 3,
    gsig_max:   int   = 10,     # hard cap: 10 px → 20 px diameter
    fallback:   int   = 6,
    logger:     logging.Logger = logger,
) -> int:
    """Estimate gSig by brightness-weighted LoG blob detection on the PNR image.

    Only blobs whose peak PNR ≥ ``pnr_pct`` percentile of the image are
    retained (soma bodies have high PNR; thin dendrites do not).
    The gSig estimate is the 25th percentile of retained blob radii, capped
    at ``gsig_max`` (default 10 px = 20 px diameter) to prevent merged soma
    clusters from inflating the estimate.

    Falls back to ``fallback`` (default 6) if fewer than 5 blobs survive.
    """
    try:
        from skimage.feature import blob_log as _blob_log
    except ImportError:
        logger.warning("  scikit-image not available — using fallback gSig")
        return fallback

    p_bright = float(np.percentile(pnr, pnr_pct))
    pnr_norm = np.clip(pnr, 0, np.percentile(pnr, 99.9))
    p_max    = pnr_norm.max()
    if p_max < 1e-10:
        logger.warning("  PNR image is flat — using fallback gSig")
        return fallback
    pnr_norm = pnr_norm / p_max

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

    h, w = pnr.shape
    bright_blobs = []
    for r, c, sigma in blobs:
        r_px, c_px = int(round(r)), int(round(c))
        if 0 <= r_px < h and 0 <= c_px < w:
            if float(pnr[r_px, c_px]) >= p_bright:
                bright_blobs.append((r, c, sigma))

    logger.info(
        f"  Blob detection: {len(blobs)} total → "
        f"{len(bright_blobs)} bright (PNR ≥ {p_bright:.1f})"
    )

    if len(bright_blobs) < 5:
        logger.warning(
            f"  Only {len(bright_blobs)} bright blobs — using fallback gSig={fallback}")
        return fallback

    sigmas = np.array([b[2] for b in bright_blobs])
    radii  = sigmas * np.sqrt(2)
    gSig   = int(round(float(np.percentile(radii, 25))))
    gSig   = int(np.clip(gSig, gsig_min, gsig_max))

    logger.info(
        f"  Bright blob radii: p25={np.percentile(radii,25):.1f}  "
        f"median={np.median(radii):.1f}  p75={np.percentile(radii,75):.1f} px"
        f"  → gSig={gSig} (cap [{gsig_min},{gsig_max}])"
    )
    return gSig


# ── Threshold estimation ──────────────────────────────────────────────────────

def _corr_threshold_upper_tail(
    cn:     np.ndarray,
    *,
    tail_pct: float = 92.0,
    logger: logging.Logger = logger,
) -> float:
    """Estimate min_corr from the upper tail of the Cn distribution.

    For sparse 2P data the Cn histogram is dominated by background pixels
    (large peak at low Cn) with a long heavy tail of active-neuron pixels.
    The histogram is often *not* bimodal — soma pixels are too few to form a
    visible second mode — so valley detection is unreliable.

    Instead we use the ``tail_pct`` percentile of *all* Cn pixels (including
    the zero-valued border pixels).  The default p92 of ALL pixels places the
    threshold well into the active-neuron tail while typically leaving at least
    ~8% of the FOV as candidate seed pixels.

    The result is clipped to [p80, p98] of the full Cn image as a sanity bound.
    """
    all_cn  = cn.ravel()
    # p92 of ALL pixels (incl. zero / border)
    threshold = float(np.percentile(all_cn, tail_pct))

    lo = float(np.percentile(all_cn, 80))
    hi = float(np.percentile(all_cn, 98))
    threshold = float(np.clip(threshold, lo, hi))

    # Count candidate seed pixels so the user can sanity-check
    n_seeds   = int((cn >= threshold).sum())
    frac_seeds = n_seeds / cn.size
    logger.info(
        f"  min_corr: p{tail_pct:.0f} of all Cn = {threshold:.3f}  "
        f"({n_seeds} pixels = {frac_seeds*100:.1f}% of FOV)"
    )
    return threshold


def _pnr_threshold_from_signal_pixels(
    cn:        np.ndarray,
    pnr:       np.ndarray,
    min_corr:  float,
    *,
    signal_pnr_pct: float = 10.0,
    pnr_floor:      float = 3.0,
    logger:    logging.Logger = logger,
) -> float:
    """Estimate min_pnr from the signal pixel population.

    Masks to pixels where Cn ≥ min_corr, then returns the ``signal_pnr_pct``
    percentile of PNR at those pixels.  Falls back to the 85th percentile
    of all PNR > 1 pixels if fewer than 100 signal pixels are found.
    """
    signal_mask = cn >= min_corr
    n_signal    = int(signal_mask.sum())

    if n_signal >= 100:
        signal_pnr  = pnr[signal_mask]
        pnr_99      = float(np.percentile(signal_pnr, 99))
        signal_pnr  = signal_pnr[signal_pnr < pnr_99]
        threshold   = float(np.percentile(signal_pnr, signal_pnr_pct))
        logger.info(
            f"  min_pnr: {n_signal} signal pixels (Cn≥{min_corr:.2f})  "
            f"p10={threshold:.1f}  median={np.median(signal_pnr):.1f}"
        )
    else:
        all_pnr   = pnr[pnr > 1].ravel()
        threshold = float(np.percentile(all_pnr, 85))
        logger.warning(
            f"  min_pnr: only {n_signal} signal pixels — fallback p85={threshold:.1f}")

    threshold = max(pnr_floor, threshold)
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
    """Save a four-panel Cn/PNR inspection figure."""
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

        # Images
        ax_cn  = _dax(gs[0, 0]); ax_cn.axis("off")
        ax_pnr = _dax(gs[0, 1]); ax_pnr.axis("off")

        im_cn = ax_cn.imshow(cn, cmap="inferno", aspect="equal",
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

        # Histograms
        ax_ch = _dax(gs[1, 0])
        cn_all  = cn.ravel()
        cn_pos  = cn_all[cn_all > 0]
        ax_ch.hist(cn_pos, bins=200, color="#4fc3f7", edgecolor="none", alpha=0.8,
                   range=(0, float(np.percentile(cn_pos, 99.5))))
        ax_ch.axvline(min_corr, color="white", lw=1.5, ls="--",
                      label=f"min_corr={min_corr:.2f}")
        ax_ch.set_xlabel("Cn value", color="0.6", fontsize=7)
        ax_ch.set_ylabel("pixels",   color="0.6", fontsize=7)
        ax_ch.set_title("Cn distribution  (threshold = p92 of all pixels)",
                        color="0.85", fontsize=8)
        ax_ch.legend(fontsize=7, facecolor="0.15", labelcolor="0.85")

        ax_ph = _dax(gs[1, 1])
        pnr_all    = pnr[(pnr > 1) & (pnr < np.percentile(pnr, 99.5))].ravel()
        pnr_signal = pnr[cn >= min_corr].ravel()
        pnr_signal = pnr_signal[(pnr_signal > 1) &
                                 (pnr_signal < np.percentile(pnr_signal, 99.5))]
        ax_ph.hist(pnr_all,    bins=150, color="#444", edgecolor="none",
                   alpha=0.6, label="all pixels")
        ax_ph.hist(pnr_signal, bins=100, color="#f48fb1", edgecolor="none",
                   alpha=0.85, label=f"Cn≥{min_corr:.2f}")
        ax_ph.axvline(min_pnr, color="white", lw=1.5, ls="--",
                      label=f"min_pnr={min_pnr:.1f}")
        ax_ph.set_xlabel("PNR value", color="0.6", fontsize=7)
        ax_ph.set_ylabel("pixels",    color="0.6", fontsize=7)
        ax_ph.set_title("PNR distribution  (pink = signal pixels Cn≥min_corr)",
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
    json_path:   Union[str, Path],
    suggestions: dict,
    *,
    section: str  = "cnmf",
    dry_run: bool = False,
) -> None:
    """Write parameter suggestions into an existing pipeline JSON file.

    Plain keys go into *section* (default ``"cnmf"``); dotted keys
    (``"motion_correction.max_shifts"``) override the section.
    Preserves all other JSON content.
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
