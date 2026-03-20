"""
caiman/utils/params_estimator.py
=================================
Data-driven estimation of CaImAn pipeline parameters from a movie sample,
with optional species and magnification priors.

Given a small temporal subsample of the motion-corrected movie, ``estimate_params``
computes the Cn/PNR images and from them estimates:

- **gSig** — Gaussian half-width of the PSF (soma radius proxy).  Measured by
  brightness-weighted LoG blob detection on the unfiltered PNR image.  Only
  blobs whose peak PNR ≥ the image 87th percentile are retained (soma bodies
  have high PNR; dendrites do not).  The estimate is the 25th percentile of
  retained blob radii, clipped to the species × magnification prior range.
- **gSiz** — Spatial support: ``4 × gSig + 1`` (always odd).
- **rf** — Patch half-size.  Chosen as ``ceil(gSiz × 1.7)``, rounded to an
  even number, with a hard lower bound of ``ceil(ring_size_factor × gSiz) + 4``
  to ensure a useful ring annulus.  Empirically calibrated: mouse cortex with
  18–20 px soma diameter (gSig=5, gSiz=21) → rf=36 validated in session data.
- **stride** — Half of rf (50% patch overlap).
- **ring_size_factor** — Fixed at 0.9 (validated for 2P sparse recordings).
- **merge_thr** — Returned at 0.85 (0.7 merges neighbouring cells).
- **min_corr / min_pnr** — Seed thresholds estimated from Cn/PNR distributions.
- **K** — Seeds per patch, scaled to expected neuron density.

Species × magnification priors
--------------------------------
``species`` ∈ {"mouse", "rat"}, ``magnification`` ∈ {"20x", "40x"}.
These constrain the gSig search range and set the blob-detection fallback:

  mouse 20x: gSig 3–9 px  (10-15 µm soma @ ~0.8 µm/px, FOV ~400 µm)
  mouse 40x: gSig 6–14 px (10-15 µm soma @ ~0.4 µm/px, FOV ~200 µm)
  rat   20x: gSig 4–11 px (12-18 µm soma @ ~0.8 µm/px)
  rat   40x: gSig 7–16 px (12-18 µm soma @ ~0.4 µm/px)

All estimates are starting points — inspect the QC figure and tune from there.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("caiman")

# ── Species × magnification priors ────────────────────────────────────────────
_PRIORS: dict[tuple, dict] = {
    ("mouse", "20x"): {"gsig_min": 3, "gsig_max":  9, "fallback": 6},
    ("mouse", "40x"): {"gsig_min": 6, "gsig_max": 14, "fallback": 9},
    ("rat",   "20x"): {"gsig_min": 4, "gsig_max": 11, "fallback": 7},
    ("rat",   "40x"): {"gsig_min": 7, "gsig_max": 16, "fallback": 10},
}

_RING_SIZE_FACTOR = 0.9   # validated for 2P sparse recordings
_RF_SCALE         = 1.7   # rf ≈ 1.7 × gSiz — calibrated from mouse/20x gSig=5 rf=36
_MERGE_THR        = 0.85  # validated for 2P soma (0.7 merges neighbouring cells)


# ── Public entry point ────────────────────────────────────────────────────────

def estimate_params(
    fname_mc:      Union[str, Path],
    *,
    species:       str                       = "mouse",
    magnification: str                       = "20x",
    gSig_hint:     Optional[int]             = None,
    n_frames:      int                       = 500,
    fr:            float                     = 30.0,
    out_path:      Optional[Union[str, Path]] = None,
    logger:        logging.Logger            = logger,
) -> dict:
    """Estimate CNMF parameters from a subsampled movie.

    Parameters
    ----------
    fname_mc : str or Path
        Path to the F-order motion-corrected mmap (``*rig*order_F*.mmap``).
    species : {"mouse", "rat"}
        Sets the gSig search bounds.
    magnification : {"20x", "40x"}
        Combined with species to bound gSig.
    gSig_hint : int, optional
        Prior soma half-width in pixels.  Skips blob detection.
    n_frames : int
        Frames to subsample for Cn/PNR.  500 is fast (~2 s).
    fr : float
        Frame rate in Hz — used to compute dff_frames_window.
    out_path : str or Path, optional
        Save Cn/PNR inspection figure here.
    logger : logging.Logger

    Returns
    -------
    dict
        Ready-to-use suggestions for the pipeline JSON cnmf section.
    """
    import caiman.mmapping as _mmap

    prior_key = (species.lower(), magnification.lower())
    if prior_key not in _PRIORS:
        raise ValueError(
            f"Unknown (species, magnification)={prior_key!r}. "
            f"Valid: {list(_PRIORS)}"
        )
    prior = _PRIORS[prior_key]
    logger.info(
        f"Parameter estimation: species={species}  magnification={magnification}  "
        f"gSig prior=[{prior['gsig_min']}, {prior['gsig_max']}] px"
    )

    logger.info(f"  Loading {n_frames} frames from {fname_mc}")
    Yr, dims, T = _mmap.load_memmap(str(fname_mc))
    step   = max(1, T // n_frames)
    frames = np.reshape(Yr.T, [T] + list(dims), order="F")[::step]
    del Yr
    logger.info(f"  Loaded {len(frames)} frames  dims={dims}  fr={fr} Hz")

    # ── Cn / PNR ─────────────────────────────────────────────────────────────
    from caiman.summary_images import correlation_pnr as _corr_pnr

    if gSig_hint is not None:
        gSig = int(np.clip(round(gSig_hint), prior["gsig_min"], prior["gsig_max"]))
        logger.info(f"  gSig: hint={gSig_hint} → clipped to prior → {gSig} px")
        cn, pnr = _corr_pnr(frames, gSig=[gSig, gSig], center_psf=True,
                             swap_dim=False)
    else:
        logger.info("  Pass 1: unfiltered PNR for gSig estimation")
        _cn_raw, pnr_raw = _corr_pnr(frames, gSig=None, center_psf=False,
                                      swap_dim=False)
        gSig = _estimate_gsig_from_pnr(pnr_raw, prior=prior, logger=logger)
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
    gSiz = gSig * 4 + 1   # must be odd; 4×gSig+1 is the CaImAn convention

    # Lower bound: ring annulus must fit with margin
    rf_ring_min = int(math.ceil(_RING_SIZE_FACTOR * gSiz)) + 4
    # Target: 1.7 × gSiz — empirically calibrated from validated session:
    # mouse, 18-20 px soma diameter, gSig=5, gSiz=21 → rf=36 (ceil(21×1.7)=36).
    rf_target   = int(math.ceil(gSiz * _RF_SCALE))

    rf = max(rf_ring_min, rf_target)
    if rf % 2 != 0:
        rf += 1   # CaImAn expects even half-sizes

    stride  = rf // 2
    ring_px = _RING_SIZE_FACTOR * gSiz
    margin  = rf - ring_px

    # Informational: patch count on a typical 512×512 FOV
    step_px           = 2 * rf - stride   # = rf  (50% overlap)
    n_per_dim         = max(1, math.ceil((512 - 2 * rf) / step_px) + 2)
    n_patches_typical = n_per_dim ** 2

    logger.info(
        f"  Geometry: gSig={gSig}  gSiz={gSiz}  rf={rf}  stride={stride}  "
        f"ring={ring_px:.1f}px  margin={margin:.1f}px  "
        f"~{n_patches_typical} patches on 512×512 FOV"
    )

    # ── Thresholds ────────────────────────────────────────────────────────────
    # Estimated in the filtered Cn space from correlation_pnr (roughly [0,1]).
    # The precomputed cn_full used by greedyROI_corr is in a different scale
    # (0.88–1.35), so these thresholds are effectively permissive during fitting
    # — seeding is driven by v_search = Cn × PNR rank ordering.
    min_corr = round(float(_corr_threshold_upper_tail(cn, logger=logger)), 2)
    min_pnr  = round(float(_pnr_threshold_from_signal_pixels(
        cn, pnr, min_corr, logger=logger)), 1)

    # ── K: seeds per patch ────────────────────────────────────────────────────
    # 3× expected neurons per patch, clamped to [20, 60].
    n_neurons_typical = 70
    neurons_per_patch = max(1.0, n_neurons_typical / max(1, n_patches_typical))
    K = int(np.clip(round(neurons_per_patch * 3), 20, 60))
    logger.info(
        f"  K={K}  (estimated {neurons_per_patch:.1f} neurons/patch × 3 headroom)"
    )

    # ── dF/F window ───────────────────────────────────────────────────────────
    # Rolling baseline window ~17 s, clamped to [200, 2000] frames
    dff_window = int(np.clip(round(fr * 17), 200, 2000))

    # ── Assemble ──────────────────────────────────────────────────────────────
    suggestions = {
        # Spatial filter
        "gSig":             [gSig, gSig],
        "gSiz":             [gSiz, gSiz],
        # Patch geometry (compact patches validated to avoid over-merging)
        "rf":               rf,
        "stride":           stride,
        # Background model (validated ring factor for 2P sparse)
        "ring_size_factor": _RING_SIZE_FACTOR,
        # Merging (0.7 merges neighbouring somas — use 0.85)
        "merge_thr":        _MERGE_THR,
        # Seed thresholds
        "min_corr":         min_corr,
        "min_pnr":          min_pnr,
        # Seeds per patch
        "K":                K,
        # ssub=1: avoids shape mismatch in precomputed filt inject (ssub>1 bug).
        # tsub=2: halves temporal load with negligible quality loss at 30 Hz.
        "ssub":             1,
        "tsub":             2,
        # nb_patch=0 required for corr_pnr + gnb>0 (avoids scale mismatch)
        "nb_patch":         0,
        # del_duplicates=False for sparse 2P — let merge_comps handle duplicates
        "del_duplicates":   False,
        # normalize_init must be False for corr_pnr + ring model
        "normalize_init":   False,
        "rolling_sum":      True,
        "ssub_B":           2,
        # GCaMP baseline is non-negative
        "bas_nonneg":       True,
        # 0.1 (default) clips weak neurons; 0.05 is more inclusive
        "maxthr":           0.05,
        # keep largest connected component per footprint (soma recording)
        "extract_cc":       True,
        # dF/F baseline
        "dff_quantile_min":  8,
        "dff_frames_window": dff_window,
    }
    logger.info(f"  Suggestions: {suggestions}")

    if out_path is not None:
        _save_inspection_figure(
            cn, pnr, suggestions, gSig, species, magnification, out_path, logger
        )

    return suggestions


# ── gSig estimation ───────────────────────────────────────────────────────────

def _estimate_gsig_from_pnr(
    pnr:       np.ndarray,
    *,
    prior:     dict,
    num_sigma: int   = 12,
    threshold: float = 0.10,
    pnr_pct:   float = 87.0,
    logger:    logging.Logger = logger,
) -> int:
    """Estimate gSig by brightness-weighted LoG blob detection on the PNR image."""
    gsig_min = prior["gsig_min"]
    gsig_max = prior["gsig_max"]
    fallback = prior["fallback"]

    min_sigma = gsig_min / math.sqrt(2)
    max_sigma = gsig_max / math.sqrt(2)

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

    blobs = _blob_log(pnr_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                      num_sigma=num_sigma, threshold=threshold)

    if len(blobs) == 0:
        logger.warning(f"  No blobs found — using fallback gSig={fallback}")
        return fallback

    h, w = pnr.shape
    bright_blobs = [
        (r, c, sigma)
        for r, c, sigma in blobs
        if 0 <= int(round(r)) < h
        and 0 <= int(round(c)) < w
        and float(pnr[int(round(r)), int(round(c))]) >= p_bright
    ]
    logger.info(
        f"  Blob detection: {len(blobs)} total → {len(bright_blobs)} bright "
        f"(PNR ≥ {p_bright:.1f})  sigma=[{min_sigma:.1f},{max_sigma:.1f}]"
    )

    if len(bright_blobs) < 5:
        logger.warning(
            f"  Only {len(bright_blobs)} bright blobs — using fallback gSig={fallback}")
        return fallback

    sigmas = np.array([b[2] for b in bright_blobs])
    radii  = sigmas * math.sqrt(2)
    gSig   = int(np.clip(round(float(np.percentile(radii, 25))), gsig_min, gsig_max))

    logger.info(
        f"  Blob radii: p25={np.percentile(radii,25):.1f}  "
        f"median={np.median(radii):.1f}  p75={np.percentile(radii,75):.1f} px  "
        f"→ gSig={gSig} (prior=[{gsig_min},{gsig_max}])"
    )
    return gSig


# ── Threshold estimation ──────────────────────────────────────────────────────

def _corr_threshold_upper_tail(
    cn:       np.ndarray,
    *,
    tail_pct: float = 92.0,
    logger:   logging.Logger = logger,
) -> float:
    """p92 of all Cn pixels as the min_corr threshold."""
    all_cn    = cn.ravel()
    threshold = float(np.percentile(all_cn, tail_pct))
    lo        = float(np.percentile(all_cn, 80))
    hi        = float(np.percentile(all_cn, 98))
    threshold = float(np.clip(threshold, lo, hi))

    n_seeds    = int((cn >= threshold).sum())
    frac_seeds = n_seeds / cn.size
    logger.info(
        f"  min_corr: p{tail_pct:.0f}={threshold:.3f}  "
        f"({n_seeds} pixels = {frac_seeds*100:.1f}% of FOV)"
    )
    return threshold


def _pnr_threshold_from_signal_pixels(
    cn:       np.ndarray,
    pnr:      np.ndarray,
    min_corr: float,
    *,
    signal_pnr_pct: float = 10.0,
    pnr_floor:      float = 3.0,
    logger:   logging.Logger = logger,
) -> float:
    """p10 of PNR at signal pixels (Cn ≥ min_corr), floor 3.0."""
    signal_mask = cn >= min_corr
    n_signal    = int(signal_mask.sum())

    if n_signal >= 100:
        signal_pnr = pnr[signal_mask]
        pnr_99     = float(np.percentile(signal_pnr, 99))
        signal_pnr = signal_pnr[signal_pnr < pnr_99]
        threshold  = float(np.percentile(signal_pnr, signal_pnr_pct))
        logger.info(
            f"  min_pnr: {n_signal} signal pixels (Cn≥{min_corr:.2f})  "
            f"p10={threshold:.1f}  median={np.median(signal_pnr):.1f}"
        )
    else:
        threshold = float(np.percentile(pnr[pnr > 1].ravel(), 85))
        logger.warning(
            f"  min_pnr: only {n_signal} signal pixels — fallback p85={threshold:.1f}"
        )

    threshold = max(pnr_floor, threshold)
    logger.info(f"  min_pnr → {threshold:.1f}")
    return threshold


# ── Inspection figure ─────────────────────────────────────────────────────────

def _save_inspection_figure(
    cn:            np.ndarray,
    pnr:           np.ndarray,
    suggestions:   dict,
    gSig:          int,
    species:       str,
    magnification: str,
    out_path:      Union[str, Path],
    logger:        logging.Logger,
) -> None:
    """Four-panel Cn/PNR inspection figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        min_corr = suggestions["min_corr"]
        min_pnr  = suggestions["min_pnr"]
        rf       = suggestions["rf"]
        gSiz     = suggestions["gSiz"][0]
        ring_px  = _RING_SIZE_FACTOR * gSiz

        fig = plt.figure(figsize=(16, 8), facecolor="k")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                                left=0.05, right=0.97, top=0.92, bottom=0.06)

        def _dax(spec):
            ax = fig.add_subplot(spec)
            ax.set_facecolor("k")
            ax.tick_params(colors="0.6", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("0.3")
            return ax

        # Images
        ax_cn  = _dax(gs[0, 0]); ax_cn.axis("off")
        ax_pnr = _dax(gs[0, 1]); ax_pnr.axis("off")

        im_cn = ax_cn.imshow(cn, cmap="inferno", aspect="equal", interpolation="nearest")
        ax_cn.set_title(f"Correlation (Cn)  |  min_corr={min_corr:.2f}",
                        color="0.85", fontsize=8)
        cb = plt.colorbar(im_cn,
                          cax=make_axes_locatable(ax_cn).append_axes("right","3%",pad=0.05))
        cb.ax.tick_params(colors="0.6", labelsize=6)

        pnr_clip = np.clip(pnr, 0, np.percentile(pnr, 99.5))
        im_pnr   = ax_pnr.imshow(pnr_clip, cmap="inferno", aspect="equal",
                                  interpolation="nearest")
        ax_pnr.set_title(
            f"PNR  |  min_pnr={min_pnr:.1f}  |  gSig≈{gSig} px  "
            f"|  {species} {magnification}",
            color="0.85", fontsize=8)
        cb2 = plt.colorbar(im_pnr,
                           cax=make_axes_locatable(ax_pnr).append_axes("right","3%",pad=0.05))
        cb2.ax.tick_params(colors="0.6", labelsize=6)

        # Histograms
        ax_ch = _dax(gs[1, 0])
        cn_pos = cn.ravel(); cn_pos = cn_pos[cn_pos > 0]
        ax_ch.hist(cn_pos, bins=200, color="#4fc3f7", edgecolor="none", alpha=0.8,
                   range=(0, float(np.percentile(cn_pos, 99.5))))
        ax_ch.axvline(min_corr, color="white", lw=1.5, ls="--",
                      label=f"min_corr={min_corr:.2f}")
        ax_ch.set_xlabel("Cn value", color="0.6", fontsize=7)
        ax_ch.set_ylabel("pixels",   color="0.6", fontsize=7)
        ax_ch.set_title("Cn distribution", color="0.85", fontsize=8)
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

        fig.suptitle(
            f"Parameter Estimation  |  gSig={gSig}  gSiz={gSiz}  "
            f"rf={rf}  ring={ring_px:.1f}px  stride={suggestions['stride']}  "
            f"|  min_corr={min_corr}  min_pnr={min_pnr}",
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
    """Write parameter suggestions into an existing pipeline JSON.

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
