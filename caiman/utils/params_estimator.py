"""
caiman/utils/params_estimator.py
=================================
Data-driven estimation of CaImAn pipeline parameters from a movie sample.

Given a small temporal subsample of the movie, ``estimate_params`` computes the
Cn/PNR images and from them estimates:

- **gSig** — Gaussian half-width of the PSF (soma radius proxy).  Measured by
  LoG blob detection on the PNR image; the median blob radius is returned.
- **gSiz** — Spatial support, set to ``4 × gSig + 1`` (standard rule of thumb).
- **min_corr** — Suggested Cn threshold, set at the inflection point of the
  Cn histogram (typically the shoulder between background and cell pixels).
- **min_pnr** — Suggested PNR threshold, set similarly.
- **rf** — Patch half-size, set to ``3 × gSiz`` (must be > gSiz; ring
  constraint ``ring_size_factor × gSiz < rf`` is checked automatically).

All estimates are *starting points* — inspect the QC figures and tune from
there.  ``estimate_params`` also writes a PNR/Cn inspection figure so you can
visually verify the thresholds before committing to a full run.

Usage
-----
    from caiman.utils.params_estimator import estimate_params

    suggestions = estimate_params(
        fname_mc   = fname_mc,       # F-order MC mmap path
        gSig_hint  = None,           # set if you already have a rough estimate
        n_frames   = 500,            # frames to subsample for estimation
        out_path   = outdir / f"{session}_qc_00_param_estimate.png",
        logger     = logger,
    )

    # suggestions is a dict ready to be merged into the JSON cnmf section:
    # {
    #   "gSig":     [8, 8],
    #   "gSiz":     [33, 33],
    #   "rf":       36,
    #   "stride":   18,
    #   "min_corr": 0.52,
    #   "min_pnr":  6.1,
    # }

Integrate with new_session.py
------------------------------
Pass ``--estimate-params`` to ``new_session.py`` to run estimation automatically
when creating a session, writing the suggested values directly into the JSON.
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
    gSig_hint:   Optional[int]          = None,
    n_frames:    int                     = 500,
    fr:          float                   = 30.0,
    ring_size_factor: float              = 0.9,
    out_path:    Optional[Union[str, Path]] = None,
    logger:      logging.Logger          = logger,
) -> dict:
    """Estimate CNMF parameters from a subsampled movie.

    Computes Cn and PNR images from *n_frames* evenly-spaced frames of the
    motion-corrected movie, then derives ``gSig``, ``gSiz``, ``rf``,
    ``stride``, ``min_corr``, and ``min_pnr`` from those images.

    Parameters
    ----------
    fname_mc
        Path to the F-order motion-corrected mmap
        (``*rig*order_F*.mmap``).
    gSig_hint
        Prior estimate of the soma half-width in pixels.  If ``None``, gSig
        is estimated from the PNR image via LoG blob detection.
    n_frames
        Number of frames to subsample for the Cn/PNR computation.  500 is
        fast (~2 s) and gives a good estimate; use 1000+ for noisy data.
    fr
        Acquisition frame rate in Hz.  Used only for log messages.
    ring_size_factor
        Must match the value in the JSON ``cnmf`` section.  Used to check that
        the suggested ``rf`` satisfies ``ring_size_factor × gSiz < rf``.
    out_path
        If given, save a Cn/PNR inspection figure to this path.  The figure
        shows the images, histograms, and the estimated thresholds.
    logger
        Logger to write progress messages to.

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
    logger.info(f"  Loaded {len(frames)} frames, dims={dims}, fr={fr} Hz")

    # ── Compute Cn and PNR ────────────────────────────────────────────────────
    from caiman.summary_images import correlation_pnr as _corr_pnr

    gSig_for_compute = [gSig_hint, gSig_hint] if gSig_hint else None
    cn, pnr = _corr_pnr(frames, gSig=gSig_for_compute, center_psf=True,
                         swap_dim=False)
    cn[np.isnan(cn)] = 0

    logger.info(
        f"  Cn:  mean={cn.mean():.3f}  p75={np.percentile(cn, 75):.3f}  "
        f"max={cn.max():.3f}"
    )
    logger.info(
        f"  PNR: mean={pnr.mean():.1f}  p75={np.percentile(pnr, 75):.1f}  "
        f"max={np.percentile(pnr, 99.5):.1f}"
    )

    # ── Estimate gSig ─────────────────────────────────────────────────────────
    if gSig_hint is not None:
        gSig = int(round(gSig_hint))
        logger.info(f"  gSig: using hint = {gSig} px")
    else:
        gSig = _estimate_gsig_from_pnr(pnr, logger=logger)

    gSiz   = gSig * 4 + 1
    rf_min = int(np.ceil(gSiz * ring_size_factor)) + 1   # hard minimum
    # Round rf up to next multiple of gSig for clean tile alignment
    rf     = max(rf_min, int(np.ceil(gSiz * 1.5 / gSig) * gSig))
    stride = rf // 2

    logger.info(
        f"  Geometry: gSig={gSig}  gSiz={gSiz}  rf={rf}  stride={stride}  "
        f"(ring check: {ring_size_factor}×{gSiz}={ring_size_factor*gSiz:.1f} < {rf} ✓)"
    )

    # ── Estimate thresholds ───────────────────────────────────────────────────
    min_corr = _threshold_from_histogram(
        cn[cn > 0.05], n_bins=200, low_pct=50, high_pct=99,
        label="min_corr", logger=logger)
    min_pnr  = _threshold_from_histogram(
        pnr[(pnr > 1) & (pnr < np.percentile(pnr, 99.5))], n_bins=200,
        low_pct=50, high_pct=99,
        label="min_pnr",  logger=logger)

    # Round to 1 d.p. for readability in the JSON
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

    # ── Optional figure ───────────────────────────────────────────────────────
    if out_path is not None:
        _save_inspection_figure(
            cn, pnr, suggestions, gSig, out_path, logger)

    return suggestions


# ── gSig estimation ───────────────────────────────────────────────────────────

def _estimate_gsig_from_pnr(
    pnr: np.ndarray,
    *,
    min_sigma:  float = 2.0,
    max_sigma:  float = 20.0,
    num_sigma:  int   = 10,
    threshold:  float = 0.15,
    logger:     logging.Logger = logger,
) -> int:
    """Estimate gSig by LoG blob detection on the PNR image.

    Uses ``skimage.feature.blob_log`` on a percentile-normalised PNR image.
    Falls back to ``gSig=5`` if fewer than 5 blobs are found (dataset may be
    too sparse or noisy for reliable estimation).

    Parameters
    ----------
    pnr
        2-D PNR image.
    min_sigma, max_sigma
        Blob radius search range in pixels.
    num_sigma
        Number of intermediate sigma values to test.
    threshold
        Blob detection threshold (fraction of maximum response).

    Returns
    -------
    int
        Estimated gSig in pixels.
    """
    try:
        from skimage.feature import blob_log as _blob_log
    except ImportError:
        logger.warning("  scikit-image not available — using default gSig=5")
        return 5

    # Normalise to [0, 1] using percentile clip for robustness
    pnr_clip = np.clip(pnr, 0, np.percentile(pnr, 99.5))
    p_max    = pnr_clip.max()
    if p_max < 1e-10:
        logger.warning("  PNR image is flat — using default gSig=5")
        return 5

    pnr_norm = pnr_clip / p_max

    blobs = _blob_log(
        pnr_norm,
        min_sigma  = min_sigma,
        max_sigma  = max_sigma,
        num_sigma  = num_sigma,
        threshold  = threshold,
    )

    if len(blobs) < 5:
        logger.warning(
            f"  Only {len(blobs)} blobs found — defaulting to gSig=5 "
            f"(try lower threshold or increase n_frames)"
        )
        return 5

    # blob_log sigma column ≈ radius/√2 — convert to half-width
    sigmas = blobs[:, 2]
    radii  = sigmas * np.sqrt(2)
    gSig   = int(round(np.median(radii)))
    gSig   = max(2, gSig)   # floor at 2 px

    logger.info(
        f"  Blob detection: {len(blobs)} blobs  "
        f"radius median={np.median(radii):.1f} px  → gSig={gSig}"
    )
    return gSig


# ── Threshold estimation ──────────────────────────────────────────────────────

def _threshold_from_histogram(
    values:   np.ndarray,
    n_bins:   int,
    low_pct:  float,
    high_pct: float,
    label:    str   = "",
    logger:   logging.Logger = logger,
) -> float:
    """Estimate a threshold at the inflection point of the value distribution.

    Fits a histogram of *values* and finds the gradient inflection between
    background (low values, high density) and signal (high values, low density).
    Falls back to the 65th percentile if no clear inflection is found.

    Parameters
    ----------
    values
        1-D array of pixel values (pre-masked to remove trivial zeros).
    n_bins
        Number of histogram bins.
    low_pct, high_pct
        Percentile range used to clip the histogram before fitting.
    label
        Name of the parameter (for logging only).
    """
    if len(values) == 0:
        return 0.5

    vlo = float(np.percentile(values, low_pct))
    vhi = float(np.percentile(values, high_pct))
    if vhi <= vlo:
        return vlo

    hist, edges = np.histogram(
        np.clip(values, vlo, vhi), bins=n_bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Smooth histogram, then find the steepest negative gradient
    # (inflection = where the density starts falling sharply = threshold)
    from scipy.ndimage import gaussian_filter1d as _gf1d
    smooth = _gf1d(hist.astype(float), sigma=4)
    grad   = np.gradient(smooth)

    # Search in the lower 60% of the value range (threshold should be near
    # the background/signal boundary, not at the far tail)
    search_end = int(len(grad) * 0.6)
    if search_end < 2:
        search_end = len(grad)

    min_grad_idx = int(np.argmin(grad[:search_end]))
    threshold    = float(centres[min_grad_idx])

    # Sanity clip: keep between 25th and 85th percentile
    lo_bound = float(np.percentile(values, 25))
    hi_bound = float(np.percentile(values, 85))
    threshold = float(np.clip(threshold, lo_bound, hi_bound))

    logger.info(
        f"  {label}: histogram inflection at {threshold:.3f} "
        f"(range [{vlo:.3f}, {vhi:.3f}])"
    )
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
        ax_ch.hist(cn_vals, bins=150, color="#4fc3f7", edgecolor="none", alpha=0.8,
                   range=(float(np.percentile(cn_vals, 1)),
                          float(np.percentile(cn_vals, 99))))
        ax_ch.axvline(min_corr, color="white", lw=1.5, ls="--",
                      label=f"min_corr={min_corr:.2f}")
        ax_ch.set_xlabel("Cn value", color="0.6", fontsize=7)
        ax_ch.set_ylabel("pixels",   color="0.6", fontsize=7)
        ax_ch.set_title("Cn distribution", color="0.85", fontsize=8)
        ax_ch.legend(fontsize=7, facecolor="0.15", labelcolor="0.85")

        ax_ph = _dax(gs[1, 1])
        pnr_vals = pnr[(pnr > 1) & (pnr < np.percentile(pnr, 99.5))].ravel()
        ax_ph.hist(pnr_vals, bins=150, color="#f48fb1", edgecolor="none", alpha=0.8)
        ax_ph.axvline(min_pnr, color="white", lw=1.5, ls="--",
                      label=f"min_pnr={min_pnr:.1f}")
        ax_ph.set_xlabel("PNR value", color="0.6", fontsize=7)
        ax_ph.set_ylabel("pixels",    color="0.6", fontsize=7)
        ax_ph.set_title("PNR distribution", color="0.85", fontsize=8)
        ax_ph.legend(fontsize=7, facecolor="0.15", labelcolor="0.85")

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
    dry_run: bool = False,
) -> None:
    """Write parameter suggestions into an existing pipeline JSON file.

    Only the keys present in *suggestions* are updated; all other values
    in the ``cnmf`` section are preserved.

    Parameters
    ----------
    json_path
        Path to ``<session>_pipeline.json``.
    suggestions
        Dict returned by :func:`estimate_params`.
    dry_run
        If ``True``, print what would be written without modifying the file.
    """
    import json

    path = Path(json_path)
    raw  = json.loads(path.read_text())
    cnmf = raw.setdefault("cnmf", {})

    changed = {}
    for k, v in suggestions.items():
        if cnmf.get(k) != v:
            changed[k] = (cnmf.get(k), v)
        cnmf[k] = v

    if not changed:
        logger.info("apply_suggestions: no changes needed")
        return

    if dry_run:
        logger.info("apply_suggestions (dry run):")
        for k, (old, new) in changed.items():
            logger.info(f"  {k}: {old!r} → {new!r}")
        return

    path.write_text(json.dumps(raw, indent=4) + "\n")
    logger.info(f"apply_suggestions: updated {path.name}")
    for k, (old, new) in changed.items():
        logger.info(f"  {k}: {old!r} → {new!r}")
