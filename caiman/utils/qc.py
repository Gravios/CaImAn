"""
caiman/utils/qc.py
==================
Quality-control figure generation for CaImAn pipelines.

Each function saves one PNG (lossless, dpi=150) and returns the output path.
Nothing is displayed — matplotlib runs in headless Agg mode.  Every public
function is wrapped with ``@_guard`` so a plotting failure can never crash
the pipeline; it logs a warning and returns ``None`` instead.

All figures use a dark theme (black background) suitable for archiving and
for direct inclusion in lab presentations.

Public API
----------
qc_raw_sample(fname, out_path)
    Grid of evenly-spaced raw frames.

qc_motion_correction(mc, out_path)
    Shift traces, magnitude histogram, and mean-frame comparison.

qc_correlation_image(Cn, out_path)
    Local correlation image with statistics annotation.

qc_cnmf_fit(cnm, Cn, out_path)
    Max-projection + centroid overlay from the initial CNMF fit.

qc_cnmf_refit(cnm2, Cn, out_path)
    Same as above for the refit object.

qc_component_evaluation(cnm2, Cn, out_path)
    Accepted / rejected footprints and SNR / r-value distributions.

qc_traces(cnm2, fr, out_path)
    Stacked normalised dF/F (or C) traces.

qc_pnr_image(Cn, pnr, out_path)
    Side-by-side Cn and PNR images for corr_pnr threshold tuning.

save_all_post_cnmf(cnm2, Cn, fr, out_dir, session)
    Convenience wrapper: calls refit + evaluation + traces in one shot.
"""

from __future__ import annotations

import functools
import logging
import traceback
from pathlib import Path
from typing import Optional, Union

import numpy as np

# ── matplotlib: headless Agg mode ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger("caiman")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _guard(fn):
    """Decorator: catch all exceptions so a QC failure never crashes the pipeline."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.warning(f"QC [{fn.__name__}] failed:\n{traceback.format_exc()}")
            return None
    return wrapper


def _save(fig: plt.Figure, path: Union[str, Path]) -> str:
    """Save *fig* to *path* at 150 dpi, close it, and log the result."""
    path = str(path)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="k")
    plt.close(fig)
    logger.info(f"QC figure saved: {path}")
    return path


def _percentile_clip(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Clip *img* to the [*lo*, *hi*] percentile range for display."""
    vlo, vhi = np.nanpercentile(img, [lo, hi])
    return np.clip(img, vlo, vhi)


def _dark_fig(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[int, int] = (10, 6),
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a dark-themed figure with *nrows* × *ncols* axes.

    Returns ``(fig, axes)`` where *axes* is always a 1-D ndarray regardless
    of layout, so callers can index with ``axes[0]``, ``axes[1]``, etc.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor="k", **kwargs)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("k")
        ax.tick_params(colors="0.6", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("0.3")
    return fig, np.atleast_1d(axes).ravel()


def _dark_ax(fig: plt.Figure, spec) -> plt.Axes:
    """Add a single dark-themed axis to *fig* at GridSpec position *spec*."""
    ax = fig.add_subplot(spec)
    ax.set_facecolor("k")
    ax.tick_params(colors="0.6", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("0.3")
    return ax


def _imshow(
    ax: plt.Axes,
    img: np.ndarray,
    cmap: str = "gray",
    title: str = "",
    colorbar: bool = False,
    **kwargs,
) -> plt.cm.ScalarMappable:
    """Display *img* on *ax* with optional title and colorbar.

    Returns the ``AxesImage`` so callers can adjust the norm if needed.
    """
    im = ax.imshow(img, cmap=cmap, aspect="equal",
                   interpolation="nearest", **kwargs)
    if title:
        ax.set_title(title, color="0.85", fontsize=8, pad=3)
    ax.axis("off")
    if colorbar:
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3%", pad=0.05)
        cb  = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(colors="0.6", labelsize=6)
    return im


def _contour_overlay(
    ax: plt.Axes,
    A_dense: np.ndarray,
    d1: int,
    d2: int,
    indices: np.ndarray,
    color: str,
    threshold_frac: float = 0.3,
) -> None:
    """Draw one contour per component in *indices* on *ax*.

    Parameters
    ----------
    A_dense
        Dense spatial footprint matrix, shape ``(d1*d2, K)``.
    d1, d2
        Spatial dimensions of the FOV.
    indices
        Component indices to draw contours for.
    color
        Contour line colour string.
    threshold_frac
        Contour drawn at ``component.max() × threshold_frac``.
    """
    A_vol = A_dense.reshape(d1, d2, -1, order="F")
    for k in indices:
        comp = A_vol[:, :, k]
        if comp.max() < 1e-10:
            continue
        try:
            ax.contour(comp, levels=[comp.max() * threshold_frac],
                       colors=[color], linewidths=0.5, alpha=0.8)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Raw TIFF sample
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_raw_sample(
    fname: Union[str, Path],
    out_path: Union[str, Path],
    n_frames: int = 9,
) -> Optional[str]:
    """Save a grid of evenly-spaced raw frames.

    Parameters
    ----------
    fname
        Path to the input TIFF stack.
    out_path
        Output PNG path.
    n_frames
        Number of frames to sample (default 9 → 3×3 grid).

    Returns
    -------
    str or None
        Absolute path of the saved PNG, or ``None`` on failure.
    """
    import tifffile
    with tifffile.TiffFile(str(fname)) as tf:
        T   = len(tf.pages)
        idx = np.linspace(0, T - 1, n_frames, dtype=int)
        frames = np.stack([tf.pages[i].asarray() for i in idx])

    nc  = 3
    nr  = int(np.ceil(n_frames / nc))
    fig, axes = _dark_fig(nr, nc, figsize=(nc * 4, nr * 3 + 0.6))

    vlo = np.nanpercentile(frames, 1)
    vhi = np.nanpercentile(frames, 99.5)

    for k, ax in enumerate(axes):
        if k < n_frames:
            _imshow(ax, frames[k], vmin=vlo, vmax=vhi, title=f"frame {idx[k]}")
        else:
            ax.set_visible(False)

    fig.suptitle("Raw TIFF — frame sample", color="0.85", fontsize=10, y=1.01)
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Motion correction
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_motion_correction(mc, out_path: Union[str, Path]) -> Optional[str]:
    """Save a motion-correction summary figure.

    Four panels:
    - Rigid shift traces (row and column per frame)
    - Shift magnitude histogram with statistics
    - Mean raw frame
    - Mean corrected frame
    - Corrected − Raw difference image

    Parameters
    ----------
    mc
        CaImAn ``MotionCorrect`` object after ``motion_correct()`` has run.
    out_path
        Output PNG path.

    Returns
    -------
    str or None
        Absolute path of the saved PNG, or ``None`` on failure.
    """
    import caiman as cm
    import tifffile

    shifts_rig = np.array(mc.shifts_rig)           # (T, 2) — [row, col]
    T  = len(shifts_rig)
    t  = np.arange(T)
    mag = np.hypot(shifts_rig[:, 0], shifts_rig[:, 1])

    # Mean raw — subsampled for speed (≤300 frames)
    with tifffile.TiffFile(mc.fname[0]) as tf:
        n_raw = len(tf.pages)
        step  = max(1, n_raw // 300)
        raw_frames = np.stack(
            [tf.pages[i].asarray() for i in range(0, n_raw, step)]
        ).astype(np.float32)
    mean_raw = raw_frames.mean(axis=0)
    del raw_frames

    # Mean corrected — F-order mmap, shape (d1*d2, T)
    corr, dims_mc, T_mc = cm.mmapping.load_memmap(mc.mmap_file[0])
    step_c    = max(1, T_mc // 300)
    mean_corr = np.mean(corr[:, ::step_c], axis=1).reshape(dims_mc, order="F")
    del corr

    diff  = mean_corr.astype(np.float32) - mean_raw
    vlim  = float(np.nanpercentile(np.abs(diff), 99))

    fig = plt.figure(figsize=(16, 10), facecolor="k")
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.93, bottom=0.06,
    )

    # Row 0: shift traces + histogram
    ax_shifts = _dark_ax(fig, gs[0, :2])
    ax_shifts.plot(t, shifts_rig[:, 0], color="#4fc3f7", lw=0.6, label="row (y)")
    ax_shifts.plot(t, shifts_rig[:, 1], color="#f48fb1", lw=0.6, label="col (x)")
    ax_shifts.axhline(0, color="0.4", lw=0.5, ls="--")
    ax_shifts.set_xlabel("frame",       color="0.6", fontsize=7)
    ax_shifts.set_ylabel("shift (px)",  color="0.6", fontsize=7)
    ax_shifts.set_title("Rigid shifts per frame", color="0.85", fontsize=8)
    ax_shifts.legend(fontsize=7, facecolor="0.15", labelcolor="0.85", framealpha=0.8)

    ax_hist = _dark_ax(fig, gs[0, 2])
    ax_hist.hist(mag, bins=50, color="#4fc3f7", edgecolor="none", alpha=0.85)
    ax_hist.set_xlabel("shift magnitude (px)", color="0.6", fontsize=7)
    ax_hist.set_ylabel("frames",               color="0.6", fontsize=7)
    ax_hist.set_title("Shift magnitude", color="0.85", fontsize=8)
    stats = (f"median={np.median(mag):.2f}  "
             f"p95={np.percentile(mag, 95):.2f}  "
             f"max={mag.max():.2f} px")
    ax_hist.text(0.97, 0.95, stats, transform=ax_hist.transAxes,
                 ha="right", va="top", fontsize=6, color="0.7")

    # Rows 1–2: mean images
    vlo_r, vhi_r = np.nanpercentile(mean_raw,  [1, 99.5])
    vlo_c, vhi_c = np.nanpercentile(mean_corr, [1, 99.5])

    ax_raw  = _dark_ax(fig, gs[1:, 0])
    ax_cor  = _dark_ax(fig, gs[1:, 1])
    ax_diff = _dark_ax(fig, gs[1:, 2])
    for ax in (ax_raw, ax_cor, ax_diff):
        ax.axis("off")

    _imshow(ax_raw,  mean_raw,  vmin=vlo_r, vmax=vhi_r, title="Mean raw",       colorbar=True)
    _imshow(ax_cor,  mean_corr, vmin=vlo_c, vmax=vhi_c, title="Mean corrected", colorbar=True)
    _imshow(ax_diff, diff, cmap="RdBu_r", vmin=-vlim,   vmax=vlim,
            title="Corrected − Raw", colorbar=True)

    fig.suptitle("Motion Correction QC", color="0.85", fontsize=11)
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Correlation image
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_correlation_image(Cn: np.ndarray, out_path: Union[str, Path]) -> Optional[str]:
    """Save the local correlation image with statistics annotation.

    Parameters
    ----------
    Cn
        2-D correlation image array, shape ``(d1, d2)``.
    out_path
        Output PNG path.

    Returns
    -------
    str or None
        Absolute path of the saved PNG, or ``None`` on failure.
    """
    fig, axes = _dark_fig(1, 1, figsize=(7, 6))
    stats = (f"mean={np.nanmean(Cn):.3f}  "
             f"p99={np.nanpercentile(Cn, 99):.3f}  "
             f"max={np.nanmax(Cn):.3f}")
    _imshow(axes[0], Cn, cmap="inferno",
            title=f"Local Correlation Image\n{stats}", colorbar=True)
    fig.suptitle("Summary Image QC", color="0.85", fontsize=11)
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b — PNR image (corr_pnr threshold tuning)
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_pnr_image(
    Cn: np.ndarray,
    pnr: np.ndarray,
    out_path: Union[str, Path],
    min_corr: Optional[float] = None,
    min_pnr: Optional[float] = None,
) -> Optional[str]:
    """Save side-by-side Cn and PNR images for ``corr_pnr`` threshold tuning.

    Optionally overlays the ``min_corr`` / ``min_pnr`` threshold lines on
    histograms of each image so you can visually verify the seed pixel density.

    Parameters
    ----------
    Cn
        Local correlation image, shape ``(d1, d2)``.
    pnr
        Peak-to-noise ratio image, shape ``(d1, d2)``.
    out_path
        Output PNG path.
    min_corr
        If given, annotated as a vertical line on the Cn histogram.
    min_pnr
        If given, annotated as a vertical line on the PNR histogram.

    Returns
    -------
    str or None
        Absolute path of the saved PNG, or ``None`` on failure.
    """
    fig = plt.figure(figsize=(16, 8), facecolor="k")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                            left=0.05, right=0.97, top=0.93, bottom=0.06)

    # Images — top row
    ax_cn  = _dark_ax(fig, gs[0, 0]); ax_cn.axis("off")
    ax_pnr = _dark_ax(fig, gs[0, 1]); ax_pnr.axis("off")
    _imshow(ax_cn,  Cn,  cmap="inferno",
            title=f"Correlation (Cn)  [{np.nanmin(Cn):.2f} – {np.nanmax(Cn):.2f}]",
            colorbar=True)
    _imshow(ax_pnr, pnr, cmap="inferno",
            title=f"PNR  [{np.nanmin(pnr):.1f} – {np.nanmax(pnr):.1f}]",
            colorbar=True)

    # Histograms — bottom row
    ax_ch = _dark_ax(fig, gs[1, 0])
    ax_ch.hist(Cn.ravel(), bins=100, color="#4fc3f7", edgecolor="none", alpha=0.8)
    ax_ch.set_xlabel("Cn value", color="0.6", fontsize=7)
    ax_ch.set_ylabel("pixels",   color="0.6", fontsize=7)
    ax_ch.set_title("Cn distribution", color="0.85", fontsize=8)
    if min_corr is not None:
        ax_ch.axvline(min_corr, color="white", lw=1.2, ls="--",
                      label=f"min_corr={min_corr:.2f}")
        ax_ch.legend(fontsize=6, facecolor="0.15", labelcolor="0.85")

    ax_ph = _dark_ax(fig, gs[1, 1])
    pnr_clipped = np.clip(pnr.ravel(), 0, np.nanpercentile(pnr, 99.5))
    ax_ph.hist(pnr_clipped, bins=100, color="#f48fb1", edgecolor="none", alpha=0.8)
    ax_ph.set_xlabel("PNR value", color="0.6", fontsize=7)
    ax_ph.set_ylabel("pixels",    color="0.6", fontsize=7)
    ax_ph.set_title("PNR distribution", color="0.85", fontsize=8)
    if min_pnr is not None:
        ax_ph.axvline(min_pnr, color="white", lw=1.2, ls="--",
                      label=f"min_pnr={min_pnr:.1f}")
        ax_ph.legend(fontsize=6, facecolor="0.15", labelcolor="0.85")

    fig.suptitle("Cn / PNR Threshold Inspection", color="0.85", fontsize=11)
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 / 5 — CNMF footprints (shared renderer + two thin wrappers)
# ─────────────────────────────────────────────────────────────────────────────

def _qc_footprints(
    cnm,
    Cn: Optional[np.ndarray],
    out_path: Union[str, Path],
    title: str = "Spatial footprints",
) -> Optional[str]:
    """Render max-projection and centroid-on-Cn footprint panels.

    Internal helper shared by :func:`qc_cnmf_fit` and :func:`qc_cnmf_refit`.
    Not decorated with ``@_guard`` — callers handle exception wrapping.
    """
    A      = cnm.estimates.A
    K      = A.shape[1]
    d1, d2 = cnm.dims

    A_dense = np.asarray(A.todense())                   # (d1*d2, K)
    A_max   = A_dense.max(axis=1).reshape(d1, d2, order="F")

    # Colormap scaling: use only non-zero pixels (neuron pixels) for the
    # vmax estimate.  With sparse 2P recordings the FOV is >80% background
    # (A_max=0), so percentile_clip(A_max) would map vmax≈0 → black image.
    _nz = A_max[A_max > 0]
    if _nz.size > 0:
        _vmax = float(np.percentile(_nz, 99))
        _vmin = 0.0
    else:
        _vmax = None
        _vmin = None

    ncols   = 2 if Cn is not None else 1
    fig, axes = _dark_fig(1, ncols, figsize=(ncols * 7, 6))

    _imshow(axes[0], A_max, vmin=_vmin, vmax=_vmax,
            cmap="hot", title=f"Max footprint projection  (K={K})")

    if Cn is not None and ncols > 1:
        axes[1].axis("off")
        _imshow(axes[1], Cn, cmap="gray",
                title=f"Footprint centroids on Cn  (K={K})")
        _contour_overlay(axes[1], A_dense, d1, d2,
                         np.arange(K), color="#00e5ff")

    fig.suptitle(title, color="0.85", fontsize=11)
    return _save(fig, out_path)


@_guard
def qc_cnmf_fit(cnm, Cn: Optional[np.ndarray], out_path: Union[str, Path]) -> Optional[str]:
    """Save spatial footprints from the initial CNMF fit.

    Parameters
    ----------
    cnm
        CaImAn ``CNMF`` object after ``fit()``.
    Cn
        Correlation image for the background panel.  ``None`` suppresses it.
    out_path
        Output PNG path.

    Returns
    -------
    str or None
        Absolute path of the saved PNG, or ``None`` on failure.
    """
    return _qc_footprints(cnm, Cn, out_path,
                          title="CNMF initial fit — spatial footprints")


@_guard
def qc_cnmf_refit(cnm2, Cn: Optional[np.ndarray], out_path: Union[str, Path]) -> Optional[str]:
    """Save spatial footprints after the full-AR refit.

    Parameters
    ----------
    cnm2
        CaImAn ``CNMF`` object after ``refit()``.
    Cn
        Correlation image for the background panel.  ``None`` suppresses it.
    out_path
        Output PNG path.

    Returns
    -------
    str or None
        Absolute path of the saved PNG, or ``None`` on failure.
    """
    return _qc_footprints(cnm2, Cn, out_path,
                          title="CNMF refit — spatial footprints (full AR)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Component evaluation
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_component_evaluation(
    cnm2,
    Cn: Optional[np.ndarray],
    out_path: Union[str, Path],
) -> Optional[str]:
    """Save a component evaluation summary figure.

    Four panels (panels 3–4 only rendered when data is available):
    - Accepted footprints on Cn (green contours)
    - Rejected footprints on Cn (red contours)
    - SNR distribution: accepted vs rejected, with threshold line
    - Spatial correlation (r_values) distribution with threshold line

    Parameters
    ----------
    cnm2
        CaImAn ``CNMF`` object after ``evaluate_components()``.
    Cn
        Correlation image background.  ``None`` uses a blank black image.
    out_path
        Output PNG path.

    Returns
    -------
    str or None
        Absolute path of the saved PNG, or ``None`` on failure.
    """
    est      = cnm2.estimates
    idx_good = est.idx_components     if est.idx_components     is not None else []
    idx_bad  = est.idx_components_bad if est.idx_components_bad is not None else []
    d1, d2   = cnm2.dims

    A_dense  = np.asarray(est.A.todense())

    has_snr  = getattr(est, "SNR_comp",  None) is not None
    has_rval = getattr(est, "r_values",  None) is not None

    ncols = 2 + int(has_snr) + int(has_rval)
    fig   = plt.figure(figsize=(ncols * 5, 6), facecolor="k")
    gs    = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.3,
                              left=0.05, right=0.97)

    bg = _percentile_clip(Cn) if Cn is not None else np.zeros((d1, d2))

    ax_good = _dark_ax(fig, gs[0, 0]); ax_good.axis("off")
    ax_bad  = _dark_ax(fig, gs[0, 1]); ax_bad.axis("off")

    _imshow(ax_good, bg, cmap="gray", title=f"Accepted  (n={len(idx_good)})")
    _contour_overlay(ax_good, A_dense, d1, d2, idx_good, color="#69ff47")

    _imshow(ax_bad, bg, cmap="gray", title=f"Rejected  (n={len(idx_bad)})")
    _contour_overlay(ax_bad, A_dense, d1, d2, idx_bad, color="#ff4747")

    col = 2
    if has_snr:
        snr    = np.asarray(est.SNR_comp)
        ax_snr = _dark_ax(fig, gs[0, col])
        bins   = np.linspace(0, np.nanpercentile(snr, 99), 40)
        if len(idx_good):
            ax_snr.hist(snr[idx_good], bins=bins, color="#69ff47",
                        alpha=0.75, label=f"accept (n={len(idx_good)})")
        if len(idx_bad):
            ax_snr.hist(snr[idx_bad],  bins=bins, color="#ff4747",
                        alpha=0.75, label=f"reject (n={len(idx_bad)})")
        thr_snr = cnm2.params.get("quality", "min_SNR")
        if thr_snr is not None:
            ax_snr.axvline(thr_snr, color="white", lw=1, ls="--",
                           alpha=0.6, label=f"thr={thr_snr}")
        ax_snr.set_xlabel("SNR",         color="0.6", fontsize=7)
        ax_snr.set_ylabel("components",  color="0.6", fontsize=7)
        ax_snr.set_title("SNR distribution", color="0.85", fontsize=8)
        ax_snr.legend(fontsize=6, facecolor="0.15", labelcolor="0.85")
        col += 1

    if has_rval:
        rv    = np.asarray(est.r_values)
        ax_rv = _dark_ax(fig, gs[0, col])
        bins  = np.linspace(-1, 1, 40)
        if len(idx_good):
            ax_rv.hist(rv[idx_good], bins=bins, color="#69ff47",
                       alpha=0.75, label=f"accept (n={len(idx_good)})")
        if len(idx_bad):
            ax_rv.hist(rv[idx_bad],  bins=bins, color="#ff4747",
                       alpha=0.75, label=f"reject (n={len(idx_bad)})")
        thr_rv = cnm2.params.get("quality", "rval_thr")
        if thr_rv is not None:
            ax_rv.axvline(thr_rv, color="white", lw=1, ls="--",
                          alpha=0.6, label=f"thr={thr_rv}")
        ax_rv.set_xlabel("r value",      color="0.6", fontsize=7)
        ax_rv.set_ylabel("components",   color="0.6", fontsize=7)
        ax_rv.set_title("Spatial correlation (r_values)", color="0.85", fontsize=8)
        ax_rv.legend(fontsize=6, facecolor="0.15", labelcolor="0.85")

    fig.suptitle("Component Evaluation QC", color="0.85", fontsize=11)
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Traces
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_traces(
    cnm2,
    fr: float,
    out_path: Union[str, Path],
    n_show: int = 20,
) -> Optional[str]:
    """Save stacked normalised traces for the first *n_show* accepted components.

    Uses ``F_dff`` if available, otherwise falls back to denoised ``C``.

    Parameters
    ----------
    cnm2
        CaImAn ``CNMF`` object after ``select_components()`` and (optionally)
        ``detrend_df_f()``.
    fr
        Acquisition frame rate in Hz.
    out_path
        Output PNG path.
    n_show
        Maximum number of traces to display (default 20).

    Returns
    -------
    str or None
        Absolute path of the saved PNG, or ``None`` on failure.
    """
    est = cnm2.estimates

    if getattr(est, "F_dff", None) is not None:
        traces, ylabel = est.F_dff, "dF/F"
    else:
        traces, ylabel = est.C, "C (a.u.)"

    K = min(n_show, traces.shape[0])
    T = traces.shape[1]
    t = np.arange(T) / fr           # seconds

    fig, ax = plt.subplots(1, 1, figsize=(14, 1.0 + K * 0.55), facecolor="k")
    ax.set_facecolor("k")
    ax.tick_params(colors="0.6", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("0.3")

    spacing = 0.0
    for k in range(K):
        tr      = traces[k].copy()
        tr     -= tr.min()
        peak    = tr.max()
        if peak > 0:
            tr /= peak
        ax.plot(t, tr + spacing, lw=0.7, color="#4fc3f7", alpha=0.85)
        ax.text(t[-1] * 1.003, spacing + 0.3,
                f"#{k}", color="0.6", fontsize=5, va="center")
        spacing += 1.2

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (s)",                            color="0.6", fontsize=8)
    ax.set_ylabel(f"component  [{ylabel}, normalised]",  color="0.6", fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"Traces — first {K} accepted components", color="0.85", fontsize=9)
    fig.suptitle("Traces QC", color="0.85", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def save_all_post_cnmf(
    cnm2,
    Cn: Optional[np.ndarray],
    fr: float,
    out_dir: Union[str, Path],
    session: str,
) -> dict[str, Optional[str]]:
    """Save all post-CNMF QC figures in one call.

    Calls :func:`qc_cnmf_refit`, :func:`qc_component_evaluation`, and
    :func:`qc_traces` sequentially.

    Parameters
    ----------
    cnm2
        Evaluated and selected CNMF object.
    Cn
        Correlation image.
    fr
        Frame rate in Hz.
    out_dir
        Directory to write QC images into.
    session
        Session name prefix for filenames.

    Returns
    -------
    dict
        Mapping of ``{"refit": path, "evaluation": path, "traces": path}``.
        Values are the saved PNG paths or ``None`` on failure.
    """
    d = Path(out_dir)
    return {
        "refit":      qc_cnmf_refit(cnm2, Cn,
                          str(d / f"{session}_qc_05_refit_footprints.png")),
        "evaluation": qc_component_evaluation(cnm2, Cn,
                          str(d / f"{session}_qc_06_evaluation.png")),
        "traces":     qc_traces(cnm2, fr,
                          str(d / f"{session}_qc_07_traces.png")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# QCRunner — config-driven orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class QCRunner:
    """Config-driven QC orchestrator.

    Constructed once from the pipeline ``ParamBag``, session name, and output
    directory.  Each method takes only the data objects that change at that
    pipeline step — output paths, frame rate, and threshold values are
    resolved automatically from the stored config.

    This eliminates the repeated ``str(outdir / f"{session}_qc_NN_*.png")``
    boilerplate at every call site, and ensures QC figures always use the
    same parameter values as the pipeline run that produced them.

    Parameters
    ----------
    P
        ``ParamBag`` from :func:`~caiman.utils.params_io.load_pipeline_params`.
    session
        Session identifier (script stem minus ``_pipeline``).
    outdir
        Directory where QC PNGs are written — same folder as the pipeline
        outputs.

    Examples
    --------
    >>> qc = QCRunner(_P, session, outdir)
    >>>
    >>> qc.raw_sample(fnames)
    >>> qc.motion_correction(mc)
    >>> qc.correlation_image(Cn)
    >>> qc.cnmf_fit(cnm, Cn)
    >>> qc.cnmf_refit(cnm2, Cn)
    >>> qc.component_evaluation(cnm2, Cn)
    >>> qc.traces(cnm2)
    """

    # Filename templates — NN is zero-padded step number, label is human name.
    _STEPS = {
        "raw_sample":          ("01", "raw_sample"),
        "motion_correction":   ("02", "motion_correction"),
        "correlation_image":   ("03", "correlation_image"),
        "pnr_image":           ("03b", "pnr_image"),
        "cnmf_fit":            ("04", "fit_footprints"),
        "cnmf_refit":          ("05", "refit_footprints"),
        "component_evaluation":("06", "evaluation"),
        "traces":              ("07", "traces"),
    }

    def __init__(self, P, session: str, outdir: Union[str, Path]) -> None:
        self._P       = P
        self._session = session
        self._outdir  = Path(outdir)
        # Cache frequently-used leaves from the ParamBag
        self._fr        = float(P.data.fr)
        self._min_corr  = float(getattr(P.cnmf, "min_corr", 0.5))
        self._min_pnr   = float(getattr(P.cnmf, "min_pnr",  6.0))

    # ── path helpers ──────────────────────────────────────────────────────────

    def _path(self, step_key: str) -> str:
        """Return the canonical output path for *step_key*."""
        nn, label = self._STEPS[step_key]
        return str(self._outdir / f"{self._session}_qc_{nn}_{label}.png")

    # ── public methods ────────────────────────────────────────────────────────

    def raw_sample(
        self,
        fname: Union[str, Path],
        n_frames: int = 9,
    ) -> Optional[str]:
        """Save a grid of evenly-spaced raw frames.

        Parameters
        ----------
        fname
            Path to the input TIFF stack.
        n_frames
            Number of frames to sample (default 9 → 3×3 grid).
        """
        return qc_raw_sample(fname, self._path("raw_sample"),
                             n_frames=n_frames)

    def motion_correction(self, mc) -> Optional[str]:
        """Save the motion-correction summary figure.

        Parameters
        ----------
        mc
            ``MotionCorrect`` object after ``motion_correct()`` has run.
        """
        return qc_motion_correction(mc, self._path("motion_correction"))

    def correlation_image(self, Cn: np.ndarray) -> Optional[str]:
        """Save the local correlation image.

        Parameters
        ----------
        Cn
            2-D correlation image, shape ``(d1, d2)``.
        """
        return qc_correlation_image(Cn, self._path("correlation_image"))

    def pnr_image(
        self,
        Cn: np.ndarray,
        pnr: np.ndarray,
    ) -> Optional[str]:
        """Save side-by-side Cn / PNR images with threshold markers.

        ``min_corr`` and ``min_pnr`` are read from the JSON ``cnmf`` section.

        Parameters
        ----------
        Cn
            Local correlation image.
        pnr
            Peak-to-noise ratio image.
        """
        return qc_pnr_image(Cn, pnr, self._path("pnr_image"),
                            min_corr=self._min_corr,
                            min_pnr=self._min_pnr)

    def cnmf_fit(self, cnm, Cn: Optional[np.ndarray] = None) -> Optional[str]:
        """Save footprints from the initial CNMF fit.

        Parameters
        ----------
        cnm
            ``CNMF`` object after ``fit()``.
        Cn
            Correlation image background (optional).
        """
        return qc_cnmf_fit(cnm, Cn, self._path("cnmf_fit"))

    def cnmf_refit(self, cnm2, Cn: Optional[np.ndarray] = None) -> Optional[str]:
        """Save footprints after the full-AR refit.

        Parameters
        ----------
        cnm2
            ``CNMF`` object after ``refit()``.
        Cn
            Correlation image background (optional).
        """
        return qc_cnmf_refit(cnm2, Cn, self._path("cnmf_refit"))

    def component_evaluation(
        self,
        cnm2,
        Cn: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Save the component evaluation summary.

        Parameters
        ----------
        cnm2
            ``CNMF`` object after ``evaluate_components()``.
        Cn
            Correlation image background (optional).
        """
        return qc_component_evaluation(cnm2, Cn,
                                       self._path("component_evaluation"))

    def traces(self, cnm2, n_show: int = 20) -> Optional[str]:
        """Save stacked normalised traces.

        Frame rate is read from ``P.data.fr``.

        Parameters
        ----------
        cnm2
            ``CNMF`` object after ``select_components()`` and
            (optionally) ``detrend_df_f()``.
        n_show
            Maximum number of traces to display (default 20).
        """
        return qc_traces(cnm2, self._fr, self._path("traces"),
                         n_show=n_show)

    def all_post_cnmf(
        self,
        cnm2,
        Cn: Optional[np.ndarray] = None,
    ) -> dict[str, Optional[str]]:
        """Save refit, evaluation, and traces figures in one call.

        Parameters
        ----------
        cnm2
            Evaluated and selected ``CNMF`` object.
        Cn
            Correlation image (optional).

        Returns
        -------
        dict
            ``{"refit": path, "evaluation": path, "traces": path}``
        """
        return {
            "refit":       self.cnmf_refit(cnm2, Cn),
            "evaluation":  self.component_evaluation(cnm2, Cn),
            "traces":      self.traces(cnm2),
        }
