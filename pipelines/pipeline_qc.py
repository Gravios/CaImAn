"""
pipeline_qc.py — Quality-control figure generation for the CaImAn pipeline.

Each function takes the relevant data and saves one PNG (lossless, suitable
for archiving and publication).  Nothing is displayed; matplotlib is used in
non-interactive Agg mode so this works headless.

All functions are safe to call even when data is partial/None — they log a
warning and return without raising.

Output files are named  <session>_qc_<step>.png  and written next to the log.
"""

import logging
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("caiman")

# ── matplotlib setup (headless) ───────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="k")
    plt.close(fig)
    logger.info(f"QC figure saved: {path}")


def _percentile_clip(img: np.ndarray, lo: float = 1.0, hi: float = 99.0
                     ) -> np.ndarray:
    """Clip image to [lo, hi] percentile for display."""
    vlo, vhi = np.nanpercentile(img, [lo, hi])
    return np.clip(img, vlo, vhi)


def _dark_fig(nrows: int = 1, ncols: int = 1, figsize: tuple = (10, 6),
              **kw) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor="k", **kw)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("k")
        ax.tick_params(colors="0.6", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("0.3")
    return fig, axes


def _imshow(ax: plt.Axes, img: np.ndarray, cmap: str = "gray",
            title: str = "", colorbar: bool = False, **kw) -> None:
    im = ax.imshow(img, cmap=cmap, aspect="equal", interpolation="nearest",
                   **kw)
    if title:
        ax.set_title(title, color="0.85", fontsize=8, pad=3)
    ax.axis("off")
    if colorbar:
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3%", pad=0.05)
        cb  = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(colors="0.6", labelsize=6)


def _guard(fn):
    """Decorator: catch all exceptions so QC never crashes the pipeline."""
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            logger.warning(f"QC [{fn.__name__}] failed:\n{traceback.format_exc()}")
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Raw TIFF sample
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_raw_sample(fname: str, out_path: str, n_frames: int = 9) -> None:
    """Grid of evenly-spaced raw frames showing original fluorescence."""
    import tifffile
    with tifffile.TiffFile(fname) as tf:
        T   = len(tf.pages)
        idx = np.linspace(0, T - 1, n_frames, dtype=int)
        frames = np.stack([tf.pages[i].asarray() for i in idx])

    nc  = 3
    nr  = int(np.ceil(n_frames / nc))
    fig, axes = _dark_fig(nr, nc, figsize=(nc * 4, nr * 3 + 0.6))
    axes_flat = np.atleast_1d(axes).ravel()

    vlo = np.nanpercentile(frames, 1)
    vhi = np.nanpercentile(frames, 99.5)

    for k, ax in enumerate(axes_flat):
        if k < n_frames:
            _imshow(ax, frames[k], vmin=vlo, vmax=vhi,
                    title=f"frame {idx[k]}")
        else:
            ax.set_visible(False)

    fig.suptitle("Raw TIFF — frame sample", color="0.85", fontsize=10, y=1.01)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Motion correction QC
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_motion_correction(mc, out_path: str) -> None:
    """Four-panel MC summary:
       • Rigid shift traces (x, y vs frame)
       • Shift magnitude histogram
       • Mean raw frame vs mean corrected frame
       • Difference image (mean_raw − mean_corrected)
    """
    import caiman as cm

    # ── Shifts ────────────────────────────────────────────────────────────
    shifts_rig = np.array(mc.shifts_rig)   # (T, 2) — (row, col)
    T = len(shifts_rig)
    t = np.arange(T)
    mag = np.sqrt(shifts_rig[:, 0]**2 + shifts_rig[:, 1]**2)

    # ── Mean frames ───────────────────────────────────────────────────────
    # mc.fname[0] is the raw TIFF path — not an mmap, so load_memmap
    # cannot be used.  Read via tifffile directly (subsampled for speed).
    import tifffile as _tifffile
    with _tifffile.TiffFile(mc.fname[0]) as _tf:
        _n    = len(_tf.pages)
        _step = max(1, _n // 300)
        raw_frames = np.stack(
            [_tf.pages[i].asarray() for i in range(0, _n, _step)],
            axis=0,
        ).astype(np.float32)
    mean_raw = raw_frames.mean(axis=0)
    del raw_frames

    # Corrected frames are in the F-order mmap written by GPU MC.
    # load_memmap returns (array, dims, T) — dims is the spatial shape
    # (d1, d2).  mc.dims does not exist on MotionCorrect.
    corr, dims_mc, T_mc = cm.mmapping.load_memmap(mc.mmap_file[0])
    step_c    = max(1, T_mc // 300)
    mean_corr = np.mean(corr[:, ::step_c], axis=1).reshape(dims_mc)
    del corr

    fig = plt.figure(figsize=(14, 10), facecolor="k")
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.93, bottom=0.06,
    )

    def _darkax(spec):
        ax = fig.add_subplot(spec)
        ax.set_facecolor("k")
        ax.tick_params(colors="0.6", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.3")
        return ax

    # Row 0: shift traces
    ax_y = _darkax(gs[0, :2])
    ax_y.plot(t, shifts_rig[:, 0], color="#4fc3f7", lw=0.6, label="row (y)")
    ax_y.plot(t, shifts_rig[:, 1], color="#f48fb1", lw=0.6, label="col (x)")
    ax_y.axhline(0, color="0.4", lw=0.5, ls="--")
    ax_y.set_xlabel("frame", color="0.6", fontsize=7)
    ax_y.set_ylabel("shift (px)", color="0.6", fontsize=7)
    ax_y.set_title("Rigid shifts per frame", color="0.85", fontsize=8)
    ax_y.legend(fontsize=7, facecolor="0.15", labelcolor="0.85",
                framealpha=0.8)

    ax_hist = _darkax(gs[0, 2])
    ax_hist.hist(mag, bins=50, color="#4fc3f7", edgecolor="none", alpha=0.85)
    ax_hist.set_xlabel("shift magnitude (px)", color="0.6", fontsize=7)
    ax_hist.set_ylabel("frames", color="0.6", fontsize=7)
    ax_hist.set_title("Shift magnitude distribution", color="0.85", fontsize=8)

    # Stats annotation
    txt = (f"median={np.median(mag):.2f}  "
           f"p95={np.percentile(mag, 95):.2f}  "
           f"max={mag.max():.2f} px")
    ax_hist.text(0.97, 0.95, txt, transform=ax_hist.transAxes,
                 ha="right", va="top", fontsize=6, color="0.7")

    # Rows 1-2: mean images
    vlo_r, vhi_r = np.nanpercentile(mean_raw,  [1, 99.5])
    vlo_c, vhi_c = np.nanpercentile(mean_corr, [1, 99.5])
    diff = mean_corr - mean_raw
    vlim = np.nanpercentile(np.abs(diff), 99)

    ax_raw  = fig.add_subplot(gs[1:, 0]);  ax_raw.set_facecolor("k");  ax_raw.axis("off")
    ax_cor  = fig.add_subplot(gs[1:, 1]);  ax_cor.set_facecolor("k");  ax_cor.axis("off")
    ax_diff = fig.add_subplot(gs[1:, 2]);  ax_diff.set_facecolor("k"); ax_diff.axis("off")

    _imshow(ax_raw,  mean_raw,  vmin=vlo_r, vmax=vhi_r,
            title="Mean raw",       colorbar=True)
    _imshow(ax_cor,  mean_corr, vmin=vlo_c, vmax=vhi_c,
            title="Mean corrected", colorbar=True)
    _imshow(ax_diff, diff, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
            title="Corrected − Raw (mean)", colorbar=True)

    fig.suptitle("Motion Correction QC", color="0.85", fontsize=11)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Correlation image
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_correlation_image(Cn: np.ndarray, out_path: str) -> None:
    """Single-panel correlation image with colorbar and statistics."""
    fig, ax = _dark_fig(1, 1, figsize=(7, 6))
    _imshow(ax, Cn, cmap="inferno", title="Local correlation image (Cn)",
            colorbar=True)
    txt = (f"mean={np.nanmean(Cn):.3f}  "
           f"p99={np.nanpercentile(Cn, 99):.3f}  "
           f"max={np.nanmax(Cn):.3f}")
    ax.set_title(f"Local Correlation Image\n{txt}", color="0.85", fontsize=9)
    fig.suptitle("Summary Image QC", color="0.85", fontsize=11)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — CNMF initial fit
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_cnmf_fit(cnm, Cn: Optional[np.ndarray], out_path: str) -> None:
    """Spatial footprints from initial fit overlaid on correlation image."""
    _qc_footprints(cnm, Cn, out_path, title="CNMF initial fit — spatial footprints")


def _qc_footprints(cnm, Cn: Optional[np.ndarray], out_path: str,
                   title: str = "Spatial footprints") -> None:
    """Shared renderer for fit and refit footprint QC."""
    A   = cnm.estimates.A
    nr  = A.shape[1]
    d1, d2 = cnm.dims

    # Max-projection of all footprints
    A_dense = np.array(A.todense())         # (d, K)
    A_max   = A_dense.max(axis=1).reshape(d1, d2, order="F")

    # Contour overlay: for each component find centroid
    A_comp  = A_dense.reshape(d1, d2, nr, order="F")   # (d1, d2, K)

    fig, axes = _dark_fig(1, 2 if Cn is not None else 1,
                          figsize=(14 if Cn is not None else 7, 6))
    axes = np.atleast_1d(axes)

    _imshow(axes[0], _percentile_clip(A_max),
            cmap="hot", title=f"Max footprint projection  (K={nr})")

    if Cn is not None and len(axes) > 1:
        axes[1].set_facecolor("k")
        axes[1].axis("off")
        _imshow(axes[1], Cn, cmap="gray",
                title=f"Footprint centroids on Cn  (K={nr})")
        # Draw one contour per component
        for k in range(nr):
            comp = A_comp[:, :, k]
            if comp.max() < 1e-10:
                continue
            thr  = comp.max() * 0.3
            try:
                axes[1].contour(comp, levels=[thr], colors=["#00e5ff"],
                                linewidths=0.5, alpha=0.7)
            except Exception:
                pass

    fig.suptitle(title, color="0.85", fontsize=11)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — CNMF refit
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_cnmf_refit(cnm2, Cn: Optional[np.ndarray], out_path: str) -> None:
    """Footprints after refit with full AR model."""
    _qc_footprints(cnm2, Cn, out_path, title="CNMF refit — spatial footprints (full AR)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Component evaluation
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_component_evaluation(cnm2, Cn: Optional[np.ndarray], out_path: str) -> None:
    """
    Four-panel evaluation summary:
    • Accepted footprints on Cn (green contours)
    • Rejected footprints on Cn (red contours)
    • SNR distribution (accepted vs rejected)
    • r_values distribution (accepted vs rejected)
    """
    est  = cnm2.estimates
    idx_good = est.idx_components
    idx_bad  = est.idx_components_bad
    d1, d2   = cnm2.dims

    A_dense = np.array(est.A.todense()).reshape(d1, d2, -1, order="F")

    has_Cn   = Cn is not None
    has_snr  = hasattr(est, "SNR_comp") and est.SNR_comp is not None
    has_rval = hasattr(est, "r_values") and est.r_values is not None

    ncols = 2 + int(has_snr) + int(has_rval)
    fig   = plt.figure(figsize=(ncols * 5, 6), facecolor="k")
    gs    = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.3,
                              left=0.05, right=0.97)

    def _darkax2(spec):
        ax = fig.add_subplot(spec)
        ax.set_facecolor("k")
        ax.tick_params(colors="0.6", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.3")
        return ax

    def _contours(ax, idxs, color):
        for k in idxs:
            comp = A_dense[:, :, k]
            if comp.max() < 1e-10:
                continue
            try:
                ax.contour(comp, levels=[comp.max() * 0.3],
                           colors=[color], linewidths=0.5, alpha=0.8)
            except Exception:
                pass

    bg = Cn if has_Cn else np.zeros((d1, d2))
    bg_clipped = _percentile_clip(bg)

    ax_good = _darkax2(gs[0, 0])
    ax_good.axis("off")
    _imshow(ax_good, bg_clipped, cmap="gray",
            title=f"Accepted  (n={len(idx_good)})")
    _contours(ax_good, idx_good, "#69ff47")

    ax_bad = _darkax2(gs[0, 1])
    ax_bad.axis("off")
    _imshow(ax_bad, bg_clipped, cmap="gray",
            title=f"Rejected  (n={len(idx_bad)})")
    _contours(ax_bad, idx_bad, "#ff4747")

    col = 2
    if has_snr:
        snr = np.array(est.SNR_comp)
        ax_snr = _darkax2(gs[0, col])
        bins = np.linspace(0, np.nanpercentile(snr, 99), 40)
        if len(idx_good):
            ax_snr.hist(snr[idx_good], bins=bins, color="#69ff47",
                        alpha=0.75, label=f"accept (n={len(idx_good)})")
        if len(idx_bad):
            ax_snr.hist(snr[idx_bad],  bins=bins, color="#ff4747",
                        alpha=0.75, label=f"reject (n={len(idx_bad)})")
        ax_snr.axvline(cnm2.params.get("quality", "min_SNR"),
                       color="white", lw=1, ls="--", alpha=0.6, label="threshold")
        ax_snr.set_xlabel("SNR", color="0.6", fontsize=7)
        ax_snr.set_ylabel("components", color="0.6", fontsize=7)
        ax_snr.set_title("SNR distribution", color="0.85", fontsize=8)
        ax_snr.legend(fontsize=6, facecolor="0.15", labelcolor="0.85")
        col += 1

    if has_rval:
        rv = np.array(est.r_values)
        ax_rv = _darkax2(gs[0, col])
        bins  = np.linspace(-1, 1, 40)
        if len(idx_good):
            ax_rv.hist(rv[idx_good], bins=bins, color="#69ff47",
                       alpha=0.75, label=f"accept (n={len(idx_good)})")
        if len(idx_bad):
            ax_rv.hist(rv[idx_bad],  bins=bins, color="#ff4747",
                       alpha=0.75, label=f"reject (n={len(idx_bad)})")
        ax_rv.axvline(cnm2.params.get("quality", "rval_thr"),
                      color="white", lw=1, ls="--", alpha=0.6, label="threshold")
        ax_rv.set_xlabel("r value", color="0.6", fontsize=7)
        ax_rv.set_ylabel("components", color="0.6", fontsize=7)
        ax_rv.set_title("Spatial correlation (r_values)", color="0.85", fontsize=8)
        ax_rv.legend(fontsize=6, facecolor="0.15", labelcolor="0.85")

    fig.suptitle("Component Evaluation QC", color="0.85", fontsize=11)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Traces sample
# ─────────────────────────────────────────────────────────────────────────────

@_guard
def qc_traces(cnm2, fr: float, out_path: str, n_show: int = 20) -> None:
    """Stacked dF/F (or C) traces for the first n_show accepted components."""
    est = cnm2.estimates

    # Prefer dF/F if available, fall back to denoised C
    if hasattr(est, "F_dff") and est.F_dff is not None:
        traces = est.F_dff
        ylabel = "dF/F"
    else:
        traces = est.C
        ylabel = "C (a.u.)"

    K  = min(n_show, traces.shape[0])
    T  = traces.shape[1]
    t  = np.arange(T) / fr   # seconds

    fig, ax = plt.subplots(1, 1, figsize=(14, 1.0 + K * 0.55), facecolor="k")
    ax.set_facecolor("k")
    ax.tick_params(colors="0.6", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("0.3")

    spacing = 0.0
    offsets = []
    for k in range(K):
        tr = traces[k]
        tr_norm = tr - tr.min()
        peak    = tr_norm.max()
        if peak > 0:
            tr_norm /= peak
        if k == 0:
            spacing = 0.0
        else:
            spacing += 1.2
        offsets.append(spacing)
        ax.plot(t, tr_norm + spacing, lw=0.7, color="#4fc3f7", alpha=0.85)
        ax.text(t[-1] + t[-1] * 0.003, spacing + 0.3,
                f"#{k}", color="0.6", fontsize=5, va="center")

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (s)", color="0.6", fontsize=8)
    ax.set_ylabel(f"component  [{ylabel}, normalised]", color="0.6", fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"Traces — first {K} components", color="0.85", fontsize=9)

    fig.suptitle("Traces QC", color="0.85", fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: save all QC figures at once from a finished cnm2 object
# ─────────────────────────────────────────────────────────────────────────────

def save_all_post_cnmf(cnm2, Cn, fr, out_dir, session):
    """Save refit, evaluation, and traces QC figures."""
    d = Path(out_dir)
    qc_cnmf_refit(
        cnm2, Cn,
        str(d / f"{session}_qc_05_refit.png"),
    )
    qc_component_evaluation(
        cnm2, Cn,
        str(d / f"{session}_qc_06_evaluation.png"),
    )
    qc_traces(
        cnm2, fr,
        str(d / f"{session}_qc_07_traces.png"),
    )
