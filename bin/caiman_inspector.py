#!/usr/bin/env python
"""
CaImAn Component Inspector  (Qt6 / PyQt6)
==========================================
Interactive GUI for reviewing, merging, and deleting CNMF components.

Requires
--------
    PyQt6, matplotlib (≥3.7), numpy, scipy
    caiman  — only needed when loading a real .hdf5 file

Usage
-----
    python bin/cnmf_inspector.py /path/to/results.hdf5
    python bin/cnmf_inspector.py                        # synthetic demo

Controls
--------
    Table        — click to select; Ctrl/Shift for multi-select
    Corr matrix  — click any off-diagonal cell to highlight that pair
                   in orange / magenta on the cell viewer
    ⊕ Merge      — merge ≥ 2 selected (footprint = pixel max, trace = mean)
    ✕ Delete     — permanently remove selected
    ↩ Undo       — Ctrl+Z  — undo last merge or delete (up to 5 steps)
    ↪ Redo       — Ctrl+Y  — redo last undone operation
    File > Save  — write modified A / C back to the .hdf5
"""

import sys
import os
import argparse
from collections import deque
import numpy as np
import scipy.sparse

HISTORY_MAXLEN = 5   # maximum undo / redo steps retained

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.colors as mcolors
from matplotlib.figure   import Figure
from matplotlib.patches  import Rectangle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QTableWidget,
    QTableWidgetItem, QLabel, QFileDialog,
    QMessageBox, QSizePolicy, QAbstractItemView, QHeaderView,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QSize, QItemSelectionModel
from PyQt6.QtGui   import (
    QAction, QColor, QBrush, QFont, QPalette, QKeySequence,
)

# ── Colour constants ──────────────────────────────────────────────────────────

PALETTE_HEX = [
    "#FF4444", "#FF8C00", "#FFEE00", "#88FF00",
    "#FF00FF", "#FF66BB", "#AA44FF", "#AAAAFF",
    "#FF6644", "#CCFF00", "#FF44CC", "#FF9944",
]

CORR_A   = "#FF8C00"   # orange  – first of a clicked corr pair
CORR_B   = "#FF00FF"   # magenta – second
CYAN     = "#00FFFF"   # default (unselected) footprint colour

ALPHA_SEL    = 0.55   # overlay alpha for selected components
ALPHA_DEF    = 0.12   # overlay alpha for unselected components
CROSS_MIN_ARM = 50    # minimum cross arm length in pixels

BG_DARK  = "#111111"
AX_BG    = "#0d0d0d"


# ── Data model ────────────────────────────────────────────────────────────────

class ComponentStore:
    """
    Mutable container for CNMF component data.
    A is stored as dense float32 (d × K).
    """

    def __init__(self, A, C, Cn, dims,
                 SNR_comp=None, r_values=None, cnn_preds=None,
                 cnm_obj=None, fpath=None):
        self._A = _to_dense(A)
        self.C  = np.asarray(C, dtype=np.float32)
        self.Cn = np.asarray(Cn, dtype=np.float32)
        self.dims = dims

        self.SNR_comp  = _opt_arr(SNR_comp)
        self.r_values  = _opt_arr(r_values)
        self.cnn_preds = _opt_arr(cnn_preds)

        self.cnm_obj = cnm_obj
        self.fpath   = fpath
        self.labels  = [f"C{i:03d}" for i in range(self._A.shape[1])]
        self._corr   = None

        # History stacks — each entry is (op_label: str, snapshot: dict)
        self._undo_stack: deque = deque(maxlen=HISTORY_MAXLEN)
        self._redo_stack: deque = deque(maxlen=HISTORY_MAXLEN)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n(self):
        return self._A.shape[1]

    @property
    def corr(self):
        if self._corr is None or self._corr.shape[0] != self.n:
            self._corr = _active_pearson_corr(self.C)
        return self._corr

    def invalidate_corr(self):
        self._corr = None

    # ── History ───────────────────────────────────────────────────────────────

    def _capture(self) -> dict:
        """Return a deep copy of all mutable component state."""
        return {
            "A":         self._A.copy(),
            "C":         self.C.copy(),
            "labels":    list(self.labels),
            "SNR_comp":  self.SNR_comp.copy()  if self.SNR_comp  is not None else None,
            "r_values":  self.r_values.copy()  if self.r_values  is not None else None,
            "cnn_preds": self.cnn_preds.copy() if self.cnn_preds is not None else None,
        }

    def _restore(self, snap: dict):
        """Replace mutable state from a snapshot."""
        self._A        = snap["A"]
        self.C         = snap["C"]
        self.labels    = snap["labels"]
        self.SNR_comp  = snap["SNR_comp"]
        self.r_values  = snap["r_values"]
        self.cnn_preds = snap["cnn_preds"]
        self.invalidate_corr()

    def _push_undo(self, op: str):
        """
        Call this BEFORE any destructive mutation.
        Saves the current state onto the undo stack and clears redo history,
        since a new branch has been created.
        """
        self._undo_stack.append((op, self._capture()))
        self._redo_stack.clear()

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_description(self) -> str:
        if not self._undo_stack:
            return "Nothing to undo"
        op, _ = self._undo_stack[-1]
        n = len(self._undo_stack)
        return f"Undo {op}  ({n} step{'s' if n > 1 else ''} available)"

    @property
    def redo_description(self) -> str:
        if not self._redo_stack:
            return "Nothing to redo"
        op, _ = self._redo_stack[-1]
        n = len(self._redo_stack)
        return f"Redo {op}  ({n} step{'s' if n > 1 else ''} available)"

    def undo(self) -> str:
        """
        Restore the state before the last destructive operation.
        The current state is pushed onto the redo stack so it can be
        re-applied.  Returns a human-readable status string.
        """
        if not self._undo_stack:
            return "Nothing to undo"
        op, snap = self._undo_stack.pop()
        self._redo_stack.append((op, self._capture()))
        self._restore(snap)
        n_left = len(self._undo_stack)
        suffix = f"  ({n_left} undo step{'s' if n_left != 1 else ''} left)" if n_left else ""
        return f"Undid {op}{suffix}"

    def redo(self) -> str:
        """
        Re-apply the most recently undone operation.
        The current state is pushed back onto the undo stack.
        Returns a human-readable status string.
        """
        if not self._redo_stack:
            return "Nothing to redo"
        op, snap = self._redo_stack.pop()
        self._undo_stack.append((op, self._capture()))
        self._restore(snap)
        n_left = len(self._redo_stack)
        suffix = f"  ({n_left} redo step{'s' if n_left != 1 else ''} left)" if n_left else ""
        return f"Redid {op}{suffix}"

    # ── Per-component helpers ─────────────────────────────────────────────────

    def footprint(self, i):
        return self._A[:, i].reshape(self.dims, order="F")

    def max_a(self, i):   return float(self._A[:, i].max())
    def peak_df(self, i): return float(self.C[i].max() - self.C[i].min())
    def snr(self, i):     return float(self.SNR_comp[i])  if self.SNR_comp  is not None else float("nan")
    def rval(self, i):    return float(self.r_values[i])  if self.r_values  is not None else float("nan")
    def cnn(self, i):     return float(self.cnn_preds[i]) if self.cnn_preds is not None else float("nan")

    # ── Mutation ──────────────────────────────────────────────────────────────

    def merge(self, indices):
        """
        Merge components at `indices`.
        Footprint = pixel-wise max. Trace = mean. Appended at end.
        Returns the new component's index.
        """
        self._push_undo("merge")
        idx = sorted(indices)
        new_a = np.zeros(self._A.shape[0], dtype=np.float32)
        for i in idx:
            np.maximum(new_a, self._A[:, i], out=new_a)
        new_c     = self.C[idx].mean(axis=0)
        new_label = "+".join(self.labels[i] for i in idx)

        keep = [j for j in range(self.n) if j not in set(idx)]
        self._A     = np.column_stack([self._A[:, keep], new_a[:, None]])
        self.C      = np.vstack([self.C[keep], new_c[None, :]])
        self.labels = [self.labels[j] for j in keep] + [new_label]

        for attr in ("SNR_comp", "r_values", "cnn_preds"):
            arr = getattr(self, attr)
            if arr is not None:
                setattr(self, attr,
                        np.append(arr[keep], float(arr[idx].mean())))
        self.invalidate_corr()
        return self.n - 1

    def delete(self, indices):
        self._push_undo("delete")
        keep = [j for j in range(self.n) if j not in set(indices)]
        self._A     = self._A[:, keep]
        self.C      = self.C[keep]
        self.labels = [self.labels[j] for j in keep]
        for attr in ("SNR_comp", "r_values", "cnn_preds"):
            arr = getattr(self, attr)
            if arr is not None:
                setattr(self, attr, arr[keep])
        self.invalidate_corr()

    # ── Save ──────────────────────────────────────────────────────────────────

    @staticmethod
    def curated_path(src_path: str) -> str:
        """Insert ``_curated`` before the extension so the original is never
        overwritten.  Idempotent: calling twice gives the same result.

        Examples::

            results.hdf5          -> results_curated.hdf5
            results_curated.hdf5  -> results_curated.hdf5
        """
        import os
        base, ext = os.path.splitext(src_path)
        if base.endswith("_curated"):
            return src_path
        return base + "_curated" + ext

    def save(self, path: str):
        """Write the curated A/C back via cnm_obj.save(), which uses
        CaImAn's own fn_relocated + save_dict_to_hdf5 serialisation path.

        Derived per-component fields (S, YrA, F_dff, etc.) are nulled out
        because they are invalid after manual edits and must be recomputed.
        idx_components is reset to 0..K-1.  A provenance entry is appended.
        """
        import time as _time
        if self.cnm_obj is None:
            raise RuntimeError("No CaImAn object available -- cannot save.")

        est = self.cnm_obj.estimates
        K   = self.n

        # Spatial / temporal
        est.A  = scipy.sparse.csc_matrix(self._A)
        est.C  = self.C
        est.nr = K

        # Null stale K-indexed derived fields
        for field in ("S", "YrA", "R", "F_dff",
                      "g", "bl", "c1", "neurons_sn", "lam",
                      "coordinates", "A_thr"):
            if getattr(est, field, None) is not None:
                setattr(est, field, None)

        # Quality metrics
        est.SNR_comp  = self.SNR_comp
        est.r_values  = self.r_values
        est.cnn_preds = self.cnn_preds

        # Reset accepted component index so file is self-consistent
        est.idx_components     = np.arange(K, dtype=int)
        est.idx_components_bad = np.array([], dtype=int)

        # Append provenance record
        if hasattr(self.cnm_obj, "provenance"):
            self.cnm_obj.provenance.append({
                "event":        "curated",
                "time":         int(_time.time()),
                "description":  "Manual curation via cnmf_inspector",
                "n_components": K,
                "source_file":  self.fpath or "",
                "save_path":    path,
            })

        # Delegate to CaImAn's own serialiser
        self.cnm_obj.save(path)
        self.fpath = path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dense(A) -> np.ndarray:
    if scipy.sparse.issparse(A):
        return np.asarray(A.todense(), dtype=np.float32)
    return np.asarray(A, dtype=np.float32)

def _opt_arr(x):
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    return arr if arr.size > 0 else None   # treat empty arrays as absent


# ── Correlation helper ───────────────────────────────────────────────────────

def _active_pearson_corr(C: np.ndarray,
                         active_thr: float = 0.0) -> np.ndarray:
    """
    Pairwise Pearson correlation computed only on *active* frames.

    For each pair (i, j) the active frame mask is:

        mask_ij = (C[i] > active_thr) | (C[j] > active_thr)

    Pearson r is then evaluated on C[i][mask_ij] vs C[j][mask_ij].
    If fewer than 3 active frames exist for a pair, that cell is set
    to NaN rather than returning a spurious value.

    Why not plain Pearson on all frames?
    ─────────────────────────────────────
    CaImAn traces have long zero-valued (or near-zero) baseline epochs.
    Including those frames inflates the apparent correlation between any
    two cells that happen to share a quiet period, and suppresses the
    true co-activation signal relative to the noise.

    Why not Spearman?
    ─────────────────
    With sparse, non-negative traces, the baseline zeros all receive the
    same average rank.  If one trace has a true zero baseline while
    another has small noise at baseline, the zero-rank plateau is broken
    in one trace but not the other, drastically lowering Spearman r even
    between near-duplicate components.  Masked Pearson on active frames
    avoids this by simply ignoring the silent epochs entirely.

    Parameters
    ----------
    C          : (K, T) float32 trace matrix
    active_thr : threshold above which a frame is considered active.
                 Defaults to 0.0, matching CaImAn's non-negative OASIS
                 output (traces are exactly 0 during silent epochs).

    Returns
    -------
    (K, K) float32 symmetric matrix; diagonal is NaN.
    """
    K, T = C.shape
    mat  = np.full((K, K), np.nan, dtype=np.float32)
    if K == 0 or T < 3:
        return mat

    active = (C > active_thr)   # (K, T) bool

    for i in range(K):
        mat[i, i] = np.nan
        for j in range(i + 1, K):
            mask = active[i] | active[j]
            n    = int(mask.sum())
            if n < 3:
                continue
            a = C[i][mask].astype(np.float64)
            b = C[j][mask].astype(np.float64)
            a -= a.mean();  b -= b.mean()
            na = np.linalg.norm(a);  nb = np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                continue
            r = float(np.dot(a, b) / (na * nb))
            r = float(np.clip(r, -1.0, 1.0))
            mat[i, j] = r
            mat[j, i] = r

    return mat


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_from_hdf5(path: str) -> ComponentStore:
    from caiman.source_extraction.cnmf.cnmf import load_CNMF
    cnm  = load_CNMF(path)
    est  = cnm.estimates
    dims = cnm.dims if (cnm.dims and all(d > 0 for d in cnm.dims)) else est.dims

    A = _to_dense(est.A)
    C = est.C

    idx = getattr(est, "idx_components", None)
    if idx is not None and len(idx) > 0:
        A = A[:, idx];  C = C[idx]
        snr  = est.SNR_comp[idx]  if est.SNR_comp  is not None else None
        rval = est.r_values[idx]  if est.r_values  is not None else None
        cnn  = est.cnn_preds[idx] if est.cnn_preds is not None else None
    else:
        snr  = getattr(est, "SNR_comp",  None)
        rval = getattr(est, "r_values",  None)
        cnn  = getattr(est, "cnn_preds", None)

    _Cn = getattr(est, "Cn", None)
    Cn  = _Cn if _Cn is not None else A.max(axis=1).reshape(dims, order="F")
    return ComponentStore(A, C, Cn, dims,
                          SNR_comp=snr, r_values=rval, cnn_preds=cnn,
                          cnm_obj=cnm, fpath=path)


# ── RGBA overlay builder ──────────────────────────────────────────────────────

def _build_overlay(store: ComponentStore,
                   sel: list, pair) -> np.ndarray:
    """Return (d1, d2, 4) RGBA float32 composite of all footprints."""
    d1, d2  = store.dims
    out     = np.zeros((d1, d2, 4), dtype=np.float32)
    sel_set = set(sel)
    pair_set = set(pair) if pair else set()

    for i in range(store.n):
        fp   = store.footprint(i)
        peak = fp.max()
        if peak < 1e-9:
            continue
        norm = fp / peak

        if pair and i in pair_set:
            hex_c = CORR_A if i == pair[0] else CORR_B
            alpha = ALPHA_SEL
        elif i in sel_set:
            hex_c = PALETTE_HEX[sel.index(i) % len(PALETTE_HEX)]
            alpha = ALPHA_SEL
        else:
            hex_c = CYAN
            alpha = ALPHA_DEF

        r, g, b = mcolors.to_rgb(hex_c)
        a_mask  = norm * alpha
        out[..., 0] = np.maximum(out[..., 0], r * a_mask)
        out[..., 1] = np.maximum(out[..., 1], g * a_mask)
        out[..., 2] = np.maximum(out[..., 2], b * a_mask)
        out[..., 3] = np.maximum(out[..., 3], a_mask)

    return np.clip(out, 0, 1)


# ── Canvas base ───────────────────────────────────────────────────────────────

class _Canvas(FigureCanvas):
    """Dark-themed matplotlib canvas that expands to fill available space."""

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        sp = self.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        sp.setVerticalPolicy(QSizePolicy.Policy.Expanding)
        self.setSizePolicy(sp)
        self.setMinimumSize(QSize(80, 80))


# ── Cell viewer ───────────────────────────────────────────────────────────────

class CellViewer(_Canvas):
    """
    Top pane — Cn greyscale background with RGBA footprint overlay.

    Colours
    -------
      Unselected  → dim cyan
      Selected    → bright palette colours (no cyan)
      Corr pair   → orange / magenta

    Interaction
    -----------
      Left-click          → select the nearest component (by footprint centroid)
      Ctrl + Left-click   → toggle a second component into / out of the selection
    """

    # Emitted when the user clicks a component in the image.
    # Arguments: component index (int), Ctrl held (bool)
    component_clicked = pyqtSignal(int, bool)

    def __init__(self, store: ComponentStore, parent=None):
        super().__init__(parent)
        self.store       = store
        self._sel        = []
        self._pair       = None
        self._centroids  = None   # (cy, cx) list, lazily computed, invalidated by n-change

        # Zero-margin layout so the image fills the canvas
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")

        self._invert_bg = False   # toggled by toolbar button
        self.mpl_connect("button_press_event", self._on_mpl_click)
        self._redraw()

    # ── Selection setters ─────────────────────────────────────────────────────

    def set_selection(self, sel):
        self._sel  = list(sel)
        self._pair = None
        self._redraw()

    def set_pair(self, pair):
        self._pair = pair
        self._redraw()

    def toggle_bg(self):
        """Invert the Cn background greyscale (does not affect overlays or crosses)."""
        self._invert_bg = not self._invert_bg
        self._redraw()

    # ── Centroid cache ────────────────────────────────────────────────────────

    def _get_centroids(self):
        """
        Return a list of (cy, cx) centroid coordinates for each component,
        in data-space (row, col) matching origin='lower'.
        Recomputed whenever the number of components changes.
        """
        s = self.store
        if self._centroids is not None and len(self._centroids) == s.n:
            return self._centroids

        d1, d2 = s.dims
        # For Fortran-order flattening: pixel index p → row = p % d1, col = p // d1
        p     = np.arange(d1 * d2)
        rows  = (p % d1).astype(np.float32)
        cols  = (p // d1).astype(np.float32)

        centroids = []
        for i in range(s.n):
            fp    = s._A[:, i]
            total = fp.sum()
            if total > 1e-9:
                cy = float((fp * rows).sum() / total)
                cx = float((fp * cols).sum() / total)
            else:
                cy, cx = d1 / 2.0, d2 / 2.0
            centroids.append((cy, cx))

        self._centroids = centroids
        return self._centroids

    # ── Cross-arm geometry ────────────────────────────────────────────────────

    def _get_cross_params(self):
        """
        Return a list of (cy, cx, arm) for each component, where:
          cy, cx  — footprint centroid in data coordinates (row, col)
          arm     — half-length of each cross arm in pixels

        arm = max(CROSS_MIN_ARM, 1.5 × max(footprint_height, footprint_width))

        The footprint bounding box is derived from pixels whose weight exceeds
        20 % of the component peak, matching the visual extents of the overlay.
        Cached and invalidated when the component count changes.
        """
        s = self.store
        # Reuse centroid cache check — both have the same invalidation condition
        if (self._centroids is not None and len(self._centroids) == s.n
                and hasattr(self, '_cross_params')
                and len(self._cross_params) == s.n):
            return self._cross_params

        centroids = self._get_centroids()   # ensures self._centroids is populated
        d1, d2    = s.dims
        params    = []
        for i in range(s.n):
            cy, cx = centroids[i]
            fp     = s.footprint(i)          # (d1, d2) in data-space (row, col)
            peak   = fp.max()
            if peak < 1e-9:
                params.append((cy, cx, CROSS_MIN_ARM))
                continue
            mask = fp > peak * 0.2
            rows_on, cols_on = np.where(mask)
            if rows_on.size == 0:
                params.append((cy, cx, CROSS_MIN_ARM))
                continue
            height = int(rows_on.max() - rows_on.min()) + 1
            width  = int(cols_on.max() - cols_on.min()) + 1
            arm    = max(CROSS_MIN_ARM, 1.5 * max(height, width))
            params.append((cy, cx, arm))

        self._cross_params = params
        return self._cross_params

    # ── Click handler ─────────────────────────────────────────────────────────

    def _on_mpl_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # In origin='lower' axes: xdata = column, ydata = row
        click_col = float(event.xdata)
        click_row = float(event.ydata)

        centroids = self._get_centroids()
        if not centroids:
            return

        # Nearest centroid by squared Euclidean distance
        dists   = [(click_row - cy) ** 2 + (click_col - cx) ** 2
                   for cy, cx in centroids]
        nearest = int(np.argmin(dists))

        ctrl = "control" in (event.modifiers or frozenset())
        self.component_clicked.emit(nearest, ctrl)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _redraw(self):
        ax = self.ax
        ax.cla()
        ax.set_facecolor("black")
        ax.set_xticks([]);  ax.set_yticks([])
        s = self.store

        bg_cmap = "gray_r" if self._invert_bg else "gray"
        ax.imshow(s.Cn, cmap=bg_cmap, origin="lower",
                  interpolation="nearest", aspect="equal")
        rgba = _build_overlay(s, self._sel, self._pair)
        ax.imshow(rgba, origin="lower", interpolation="nearest", aspect="equal")

        # Draw a cross only on currently selected / pair components.
        # When nothing is selected no crosses are drawn at all.
        sel_set  = set(self._sel)
        pair_set = set(self._pair) if self._pair else set()
        active   = sel_set | pair_set
        if active:
            cross_p = self._get_cross_params()
            for i in active:
                if i >= len(cross_p):
                    continue
                cy, cx, arm = cross_p[i]
                if self._pair and i in pair_set:
                    color = CORR_A if i == self._pair[0] else CORR_B
                else:
                    k     = self._sel.index(i) if i in sel_set else 0
                    color = PALETTE_HEX[k % len(PALETTE_HEX)]
                # Horizontal arm (x = col axis, y = row axis in data space)
                ax.plot([cx - arm, cx + arm], [cy, cy],
                        color=color, lw=0.9, alpha=0.85,
                        solid_capstyle="butt")
                # Vertical arm
                ax.plot([cx, cx], [cy - arm, cy + arm],
                        color=color, lw=0.9, alpha=0.85,
                        solid_capstyle="butt")

        self.draw_idle()


# ── Trace viewer ──────────────────────────────────────────────────────────────

class TraceViewer(_Canvas):
    """Bottom-left — stacked normalised traces for selected components."""

    # Fixed axes position derived from subplots_adjust margins:
    #   left=0.05  right=0.97  →  width  = 0.92
    #   bottom=0.12  top=0.95  →  height = 0.83
    _AX_POS = [0.05, 0.12, 0.92, 0.83]   # [left, bottom, width, height]

    def __init__(self, store: ComponentStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.ax    = self.fig.add_subplot(111)
        self.ax.set_facecolor(AX_BG)
        self.ax.set_position(self._AX_POS)   # pin immediately
        self._sel  = []
        self._redraw()

    def set_selection(self, sel):
        self._sel = list(sel)
        self._redraw()

    def _redraw(self):
        ax = self.ax
        ax.cla()
        ax.set_position(self._AX_POS)   # cla() resets bbox; re-pin it
        ax.set_facecolor(AX_BG)
        s = self.store

        if not self._sel:
            ax.text(0.5, 0.5, "Select component(s)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#555555", fontsize=9)
            ax.set_xticks([]);  ax.set_yticks([])
            self.draw_idle()
            return

        try:
            T = s.C.shape[1]
            t = np.arange(T)
            offset = 0.0
            for k, i in enumerate(self._sel):
                color = PALETTE_HEX[k % len(PALETTE_HEX)]
                trace = s.C[i]
                span  = trace.max() - trace.min()
                tr    = (trace - trace.min()) / (span + 1e-9)
                ax.plot(t, tr + offset, color=color, lw=0.7, label=s.labels[i])
                offset += 1.3

            ax.set_xlim(0, T)
            ax.set_ylim(-0.3, offset)
            ax.set_xlabel("Frame", color="#888888", fontsize=7)
            ax.tick_params(colors="#666666", labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")
            ax.legend(fontsize=6, loc="upper right",
                      facecolor=BG_DARK, edgecolor="#333333",
                      labelcolor="white", framealpha=0.8)
        except Exception as exc:
            ax.cla()
            ax.set_facecolor(AX_BG)
            ax.text(0.5, 0.5, f"Draw error: {exc}",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#ff4444", fontsize=7, wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
        finally:
            self.draw_idle()   # always flush, even if plotting raised


# ── Correlation matrix ────────────────────────────────────────────────────────

class CorrMatrix(_Canvas):
    """
    Bottom-right — pairwise correlation heat-map.
    Clicking any off-diagonal cell emits pair_clicked(i, j) and draws
    orange / magenta cross-hairs and bounding boxes on that pair.
    """

    pair_clicked = pyqtSignal(int, int)

    # Fixed layout constants (in figure-fraction coordinates):
    #   image axes: [left, bottom, width, height]
    #   colorbar:   thin strip to the right, separated by a gap
    _IM_POS  = [0.08, 0.12, 0.78, 0.84]   # image axes position
    _CB_POS  = [0.89, 0.12, 0.02, 0.84]   # colorbar axes position (fixed, never moves)

    def __init__(self, store: ComponentStore, parent=None):
        super().__init__(parent)
        self.store      = store
        self._highlight = None

        # Create both axes once at fixed positions.
        # Using cax= in colorbar() keeps the image axes bbox stable —
        # fig.colorbar(ax=ax) steals space from ax on every call and
        # causes the plot to shrink with each redraw.
        self.ax    = self.fig.add_axes(self._IM_POS)
        self._cbax = self.fig.add_axes(self._CB_POS)
        self.ax.set_facecolor(BG_DARK)
        self._cbax.set_facecolor(BG_DARK)

        self.mpl_connect("button_press_event", self._on_click)
        self._redraw()

    def refresh(self):
        self._highlight = None
        self._redraw()

    def _redraw(self):
        ax = self.ax
        ax.cla()
        self._cbax.cla()   # clear but keep the axes; its position is fixed
        s    = self.store
        corr = s.corr.copy()

        # Auto colour axis: symmetric around zero, clipped to [-1, 1].
        # Use the 99th-percentile absolute off-diagonal value so a single
        # perfectly-correlated pair does not collapse the dynamic range.
        off_diag = corr[~np.isnan(corr)]
        clim = float(np.clip(
            np.percentile(np.abs(off_diag), 99), 0.05, 1.0
        )) if off_diag.size > 0 else 1.0

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-clim, vmax=clim,
                       origin="upper", aspect="auto",
                       interpolation="nearest")

        # cax= draws into our pre-allocated axes — image axes bbox never changes
        cb = self.fig.colorbar(im, cax=self._cbax)
        cb.ax.tick_params(colors="#888888", labelsize=6)
        cb.outline.set_edgecolor("#333333")

        if self._highlight:
            hi, hj = self._highlight
            for band, col in [(hi, CORR_A), (hj, CORR_B)]:
                ax.axhline(band, color=col, lw=1.0, alpha=0.6)
                ax.axvline(band, color=col, lw=1.0, alpha=0.6)
            ax.add_patch(Rectangle((hj - 0.5, hi - 0.5), 1, 1,
                                   lw=2.0, edgecolor=CORR_A, facecolor="none"))
            ax.add_patch(Rectangle((hi - 0.5, hj - 0.5), 1, 1,
                                   lw=2.0, edgecolor=CORR_B, facecolor="none"))

        n    = s.n
        step = max(1, n // 12)
        tks  = list(range(0, n, step))
        lbls = [s.labels[t] for t in tks]
        ax.set_xticks(tks)
        ax.set_xticklabels(lbls, rotation=45, ha="right",
                           fontsize=5, color="#aaaaaa")
        ax.set_yticks(tks)
        ax.set_yticklabels(lbls, fontsize=5, color="#aaaaaa")
        ax.tick_params(colors="#666666")
        self.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        j = int(round(event.xdata))
        i = int(round(event.ydata))
        n = self.store.n
        if 0 <= i < n and 0 <= j < n and i != j:
            self._highlight = (i, j)
            self._redraw()
            self.pair_clicked.emit(i, j)



# ── Distance matrix ───────────────────────────────────────────────────────────

class DistMatrix(_Canvas):
    """
    Top-right pane — pairwise Euclidean distance between component centroids.

    The matrix is symmetric; the diagonal is NaN.  Colour scale auto-adjusts
    to the 99th percentile of off-diagonal values each redraw.

    A linear / log10 toggle is available via set_log(bool).  In log mode zero
    distances (identical centroids) are shown as NaN (white) to avoid -inf.
    Clicking a cell highlights that pair in orange / magenta on the cell viewer,
    identical to the correlation matrix click behaviour.

    The matrix is recomputed from scratch whenever store.n changes (after any
    merge, delete, undo, or redo), because the centroid list is invalidated at
    that point.
    """

    pair_clicked = pyqtSignal(int, int)

    # Fixed layout — mirrors CorrMatrix proportions
    _IM_POS = [0.08, 0.12, 0.78, 0.84]
    _CB_POS = [0.89, 0.12, 0.02, 0.84]

    def __init__(self, store: ComponentStore, parent=None):
        super().__init__(parent)
        self.store      = store
        self._highlight = None
        self._log       = False

        self.ax    = self.fig.add_axes(self._IM_POS)
        self._cbax = self.fig.add_axes(self._CB_POS)
        self.ax.set_facecolor(BG_DARK)
        self._cbax.set_facecolor(BG_DARK)

        self.mpl_connect("button_press_event", self._on_click)
        self._redraw()

    # ── Public interface ──────────────────────────────────────────────────────

    def refresh(self):
        """Full redraw — call after any store mutation."""
        self._highlight = None
        self._redraw()

    def set_log(self, log: bool):
        """Switch between linear and log10 colour scale."""
        if log != self._log:
            self._log = log
            self._redraw()

    # ── Geometry ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_dist(store: ComponentStore) -> np.ndarray:
        """
        Return (K, K) symmetric matrix of Euclidean centroid distances in px.
        Diagonal is NaN.  Uses the footprint centroid (weighted centroid of
        the spatial component A) in (row, col) data space.
        """
        s   = store
        K   = s.n
        mat = np.full((K, K), np.nan, dtype=np.float32)
        if K == 0:
            return mat

        d1, d2 = s.dims
        p      = np.arange(d1 * d2, dtype=np.float32)
        rows   = p % d1
        cols   = p // d1

        cy_cx = np.empty((K, 2), dtype=np.float32)
        for i in range(K):
            fp    = s._A[:, i]
            total = fp.sum()
            if total > 1e-9:
                cy_cx[i, 0] = (fp * rows).sum() / total
                cy_cx[i, 1] = (fp * cols).sum() / total
            else:
                cy_cx[i, 0] = d1 / 2.0
                cy_cx[i, 1] = d2 / 2.0

        for i in range(K):
            for j in range(i + 1, K):
                d = float(np.sqrt(((cy_cx[i] - cy_cx[j]) ** 2).sum()))
                mat[i, j] = d
                mat[j, i] = d
        return mat

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _redraw(self):
        ax = self.ax
        ax.cla()
        self._cbax.cla()

        s    = self.store
        dist = self._compute_dist(s)

        if self._log:
            # log10; zero distances (identical centroids) → NaN (shown as white)
            with np.errstate(divide="ignore", invalid="ignore"):
                disp = np.where(dist > 0, np.log10(dist), np.nan)
            cb_label = "log₁₀ distance (px)"
        else:
            disp     = dist
            cb_label = "Distance (px)"

        off = disp[~np.isnan(disp)]
        if off.size > 0:
            vmax = float(np.clip(np.percentile(off, 99), 1e-3, None))
            vmin = float(off.min()) if self._log else 0.0
        else:
            vmin, vmax = 0.0, 1.0

        im = ax.imshow(disp, cmap="viridis_r", vmin=vmin, vmax=vmax,
                       origin="upper", aspect="auto",
                       interpolation="nearest")
        cb = self.fig.colorbar(im, cax=self._cbax)
        cb.set_label(cb_label, color="#aaaaaa", fontsize=5, labelpad=4)
        cb.ax.tick_params(colors="#888888", labelsize=6)
        cb.outline.set_edgecolor("#333333")

        if self._highlight:
            hi, hj = self._highlight
            for band, col in [(hi, CORR_A), (hj, CORR_B)]:
                ax.axhline(band, color=col, lw=1.0, alpha=0.6)
                ax.axvline(band, color=col, lw=1.0, alpha=0.6)
            ax.add_patch(Rectangle((hj - 0.5, hi - 0.5), 1, 1,
                                   lw=2.0, edgecolor=CORR_A, facecolor="none"))
            ax.add_patch(Rectangle((hi - 0.5, hj - 0.5), 1, 1,
                                   lw=2.0, edgecolor=CORR_B, facecolor="none"))

        n    = s.n
        step = max(1, n // 12)
        tks  = list(range(0, n, step))
        lbls = [s.labels[t] for t in tks]
        ax.set_xticks(tks)
        ax.set_xticklabels(lbls, rotation=45, ha="right",
                           fontsize=5, color="#aaaaaa")
        ax.set_yticks(tks)
        ax.set_yticklabels(lbls, fontsize=5, color="#aaaaaa")
        ax.tick_params(colors="#666666")
        ax.set_title("Centroid distance", color="#888888",
                     fontsize=6, pad=3)
        self.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        j = int(round(event.xdata))
        i = int(round(event.ydata))
        n = self.store.n
        if 0 <= i < n and 0 <= j < n and i != j:
            self._highlight = (i, j)
            self._redraw()
            self.pair_clicked.emit(i, j)

# ── Component table ───────────────────────────────────────────────────────────

class ComponentTable(QTableWidget):
    """
    Left panel — one row per component with quality metrics.
    Selected rows are lightly tinted with their palette colour.
    A corr-pair click tints those two rows in orange / magenta.
    """

    _FIXED    = ("Label", "Max A", "Peak ΔF")
    _OPTIONAL = (("SNR",   "SNR_comp"),
                 ("r_val", "r_values"),
                 ("CNN",   "cnn_preds"))

    def __init__(self, store: ComponentStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(False)
        self.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setDefaultSectionSize(20)
        f = QFont()
        f.setPointSize(8)
        self.setFont(f)
        self._populate()

    # ── Public ────────────────────────────────────────────────────────────────

    def refresh(self):
        self._populate()

    def tint_selection(self, sel: list):
        """Palette tint on selected rows; dark reset on others.
        Note: no blockSignals here — setBackground emits itemChanged,
        which is not connected to any slot, so it is safe to let it fire.
        Blocking table signals here was causing itemSelectionChanged to be
        re-emitted after blockSignals(False) in PyQt6, overwriting _sel.
        """
        dark = QColor(30, 30, 30)
        for row in range(self.store.n):
            if row in sel:
                k  = sel.index(row)
                hx = PALETTE_HEX[k % len(PALETTE_HEX)].lstrip("#")
                bg = QColor(int(hx[0:2], 16),
                            int(hx[2:4], 16),
                            int(hx[4:6], 16), 55)
            else:
                bg = dark
            self._set_row_bg(row, bg)

    def tint_pair(self, i: int, j: int):
        """Orange / magenta tint for a corr-clicked pair.
        Note: no blockSignals — see tint_selection for rationale.
        """
        dark = QColor(30, 30, 30)
        for row in range(self.store.n):
            if   row == i: bg = QColor(255, 140,   0, 80)
            elif row == j: bg = QColor(255,   0, 255, 80)
            else:          bg = dark
            self._set_row_bg(row, bg)

    # ── Private ───────────────────────────────────────────────────────────────

    def _col_names(self):
        cols = list(self._FIXED)
        for label, attr in self._OPTIONAL:
            if getattr(self.store, attr, None) is not None:
                cols.append(label)
        return cols

    def _populate(self):
        s    = self.store
        cols = self._col_names()
        self.blockSignals(True)
        self.clear()
        self.setColumnCount(len(cols))
        self.setHorizontalHeaderLabels(cols)
        self.setRowCount(s.n)
        dark = QColor(30, 30, 30)
        for i in range(s.n):
            vals = [s.labels[i],
                    f"{s.max_a(i):.3f}",
                    f"{s.peak_df(i):.1f}"]
            for _, attr in self._OPTIONAL:
                if getattr(s, attr, None) is not None:
                    vals.append(f"{getattr(s, attr)[i]:.3f}")
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setBackground(QBrush(dark))
                self.setItem(i, c, item)
        self.blockSignals(False)

    def _set_row_bg(self, row: int, color: QColor):
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                item.setBackground(QBrush(color))


# ── Main window ───────────────────────────────────────────────────────────────

class InspectorWindow(QMainWindow):

    def __init__(self, store: ComponentStore):
        super().__init__()
        self.store = store
        self.setWindowTitle("CaImAn Component Inspector")
        self.resize(1760, 980)
        self._pair = None

        _apply_dark_palette()
        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._wire_signals()
        self._full_refresh()

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        for label, shortcut, slot in [
            ("&Open…",      "Ctrl+O",       self._open_file),
            ("&Save",       "Ctrl+S",       self._save_file),
            ("Save &As…",   "Ctrl+Shift+S", self._save_as_file),
        ]:
            a = QAction(label, self)
            a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            fm.addAction(a)
        fm.addSeparator()
        a = QAction("&Quit", self)
        a.setShortcut(QKeySequence("Ctrl+Q"))
        a.triggered.connect(self.close)
        fm.addAction(a)

        vm = mb.addMenu("&View")
        a  = QAction("&Refresh", self)
        a.setShortcut(QKeySequence("F5"))
        a.triggered.connect(self._full_refresh)
        vm.addAction(a)

        em = mb.addMenu("&Edit")
        self.act_undo = QAction("↩  Undo", self)
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_undo.setEnabled(False)
        self.act_undo.triggered.connect(self._do_undo)
        em.addAction(self.act_undo)
        self.act_redo = QAction("↪  Redo", self)
        self.act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self.act_redo.setEnabled(False)
        self.act_redo.triggered.connect(self._do_redo)
        em.addAction(self.act_redo)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        self.act_merge = QAction("⊕  Merge selected", self)
        self.act_merge.setToolTip(
            "Merge ≥ 2 selected components.\n"
            "Spatial footprint = pixel-wise maximum.\n"
            "Temporal trace    = mean of selected traces.\n"
            "Merged component is appended at end of the list.")
        self.act_merge.setEnabled(False)
        self.act_merge.triggered.connect(self._do_merge)
        tb.addAction(self.act_merge)

        self.act_delete = QAction("✕  Delete selected", self)
        self.act_delete.setToolTip("Permanently remove selected components.")
        self.act_delete.setEnabled(False)
        self.act_delete.triggered.connect(self._do_delete)
        tb.addAction(self.act_delete)

        tb.addSeparator()

        self.act_undo_tb = QAction("↩  Undo", self)
        self.act_undo_tb.setToolTip("Nothing to undo")
        self.act_undo_tb.setEnabled(False)
        self.act_undo_tb.triggered.connect(self._do_undo)
        tb.addAction(self.act_undo_tb)

        self.act_redo_tb = QAction("↪  Redo", self)
        self.act_redo_tb.setToolTip("Nothing to redo")
        self.act_redo_tb.setEnabled(False)
        self.act_redo_tb.triggered.connect(self._do_redo)
        tb.addAction(self.act_redo_tb)

        tb.addSeparator()

        self._sel_lbl = QLabel("  0 selected  |  0 total")
        self._sel_lbl.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        tb.addWidget(self._sel_lbl)

        tb.addSeparator()

        self.act_invert_bg = QAction("⬛  Invert BG", self)
        self.act_invert_bg.setToolTip(
            "Invert the cell viewer background greyscale.\n"
            "Overlays and crosses are not affected.")
        self.act_invert_bg.setCheckable(True)
        self.act_invert_bg.setChecked(False)
        self.act_invert_bg.triggered.connect(
            lambda: self.cell_view.toggle_bg())
        tb.addAction(self.act_invert_bg)

        self.act_dist_log = QAction("log₁₀  Dist", self)
        self.act_dist_log.setToolTip(
            "Toggle distance matrix between linear and log\u2081\u2080 scale.\n"
            "Log scale compresses the large range between close\n"
            "and far component pairs.")
        self.act_dist_log.setCheckable(True)
        self.act_dist_log.setChecked(False)
        self.act_dist_log.triggered.connect(
            lambda checked: self.dist_view.set_log(checked))
        tb.addAction(self.act_dist_log)

    # ── Central layout ────────────────────────────────────────────────────────

    def _build_central(self):
        s = self.store

        self.table      = ComponentTable(s)
        self.cell_view  = CellViewer(s)
        self.dist_view  = DistMatrix(s)
        self.trace_view = TraceViewer(s)
        self.corr_view  = CorrMatrix(s)

        # Top row: cell viewer (left) | distance matrix (right)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.cell_view)
        top.addWidget(self.dist_view)
        top.setSizes([1, 1])

        # Bottom row: trace (left) | corr (right)
        bot = QSplitter(Qt.Orientation.Horizontal)
        bot.addWidget(self.trace_view)
        bot.addWidget(self.corr_view)
        bot.setSizes([1, 1])

        # Right column: top row | bottom row
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(top)
        right.addWidget(bot)
        right.setSizes([3, 2])

        # Root: table (left, narrow) | right column
        root = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.table)
        root.addWidget(right)
        root.setSizes([1, 5])

        self.setCentralWidget(root)
        self.statusBar().showMessage(f"Loaded {s.n} components")

    # ── Signals ───────────────────────────────────────────────────────────────

    def _wire_signals(self):
        self.table.itemSelectionChanged.connect(self._on_table_sel)
        self.corr_view.pair_clicked.connect(self._on_corr_click)
        self.dist_view.pair_clicked.connect(self._on_corr_click)
        self.cell_view.component_clicked.connect(self._on_cell_click)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_table_sel(self):
        # Re-entrancy guard: tint_selection calls setBackground() which emits
        # itemChanged on the model. In PyQt6, this can indirectly cause the
        # selection model to re-emit, triggering a second call to this slot
        # with an empty selection that would wipe out the trace viewer.
        if getattr(self, "_in_table_sel", False):
            return
        self._in_table_sel = True
        try:
            sel = sorted({idx.row() for idx in self.table.selectedIndexes()})
            self.cell_view.set_selection(sel)
            self.trace_view.set_selection(sel)
            self.table.tint_selection(sel)
            n = len(sel)
            self.act_merge.setEnabled(n >= 2)
            self.act_delete.setEnabled(n >= 1)
            self._sel_lbl.setText(f"  {n} selected  |  {self.store.n} total")
        finally:
            self._in_table_sel = False

    def _on_corr_click(self, i: int, j: int):
        self._pair = (i, j)
        self.cell_view.set_pair((i, j))
        self.trace_view.set_selection([i, j])   # show both traces
        self.table.tint_pair(i, j)
        r = self.store.corr[i, j]
        self.statusBar().showMessage(
            f"Pair  {self.store.labels[i]}  ↔  {self.store.labels[j]}   "
            f"r = {r:+.4f}")

    def _on_cell_click(self, idx: int, ctrl: bool):
        """
        Handle a click on a component footprint in the cell viewer.

        Without Ctrl  → select *only* the clicked component.
        With Ctrl     → toggle the clicked component in/out of the current
                        multi-selection, leaving all other selected rows intact.

        The selection is applied programmatically to the table (so it is
        the single source of truth), then _on_table_sel is called manually
        to propagate the change to all dependent views.
        """
        cur = sorted({i.row() for i in self.table.selectedIndexes()})
        if ctrl:
            if idx in cur:
                new_sel = [r for r in cur if r != idx]
            else:
                new_sel = sorted(cur + [idx])
        else:
            new_sel = [idx]

        # Apply to table selection model without triggering the signal
        # (we call _on_table_sel manually below)
        sm = self.table.selectionModel()
        self.table.blockSignals(True)
        sm.clearSelection()
        flags = (QItemSelectionModel.SelectionFlag.Select
                 | QItemSelectionModel.SelectionFlag.Rows)
        for row in new_sel:
            sm.select(self.table.model().index(row, 0), flags)
        self.table.blockSignals(False)

        # Scroll to the last-selected row so it is visible
        if new_sel:
            self.table.scrollToItem(
                self.table.item(new_sel[-1], 0),
                QAbstractItemView.ScrollHint.EnsureVisible)

        # Propagate to all views
        self._on_table_sel()
        self.statusBar().showMessage(
            f"Selected  {self.store.labels[idx]}  "
            f"({'+ ' if ctrl and len(new_sel) > 1 else ''}"
            f"{len(new_sel)} component{'s' if len(new_sel) != 1 else ''})")

    # ── Merge / Delete ────────────────────────────────────────────────────────

    def _do_merge(self):
        sel = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if len(sel) < 2:
            return
        n_sel   = len(sel)
        new_idx = self.store.merge(sel)
        self._full_refresh()
        self.table.blockSignals(True)
        self.table.selectRow(new_idx)
        self.table.scrollToItem(
            self.table.item(new_idx, 0),
            QAbstractItemView.ScrollHint.EnsureVisible)
        self.table.blockSignals(False)
        self._on_table_sel()
        self.statusBar().showMessage(
            f"Merged {n_sel} → {self.store.labels[new_idx]}  "
            f"({self.store.n} components)")

    def _do_delete(self):
        sel = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not sel:
            return
        n_del = len(sel)
        self.store.delete(sel)
        self._full_refresh()
        self.statusBar().showMessage(
            f"Deleted {n_del}  ({self.store.n} remaining)")

    # ── Undo / Redo ───────────────────────────────────────────────────────────

    def _do_undo(self):
        msg = self.store.undo()
        self._full_refresh()
        self.statusBar().showMessage(msg)

    def _do_redo(self):
        msg = self.store.redo()
        self._full_refresh()
        self.statusBar().showMessage(msg)

    def _update_history_actions(self):
        """Sync enabled state and tooltips of all four undo/redo actions."""
        s = self.store
        for act in (self.act_undo, self.act_undo_tb):
            act.setEnabled(s.can_undo)
            act.setToolTip(s.undo_description)
        for act in (self.act_redo, self.act_redo_tb):
            act.setEnabled(s.can_redo)
            act.setToolTip(s.redo_description)

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CaImAn results", "",
            "HDF5 files (*.hdf5 *.h5);;All files (*)")
        if not path:
            return
        try:
            store = load_from_hdf5(path)
            self._swap_store(store)
            self.statusBar().showMessage(
                f"Loaded {store.n} components from {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            import traceback; traceback.print_exc()

    def _save_file(self):
        if self.store.fpath:
            path = ComponentStore.curated_path(self.store.fpath)
            self._do_save(path)
        else:
            self._save_as_file()

    def _save_as_file(self):
        # Pre-fill the dialog with the _curated path from the source file
        default = (ComponentStore.curated_path(self.store.fpath)
                   if self.store.fpath else "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save curated CaImAn results", default,
            "HDF5 files (*.hdf5 *.h5)")
        if path:
            self._do_save(path)

    def _do_save(self, path: str):
        try:
            self.store.save(path)
            self.statusBar().showMessage(f"Saved → {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _swap_store(self, new_store: ComponentStore):
        self.store = new_store
        for w in (self.table, self.cell_view,
                  self.trace_view, self.corr_view,
                  self.dist_view):
            w.store = new_store
        self._full_refresh()

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _full_refresh(self):
        self.table.blockSignals(True)
        self.table.refresh()
        self.table.clearSelection()
        self.table.blockSignals(False)
        self.cell_view.set_selection([])
        self.trace_view.set_selection([])
        self.corr_view.refresh()
        self.dist_view.refresh()
        self.act_merge.setEnabled(False)
        self.act_delete.setEnabled(False)
        self._sel_lbl.setText(f"  0 selected  |  {self.store.n} total")
        self.statusBar().showMessage(f"{self.store.n} components")
        self._update_history_actions()


# ── Dark palette (module-level so it can be called before QMainWindow) ────────

def _apply_dark_palette():
    app = QApplication.instance()
    if app is None:
        return
    app.setStyle("Fusion")
    p  = QPalette()
    CR = QPalette.ColorRole
    for role, rgb in [
        (CR.Window,          (28,  28,  28)),
        (CR.WindowText,      (210, 210, 210)),
        (CR.Base,            (18,  18,  18)),
        (CR.AlternateBase,   (32,  32,  32)),
        (CR.Text,            (210, 210, 210)),
        (CR.Button,          (45,  45,  45)),
        (CR.ButtonText,      (210, 210, 210)),
        (CR.Highlight,       (55,  95,  155)),
        (CR.HighlightedText, (255, 255, 255)),
        (CR.ToolTipBase,     (40,  40,  40)),
        (CR.ToolTipText,     (210, 210, 210)),
    ]:
        p.setColor(role, QColor(*rgb))
    app.setPalette(p)


# ── Demo data ─────────────────────────────────────────────────────────────────

def _make_demo_store() -> ComponentStore:
    print("[demo]  No file specified — running with synthetic data.")
    print("[demo]  Injected duplicate pair: C003 ≈ C005")
    rng  = np.random.default_rng(42)
    dims = (100, 100)
    K, T = 35, 2000
    d    = dims[0] * dims[1]

    A = np.zeros((d, K), dtype=np.float32)
    for k in range(K):
        cy, cx = rng.integers(8, 92, size=2)
        y, x   = np.ogrid[:dims[0], :dims[1]]
        sigma  = rng.uniform(4, 9)
        mask   = (y - cy) ** 2 + (x - cx) ** 2 < sigma ** 2
        A[mask.ravel(order="F"), k] = rng.uniform(
            0.5, 1.0, mask.sum()).astype(np.float32)

    C    = rng.random((K, T)).astype(np.float32)
    C[5] = C[3] + rng.random(T).astype(np.float32) * 0.03   # duplicate pair
    snr  = np.abs(rng.normal(3.0, 1.5, K)).astype(np.float32)
    rval = np.clip(rng.normal(0.7, 0.2, K), -1, 1).astype(np.float32)
    cnn  = np.clip(rng.normal(0.75, 0.15, K), 0, 1).astype(np.float32)
    Cn   = A.max(axis=1).reshape(dims, order="F")

    return ComponentStore(A, C, Cn, dims,
                          SNR_comp=snr, r_values=rval, cnn_preds=cnn)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CaImAn Component Inspector (Qt6 / PyQt6)")
    parser.add_argument("results", nargs="?",
                        help="Path to a CaImAn .hdf5 results file")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    _apply_dark_palette()

    if args.results:
        try:
            store = load_from_hdf5(args.results)
            print(f"Loaded {store.n} components from {args.results}")
        except Exception as exc:
            print(f"Error loading '{args.results}': {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        store = _make_demo_store()

    win = InspectorWindow(store)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
