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

TRACE_DENOISED = "C (denoised)"   # canonical trace key; this one is saved back

import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QTableWidget,
    QTableWidgetItem, QLabel, QFileDialog, QComboBox,
    QMessageBox, QSizePolicy, QAbstractItemView, QHeaderView,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QSize, QItemSelectionModel
from PyQt6.QtGui   import (
    QAction, QColor, QBrush, QFont, QPalette, QKeySequence,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QSize, QItemSelectionModel

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
CROSS_MIN_ARM = 25    # minimum cross arm half-length in pixels

BG_DARK  = "#111111"
AX_BG    = "#0d0d0d"



os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")
pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)

# ── Colour maps ───────────────────────────────────────────────────────────────

def _cmap_rdbu_r():
    """Red-white-blue diverging, red = +1, blue = -1."""
    stops  = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [(  5,  48,  97, 255),
              ( 67, 147, 195, 255),
              (247, 247, 247, 255),
              (214,  96,  77, 255),
              (103,   0,  31, 255)]
    return pg.ColorMap(pos=stops, color=colors)

def _cmap_viridis_r():
    """Reversed viridis: yellow (near) → purple (far)."""
    stops  = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [(253, 231,  37, 255),
              ( 94, 201,  98, 255),
              ( 33, 144, 141, 255),
              ( 59,  82, 139, 255),
              ( 68,   1,  84, 255)]
    return pg.ColorMap(pos=stops, color=colors)

CMAP_RDBU_R    = _cmap_rdbu_r()
CMAP_VIRIDIS_R = _cmap_viridis_r()


# ── RGBA overlay builder (pyqtgraph version) ──────────────────────────────────

def _build_overlay(store: 'ComponentStore',
                   sel: list, pair) -> np.ndarray:
    """
    Return (d1, d2, 4) RGBA uint8 composite of selected/pair footprints.
    Unselected components are skipped entirely.
    """
    d1, d2  = store.dims
    out     = np.zeros((d1, d2, 4), dtype=np.float32)
    sel_set  = set(sel)
    pair_set = set(pair) if pair else set()

    for i in range(store.n):
        if pair and i in pair_set:
            hex_c = CORR_A if i == pair[0] else CORR_B
            alpha = ALPHA_SEL
        elif i in sel_set:
            hex_c = PALETTE_HEX[sel.index(i) % len(PALETTE_HEX)]
            alpha = ALPHA_SEL
        else:
            continue

        fp   = store.footprint(i)
        peak = fp.max()
        if peak < 1e-9:
            continue
        norm = fp / peak

        h  = hex_c.lstrip('#')
        r  = int(h[0:2], 16) / 255.0
        g  = int(h[2:4], 16) / 255.0
        b  = int(h[4:6], 16) / 255.0
        am = norm * alpha
        out[..., 0] = np.maximum(out[..., 0], r * am)
        out[..., 1] = np.maximum(out[..., 1], g * am)
        out[..., 2] = np.maximum(out[..., 2], b * am)
        out[..., 3] = np.maximum(out[..., 3], am)

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)



# ── Data model ────────────────────────────────────────────────────────────────

class ComponentStore:
    """
    Mutable container for CNMF component data.
    A is stored as dense float32 (d × K).
    """

    def __init__(self, A, C, Cn, dims,
                 SNR_comp=None, r_values=None, cnn_preds=None,
                 cnm_obj=None, fpath=None, traces=None):
        self._A = _to_dense(A)
        self.dims = dims

        # ── Temporal traces (display registry) ───────────────────────────────
        # Every entry is a (K, T) array.  TRACE_DENOISED is the canonical
        # trace written back on save; the others (ΔF/F, raw, deconvolved) are
        # for display only and are selected via the trace dropdown.  self.C
        # always mirrors the denoised entry so save / correlation / merge math
        # are unaffected by the display choice.
        if traces is None:
            traces = {TRACE_DENOISED: np.asarray(C, dtype=np.float32)}
        self.traces = {k: np.asarray(v, dtype=np.float32) for k, v in traces.items()}
        if TRACE_DENOISED not in self.traces:
            self.traces[TRACE_DENOISED] = np.asarray(C, dtype=np.float32)
        self.disp_key = TRACE_DENOISED
        self.C = self.traces[TRACE_DENOISED]

        # ── Background images ────────────────────────────────────────────────
        # `Cn` is the *active* greyscale shown behind the footprints.  A named
        # registry holds every available image so the user can switch live
        # (correlation image, footprint max-projection, a loaded data
        # projection, etc).  self.Cn always mirrors the active entry so the
        # rest of the code can keep reading store.Cn unchanged.
        self.backgrounds: dict = {}
        self.bg_key = None
        self.add_background("correlation",
                            np.asarray(Cn, dtype=np.float32), make_active=True)
        fp_max = self._A.max(axis=1).reshape(dims, order="F")
        self.add_background("footprint max", fp_max)
        # If the correlation image is (near-)flat — e.g. SUPPORT-denoised data
        # saturates Cn to ~1.0 — fall back to the footprint projection, which
        # at least reveals where the components sit.
        if not _has_contrast(self.Cn):
            self.set_background("footprint max")

        self.SNR_comp  = _opt_arr(SNR_comp)
        self.r_values  = _opt_arr(r_values)
        self.cnn_preds = _opt_arr(cnn_preds)

        self.cnm_obj = cnm_obj
        self.fpath   = fpath
        self.labels  = [f"C{i:03d}" for i in range(self._A.shape[1])]
        self.hidden  = np.zeros(self._A.shape[1], dtype=bool)  # per-component visibility
        self._corr   = None

        # History stacks — each entry is (op_label: str, snapshot: dict)
        self._undo_stack: deque = deque(maxlen=HISTORY_MAXLEN)
        self._redo_stack: deque = deque(maxlen=HISTORY_MAXLEN)

    # ── Background images ──────────────────────────────────────────────────────

    def _coerce_bg(self, img) -> np.ndarray:
        """Coerce an arbitrary image to a 2-D float32 array matching dims.

        A 3-D stack (frames, h, w) is collapsed to a max-projection, so a raw
        movie file can be dropped in directly as an 'actual cells' background.
        A transposed image is auto-corrected.
        """
        a = np.asarray(img, dtype=np.float32).squeeze()
        if a.ndim == 3:
            a = a.max(axis=0)
        if a.ndim != 2:
            raise ValueError(f"background must be 2-D (got shape {a.shape})")
        if a.shape == tuple(self.dims):
            return np.ascontiguousarray(a)
        if a.shape == tuple(self.dims)[::-1]:
            return np.ascontiguousarray(a.T)
        raise ValueError(
            f"background shape {a.shape} does not match dims {tuple(self.dims)}")

    def add_background(self, name: str, img, make_active: bool = False) -> str:
        """Register a named background image (overwriting any same-named one)."""
        self.backgrounds[name] = self._coerce_bg(img)
        if make_active or self.bg_key is None:
            self.set_background(name)
        return name

    def set_background(self, name: str):
        if name not in self.backgrounds:
            raise KeyError(name)
        self.bg_key = name
        self.Cn = self.backgrounds[name]

    def background_names(self) -> list:
        return list(self.backgrounds.keys())

    # ── Temporal trace selection (display only) ────────────────────────────────

    def trace_names(self) -> list:
        return list(self.traces.keys())

    def set_trace(self, name: str):
        if name not in self.traces:
            raise KeyError(name)
        self.disp_key = name

    def disp_trace(self, i):
        """The currently displayed trace for component i."""
        return self.traces[self.disp_key][i]

    @property
    def disp_T(self) -> int:
        return self.traces[self.disp_key].shape[1]

    # ── Per-component visibility (view only, not saved) ────────────────────────

    def is_hidden(self, i) -> bool:
        return bool(self.hidden[i])

    def toggle_hidden(self, indices):
        for i in indices:
            self.hidden[i] = not self.hidden[i]

    def set_hidden(self, indices, flag: bool):
        for i in indices:
            self.hidden[i] = bool(flag)

    def show_all(self):
        """Make every component visible again."""
        self.hidden[:] = False

    @property
    def n_hidden(self) -> int:
        return int(self.hidden.sum())


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
            "traces":    {k: v.copy() for k, v in self.traces.items()},
            "labels":    list(self.labels),
            "hidden":    self.hidden.copy(),
            "SNR_comp":  self.SNR_comp.copy()  if self.SNR_comp  is not None else None,
            "r_values":  self.r_values.copy()  if self.r_values  is not None else None,
            "cnn_preds": self.cnn_preds.copy() if self.cnn_preds is not None else None,
        }

    def _restore(self, snap: dict):
        """Replace mutable state from a snapshot."""
        self._A        = snap["A"]
        self.traces    = snap["traces"]
        self.C         = self.traces[TRACE_DENOISED]
        self.labels    = snap["labels"]
        self.hidden    = snap.get("hidden",
                                  np.zeros(self._A.shape[1], dtype=bool))
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
        new_label = "+".join(self.labels[i] for i in idx)

        keep = [j for j in range(self.n) if j not in set(idx)]
        self._A     = np.column_stack([self._A[:, keep], new_a[:, None]])
        for name, arr in self.traces.items():
            new_row = arr[idx].mean(axis=0)
            self.traces[name] = np.vstack([arr[keep], new_row[None, :]])
        self.C      = self.traces[TRACE_DENOISED]
        self.labels = [self.labels[j] for j in keep] + [new_label]
        # Merged component is hidden only if every constituent was hidden.
        self.hidden = np.append(self.hidden[keep], bool(self.hidden[idx].all()))

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
        for name, arr in self.traces.items():
            self.traces[name] = arr[keep]
        self.C      = self.traces[TRACE_DENOISED]
        self.labels = [self.labels[j] for j in keep]
        self.hidden = self.hidden[keep]
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


def _has_contrast(img, eps: float = 1e-6) -> bool:
    """True if an image has usable greyscale contrast.

    Denoised data can saturate the correlation image to a near-constant
    value (Cn ~ 1.0); such an image makes a useless background.
    """
    a = np.asarray(img, dtype=np.float32)
    if a.size == 0 or not np.isfinite(a).any():
        return False
    finite = a[np.isfinite(a)]
    ptp = float(finite.max() - finite.min())
    return ptp > eps and float(finite.std()) > eps


def _load_image_file(path: str) -> np.ndarray:
    """Load a background image from .npy/.npz/.tif/.tiff/.png/.jpg.

    Returns the raw array (2-D, or 3-D stack to be collapsed by the store).
    Stacks are returned as-is so the caller can max-project them.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        with np.load(path) as z:
            for key in ("projection", "img", "image", "Cn", "arr_0"):
                if key in z:
                    return z[key]
            return z[z.files[0]]
    if ext in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    if ext in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        return np.asarray(Image.open(path).convert("F"))
    raise ValueError(f"unsupported background format: {ext}")


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
    import os as _os
    # Ensure CAIMAN_TEMP is set to a writable directory before load_CNMF
    # calls fn_relocated.  If the current value is missing or unwritable,
    # fall back to the directory that contains the HDF5 file.
    _ct = _os.environ.get("CAIMAN_TEMP", "")
    if not (_ct and _os.path.isdir(_ct) and _os.access(_ct, _os.W_OK)):
        _fallback = _os.path.abspath(_os.path.dirname(path)) or "."
        _os.environ["CAIMAN_TEMP"] = _fallback

    from caiman.source_extraction.cnmf.cnmf import load_CNMF
    cnm  = load_CNMF(path)
    est  = cnm.estimates
    dims = cnm.dims if (cnm.dims and all(d > 0 for d in cnm.dims)) else est.dims

    A = _to_dense(est.A)
    C = est.C

    idx = getattr(est, "idx_components", None)

    def _sel_rows(arr):
        """Slice an optional (K, T) trace array by idx_components."""
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return None
        return arr[idx] if (idx is not None and len(idx) > 0) else arr

    if idx is not None and len(idx) > 0:
        A = A[:, idx];  C = C[idx]
        snr  = est.SNR_comp[idx]  if est.SNR_comp  is not None else None
        rval = est.r_values[idx]  if est.r_values  is not None else None
        cnn  = est.cnn_preds[idx] if est.cnn_preds is not None else None
    else:
        snr  = getattr(est, "SNR_comp",  None)
        rval = getattr(est, "r_values",  None)
        cnn  = getattr(est, "cnn_preds", None)

    # Build the trace registry: denoised C is canonical; add ΔF/F, the
    # residual-added raw trace, and deconvolved spikes when available and
    # shape-compatible.  Display-only — the dropdown switches between them.
    Cd     = np.asarray(C, dtype=np.float32)
    traces = {TRACE_DENOISED: Cd}
    F_dff  = _sel_rows(getattr(est, "F_dff", None))
    YrA    = _sel_rows(getattr(est, "YrA",   None))
    S      = _sel_rows(getattr(est, "S",     None))
    if F_dff is not None and F_dff.shape == Cd.shape:
        traces["ΔF/F"] = F_dff
    if YrA is not None and YrA.shape == Cd.shape:
        traces["C + residual (raw)"] = Cd + YrA
    if S is not None and S.shape == Cd.shape and np.any(S):
        traces["S (deconvolved)"] = S

    _Cn = getattr(est, "Cn", None)
    Cn  = _Cn if _Cn is not None else A.max(axis=1).reshape(dims, order="F")
    store = ComponentStore(A, C, Cn, dims,
                           SNR_comp=snr, r_values=rval, cnn_preds=cnn,
                           cnm_obj=cnm, fpath=path, traces=traces)
    _attach_sibling_backgrounds(store, path)
    return store


def _attach_sibling_backgrounds(store: 'ComponentStore', path: str):
    """Register data-projection images sitting next to the results file.

    Companion images share the *session prefix* with the results file
    (e.g. ``<session>_mean.npy`` next to ``<session>_results.hdf5``); a small
    token vocabulary is also accepted.  Candidates must be 2-D and match the
    FOV (enforced by the store's coercion).  If the correlation image is flat
    — e.g. denoised data saturates Cn — the best-ranked projection is
    auto-activated so the viewer shows the actual cells.
    """
    import glob
    import re as _re
    folder = _os_dirname(path)
    stem   = os.path.splitext(os.path.basename(path))[0]          # <session>_results
    # Strip a trailing results/cnmf/curated tail to recover the session prefix.
    prefix = _re.sub(r'[._-]?(results?|cnmf?|cnm|estimates?|curated)$', '',
                     stem, flags=_re.I)
    tokens = ("projection", "percentile", "mean", "median",
              "max", "pnr", "summary", "anat", "template", "_cn")
    exts   = (".npy", ".npz", ".tif", ".tiff", ".png", ".jpg", ".jpeg")

    found = []   # (label, lowername)
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        if os.path.abspath(f) == os.path.abspath(path):
            continue
        low       = os.path.basename(f).lower()
        name_stem = os.path.splitext(os.path.basename(f))[0]
        if not low.endswith(exts):
            continue
        shares_prefix = bool(prefix) and name_stem.lower().startswith(prefix.lower())
        has_token     = any(tok in low for tok in tokens)
        if not (shares_prefix or has_token):
            continue
        try:
            img = _load_image_file(f)
            if np.asarray(img).squeeze().ndim != 2:   # skip stacks at auto-load
                continue
            label = name_stem
            if shares_prefix:
                label = name_stem[len(prefix):].lstrip("._- ") or name_stem
            store.add_background(label, img)           # coercion dim-checks
            found.append((label, low))
        except Exception:
            continue

    if found and not _has_contrast(store.backgrounds.get("correlation")):
        def _rank(item):
            low = item[1]
            for r, tok in enumerate(tokens):
                if tok in low:
                    return r
            return len(tokens)
        store.set_background(sorted(found, key=_rank)[0][0])


def _os_dirname(path: str) -> str:
    return os.path.abspath(os.path.dirname(path)) or "."


# ── RGBA overlay builder ──────────────────────────────────────────────────────

def _build_overlay(store: ComponentStore,
                   sel: list, pair) -> np.ndarray:
    """Return (d1, d2, 4) RGBA float32 composite of all footprints."""
    d1, d2  = store.dims
    out     = np.zeros((d1, d2, 4), dtype=np.float32)
    sel_set = set(sel)
    pair_set = set(pair) if pair else set()

    for i in range(store.n):
        if store.hidden[i]:
            continue
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
            continue   # only draw overlay for selected / pair components

        _h = hex_c.lstrip('#'); r = int(_h[0:2],16)/255.0; g = int(_h[2:4],16)/255.0; b = int(_h[4:6],16)/255.0
        a_mask  = norm * alpha
        out[..., 0] = np.maximum(out[..., 0], r * a_mask)
        out[..., 1] = np.maximum(out[..., 1], g * a_mask)
        out[..., 2] = np.maximum(out[..., 2], b * a_mask)
        out[..., 3] = np.maximum(out[..., 3], a_mask)

    return np.clip(out, 0, 1)


def _build_roi_overlay(store: ComponentStore) -> np.ndarray:
    """
    Return (d1, d2, 4) RGBA composite of *every* footprint in dim cyan.

    This is the persistent "all ROIs" base layer.  It is rebuilt from the
    store on each mutation (merge / delete / undo / redo), so deleted or
    merged components visibly disappear / appear in the cell viewer.
    """
    d1, d2 = store.dims
    out    = np.zeros((d1, d2, 4), dtype=np.float32)

    _h = CYAN.lstrip('#')
    r = int(_h[0:2], 16) / 255.0
    g = int(_h[2:4], 16) / 255.0
    b = int(_h[4:6], 16) / 255.0

    for i in range(store.n):
        if store.hidden[i]:
            continue
        fp   = store.footprint(i)
        peak = fp.max()
        if peak < 1e-9:
            continue
        a_mask = (fp / peak) * ALPHA_DEF
        out[..., 0] = np.maximum(out[..., 0], r * a_mask)
        out[..., 1] = np.maximum(out[..., 1], g * a_mask)
        out[..., 2] = np.maximum(out[..., 2], b * a_mask)
        out[..., 3] = np.maximum(out[..., 3], a_mask)

    return np.clip(out, 0, 1)




# ── Cell viewer ───────────────────────────────────────────────────────────────

class CellViewer(pg.GraphicsLayoutWidget):
    """
    Top-left pane — Cn greyscale background with RGBA footprint overlay
    and per-component crosses on selected / pair components.

    Interaction (pyqtgraph built-in)
    ---------------------------------
      Right-drag     → pan
      Scroll wheel   → zoom centred on cursor
      Right-click    → context menu (includes "View All" / reset zoom)
      Left-click     → select component under cursor (footprint hit-test)
      Ctrl+Left      → toggle component in/out of selection
    """

    component_clicked = pyqtSignal(int, bool)

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(parent)
        self.store      = store
        self._sel       = []
        self._pair      = None
        self._centroids = None
        self._cross_params_cache = None
        self._invert_bg = False

        self.setBackground(BG_DARK)

        # ViewBox: aspect locked, Y not inverted → row 0 at bottom (origin='lower')
        self.vb = self.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.vb.invertY(False)

        # Background (Cn)
        self._img_bg = pg.ImageItem()
        self._img_bg.setZValue(0)
        self.vb.addItem(self._img_bg)

        # Persistent "all ROIs" layer (dim cyan, every footprint)
        self._show_all = True
        self._img_all  = pg.ImageItem()
        self._img_all.setZValue(0.5)
        self.vb.addItem(self._img_all)

        # Footprint overlay (RGBA) — selected / pair only, on top
        self._img_ov = pg.ImageItem()
        self._img_ov.setZValue(1)
        self.vb.addItem(self._img_ov)

        # Cross lines — managed as a list of PlotDataItems
        self._cross_items: list = []

        # Click detection via scene
        self.vb.scene().sigMouseClicked.connect(self._on_scene_click)

        self.refresh_rois()
        self._redraw()

    # ── Public interface ──────────────────────────────────────────────────────

    def set_selection(self, sel):
        self._sel  = list(sel)
        self._pair = None
        self._redraw()

    def set_pair(self, pair):
        self._pair = pair
        self._redraw()

    def refresh_rois(self):
        """
        Rebuild the persistent "all ROIs" base layer from the current store.

        Must be called after any component mutation (merge / delete / undo /
        redo / store swap) so the cell viewer reflects the new component set.
        Also drops the centroid / cross-arm caches, which are keyed on
        component count and would otherwise go stale when undo / redo restores
        a state with a coincidentally equal count.
        """
        self._centroids          = None
        self._cross_params_cache = None
        if self._show_all:
            self._img_all.setImage(_build_roi_overlay(self.store))
            self._img_all.setVisible(True)
        else:
            self._img_all.setVisible(False)

    def set_show_all(self, flag: bool):
        self._show_all = bool(flag)
        self.refresh_rois()

    def refresh_background(self):
        """Redraw after the store's active background image has changed."""
        self._redraw()

    def refresh_visibility(self):
        """Redraw after per-component visibility (hidden flags) has changed."""
        self.refresh_rois()
        self._redraw()


    def toggle_bg(self):
        self._invert_bg = not self._invert_bg
        self._redraw()

    def reset_zoom(self):
        self.vb.autoRange()

    # ── Scene click → component selection ────────────────────────────────────

    def _on_scene_click(self, event):
        if event.double():
            self.reset_zoom()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pt  = self.vb.mapSceneToView(event.scenePos())
        col = int(round(pt.x()))
        row = int(round(pt.y()))

        s   = self.store
        d1, d2 = s.dims
        if not (0 <= row < d1 and 0 <= col < d2):
            return

        best_idx   = None
        best_value = 0.0
        for i in range(s.n):
            fp   = s.footprint(i)
            peak = fp.max()
            if peak < 1e-9:
                continue
            val = float(fp[row, col])
            if val >= peak * 0.10 and val > best_value:
                best_value = val
                best_idx   = i

        if best_idx is None:
            return

        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self.component_clicked.emit(best_idx, ctrl)

    # ── Centroid / cross-arm caches ───────────────────────────────────────────

    def _get_centroids(self):
        s = self.store
        if self._centroids is not None and len(self._centroids) == s.n:
            return self._centroids

        d1, d2 = s.dims
        p      = np.arange(d1 * d2)
        rows   = (p % d1).astype(np.float32)
        cols   = (p // d1).astype(np.float32)

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

    def _get_cross_params(self):
        s = self.store
        if (self._centroids is not None and len(self._centroids) == s.n
                and self._cross_params_cache is not None
                and len(self._cross_params_cache) == s.n):
            return self._cross_params_cache

        centroids = self._get_centroids()
        params    = []
        for i in range(s.n):
            cy, cx = centroids[i]
            fp     = s.footprint(i)
            peak   = fp.max()
            if peak < 1e-9:
                params.append((cy, cx, CROSS_MIN_ARM))
                continue
            mask     = fp > peak * 0.2
            ry, cx_ = np.where(mask)
            if ry.size == 0:
                params.append((cy, cx, CROSS_MIN_ARM))
                continue
            h   = int(ry.max() - ry.min()) + 1
            w   = int(cx_.max() - cx_.min()) + 1
            arm = max(CROSS_MIN_ARM, 0.75 * max(h, w))
            params.append((cy, cx, arm))

        self._cross_params_cache = params
        return self._cross_params_cache

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _redraw(self):
        s = self.store

        # Background
        cn = s.Cn.copy()
        if self._invert_bg:
            cn = cn.max() - cn
        lo, hi = cn.min(), cn.max()
        if hi > lo:
            cn8 = ((cn - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            cn8 = np.zeros_like(cn, dtype=np.uint8)
        self._img_bg.setImage(cn8)

        # RGBA overlay (selected + pair only)
        rgba = _build_overlay(s, self._sel, self._pair)
        self._img_ov.setImage(rgba)

        # Crosses
        self._draw_crosses()

    def _draw_crosses(self):
        # Remove old cross line items
        for item in self._cross_items:
            self.vb.removeItem(item)
        self._cross_items = []

        sel_set  = set(self._sel)
        pair_set = set(self._pair) if self._pair else set()
        active   = {i for i in (sel_set | pair_set)
                    if not self.store.hidden[i]}
        if not active:
            return

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

            pen = pg.mkPen(color, width=1.5)
            h = pg.PlotDataItem([cx - arm, cx + arm], [cy,       cy      ], pen=pen)
            v = pg.PlotDataItem([cx,       cx      ], [cy - arm, cy + arm], pen=pen)
            h.setZValue(2)
            v.setZValue(2)
            self.vb.addItem(h)
            self.vb.addItem(v)
            self._cross_items.extend([h, v])


# ── Trace viewer ──────────────────────────────────────────────────────────────

class TraceViewer(pg.PlotWidget):
    """Bottom-left — stacked normalised traces for selected components."""

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(parent, background=BG_DARK)
        self.store = store
        self._sel  = []
        self.getAxis('bottom').setLabel('Frame', color='#888888')
        self.getAxis('bottom').setTextPen(pg.mkPen('#888888'))
        self.getAxis('left').hide()
        self._legend = self.addLegend(labelTextColor='white',
                                      brush=pg.mkBrush(BG_DARK + 'cc'),
                                      pen=pg.mkPen('#333333'))
        self._redraw()

    def set_selection(self, sel):
        self._sel = list(sel)
        self._redraw()

    def refresh(self):
        """Redraw after the active display trace has changed."""
        self._redraw()

    def _redraw(self):
        self.clear()
        self._legend = self.addLegend(labelTextColor='white',
                                      brush=pg.mkBrush(BG_DARK + 'cc'),
                                      pen=pg.mkPen('#333333'))
        s = self.store
        if not self._sel:
            ti = pg.TextItem("Select component(s)", color='#555555',
                             anchor=(0.5, 0.5))
            self.addItem(ti)
            self.getViewBox().setRange(xRange=(0, 1), yRange=(0, 1))
            return

        try:
            T      = s.disp_T
            t      = np.arange(T, dtype=np.float32)
            offset = 0.0
            for k, i in enumerate(self._sel):
                color = PALETTE_HEX[k % len(PALETTE_HEX)]
                trace = s.disp_trace(i)
                span  = trace.max() - trace.min()
                tr    = (trace - trace.min()) / (span + 1e-9)
                self.plot(t, (tr + offset).astype(np.float32),
                          pen=pg.mkPen(color, width=1),
                          name=s.labels[i])
                offset += 1.3
            self.setXRange(0, T, padding=0)
            self.setYRange(-0.3, offset, padding=0)
        except Exception as exc:
            self.clear()
            ti = pg.TextItem(f"Draw error: {exc}", color='#ff4444',
                             anchor=(0.5, 0.5))
            self.addItem(ti)


# ── Shared matrix widget base ─────────────────────────────────────────────────

class _MatrixWidget(pg.GraphicsLayoutWidget):
    """
    Base for CorrMatrix and DistMatrix.
    Displays a K×K symmetric matrix as a heat-map with a fixed colorbar.
    Clicking any off-diagonal cell emits pair_clicked(i, j).
    """

    pair_clicked = pyqtSignal(int, int)

    def __init__(self, store: 'ComponentStore', cmap: pg.ColorMap,
                 parent=None):
        super().__init__(parent)
        self.store      = store
        self._cmap      = cmap
        self._highlight = None
        self._hl_items: list = []

        self.setBackground(BG_DARK)

        # Column 0: matrix ViewBox (no mouse zoom/pan — leave for table/corr UX)
        self.vb = self.addViewBox(row=0, col=0)
        self.vb.invertY(True)            # row 0 at top (matrix convention)
        self.vb.setAspectLocked(True)
        self.vb.setMouseEnabled(x=True, y=True)
        self.vb.setMenuEnabled(True)    # right-click → "View All" reset zoom

        self._img = pg.ImageItem()
        self.vb.addItem(self._img)

        # Column 1: colorbar — do NOT pass insert_in (expects PlotItem,
        # not ViewBox).  Link the image manually and place in the grid.
        self._cbar = pg.ColorBarItem(
            colorMap=self._cmap, width=12,
            pen='#333333', hoverPen='#888888', hoverBrush='#888888'
        )
        self._cbar.setImageItem(self._img)   # no insert_in
        self.addItem(self._cbar, row=0, col=1)
        self.ci.layout.setColumnFixedWidth(1, 60)

        # Click
        self.vb.scene().sigMouseClicked.connect(self._on_scene_click)

    def refresh(self):
        self._highlight = None
        for item in self._hl_items:
            self.vb.removeItem(item)
        self._hl_items = []
        self._render()

    def highlight_pair(self, i: int, j: int):
        """Highlight (i, j) without emitting pair_clicked (avoids signal loop)."""
        self._highlight = (i, j)
        self._render()

    def _levels(self, mat: np.ndarray):
        raise NotImplementedError

    def _matrix(self) -> np.ndarray:
        raise NotImplementedError

    def _render(self):
        mat    = self._matrix()          # (K, K) float32, diag=NaN
        vmin, vmax = self._levels(mat)
        # NaN → midpoint colour so diagonal is neutral
        display = np.where(np.isnan(mat), (vmin + vmax) / 2.0, mat)
        self._img.setImage(display.astype(np.float32))
        self._img.setLookupTable(self._cmap.getLookupTable(nPts=256))
        self._img.setLevels((vmin, vmax))
        self._cbar.setLevels((vmin, vmax))

        # Highlight pair.
        # Each pixel [row, col] is placed at corner (col, row) by ImageItem
        # and spans to (col+1, row+1), so its centre is at (col+0.5, row+0.5).
        # All highlight coordinates are offset by +0.5 accordingly.
        if self._highlight:
            hi, hj = self._highlight
            for item in self._hl_items:
                self.vb.removeItem(item)
            self._hl_items = []
            K = mat.shape[0]
            for band, col in [(hi, CORR_A), (hj, CORR_B)]:
                c = band + 0.5   # centre of the band-th row/col
                lh = pg.PlotDataItem(
                    [-0.5, K + 0.5], [c, c],
                    pen=pg.mkPen(col, width=1.5))
                lv = pg.PlotDataItem(
                    [c, c], [-0.5, K + 0.5],
                    pen=pg.mkPen(col, width=1.5))
                for li in (lh, lv):
                    li.setZValue(2)
                    self.vb.addItem(li)
                    self._hl_items.append(li)
            for (ri, ci_), col in [((hi, hj), CORR_A), ((hj, hi), CORR_B)]:
                box = pg.PlotDataItem(
                    [ci_, ci_ + 1, ci_ + 1, ci_, ci_],
                    [ri,  ri,      ri  + 1, ri + 1, ri],
                    pen=pg.mkPen(col, width=2))
                box.setZValue(3)
                self.vb.addItem(box)
                self._hl_items.append(box)

        # Tick labels (thin out for large K)
        n    = self.store.n
        step = max(1, n // 12)
        tks  = list(range(0, n, step))

    def _on_scene_click(self, event):
        # Require Ctrl+Left-click for pair selection so plain left-drag
        # is free for panning (handled by ViewBox natively).
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return
        pt = self.vb.mapSceneToView(event.scenePos())
        j  = int(pt.x())   # floor: pixel [r,c] occupies [c, c+1)
        i  = int(pt.y())   # floor: pixel [r,c] occupies [r, r+1)
        n  = self.store.n
        if 0 <= i < n and 0 <= j < n and i != j:
            self._highlight = (i, j)
            self._render()
            self.pair_clicked.emit(i, j)


# ── Correlation matrix ────────────────────────────────────────────────────────

class CorrMatrix(_MatrixWidget):
    """Bottom-right — pairwise Pearson correlation heat-map (RdBu_r)."""

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(store, CMAP_RDBU_R, parent)
        self._render()

    def _matrix(self) -> np.ndarray:
        return self.store.corr.copy()

    def _levels(self, mat):
        off  = mat[~np.isnan(mat)]
        clim = float(np.clip(np.percentile(np.abs(off), 99), 0.05, 1.0)) \
               if off.size > 0 else 1.0
        return -clim, clim


# ── Distance matrix ───────────────────────────────────────────────────────────

class DistMatrix(_MatrixWidget):
    """Top-right — pairwise centroid distance heat-map (viridis_r)."""

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(store, CMAP_VIRIDIS_R, parent)
        self._log = False
        self._render()

    def set_log(self, log: bool):
        if log != self._log:
            self._log = log
            self._render()

    def _matrix(self) -> np.ndarray:
        dist = DistMatrix._compute_dist(self.store)
        if self._log:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(dist > 0, np.log10(dist), np.nan)
        return dist

    def _levels(self, mat):
        off = mat[~np.isnan(mat)]
        if off.size == 0:
            return 0.0, 1.0
        if self._log:
            with np.errstate(divide='ignore', invalid='ignore'):
                off = np.where(off > 0, np.log10(off), np.nan)
            off = off[~np.isnan(off)]
        vmax = float(np.clip(np.percentile(off, 99), 1e-3, None)) \
               if off.size > 0 else 1.0
        vmin = float(off.min()) if self._log and off.size > 0 else 0.0
        return vmin, vmax

    @staticmethod
    def _compute_dist(store: 'ComponentStore') -> np.ndarray:
        K   = store.n
        mat = np.full((K, K), np.nan, dtype=np.float32)
        if K == 0:
            return mat
        d1, d2 = store.dims
        p      = np.arange(d1 * d2, dtype=np.float32)
        rows   = p % d1
        cols   = p // d1
        cy_cx  = np.empty((K, 2), dtype=np.float32)
        for i in range(K):
            fp    = store._A[:, i]
            total = fp.sum()
            if total > 1e-9:
                cy_cx[i, 0] = (fp * rows).sum() / total
                cy_cx[i, 1] = (fp * cols).sum() / total
            else:
                cy_cx[i] = [d1 / 2.0, d2 / 2.0]
        for i in range(K):
            for j in range(i + 1, K):
                d = float(np.sqrt(((cy_cx[i] - cy_cx[j]) ** 2).sum()))
                mat[i, j] = mat[j, i] = d
        return mat



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
        self.mark_hidden()

    def mark_hidden(self):
        """Dim the text of hidden rows so it is clear which are not shown."""
        grey   = QColor(105, 105, 105)
        normal = QColor(220, 220, 220)
        for row in range(self.store.n):
            fg = grey if self.store.is_hidden(row) else normal
            for col in range(self.columnCount()):
                it = self.item(row, col)
                if it:
                    it.setForeground(QBrush(fg))

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
        vm.addSeparator()
        a = QAction("Load &background image…", self)
        a.setToolTip("Load a data projection / anatomical image (.npy/.tif/.png) "
                     "to show behind the footprints.")
        a.triggered.connect(self._load_background)
        vm.addAction(a)
        vm.addSeparator()
        a = QAction("&Show all hidden ROIs", self)
        a.setToolTip("Make every hidden component visible again.")
        a.triggered.connect(self._show_all_hidden)
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

        self.act_toggle_vis = QAction("◌  Hide / show", self)
        self.act_toggle_vis.setToolTip(
            "Toggle visibility of the selected component(s) in the cell view "
            "(V).\nHidden components stay in the data — this is not a delete.")
        self.act_toggle_vis.setShortcut(QKeySequence("V"))
        self.act_toggle_vis.setEnabled(False)
        self.act_toggle_vis.triggered.connect(self._toggle_visibility)
        tb.addAction(self.act_toggle_vis)
        self.addAction(self.act_toggle_vis)   # keep shortcut active app-wide

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

        self.act_show_rois = QAction("◎  All ROIs", self)
        self.act_show_rois.setToolTip(
            "Show every component footprint as a dim cyan overlay.\n"
            "Updates live as components are merged or deleted.")
        self.act_show_rois.setCheckable(True)
        self.act_show_rois.setChecked(True)
        self.act_show_rois.triggered.connect(
            lambda checked: self.cell_view.set_show_all(checked))
        tb.addAction(self.act_show_rois)

        tb.addSeparator()
        tb.addWidget(QLabel("  BG: "))
        self.bg_combo = QComboBox(self)
        self.bg_combo.setToolTip(
            "Background image shown behind the footprints.\n"
            "Use View ▸ Load background image… to add a data projection.")
        self.bg_combo.setMinimumWidth(140)
        self.bg_combo.currentTextChanged.connect(self._on_bg_changed)
        tb.addWidget(self.bg_combo)
        self._refresh_bg_combo()

        tb.addWidget(QLabel("  Trace: "))
        self.trace_combo = QComboBox(self)
        self.trace_combo.setToolTip(
            "Temporal trace shown in the trace viewer.\n"
            "Display-only; the denoised C is always what gets saved.")
        self.trace_combo.setMinimumWidth(150)
        self.trace_combo.currentTextChanged.connect(self._on_trace_changed)
        tb.addWidget(self.trace_combo)
        self._refresh_trace_combo()

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

        self.act_reset_zoom = QAction("⟳  Reset zoom", self)
        self.act_reset_zoom.setToolTip(
            "Reset cell viewer to full FOV.\n"
            "Shortcut: double left-click anywhere in the image.")
        self.act_reset_zoom.triggered.connect(
            lambda: self.cell_view.reset_zoom())
        tb.addAction(self.act_reset_zoom)

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

        # Link corr and dist ViewBoxes so zoom/pan stays in sync
        self.corr_view.vb.setXLink(self.dist_view.vb)
        self.corr_view.vb.setYLink(self.dist_view.vb)

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
            self.act_toggle_vis.setEnabled(n >= 1)
            self._sel_lbl.setText(f"  {n} selected  |  {self.store.n} total")
            # Update matrix highlights from table selection:
            # exactly 2 selected → highlight that pair; anything else → clear.
            if n == 2:
                self.corr_view.highlight_pair(sel[0], sel[1])
                self.dist_view.highlight_pair(sel[0], sel[1])
            else:
                self.corr_view.refresh()
                self.dist_view.refresh()
        finally:
            self._in_table_sel = False

    def _on_corr_click(self, i: int, j: int):
        """
        Handle a click on the correlation or distance matrix.

        Both components are programmatically selected in the table so that
        the Merge and Delete toolbar buttons become active immediately —
        tint_pair() alone is purely cosmetic and does not update the Qt
        selection model, leaving the buttons disabled.

        set_pair() must be called *after* _on_table_sel() because
        cell_view.set_selection() (called inside _on_table_sel) resets
        self._pair to None; re-applying set_pair() afterwards restores
        the orange / magenta pair overlay on top of the palette colours.
        """
        self._pair = (i, j)

        # Programmatically select both rows in the Qt selection model so
        # the merge / delete buttons are enabled via _on_table_sel.
        sm    = self.table.selectionModel()
        flags = (QItemSelectionModel.SelectionFlag.Select
                 | QItemSelectionModel.SelectionFlag.Rows)
        self.table.blockSignals(True)
        sm.clearSelection()
        for row in (i, j):
            if 0 <= row < self.store.n:
                sm.select(self.table.model().index(row, 0), flags)
        self.table.blockSignals(False)

        # Propagate selection → enables toolbar, updates trace viewer,
        # and calls cell_view.set_selection([i, j]) which clears _pair.
        self._on_table_sel()

        # Re-apply the pair overlay (orange / magenta) on top of the
        # palette selection colours that _on_table_sel just painted.
        self.cell_view.set_pair((i, j))
        self.table.tint_pair(i, j)

        # Mirror the highlight to both matrices (whichever was clicked
        # already updated itself via _on_scene_click; this call is
        # idempotent so calling it twice on the same widget is harmless).
        self.corr_view.highlight_pair(i, j)
        self.dist_view.highlight_pair(i, j)

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

    # ── Background image ──────────────────────────────────────────────────────

    def _refresh_bg_combo(self):
        """Repopulate the background selector to match the current store."""
        self.bg_combo.blockSignals(True)
        self.bg_combo.clear()
        self.bg_combo.addItems(self.store.background_names())
        if self.store.bg_key is not None:
            self.bg_combo.setCurrentText(self.store.bg_key)
        self.bg_combo.blockSignals(False)

    def _on_bg_changed(self, name: str):
        if not name or name == self.store.bg_key:
            return
        try:
            self.store.set_background(name)
        except KeyError:
            return
        self.cell_view.refresh_background()
        self.statusBar().showMessage(f"Background → {name}")

    # ── Trace selection ───────────────────────────────────────────────────────

    def _refresh_trace_combo(self):
        """Repopulate the trace selector to match the current store."""
        self.trace_combo.blockSignals(True)
        self.trace_combo.clear()
        self.trace_combo.addItems(self.store.trace_names())
        if self.store.disp_key is not None:
            self.trace_combo.setCurrentText(self.store.disp_key)
        # A single available trace gives the user nothing to pick.
        self.trace_combo.setEnabled(self.trace_combo.count() > 1)
        self.trace_combo.blockSignals(False)

    def _on_trace_changed(self, name: str):
        if not name or name == self.store.disp_key:
            return
        try:
            self.store.set_trace(name)
        except KeyError:
            return
        self.trace_view.refresh()
        self.statusBar().showMessage(f"Trace → {name}")

    # ── Visibility ────────────────────────────────────────────────────────────

    def _toggle_visibility(self):
        sel = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not sel:
            self.statusBar().showMessage("Select component(s) to hide / show")
            return
        self.store.toggle_hidden(sel)
        self.cell_view.refresh_visibility()
        self.table.mark_hidden()
        self.statusBar().showMessage(
            f"{self.store.n_hidden} hidden  |  {self.store.n} total")

    def _show_all_hidden(self):
        if self.store.n_hidden == 0:
            self.statusBar().showMessage("No hidden components")
            return
        self.store.show_all()
        self.cell_view.refresh_visibility()
        self.table.mark_hidden()
        self.statusBar().showMessage(f"All {self.store.n} components visible")

    def _load_background(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load background image", "",
            "Images (*.npy *.npz *.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if not path:
            return
        try:
            img  = _load_image_file(path)
            name = os.path.splitext(os.path.basename(path))[0]
            self.store.add_background(name, img, make_active=True)
        except Exception as exc:
            QMessageBox.critical(self, "Background error", str(exc))
            return
        self._refresh_bg_combo()
        self.cell_view.refresh_background()
        self.statusBar().showMessage(f"Loaded background ‘{name}’")

    def _swap_store(self, new_store: ComponentStore):
        self.store = new_store
        for w in (self.table, self.cell_view,
                  self.trace_view, self.corr_view,
                  self.dist_view):
            w.store = new_store
        self._refresh_bg_combo()
        self._refresh_trace_combo()
        self.cell_view.refresh_background()
        self._full_refresh()

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _full_refresh(self):
        self.table.blockSignals(True)
        self.table.refresh()
        self.table.clearSelection()
        self.table.blockSignals(False)
        self.cell_view.set_selection([])
        self.cell_view.refresh_rois()
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
    parser.add_argument("--background", "--bg", dest="background", default=None,
                        help="Image to show behind the footprints "
                             "(.npy/.npz/.tif/.png; a 3-D stack is max-projected)")
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

    if args.background:
        try:
            img  = _load_image_file(args.background)
            name = os.path.splitext(os.path.basename(args.background))[0]
            store.add_background(name, img, make_active=True)
            print(f"Background image: {args.background}")
        except Exception as exc:
            print(f"Warning: could not load background "
                  f"'{args.background}': {exc}", file=sys.stderr)

    win = InspectorWindow(store)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
