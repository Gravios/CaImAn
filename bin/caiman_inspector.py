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
    caiman-inspector                         # session in the current folder:
                                             #   <dir>_results.hdf5 + <dir>.tif
    caiman-inspector /path/to/<session>/     # same, for another folder
    caiman-inspector /path/to/results.hdf5
    caiman-inspector /path/to/<session>.tif  # stack + its results
    caiman-inspector --demo                  # synthetic demo
    (python bin/caiman_inspector.py … works the same)

Controls
--------
    Table        — click to select; Ctrl/Shift for multi-select;
                   click a column header to sort.  ROI-filter columns
                   (QC, Active, Fail, Area, Solid, Ecc, Edge, Rate,
                   Pk ΔF/F, Skew, Act %) appear when the results file has a
                   roi_filter group; Incl then starts from the QC verdict.
                   Analysis ▸ Run ROI filter… (Ctrl+Shift+F) adds or
                   recomputes them, e.g. for sessions processed before the
                   pipeline step existed; thresholds live in Edit ▸
                   Preferences… (Ctrl+,), ~/.config/caiman/inspector_prefs.json
    Movie bar    — under the cell viewer: play / scrub the raw stack
                   (<session>.tif / .msr next to the results is found;
                   View > Open movie… / Ctrl+M for another).  With the
                   image focused: Space play/pause, ←/→ ±1, Shift+←/→ ±10,
                   Home/End, Shift+wheel steps frames.  Drag the yellow
                   cursor in the trace view to seek.
    Corr matrix  — click any off-diagonal cell to highlight that pair
                   in orange / magenta on the cell viewer
    ⊕ Merge      — merge ≥ 2 selected (footprint = pixel max, trace = mean)
    ✕ Delete     — permanently remove selected
    ↩ Undo       — Ctrl+Z  — undo last merge or delete (up to 5 steps)
    ↪ Redo       — Ctrl+Y  — redo last undone operation
    File > Save  — write modified A / C back to the .hdf5

Event detection & curation (ported from SLabAnalysisPackage)
-------------------------------------------------------------
    Analysis > Detect events  — Ctrl+D — SLab peak detection on ΔF/F
                                (caiman.utils.event_detection); include
                                flags start at the automatic approval
    ✓ Include / exclude       — I — non-destructive approval; excluded
                                footprints are red ("Hide excluded" hides them)
    Event panel               — filtered ΔF/F, threshold, peaks, binary trace
                                of the last-selected component
    Analysis > Export         — Ctrl+E — <session>_PeakStats / _ROIStats
    Analysis > Batch detect   — stroh_results into every file of a folder
    Analysis > Blinded review — Ctrl+R — shuffled, name-blind rating;
                                Ctrl+Return saves SLabRandomization/<RATER>
    Events and include flags are saved as an SLab-compatible stroh_results
    group, so SLabAnalysisApp.py opens curated files.
"""

import sys
import os
import re
import json
import argparse
from collections import deque
import numpy as np
import scipy.sparse

HISTORY_MAXLEN = 5   # maximum undo / redo steps retained

TRACE_DENOISED = "C (denoised)"   # canonical trace key; this one is saved back
TRACE_DFF      = "ΔF/F"           # estimates.F_dff; input to event detection

import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QTableWidget,
    QTableWidgetItem, QLabel, QFileDialog, QComboBox,
    QMessageBox, QSizePolicy, QAbstractItemView, QHeaderView,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QInputDialog, QProgressDialog,
    QWidget, QHBoxLayout, QVBoxLayout, QSlider, QSpinBox, QToolButton,
    QCheckBox, QGroupBox, QPushButton, QScrollArea, QTabWidget,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QSize, QItemSelectionModel, QTimer
from PyQt6.QtGui   import (
    QAction, QColor, QBrush, QFont, QPalette, QKeySequence, QShortcut,
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
EXCLUDED = "#FF3B3B"   # footprints of components not approved (events mode)

ALPHA_SEL    = 0.55   # overlay alpha for selected components
ALPHA_DEF    = 0.12   # overlay alpha for unselected components
ALPHA_EXCL   = 0.22   # excluded components: a little stronger, so red reads
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
                 cnm_obj=None, fpath=None, traces=None,
                 orig_idx=None, fr=None, decay_time=None):
        self._A = _to_dense(A)
        self.dims = dims
        K = self._A.shape[1]

        # ── Event detection / curation (SLab stroh_results) ──────────────────
        # events   : EventResults aligned to rows, or None until detected/loaded
        # approved : manual include flag (SLab "manually_approved"); unlike
        #            delete it is non-destructive and is saved alongside A/C
        # orig_idx : row in the source file's estimates (-1 for merged rows),
        #            used to map file-level stroh_results onto the shown subset
        self.events   = None
        self.approved = np.ones(K, dtype=bool)
        # qc       : roi_filter fields {name: (K,) array} aligned to rows, or
        #            None when the file has no roi_filter group
        self.qc        = None
        self.qc_params = {}
        self.orig_idx = (np.arange(K) if orig_idx is None
                         else np.asarray(orig_idx, dtype=np.int64))
        self.fr         = fr
        self.decay_time = decay_time

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
        self.bg_sources:  dict = {}    # background name -> source file path
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
            "events":    self.events.copy() if self.events is not None else None,
            "approved":  self.approved.copy(),
            "orig_idx":  self.orig_idx.copy(),
            "qc":        ({k: v.copy() for k, v in self.qc.items()}
                          if self.qc is not None else None),
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
        self.events    = snap["events"]
        self.approved  = snap["approved"]
        self.orig_idx  = snap["orig_idx"]
        self.qc        = snap.get("qc")
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
        # Merged component is included if any constituent was; it has no
        # source row.  Its events are re-detected on the merged traces.
        self.approved = np.append(self.approved[keep], bool(self.approved[idx].any()))
        self.orig_idx = np.append(self.orig_idx[keep], -1)
        if self.events is not None:
            self.events = self.events.take(keep).append(self._detect_row(self.n - 1))
        if self.qc is not None:
            self.qc = _rf().append(_rf().take(self.qc, keep), self._qc_row(self.n - 1))
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
        self.approved = self.approved[keep]
        self.orig_idx = self.orig_idx[keep]
        if self.events is not None:
            self.events = self.events.take(keep)
        if self.qc is not None:
            self.qc = _rf().take(self.qc, keep)
        self.invalidate_corr()

    # ── ROI filter ────────────────────────────────────────────────────────────

    def _qc_row(self, i):
        """roi_filter fields for row i alone (a merged component), computed
        with the file's filter parameters.  Fields the row's data cannot
        provide are filled as unknown (NaN / 0 / False)."""
        rf = _rf()
        try:
            one = rf.evaluate(
                self._A[:, i:i + 1], self.dims,
                SNR_comp=None if self.SNR_comp is None else self.SNR_comp[i:i + 1],
                r_values=None if self.r_values is None else self.r_values[i:i + 1],
                cnn_preds=None if self.cnn_preds is None else self.cnn_preds[i:i + 1],
                F_dff=None if self.dff is None else self.dff[i:i + 1],
                C=self.C[i:i + 1],
                params=rf.resolve_params(self.qc_params))
        except Exception as exc:
            print(f"[roi_filter] merged row not evaluated: {exc}", file=sys.stderr)
            one = {}
        row = {}
        for k, v in self.qc.items():
            if k in one:
                row[k] = np.asarray(one[k], dtype=v.dtype)
            else:
                fill = (np.nan if np.issubdtype(v.dtype, np.floating)
                        else False if v.dtype == np.bool_ else 0)
                row[k] = np.array([fill], dtype=v.dtype)
        return row

    def run_roi_filter(self, cfg: dict, set_incl: bool = False) -> dict:
        """Evaluate every row with the ROI filter (undoable).

        ``cfg`` is a roi_filter section (preferences); nulls are filled from
        the loaded run's CaImAn params, then from the store (fr, decay
        time).  ``set_incl`` resets the include flags to the QC verdict.
        Returns the summary."""
        rf = _rf()
        p  = self.resolve_filter_params(cfg)
        fields = rf.evaluate(self._A, self.dims,
                             SNR_comp=self.SNR_comp, r_values=self.r_values,
                             cnn_preds=self.cnn_preds, F_dff=self.dff, C=self.C,
                             params=p)
        self._push_undo("ROI filter")
        self.qc, self.qc_params = fields, p
        if set_incl:
            self.approved = np.asarray(fields["passed"], bool).copy()
        return rf.summarize(fields)

    def resolve_filter_params(self, cfg: dict) -> dict:
        rf = _rf()
        cfg = dict(cfg)
        g = cfg.get("gSig")
        if g is not None and np.ndim(g) == 0:
            cfg["gSig"] = [float(g), float(g)]
        p = rf.resolve_params(cfg, getattr(self.cnm_obj, "params", None))
        if p.get("fr") is None and self.fr:
            p["fr"] = float(self.fr)
        if p.get("decay_time") is None and self.decay_time:
            p["decay_time"] = float(self.decay_time)
        return p

    @property
    def has_approval(self) -> bool:
        """Include flags exist once events are detected or a roi_filter
        group seeded them."""
        return self.events is not None or self.qc is not None

    # ── Events / approval ─────────────────────────────────────────────────────

    @property
    def dff(self):
        """ΔF/F traces (K, T) or None — the input event detection needs."""
        return self.traces.get(TRACE_DFF)

    @property
    def can_detect(self) -> bool:
        return self.dff is not None

    def detect_events(self, fr, decay_time, progress=None, **kw):
        """Run SLab event detection on every row; approval := auto-approval."""
        if not self.can_detect:
            raise RuntimeError("No ΔF/F traces (estimates.F_dff) in this file — "
                               "run cnm.estimates.detrend_df_f() first.")
        self.events = _ed().detect_events(self.dff, self.C, fr, decay_time,
                                          self.SNR_comp, self.cnn_preds,
                                          progress=progress, **kw)
        self.approved = self.events.auto_approved.copy()
        self.fr, self.decay_time = float(fr), float(decay_time)

    def _detect_row(self, i):
        """EventResults for row i alone, using the current events' settings
        (a blank row if ΔF/F is unavailable)."""
        ev = self.events
        if self.can_detect:
            p = ev.params
            one = _ed().detect_events(
                self.dff[i:i + 1], self.C[i:i + 1], ev.fr, ev.decay_time,
                None if self.SNR_comp is None else self.SNR_comp[i:i + 1],
                None if self.cnn_preds is None else self.cnn_preds[i:i + 1],
                threshold=ev.threshold, min_snr=p.get("min_snr", 3.0),
                min_cnn=p.get("min_cnn", 0.15), lowpass_hz=p.get("lowpass_hz"))
            one.params = dict(p)
            return one
        blank = ev.take([0]) if ev.n else ev
        blank.binary[:] = 0; blank.filtered[:] = 0
        blank.thresholds[:] = 0; blank.frequencies[:] = 0
        blank.peaks = [np.zeros((0, 4))]; blank.auto_approved[:] = False
        return blank

    def toggle_approved(self, indices):
        for i in indices:
            self.approved[i] = not self.approved[i]

    @property
    def n_approved(self) -> int:
        return int(self.approved.sum())

    def attach_events(self, res, approved):
        """Adopt file-level stroh_results, mapped through orig_idx.

        Rows whose source index is outside the stored results (or merged
        rows) are re-detected when possible.  Returns False if the stored
        results don't fit this file (row count mismatch)."""
        oi = self.orig_idx
        if res.n == self.n and np.array_equal(oi, np.arange(self.n)):
            self.events, self.approved = res, np.asarray(approved, bool).copy()
        elif oi.size and oi.max() < res.n:
            ok = oi >= 0
            if not ok.all():
                return False
            self.events = res.take(oi)
            self.approved = np.asarray(approved, bool)[oi].copy()
        else:
            return False
        if not np.isfinite(self.events.fr) and self.fr:
            self.events.fr = float(self.fr)
        if not np.isfinite(self.events.decay_time) and self.decay_time:
            self.events.decay_time = float(self.decay_time)
        self.fr = self.events.fr
        self.decay_time = self.events.decay_time
        return True

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

        # Null stale K-indexed derived fields.  ΔF/F is carried through
        # merge/delete in the trace registry (merged rows = mean of their
        # constituents' ΔF/F); it is written back rather than nulled because
        # event detection and SLabAnalysisApp both need it.
        dff = self.dff
        for field in ("S", "YrA", "R", "F_dff",
                      "g", "bl", "c1", "neurons_sn", "lam",
                      "coordinates", "A_thr"):
            if getattr(est, field, None) is not None:
                setattr(est, field, None)
        if dff is not None:
            est.F_dff = dff

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
                "n_approved":   self.n_approved,
                "n_merged_dff_rows": int((self.orig_idx < 0).sum()),
                "source_file":  self.fpath or "",
                "save_path":    path,
            })

        # Delegate to CaImAn's own serialiser
        self.cnm_obj.save(path)
        self.fpath = path
        self.orig_idx = np.arange(K)          # rows now index the saved file

        # Events + approvals as an SLab stroh_results group, rows = saved rows;
        # roi_filter fields (merged rows re-evaluated) with the include flags
        if self.events is not None or self.qc is not None:
            import h5py
            with h5py.File(path, "a") as f:
                if self.events is not None:
                    _ed().write_stroh_results(f, self.events, self.approved)
                if self.qc is not None:
                    _rf().write_group(f, self.qc, self.qc_params,
                                      manually_approved=self.approved)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dense(A) -> np.ndarray:
    if scipy.sparse.issparse(A):
        return np.asarray(A.todense(), dtype=np.float32)
    return np.asarray(A, dtype=np.float32)

def _ed():
    """``caiman.utils.event_detection``, imported lazily.

    Falls back to loading the module file from this checkout when the
    ``caiman`` package itself can't be imported (e.g. demo mode on a machine
    without the compiled CaImAn stack).
    """
    mod = sys.modules.get("_inspector_event_detection")
    if mod is not None:
        return mod
    try:
        from caiman.utils import event_detection as mod
    except Exception:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, "caiman", "utils", "event_detection.py")
        name = "_inspector_event_detection"
        spec = importlib.util.spec_from_file_location(name, p)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # @dataclass resolves its module here
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[name]
            raise
    sys.modules["_inspector_event_detection"] = mod
    return mod


def _rf():
    """``caiman.utils.roi_filter``, imported lazily (same fallback as _ed)."""
    mod = sys.modules.get("_inspector_roi_filter")
    if mod is not None:
        return mod
    try:
        from caiman.utils import roi_filter as mod
    except Exception:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, "caiman", "utils", "roi_filter.py")
        spec = importlib.util.spec_from_file_location("_inspector_roi_filter", p)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    sys.modules["_inspector_roi_filter"] = mod
    return mod


# ── Preferences (JSON, per user) ─────────────────────────────────────────────

def prefs_path() -> str:
    """``$CAIMAN_INSPECTOR_PREFS``, else ``$XDG_CONFIG_HOME/caiman/
    inspector_prefs.json`` (``~/.config`` when XDG is unset)."""
    p = os.environ.get("CAIMAN_INSPECTOR_PREFS")
    if p:
        return p
    base = (os.environ.get("XDG_CONFIG_HOME")
            or os.path.join(os.path.expanduser("~"), ".config"))
    return os.path.join(base, "caiman", "inspector_prefs.json")


def load_prefs() -> dict:
    try:
        with open(prefs_path()) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[prefs] ignoring unreadable {prefs_path()}: {exc}", file=sys.stderr)
        return {}


def save_prefs(prefs: dict) -> str:
    path = prefs_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(prefs, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def roi_filter_prefs(prefs: dict = None) -> dict:
    """roi_filter DEFAULTS (+ gSig override) overlaid with the saved section;
    unknown keys are dropped."""
    d = dict(_rf().DEFAULTS)
    d["gSig"] = None
    sec = (load_prefs() if prefs is None else prefs).get("roi_filter", {})
    for k, v in (sec if isinstance(sec, dict) else {}).items():
        if k in d:
            d[k] = v
    return d


# Field specs for the preference editor, one list per section:
#   (group, key, label, kind, (min, max, step, decimals), null_text, tooltip)
# kind: "float" | "int" | "bool" | "tristate";  null_text=None → not nullable,
# else the checkbox-off meaning ("off", "from run", …).
_RF_PREF_FIELDS = (
    ("Footprint", "footprint_thr", "Mask threshold (× peak)", "float", (0.01, 1.0, 0.05, 2), None,
     "Footprint mask = connected pixels ≥ this fraction of the footprint's peak."),
    ("Footprint", "gSig", "Expected cell radius gSig (px)", "float", (0.5, 200.0, 0.5, 1), "from run",
     "Sets the expected area π·gSig² for the area fractions.  Off: the run's init.gSig."),
    ("Footprint", "min_area_frac", "Min area (× π·gSig²)", "float", (0.0, 100.0, 0.1, 2), "off",
     "Fail footprints smaller than this multiple of the expected area."),
    ("Footprint", "max_area_frac", "Max area (× π·gSig²)", "float", (0.0, 1000.0, 0.5, 1), "off",
     "Fail footprints larger than this multiple of the expected area."),
    ("Footprint", "min_area_px", "Min area (px)", "int", (0, 10_000_000, 1, 0), "off",
     "Absolute limit; overrides the fraction."),
    ("Footprint", "max_area_px", "Max area (px)", "int", (0, 10_000_000, 10, 0), "off",
     "Absolute limit; overrides the fraction."),
    ("Footprint", "min_solidity", "Min solidity", "float", (0.0, 1.0, 0.05, 2), "off",
     "Area / convex-hull area; low for ragged or ring-shaped footprints."),
    ("Footprint", "max_eccentricity", "Max eccentricity", "float", (0.0, 1.0, 0.01, 2), "off",
     "0 = circle, → 1 = line (dendrites, scan artefacts)."),
    ("Edge", "edge_px", "Min distance to border (px)", "float", (0.0, 10_000.0, 1.0, 0), "off",
     "Fail ROIs whose centroid is closer than this to the FOV border."),
    ("CaImAn quality", "use_caiman_quality", "Apply CaImAn quality rule", "bool", None, None,
     "Pass any upper threshold and no *_lowest floor (CaImAn's evaluate_components)."),
    ("CaImAn quality", "min_SNR", "min_SNR", "float", (0.0, 1000.0, 0.1, 2), "from run", ""),
    ("CaImAn quality", "SNR_lowest", "SNR_lowest", "float", (0.0, 1000.0, 0.1, 2), "from run", ""),
    ("CaImAn quality", "rval_thr", "rval_thr", "float", (-1.0, 1.0, 0.05, 2), "from run", ""),
    ("CaImAn quality", "rval_lowest", "rval_lowest", "float", (-1.0, 1.0, 0.05, 2), "from run", ""),
    ("CaImAn quality", "use_cnn", "use_cnn", "tristate", None, "from run", ""),
    ("CaImAn quality", "min_cnn_thr", "min_cnn_thr", "float", (0.0, 1.0, 0.05, 2), "from run", ""),
    ("CaImAn quality", "cnn_lowest", "cnn_lowest", "float", (0.0, 1.0, 0.05, 2), "from run", ""),
    ("Activity", "event_threshold", "Event cutoff (ΔF/F)", "float", (0.001, 10.0, 0.05, 3), None,
     "SLab detection cap: threshold = min(10·noise, this)."),
    ("Activity", "lowpass_hz", "Low-pass cutoff (Hz)", "float", (0.1, 1000.0, 1.0, 1), "SLab default",
     "Off: SLab's normalised cutoff."),
    ("Activity", "min_events", "Min events", "int", (0, 1_000_000, 1, 0), "off",
     "Active ⇔ every set minimum is met."),
    ("Activity", "min_event_rate", "Min event rate (/min)", "float", (0.0, 1000.0, 0.1, 2), "off", ""),
    ("Activity", "min_peak_dff", "Min peak ΔF/F", "float", (0.0, 100.0, 0.05, 2), "off", ""),
    ("Activity", "min_skewness", "Min skewness", "float", (-100.0, 100.0, 0.1, 2), "off", ""),
    ("Verdict", "require_activity", "Inactive ROIs fail QC", "bool", None, None,
     "Off: activity is reported separately and silent cells can pass."),
)

# section key → (tab title, field specs, defaults loader)
PREF_SECTIONS = {
    "roi_filter": ("ROI filter", _RF_PREF_FIELDS, lambda: roi_filter_prefs({})),
}


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
        if len(_tif_shape(path)) <= 2:
            import tifffile
            return tifffile.imread(path)
        return _stack_projections(path)["stack max"]   # streamed, not imread
    if ext in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        return np.asarray(Image.open(path).convert("F"))
    raise ValueError(f"unsupported background format: {ext}")


TIF_EXTS = (".tif", ".tiff")
SIBLING_STACK_MAX_BYTES = 2 << 30   # auto-project sibling stacks only up to 2 GiB


def _tif_shape(path: str) -> tuple:
    """Shape of the first series of a TIFF, read from the header only."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        return tuple(tf.series[0].shape)


def _stack_projections(path: str, chunk: int = 256) -> dict:
    """Mean and max projections of a TIFF stack, streamed frame by frame.

    Never holds more than ``chunk`` frames in memory, so multi-GB BigTIFF
    movies work.  Leading axes (time, z, channel) are all reduced; the last
    two are (H, W).  A 2-D TIFF returns the image as both projections.
    """
    import tifffile
    shape = _tif_shape(path)
    if len(shape) <= 2:
        img = np.asarray(tifffile.imread(path), dtype=np.float32)
        return {"stack mean": img, "stack max": img}
    H, W = shape[-2:]
    acc = np.zeros((H, W), dtype=np.float64)
    mx  = np.full((H, W), -np.inf, dtype=np.float32)
    n   = 0
    try:                                        # contiguous: memmap + chunks
        arr = tifffile.memmap(path, mode="r").reshape(-1, H, W)
        for s in range(0, arr.shape[0], chunk):
            blk = np.asarray(arr[s:s + chunk], dtype=np.float32)
            acc += blk.sum(axis=0, dtype=np.float64)
            np.maximum(mx, blk.max(axis=0), out=mx)
            n += blk.shape[0]
    except Exception:                           # compressed / tiled: per page
        acc[:] = 0; mx[:] = -np.inf; n = 0
        with tifffile.TiffFile(path) as tf:
            for page in tf.pages:
                fr = np.asarray(page.asarray(), dtype=np.float32).reshape(-1, H, W)
                acc += fr.sum(axis=0, dtype=np.float64)
                np.maximum(mx, fr.max(axis=0), out=mx)
                n += fr.shape[0]
    if n == 0:
        raise ValueError(f"no frames in {path}")
    return {"stack mean": (acc / n).astype(np.float32), "stack max": mx}


def resolve_results_for_stack(stack_path: str) -> str:
    """Find the CNMF results file that belongs to a raw ``.tif`` stack.

    Looks in the stack's folder, then its parent, for (in order)
    ``<stem>_results.hdf5``, ``<stem>_caiman_results.hdf5``, then any
    ``<stem>*.hdf5`` / ``<stem>*.h5``; originals are preferred over
    ``*_curated`` copies.  Raises FileNotFoundError listing where it looked.
    """
    import glob
    stem   = os.path.splitext(os.path.basename(stack_path))[0]
    folder = _os_dirname(stack_path)
    dirs   = [folder, os.path.dirname(folder)]
    for d in dirs:                              # exact names win in either dir
        for name in (f"{stem}_results.hdf5", f"{stem}_caiman_results.hdf5"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    for d in dirs:
        cands = sorted(glob.glob(os.path.join(glob.escape(d), glob.escape(stem) + "*.hdf5"))
                       + glob.glob(os.path.join(glob.escape(d), glob.escape(stem) + "*.h5")))
        cands.sort(key=lambda p: os.path.splitext(p)[0].endswith("_curated"))
        if cands:
            return cands[0]
    raise FileNotFoundError(
        f"no CNMF results for stack '{os.path.basename(stack_path)}' — looked for "
        f"{stem}_results.hdf5 / {stem}*.hdf5 in {dirs[0]} and {dirs[1]}. "
        f"Pass the results file instead, or use --results.")


def load_any(path: str, results: str = None) -> 'ComponentStore':
    """Open a results file, or a raw ``.tif`` stack plus its results.

    For a stack, the matching results file is found with
    :func:`resolve_results_for_stack` (or taken from ``results``), and the
    stack's streamed mean / max projections are added as backgrounds with
    the mean made active, so the actual cells are visible.
    """
    if os.path.isdir(path):
        res_path, tif = discover_session(path, results)
        return load_any(tif, res_path) if tif else load_from_hdf5(res_path)
    if os.path.splitext(path)[1].lower() not in TIF_EXTS:
        return load_from_hdf5(results or path)
    res_path = results or resolve_results_for_stack(path)
    print(f"Stack {os.path.basename(path)} -> results {res_path}")
    store = load_from_hdf5(res_path)
    # The sibling scan may already have projected this very stack.
    if store.bg_sources.get("stack mean") != os.path.abspath(path):
        for name, img in _stack_projections(path).items():
            store.add_background(name, img)
            store.bg_sources[name] = os.path.abspath(path)
    store.set_background("stack mean")
    return store


RESULTS_SUFFIX = "_results.hdf5"


def discover_session(folder: str, results: str = None):
    """Find a session's results file and raw stack inside ``folder``.

    The session base name is the folder's own name (``<dir>/``), so a
    session directory laid out as::

        <base>/<base>_results.hdf5
        <base>/<base>.tif

    is found from inside it.  Results: ``<base>_results.hdf5``, else the one
    ``*_results.hdf5`` in the folder (``*_results_curated.hdf5`` copies don't
    match the pattern).  Stack: ``<base>.tif`` / ``<base>.tiff``, or None.
    ``results`` overrides the results lookup.  Returns ``(results, tif)``.
    """
    import glob
    folder = os.path.abspath(folder)
    base   = os.path.basename(folder)
    tif = next((os.path.join(folder, base + e) for e in TIF_EXTS
                if os.path.isfile(os.path.join(folder, base + e))), None)
    if results:
        return results, tif
    exact = os.path.join(folder, base + RESULTS_SUFFIX)
    if os.path.isfile(exact):
        return exact, tif
    cands = sorted(glob.glob(os.path.join(glob.escape(folder), "*" + RESULTS_SUFFIX)))
    if len(cands) == 1:
        return cands[0], tif
    if cands:
        raise FileNotFoundError(
            f"{len(cands)} *{RESULTS_SUFFIX} files in {folder} and none is "
            f"{base}{RESULTS_SUFFIX}: "
            + ", ".join(os.path.basename(c) for c in cands)
            + " — pass one explicitly.")
    raise FileNotFoundError(
        f"no {base}{RESULTS_SUFFIX} (or any *{RESULTS_SUFFIX}) in {folder}")


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

def load_from_hdf5(path: str, all_components: bool = False) -> ComponentStore:
    """Load a CNMF results file.  By default only ``idx_components`` are
    shown; ``all_components`` shows every row (as SLabAnalysisApp does —
    used by blinded review so ratings index the file's full estimates)."""
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

    idx = None if all_components else getattr(est, "idx_components", None)

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
        traces[TRACE_DFF] = F_dff
    if YrA is not None and YrA.shape == Cd.shape:
        traces["C + residual (raw)"] = Cd + YrA
    if S is not None and S.shape == Cd.shape and np.any(S):
        traces["S (deconvolved)"] = S

    _Cn = getattr(est, "Cn", None)
    Cn  = _Cn if _Cn is not None else A.max(axis=1).reshape(dims, order="F")
    orig_idx = (np.asarray(idx, dtype=np.int64)
                if idx is not None and len(idx) > 0 else np.arange(Cd.shape[0]))
    pdata = getattr(getattr(cnm, "params", None), "data", None) or {}
    store = ComponentStore(A, C, Cn, dims,
                           SNR_comp=snr, r_values=rval, cnn_preds=cnn,
                           cnm_obj=cnm, fpath=path, traces=traces,
                           orig_idx=orig_idx,
                           fr=_opt_float(pdata.get("fr")),
                           decay_time=_opt_float(pdata.get("decay_time")))
    _attach_roi_filter(store, path)        # seeds Incl from `passed` …
    _attach_stroh_results(store, path)     # … unless event approvals exist
    _attach_sibling_backgrounds(store, path)
    return store


def _opt_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def _attach_roi_filter(store: 'ComponentStore', path: str):
    """Load a ``roi_filter`` group if present, mapped through orig_idx.

    Include flags start from the group's saved manual approvals, else from
    the filter's ``passed`` verdict.  Event approvals (stroh_results), read
    afterwards, take precedence."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "roi_filter" not in f:
                return
            fields, params, appr = _rf().read_group(f["roi_filter"])
    except Exception as exc:
        print(f"[roi_filter] could not read roi_filter: {exc}", file=sys.stderr)
        return
    if not fields:
        return
    K  = int(next(iter(fields.values())).size)
    oi = store.orig_idx
    if not (oi.size and (oi >= 0).all() and oi.max() < K):
        print(f"[roi_filter] {K} rows; cannot map onto {store.n} shown "
              "components — ignored", file=sys.stderr)
        return
    store.qc        = _rf().take(fields, oi)
    store.qc_params = params
    base = appr if appr is not None else fields.get("passed")
    if base is not None:
        store.approved = np.asarray(base, bool)[oi].copy()


def _attach_stroh_results(store: 'ComponentStore', path: str):
    """Load an SLab ``stroh_results`` group (events + approvals) if present."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "stroh_results" not in f:
                return
            res, appr = _ed().read_stroh_results(f["stroh_results"])
    except Exception as exc:
        print(f"[events] could not read stroh_results: {exc}", file=sys.stderr)
        return
    if res is None:
        print("[events] stroh_results too old (< v7.1) — ignored", file=sys.stderr)
    elif res.binary.shape[1] != store.C.shape[1]:
        print("[events] stroh_results length differs from C — ignored", file=sys.stderr)
    elif not store.attach_events(res, appr):
        print(f"[events] stroh_results has {res.n} rows; cannot map onto "
              f"{store.n} shown components — ignored", file=sys.stderr)


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
    folder = _os_dirname(path)
    prefix = _session_prefix(path)
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
            label = name_stem
            if shares_prefix:
                label = name_stem[len(prefix):].lstrip("._- ") or name_stem
            # SLab rule: the raw stack <session>.tif next to the results is the
            # anatomical reference.  Project it (streamed) unless it is huge.
            if (low.endswith(TIF_EXTS) and name_stem == prefix
                    and len(_tif_shape(f)) > 2):
                if os.path.getsize(f) > SIBLING_STACK_MAX_BYTES:
                    print(f"[bg] skipping {os.path.basename(f)} "
                          f"({os.path.getsize(f) / 2**30:.1f} GiB); "
                          f"pass it with --background or open the .tif",
                          file=sys.stderr)
                    continue
                for pname, img in _stack_projections(f).items():
                    store.add_background(pname, img)
                    store.bg_sources[pname] = os.path.abspath(f)
                    found.append((pname, "mean" if "mean" in pname else "max"))
                continue
            img = _load_image_file(f)
            if np.asarray(img).squeeze().ndim != 2:   # other stacks: skip
                continue
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


def _session_prefix(path: str) -> str:
    """``<session>`` from ``<session>[_caiman][_results][_curated].hdf5``.

    Trailing results/cnmf/caiman/curated tails are stripped repeatedly, so
    ``<session>_caiman_results_curated`` → ``<session>``.
    """
    import re as _re
    prefix, prev = os.path.splitext(os.path.basename(path))[0], None
    while prefix != prev:
        prev   = prefix
        prefix = _re.sub(r'[._-]?(results?|cnmf?|cnm|caiman|estimates?|curated)$',
                         '', prefix, flags=_re.I)
    return prefix


# ── Movie source for the cell-view background (no Qt) ────────────────────────

MOVIE_EXTS = (".tif", ".tiff", ".msr")


def find_session_movie(store: 'ComponentStore'):
    """The session's raw stack, if it sits next to the results file.

    Same rule as the background scan: ``<session>.tif`` / ``.tiff`` /
    ``.msr`` (session = results name without _results/_curated tails), or a
    file named after the folder.  A stack already projected as the
    "stack mean" background is used first.
    """
    src = store.bg_sources.get("stack mean")
    if src and os.path.isfile(src) and src.lower().endswith(MOVIE_EXTS):
        return src
    if not store.fpath:
        return None
    folder = _os_dirname(store.fpath)
    for name in dict.fromkeys((_session_prefix(store.fpath),
                               os.path.basename(folder))):
        for ext in MOVIE_EXTS:
            p = os.path.join(folder, name + ext)
            if os.path.isfile(p):
                return p
    return None


class MovieSource:
    """Random-access frames of a .tif / .msr stack, oriented to ``dims``.

    Wraps :class:`caiman.utils.stack_io.StackReader` (persistent handle)
    with a small frame cache, a running mean for denoised playback, and
    fixed display levels so playback does not flicker.
    """

    CACHE_FRAMES = 64
    MAX_AVG      = 32

    def __init__(self, path: str, dims):
        from caiman.utils.stack_io import StackReader
        from collections import OrderedDict
        self.path = path
        self._r   = StackReader(path)
        self.n_frames = int(self._r.n_frames)
        shape, dims = (int(self._r.h), int(self._r.w)), tuple(int(d) for d in dims)
        if shape == dims:
            self._transpose = False
        elif shape == dims[::-1]:
            self._transpose = True
        else:
            self._r.close()
            raise ValueError(f"movie frames are {shape[0]}×{shape[1]} but the "
                             f"results FOV is {dims[0]}×{dims[1]}")
        self._cache  = OrderedDict()
        self._levels = None

    def _read(self, i: int) -> np.ndarray:
        a = np.asarray(self._r.read_frame(i), dtype=np.float32)
        return np.ascontiguousarray(a.T if self._transpose else a)

    def frame(self, i: int) -> np.ndarray:
        i = int(min(max(i, 0), self.n_frames - 1))
        f = self._cache.get(i)
        if f is None:
            f = self._read(i)
            self._cache[i] = f
            if len(self._cache) > self.CACHE_FRAMES:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(i)
        return f

    def mean(self, i: int, n: int) -> np.ndarray:
        """Mean of frames i-n+1 … i (clipped at 0)."""
        n = int(min(max(n, 1), self.MAX_AVG))
        if n == 1:
            return self.frame(i)
        lo = max(0, int(i) - n + 1)
        acc = np.zeros_like(self.frame(lo))
        for j in range(lo, int(i) + 1):
            acc += self.frame(j)
        return acc / (int(i) + 1 - lo)

    def levels(self) -> tuple:
        """Display range from robust percentiles of ~24 frames across the movie."""
        if self._levels is None:
            idx = np.unique(np.linspace(0, self.n_frames - 1,
                                        min(24, self.n_frames)).astype(int))
            sample = np.stack([self._read(int(j)) for j in idx])
            lo, hi = np.percentile(sample, [0.5, 99.8])
            self._levels = (float(lo), float(hi) if hi > lo else float(lo) + 1.0)
        return self._levels

    def close(self) -> None:
        try:
            self._r.close()
        except Exception:
            pass
        self._cache.clear()


# ── Event exports / blinded review (no Qt; called by the window) ─────────────

def export_event_tables(store: 'ComponentStore', peak_path: str) -> list:
    """Write SLab PeakStats + ROIStats tables for the *included* components.

    ``peak_path`` names the PeakStats file (.xlsx or .csv); ROIStats goes
    next to it with ``PeakStats`` → ``ROIStats`` in the name.  ``.xlsx``
    needs pandas + openpyxl and falls back to ``.csv``.  ROIStats lists every
    included ROI, including those with 0 events (SLab dropped them).
    Returns the paths written.
    """
    ed  = _ed()
    ev  = store.events
    if ev is None:
        raise RuntimeError("no events to export")
    rows  = np.flatnonzero(store.approved)
    peaks = ev.peak_table(rows)
    peaks["label"] = np.array([store.labels[r] for r in peaks["roi"]], dtype=object)
    rois  = ed.roi_stats(ev, store.approved, ev.binary.shape[1],
                         labels=store.labels, SNR_comp=store.SNR_comp,
                         cnn_preds=store.cnn_preds)
    if store.qc is not None:                 # ROI-filter fields, same rows
        rr = np.asarray(rois["ROI"], dtype=np.int64)
        for k, v in store.qc.items():
            col = np.asarray(v)[rr]
            rois[f"qc_{k}"] = col.astype(int) if col.dtype == np.bool_ else col
        if "fail_reasons" in store.qc:
            rois["qc_fail"] = np.array(
                [",".join(_rf().decode_reasons(m)) for m in store.qc["fail_reasons"][rr]],
                dtype=object)
    folder, base = os.path.split(peak_path)
    stem, ext = os.path.splitext(base)
    roi_stem = (stem.replace("PeakStats", "ROIStats") if "PeakStats" in stem
                else stem + "_ROIStats")
    ext = ext.lower() if ext.lower() in (".xlsx", ".csv") else ".xlsx"
    pd = None
    if ext == ".xlsx":
        import importlib.util
        if importlib.util.find_spec("openpyxl") is not None:
            try:
                import pandas as pd
            except ImportError:
                pd = None
        if pd is None:
            ext = ".csv"
    out = []
    for table, st in ((peaks, stem), (rois, roi_stem)):
        p = os.path.join(folder, st + ext)
        if ext == ".xlsx":
            pd.DataFrame(table).to_excel(p, index=False)
        else:
            import csv
            cols = list(table)
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(cols)
                for k in range(len(table[cols[0]]) if cols else 0):
                    w.writerow([table[c][k] for c in cols])
        out.append(p)
    return out


def blind_plan(files, rater: str):
    """Split results files for a blinded review by ``rater``.

    Returns ``(unrated, rated, skipped)``; files without a ``stroh_results``
    group (no events yet) or unreadable files are ``skipped``.
    """
    import h5py
    unrated, rated, skipped = [], [], []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                if "stroh_results" not in f:
                    skipped.append(p)
                elif "SLabRandomization" in f and rater in f["SLabRandomization"]:
                    rated.append(p)
                else:
                    unrated.append(p)
        except Exception:
            skipped.append(p)
    return unrated, rated, skipped


def write_blind_rating(path: str, rater: str, store: 'ComponentStore',
                       file_list) -> None:
    """Append ``SLabRandomization/<rater>`` to the results file — the SLab
    inter-rater layout (stroh_results fields + file_list + timestamp attr).
    Only that group is written; estimates are untouched.  Requires a store
    loaded with ``all_components=True`` and no merges/deletes, so rows are
    the file's estimates rows."""
    import h5py
    from datetime import datetime
    if not np.array_equal(store.orig_idx, np.arange(store.n)):
        raise RuntimeError("ratings must index the file's full estimates")
    with h5py.File(path, "a") as f:
        g = _ed().write_stroh_results(f, store.events, store.approved,
                                      group=f"SLabRandomization/{rater}")
        g.create_dataset("file_list",
                         data=[str(p).encode("utf-8") for p in file_list])
        g.attrs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def _build_roi_overlay(store: ComponentStore,
                       hide_excluded: bool = False) -> np.ndarray:
    """
    Return (d1, d2, 4) RGBA composite of *every* footprint in dim cyan.

    This is the persistent "all ROIs" base layer.  It is rebuilt from the
    store on each mutation (merge / delete / undo / redo), so deleted or
    merged components visibly disappear / appear in the cell viewer.
    """
    d1, d2 = store.dims
    out    = np.zeros((d1, d2, 4), dtype=np.float32)

    def _rgb(hex_c):
        _h = hex_c.lstrip('#')
        return tuple(int(_h[k:k + 2], 16) / 255.0 for k in (0, 2, 4))

    inc_rgb, exc_rgb = _rgb(CYAN), _rgb(EXCLUDED)
    use_appr = store.has_approval         # events or roi_filter seeded Incl

    for i in range(store.n):
        if store.hidden[i]:
            continue
        excluded = use_appr and not store.approved[i]
        if excluded and hide_excluded:
            continue
        r, g, b = exc_rgb if excluded else inc_rgb
        fp   = store.footprint(i)
        peak = fp.max()
        if peak < 1e-9:
            continue
        a_mask = (fp / peak) * (ALPHA_EXCL if excluded else ALPHA_DEF)
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
      Shift+Scroll   → step through movie frames (when a movie is shown)
    """

    component_clicked = pyqtSignal(int, bool)
    frame_scrolled    = pyqtSignal(int)      # ±steps from Shift+wheel

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(parent)
        self.store      = store
        self._sel       = []
        self._pair      = None
        self._centroids = None
        self._cross_params_cache = None
        self._invert_bg = False
        self._movie_img    = None     # current movie frame (float32, dims) or None
        self._movie_levels = None

        self.setBackground(BG_DARK)

        # ViewBox: aspect locked, Y not inverted → row 0 at bottom (origin='lower')
        self.vb = self.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.vb.invertY(False)

        # Background (Cn)
        self._img_bg = pg.ImageItem()
        self._img_bg.setZValue(0)
        self.vb.addItem(self._img_bg)

        # Persistent "all ROIs" layer (dim cyan, every footprint; excluded
        # components in red once events exist, optionally hidden)
        self._show_all = True
        self._hide_excluded = False
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
            self._img_all.setImage(
                _build_roi_overlay(self.store, self._hide_excluded))
            self._img_all.setVisible(True)
        else:
            self._img_all.setVisible(False)

    def set_show_all(self, flag: bool):
        self._show_all = bool(flag)
        self.refresh_rois()

    def set_hide_excluded(self, flag: bool):
        self._hide_excluded = bool(flag)
        self.refresh_rois()

    def refresh_background(self):
        """Redraw after the store's active background image has changed."""
        self._redraw()

    def set_movie_frame(self, img, levels):
        """Show a movie frame as the background (fixed display *levels*).
        Only the background layer is redrawn, so playback stays cheap."""
        self._movie_img, self._movie_levels = img, levels
        self._draw_background()

    def clear_movie(self):
        """Return to the static background image."""
        if self._movie_img is not None:
            self._movie_img = self._movie_levels = None
            self._draw_background()

    @property
    def showing_movie(self) -> bool:
        return self._movie_img is not None

    def wheelEvent(self, ev):
        if (self._movie_img is not None
                and ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            d = ev.angleDelta()
            delta = d.y() or d.x()          # some platforms turn Shift+wheel horizontal
            if delta:
                self.frame_scrolled.emit(-1 if delta > 0 else 1)
            ev.accept()
            return
        super().wheelEvent(ev)

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

    def _draw_background(self):
        if self._movie_img is not None:
            lo, hi = self._movie_levels
            img = self._movie_img
            if self._invert_bg:
                img = (lo + hi) - img
            self._img_bg.setImage(img, levels=(lo, hi), autoLevels=False)
            return
        cn = self.store.Cn.copy()
        if self._invert_bg:
            cn = cn.max() - cn
        lo, hi = cn.min(), cn.max()
        if hi > lo:
            cn8 = ((cn - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            cn8 = np.zeros_like(cn, dtype=np.uint8)
        self._img_bg.setImage(cn8, levels=(0, 255), autoLevels=False)

    def _redraw(self):
        s = self.store

        # Background (static image, or the current movie frame)
        self._draw_background()

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


# ── Movie playback bar (under the cell viewer) ──────────────────────────────

class MovieBar(QWidget):
    """Play / scrub the session's raw .tif / .msr stack as the cell-view
    background, with the ROI overlay and crosses drawn on top.

    The movie next to the results file (``<session>.tif`` / ``.msr``) is
    found automatically and opened only when first used.  Keys while the
    image or this bar has focus: Space play/pause · ←/→ ±1 frame ·
    Shift+←/→ ±10 · Home/End.  Shift+wheel over the image steps frames.
    """

    def __init__(self, cell_view: 'CellViewer', trace_view: 'TraceViewer',
                 store: 'ComponentStore', parent=None):
        super().__init__(parent)
        self.cell_view, self.trace_view = cell_view, trace_view
        self.store = store
        self.src: MovieSource | None = None
        self.path = None
        self.cur  = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        def _btn(text, tip, slot, checkable=False):
            b = QToolButton(self)
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setAutoRaise(True)
            (b.toggled if checkable else b.clicked).connect(slot)
            return b

        self.b_open  = _btn("⏏", "Open a movie (.tif / .msr) to play behind the ROIs…",
                            self.open_dialog)
        self.b_movie = _btn("Movie", "Show the movie frame as background "
                            "(off: the static BG image)", self.set_active, True)
        self.b_play  = _btn("▶", "Play / pause (Space)", self.toggle_play)
        self.slider  = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setToolTip("Scrub frames (←/→, Shift+←/→ ×10, Home/End; "
                               "Shift+wheel over the image)")
        self.slider.valueChanged.connect(self.seek)
        self.lbl = QLabel(self)
        self.lbl.setMinimumWidth(96)
        self.lbl.setStyleSheet("color: #aaaaaa; font-family: monospace;")
        self.fps = QSpinBox(self)
        self.fps.setRange(1, 1000)
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("Playback rate (default: the recording's frame rate)")
        self.fps.valueChanged.connect(self._on_fps)
        self.avg = QSpinBox(self)
        self.avg.setRange(1, MovieSource.MAX_AVG)
        self.avg.setPrefix("avg ")
        self.avg.setToolTip("Show the running mean of the last N frames "
                            "(denoises raw 2P frames)")
        self.avg.valueChanged.connect(lambda _: self.src and self.cell_view.showing_movie
                                      and self.seek(self.cur))

        # Take the cell pane's width instead of asking for one: a width hint
        # here pulls the window's splitters off their even split.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(4)
        for w in (self.b_open, self.b_movie, self.b_play):
            lay.addWidget(w)
        lay.addWidget(self.slider, 1)
        for w in (self.lbl, self.fps, self.avg):
            lay.addWidget(w)

        cell_view.frame_scrolled.connect(self.step)
        trace_view.cursor_dragged.connect(self.seek)
        self.set_store(store)

    # ── Store / file ──────────────────────────────────────────────────────────

    def set_store(self, store: 'ComponentStore'):
        """Reset for a new results file and look for its movie."""
        self.stop()
        self._close()
        self.store = store
        self.cur   = 0
        self.cell_view.clear_movie()
        self.trace_view.set_cursor(None)
        self.path = find_session_movie(store)
        fr = getattr(store, "fr", None)
        self.fps.blockSignals(True)
        self.fps.setValue(int(round(fr)) if fr and np.isfinite(fr) and fr >= 1 else 30)
        self.fps.blockSignals(False)
        self._update_ui()

    def open_dialog(self):
        start = self.path or (_os_dirname(self.store.fpath) if self.store.fpath else "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open movie", start,
            "Stacks (*.tif *.tiff *.msr);;TIFF (*.tif *.tiff);;MSR (*.msr);;All files (*)")
        if path and self.open_path(path):
            self.set_active(True)

    def open_path(self, path: str) -> bool:
        try:
            src = MovieSource(path, self.store.dims)
        except Exception as exc:
            QMessageBox.warning(self, "Movie",
                                f"Cannot open {os.path.basename(path)}:\n{exc}")
            return False
        self.stop()
        self._close()
        self.src, self.path, self.cur = src, path, 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, src.n_frames - 1)
        self.slider.setPageStep(max(1, src.n_frames // 50))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._update_ui()
        win = self.window()
        if hasattr(win, "statusBar"):
            win.statusBar().showMessage(
                f"Movie: {path}  ({src.n_frames} frames)", 8000)
        return True

    def _ensure_open(self) -> bool:
        if self.src is not None:
            return True
        return bool(self.path) and self.open_path(self.path)

    def _close(self):
        if self.src is not None:
            self.src.close()
            self.src = None

    def shutdown(self):
        self.stop()
        self._close()

    # ── Playback ──────────────────────────────────────────────────────────────

    def set_active(self, on: bool):
        """Movie frame (on) or the static background image (off)."""
        if on:
            if not self._ensure_open():
                self._set_movie_btn(False)
                return
            self.seek(self.cur)
        else:
            self.stop()
            self.cell_view.clear_movie()
            self.trace_view.set_cursor(None)
            self._set_movie_btn(False)
            self._update_ui()

    def seek(self, i: int):
        if not self._ensure_open():
            return
        src = self.src
        self.cur = int(min(max(int(i), 0), src.n_frames - 1))
        if self.slider.value() != self.cur:
            self.slider.blockSignals(True)
            self.slider.setValue(self.cur)
            self.slider.blockSignals(False)
        try:
            img = src.mean(self.cur, self.avg.value())
        except Exception as exc:
            self.stop()
            QMessageBox.warning(self, "Movie", f"Frame {self.cur}: {exc}")
            return
        self.cell_view.set_movie_frame(img, src.levels())
        self.sync_cursor()
        self._set_movie_btn(True)
        self._update_ui()

    def sync_cursor(self):
        """Trace cursor only when movie frames and trace samples align 1:1."""
        if self.src is not None and self.cell_view.showing_movie \
                and self.src.n_frames == self.store.disp_T:
            self.trace_view.set_cursor(self.cur)
        else:
            self.trace_view.set_cursor(None)

    def step(self, d: int):
        self.seek(self.cur + int(d))

    def seek_end(self):
        if self._ensure_open():
            self.seek(self.src.n_frames - 1)

    def toggle_play(self):
        if self._timer.isActive():
            self.stop()
        else:
            self.play()

    def play(self):
        if not self._ensure_open():
            return
        if not self.cell_view.showing_movie:
            self.seek(self.cur)
        self._timer.start(max(1, int(round(1000 / self.fps.value()))))
        self.b_play.setText("⏸")

    def stop(self):
        self._timer.stop()
        self.b_play.setText("▶")

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def _tick(self):
        nxt = self.cur + 1
        self.seek(0 if nxt >= self.src.n_frames else nxt)    # loops

    def _on_fps(self, v):
        if self._timer.isActive():
            self._timer.setInterval(max(1, int(round(1000 / v))))

    # ── UI state ──────────────────────────────────────────────────────────────

    def _set_movie_btn(self, on: bool):
        self.b_movie.blockSignals(True)
        self.b_movie.setChecked(on)
        self.b_movie.blockSignals(False)

    def _update_ui(self):
        have = bool(self.path)
        for w in (self.b_movie, self.b_play, self.slider, self.fps, self.avg):
            w.setEnabled(have)
        open_tip = "Open a movie (.tif / .msr) to play behind the ROIs…"
        if not have:
            self.lbl.setText("no movie")
            self.b_open.setToolTip(open_tip)
            self.b_movie.setToolTip("No movie found next to the results — ⏏ to open one")
            return
        self.b_open.setToolTip(f"{open_tip}\nCurrent: {self.path}")
        self.b_movie.setToolTip(f"Show {os.path.basename(self.path)} as the background "
                                "(off: the static BG image)")
        if self.src is None:
            self.lbl.setText("not loaded")
        else:
            self.lbl.setText(f"{self.cur:>6d} / {self.src.n_frames - 1}")


# ── Trace viewer ──────────────────────────────────────────────────────────────

class TraceViewer(pg.PlotWidget):
    """Bottom-left — stacked normalised traces for selected components.
    While a movie plays, a yellow frame cursor marks the shown frame; drag
    it to seek."""

    cursor_dragged = pyqtSignal(int)

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(parent, background=BG_DARK)
        self.store = store
        self._sel  = []
        self._cursor_frame = None
        self._cursor = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen('#ffd23f', width=1.5),
            hoverPen=pg.mkPen('#ffe680', width=3))
        self._cursor.setZValue(10)
        self._cursor.sigDragged.connect(
            lambda line: self.cursor_dragged.emit(int(round(line.value()))))
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

    def set_cursor(self, frame):
        """Show the movie frame cursor at *frame* (None hides it)."""
        self._cursor_frame = frame
        if frame is None:
            if self._cursor.scene() is not None:
                self.removeItem(self._cursor)
            return
        if self._cursor.scene() is None and self._sel:
            self.addItem(self._cursor, ignoreBounds=True)
        self._cursor.setBounds((0, max(self.store.disp_T - 1, 0)))
        self._cursor.setValue(frame)

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
            if self._cursor_frame is not None:     # clear() dropped it
                self.set_cursor(self._cursor_frame)
        except Exception as exc:
            self.clear()
            ti = pg.TextItem(f"Draw error: {exc}", color='#ff4444',
                             anchor=(0.5, 0.5))
            self.addItem(ti)


# ── Event viewer ──────────────────────────────────────────────────────────────

class EventViewer(pg.PlotWidget):
    """Below the traces — event detection for the most recently selected
    component (SLab's dF/f, filtered + peaks and binary panels in one):
    raw ΔF/F (grey), filtered ΔF/F (green), threshold (dashed), binary
    onset→peak plateaus (filled white) and detected peaks (red lines)."""

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(parent, background=BG_DARK)
        self.store = store
        self._sel  = []
        self.getAxis('bottom').setTextPen(pg.mkPen('#888888'))
        self.getAxis('left').setTextPen(pg.mkPen('#888888'))
        self.getAxis('left').setLabel('ΔF/F', color='#888888')
        self.getAxis('left').enableAutoSIPrefix(False)   # no "(x0.001)"
        self.setMouseEnabled(x=True, y=False)
        self._redraw()

    def set_selection(self, sel):
        self._sel = list(sel)
        self._redraw()

    def refresh(self):
        self._redraw()

    def _message(self, text):
        # Title only: this plot is x-linked to the trace viewer, so changing
        # the range here would squash the traces.
        self.setTitle(f"<span style='color:#666666;font-size:9pt'>{text}</span>")

    def _redraw(self):
        self.clear()
        self.setTitle(None)
        s = self.store
        if s.events is None:
            self._message("No events — Analysis ▸ Detect events…"
                          if s.can_detect else
                          "No events (file has no ΔF/F to detect on)")
            return
        if not self._sel:
            self._message("Select a component to see its events")
            return
        i  = self._sel[-1]
        ev = s.events
        t  = np.arange(ev.binary.shape[1], dtype=np.float32)
        if s.dff is not None:
            self.plot(t, s.dff[i], pen=pg.mkPen('#666666', width=1))
        self.plot(t, ev.binary[i], pen=pg.mkPen('#dddddd', width=1),
                  fillLevel=0, brush=pg.mkBrush(255, 255, 255, 70))
        self.plot(t, ev.filtered[i], pen=pg.mkPen('#44dd66', width=1.2))
        thr = float(ev.thresholds[i])
        if thr > 0:
            self.addItem(pg.InfiniteLine(
                pos=thr, angle=0,
                pen=pg.mkPen('#ccaa00', width=1, style=Qt.PenStyle.DashLine)))
        for pk in ev.peaks[i][:, 0]:
            self.addItem(pg.InfiniteLine(pos=float(pk), angle=90,
                                         pen=pg.mkPen('#ff4444', width=1)))
        flag = "✓ included" if s.approved[i] else "✗ excluded"
        auto = "auto ✓" if ev.auto_approved[i] else "auto ✗"
        self.setTitle(
            f"<span style='color:#bbbbbb;font-size:9pt'>{s.labels[i]} — "
            f"{ev.n_events(i)} events ({ev.frequencies[i]:.2f}/min), "
            f"thr {thr:.3f} — {flag} ({auto})</span>")
        self.setXRange(0, len(t), padding=0)
        self.enableAutoRange(axis='y')


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

_COMP_ROLE = Qt.ItemDataRole.UserRole        # component index of the row
_SORT_ROLE = Qt.ItemDataRole.UserRole + 1    # typed sort key
_MISS_ROLE = Qt.ItemDataRole.UserRole + 2    # True for NaN / missing values


def _num_key(v) -> tuple:
    """Numeric sort key; NaN / None compare after every number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return (1, 0.0)
    return (1, 0.0) if f != f else (0, f)


def _is_missing(key) -> bool:
    return isinstance(key, tuple) and len(key) == 2 and key[0] == 1 \
        and isinstance(key[1], float)


def _label_key(label: str) -> tuple:
    """Natural order: C002 < C010 < C002+C010."""
    return tuple((0, int(t), "") if t.isdigit() else (1, 0, t)
                 for t in re.split(r"(\d+)", str(label)) if t)


class _SortItem(QTableWidgetItem):
    """Sorts by _SORT_ROLE (numbers numerically) instead of display text.
    Missing values (NaN) stay at the bottom in both sort directions."""

    def __lt__(self, other):
        a, b = self.data(_SORT_ROLE), other.data(_SORT_ROLE)
        if a is None or b is None:
            return super().__lt__(other)
        ma, mb = bool(self.data(_MISS_ROLE)), bool(other.data(_MISS_ROLE))
        if ma != mb:
            tw = self.tableWidget()
            desc = tw is not None and tw.horizontalHeader().sortIndicatorOrder() \
                == Qt.SortOrder.DescendingOrder
            # Qt reverses the comparison for descending order; pre-flip so
            # the missing value still lands last.
            return ma if desc else mb
        return a < b


_QC_SHORT = {"empty": "empty", "area_small": "area<", "area_large": "area>",
             "solidity": "solid", "eccentricity": "ecc", "edge": "edge",
             "caiman_quality": "qual", "inactive": "inact"}


def _qc_cell(v, kind: str) -> tuple:
    """(text, sort key[, tooltip]) for one roi_filter value."""
    if kind == "bool":
        return ("✓" if v else "✗", (0 if v else 1,))
    if kind == "reasons":
        names = _rf().decode_reasons(int(v))
        return (",".join(_QC_SHORT.get(n, n) for n in names) or "—",
                (len(names), int(v)),
                ", ".join(names) or "passes every criterion")
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = float("nan")
    if f != f:
        return ("—", _num_key(f))
    text = {"int": f"{int(f)}", "f0": f"{f:.0f}", "f2": f"{f:.2f}",
            "pct": f"{100 * f:.1f}"}.get(kind, f"{f:.3f}")
    return (text, _num_key(f))


class ComponentTable(QTableWidget):
    """
    Left panel — one row per component with quality metrics.
    Selected rows are lightly tinted with their palette colour.
    A corr-pair click tints those two rows in orange / magenta.

    Click a column header to sort (again to reverse).  Sorting changes
    only the view: callers address components by store index through
    :meth:`selected_comps`, :meth:`select_comps`, :meth:`row_of` and
    :meth:`comp_of`, never by table row.  The sort order survives
    refreshes (merge, delete, event detection).
    """

    _FIXED    = ("Label", "Max A", "Peak ΔF")
    _OPTIONAL = (("SNR",   "SNR_comp"),
                 ("r_val", "r_values"),
                 ("CNN",   "cnn_preds"))

    # roi_filter fields, each shown only when present in the results file:
    # (header, field, display kind, header tooltip)
    _QC_COLS = (
        ("QC",      "passed",       "bool",
         "ROI filter verdict: footprint ∧ edge ∧ CaImAn quality (∧ active if required)"),
        ("Active",  "active",       "bool",
         "ROI filter activity: event count / rate / peak / skew thresholds met"),
        ("Fail",    "fail_reasons", "reasons",
         "Failed ROI-filter criteria (hover a cell for the full names)"),
        ("Area",    "area_px",      "int",
         "Footprint area (px) at footprint_thr × peak"),
        ("Solid",   "solidity",     "f2",
         "Solidity = area / convex-hull area (low = ragged or ring-shaped)"),
        ("Ecc",     "eccentricity", "f2",
         "Eccentricity of the footprint (0 = circle, 1 = line)"),
        ("Edge",    "edge_dist_px", "f0",
         "Centroid distance to the FOV border (px)"),
        ("Rate",    "event_rate",   "f2",
         "ROI filter: events per minute on ΔF/F (no SNR gating)"),
        ("Pk ΔF/F", "peak_dff",     "f2",
         "Peak of the smoothed ΔF/F"),
        ("Skew",    "skewness",     "f2",
         "Skewness of ΔF/F"),
        ("Act %",   "active_frac",  "pct",
         "Percent of frames inside detected events"),
    )

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
        self._row_of: dict[int, int] = {}
        hdr = self.horizontalHeader()
        hdr.setSortIndicatorShown(True)
        hdr.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)   # start in store order
        hdr.setToolTip("Click a column to sort; click again to reverse")
        self.model().layoutChanged.connect(self._reindex)
        self._populate()
        self.setSortingEnabled(True)

    # ── Public ────────────────────────────────────────────────────────────────

    def refresh(self):
        self._populate()

    def comp_of(self, row: int) -> int:
        """Store index of the component shown in table *row*."""
        it = self.item(row, 0)
        return int(it.data(_COMP_ROLE)) if it is not None else row

    def row_of(self, comp: int) -> int:
        """Table row currently showing store component *comp* (-1 if none)."""
        return self._row_of.get(comp, -1)

    def selected_comps(self) -> list:
        """Selected components as sorted store indices."""
        return sorted({self.comp_of(idx.row()) for idx in self.selectedIndexes()})

    def select_comps(self, comps, clear: bool = True, scroll_to=None) -> None:
        """Select store components *comps* without emitting
        itemSelectionChanged (callers propagate via _on_table_sel)."""
        sm    = self.selectionModel()
        flags = (QItemSelectionModel.SelectionFlag.Select
                 | QItemSelectionModel.SelectionFlag.Rows)
        self.blockSignals(True)
        if clear:
            sm.clearSelection()
        for c in comps:
            r = self.row_of(c)
            if r >= 0:
                sm.select(self.model().index(r, 0), flags)
        self.blockSignals(False)
        if scroll_to is not None and self.row_of(scroll_to) >= 0:
            self.scrollToItem(self.item(self.row_of(scroll_to), 0),
                              QAbstractItemView.ScrollHint.EnsureVisible)

    def tint_selection(self, sel: list):
        """Palette tint on selected components; dark reset on others.
        *sel* holds store indices; the palette slot follows their order.
        Note: no blockSignals here — setBackground emits itemChanged,
        which is not connected to any slot, so it is safe to let it fire.
        Blocking table signals here was causing itemSelectionChanged to be
        re-emitted after blockSignals(False) in PyQt6, overwriting _sel.
        """
        dark = QColor(30, 30, 30)
        slot = {c: k for k, c in enumerate(sel)}
        for comp in range(self.store.n):
            if comp in slot:
                hx = PALETTE_HEX[slot[comp] % len(PALETTE_HEX)].lstrip("#")
                bg = QColor(int(hx[0:2], 16),
                            int(hx[2:4], 16),
                            int(hx[4:6], 16), 55)
            else:
                bg = dark
            self._set_row_bg(self.row_of(comp), bg)

    def tint_pair(self, i: int, j: int):
        """Orange / magenta tint for a corr-clicked pair (store indices).
        Note: no blockSignals — see tint_selection for rationale.
        """
        dark = QColor(30, 30, 30)
        for comp in range(self.store.n):
            if   comp == i: bg = QColor(255, 140,   0, 80)
            elif comp == j: bg = QColor(255,   0, 255, 80)
            else:           bg = dark
            self._set_row_bg(self.row_of(comp), bg)

    # ── Private ───────────────────────────────────────────────────────────────

    def _col_names(self):
        cols = list(self._FIXED)
        for label, attr in self._OPTIONAL:
            if getattr(self.store, attr, None) is not None:
                cols.append(label)
        qc = self.store.qc
        if qc is not None:
            cols += [h for h, fld, _k, _t in self._QC_COLS if fld in qc]
        if self.store.events is not None:
            cols.append("Ev/min")
        if self.store.has_approval:
            cols.append("Incl")
        return cols

    def _populate(self):
        s    = self.store
        cols = self._col_names()
        hdr  = self.horizontalHeader()
        sort_col, sort_ord = hdr.sortIndicatorSection(), hdr.sortIndicatorOrder()
        sorting = self.isSortingEnabled()
        self.setSortingEnabled(False)          # rows must land at row == comp
        self.blockSignals(True)
        self.clear()
        self.setColumnCount(len(cols))
        self.setHorizontalHeaderLabels(cols)
        tips = {h: t for h, _f, _k, t in self._QC_COLS}
        tips["Incl"] = ("Include flag (I toggles). Starts from the ROI filter's "
                        "QC verdict, or from event auto-approval")
        for c, h in enumerate(cols):
            if h in tips:
                self.horizontalHeaderItem(c).setToolTip(tips[h])
        self.setRowCount(s.n)
        dark = QColor(30, 30, 30)
        for i in range(s.n):
            vals = [(s.labels[i], _label_key(s.labels[i])),
                    (f"{s.max_a(i):.3f}", _num_key(s.max_a(i))),
                    (f"{s.peak_df(i):.1f}", _num_key(s.peak_df(i)))]
            for _, attr in self._OPTIONAL:
                if getattr(s, attr, None) is not None:
                    v = getattr(s, attr)[i]
                    vals.append((f"{v:.3f}", _num_key(v)))
            if s.qc is not None:
                for _h, fld, kind, _t in self._QC_COLS:
                    if fld in s.qc:
                        vals.append(_qc_cell(s.qc[fld][i], kind))
            if s.events is not None:
                fq = s.events.frequencies[i]
                vals.append((f"{fq:.2f}", _num_key(fq)))
            if s.has_approval:
                vals.append(("✓" if s.approved[i] else "✗",
                             (0 if s.approved[i] else 1,)))
            for c, (text, key, *tip) in enumerate(vals):
                item = _SortItem(text)
                if tip:
                    item.setToolTip(tip[0])
                item.setData(_COMP_ROLE, i)
                item.setData(_SORT_ROLE, key)
                item.setData(_MISS_ROLE, _is_missing(key))
                item.setBackground(QBrush(dark))
                self.setItem(i, c, item)
        self.blockSignals(False)
        self._reindex()
        if sort_col >= self.columnCount():     # e.g. Incl gone after a reload
            sort_col = -1
        hdr.setSortIndicator(sort_col, sort_ord)
        self.setSortingEnabled(sorting)        # re-applies the user's sort
        self._reindex()
        self.mark_hidden()

    def _reindex(self):
        """Rebuild component → row after populate or a sort."""
        self._row_of = {}
        for r in range(self.rowCount()):
            it = self.item(r, 0)
            if it is not None and it.data(_COMP_ROLE) is not None:
                self._row_of[int(it.data(_COMP_ROLE))] = r

    def mark_hidden(self):
        """Dim hidden rows; tint excluded rows red (once events exist) and
        keep the Incl column (text and sort key) in sync with the store."""
        grey   = QColor(105, 105, 105)
        normal = QColor(220, 220, 220)
        excl   = QColor(255, 110, 110)
        s      = self.store
        has_ev = s.has_approval and self.columnCount() and \
            self.horizontalHeaderItem(self.columnCount() - 1).text() == "Incl"
        # Changing the Incl key while sorted by it would re-sort mid-loop and
        # invalidate row_of; suspend sorting and re-apply it once at the end.
        sorting = self.isSortingEnabled()
        self.setSortingEnabled(False)
        for comp in range(s.n):
            row = self.row_of(comp)
            if row < 0:
                continue
            if s.is_hidden(comp):
                fg = grey
            elif has_ev and not s.approved[comp]:
                fg = excl
            else:
                fg = normal
            if has_ev:
                it = self.item(row, self.columnCount() - 1)
                if it:
                    it.setText("✓" if s.approved[comp] else "✗")
                    it.setData(_SORT_ROLE, (0 if s.approved[comp] else 1,))
            for col in range(self.columnCount()):
                it = self.item(row, col)
                if it:
                    it.setForeground(QBrush(fg))
        self.setSortingEnabled(sorting)
        self._reindex()

    def _set_row_bg(self, row: int, color: QColor):
        if row < 0:
            return
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                item.setBackground(QBrush(color))


# ── Main window ───────────────────────────────────────────────────────────────

# ── Preference editor ─────────────────────────────────────────────────────────

def _fmt_num(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return "×".join(_fmt_num(x) for x in v)
    if isinstance(v, bool):
        return "on" if v else "off"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:g}"


class _PrefField(QWidget):
    """One preference value.  Nullable numbers get a checkbox: unchecked =
    null (shown as ``null_text``, plus the run's value when known)."""

    def __init__(self, kind, rng, null_text, hint=None, parent=None):
        super().__init__(parent)
        self.kind, self.null_text = kind, null_text
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.chk = self.spin = self.box = self.combo = self.hint = None
        if kind == "bool":
            self.box = QCheckBox(self)
            lay.addWidget(self.box)
        elif kind == "tristate":
            self.combo = QComboBox(self)
            first = null_text + (f" ({_fmt_num(hint)})" if hint is not None else "")
            self.combo.addItems([first, "on", "off"])
            lay.addWidget(self.combo)
        else:
            lo, hi, step, dec = rng
            if kind == "int":
                self.spin = QSpinBox(self)
                self.spin.setRange(int(lo), int(hi))
                self.spin.setSingleStep(int(step))
            else:
                self.spin = QDoubleSpinBox(self)
                self.spin.setRange(float(lo), float(hi))
                self.spin.setDecimals(int(dec))
                self.spin.setSingleStep(float(step))
            self.spin.setMinimumWidth(110)
            if null_text is not None:
                self.chk = QCheckBox(self)
                self.chk.setToolTip(f"Unchecked: {null_text}")
                self.hint = QLabel(null_text + (f": {_fmt_num(hint)}" if hint is not None else ""), self)
                self.hint.setStyleSheet("color: #888888;")
                self.chk.toggled.connect(self._sync)
                lay.addWidget(self.chk)
            lay.addWidget(self.spin)
            if self.hint is not None:
                lay.addWidget(self.hint)
            if hint is not None and not isinstance(hint, (list, tuple, bool)):
                try:
                    self.spin.setValue(float(hint))
                except (TypeError, ValueError):
                    pass
            elif isinstance(hint, (list, tuple)) and hint:
                self.spin.setValue(float(hint[0]))
        lay.addStretch(1)

    def _sync(self):
        on = self.chk.isChecked()
        self.spin.setEnabled(on)
        self.hint.setVisible(not on)

    def value(self):
        if self.box is not None:
            return bool(self.box.isChecked())
        if self.combo is not None:
            return {0: None, 1: True, 2: False}[self.combo.currentIndex()]
        if self.chk is not None and not self.chk.isChecked():
            return None
        v = self.spin.value()
        return int(v) if self.kind == "int" else float(v)

    def set_value(self, v):
        if self.box is not None:
            self.box.setChecked(bool(v))
            return
        if self.combo is not None:
            self.combo.setCurrentIndex(0 if v is None else 1 if v else 2)
            return
        if isinstance(v, (list, tuple)):
            v = v[0] if len(v) else None
        if self.chk is not None:
            self.chk.setChecked(v is not None)
            self._sync()
        if v is not None:
            self.spin.setValue(int(v) if self.kind == "int" else float(v))


class PreferencesDialog(QDialog):
    """Edit ▸ Preferences — one tab per entry in PREF_SECTIONS, saved as JSON
    (see prefs_path).  The ROI-filter tab can load the defaults, the
    ``roi_filter`` section of a pipeline JSON, or this file's stored
    filter parameters."""

    def __init__(self, prefs: dict, run_hints: dict = None,
                 file_params: dict = None, section: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(560, 720)
        self._prefs  = {k: (dict(v) if isinstance(v, dict) else v) for k, v in prefs.items()}
        self._fields = {}
        self._file_params = file_params
        self.tabs = QTabWidget(self)
        for sec, (title, specs, defaults) in PREF_SECTIONS.items():
            page = QWidget()
            v = QVBoxLayout(page)
            forms = {}
            for grp, key, label, kind, rng, null_text, tip in specs:
                if grp not in forms:
                    gb = QGroupBox(grp, page)
                    forms[grp] = QFormLayout(gb)
                    v.addWidget(gb)
                hint = (run_hints or {}).get(key) if null_text == "from run" else None
                w = _PrefField(kind, rng, null_text, hint, page)
                lbl = QLabel(label, page)
                if tip:
                    w.setToolTip(tip)
                    lbl.setToolTip(tip)
                forms[grp].addRow(lbl, w)
                self._fields.setdefault(sec, {})[key] = w
            if sec == "roi_filter":
                row = QHBoxLayout()
                b_def  = QPushButton("Defaults", page)
                b_json = QPushButton("From pipeline JSON…", page)
                b_file = QPushButton("From this file", page)
                b_def.setToolTip("roi_filter defaults (same as template_pipeline.json)")
                b_json.setToolTip("Load the roi_filter section of a _pipeline.json / template")
                b_file.setToolTip("The parameters stored in this results file's roi_filter group")
                b_file.setEnabled(bool(file_params))
                b_def.clicked.connect(lambda: self._apply("roi_filter", defaults()))
                b_json.clicked.connect(self._from_json)
                b_file.clicked.connect(lambda: self._apply("roi_filter", self._file_params or {}))
                for b in (b_def, b_json, b_file):
                    row.addWidget(b)
                row.addStretch(1)
                v.addLayout(row)
            v.addStretch(1)
            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, title)
            cur = defaults()
            cur.update({k: x for k, x in (self._prefs.get(sec) or {}).items() if k in cur})
            self._apply(sec, cur)
            if sec == section:
                self.tabs.setCurrentIndex(self.tabs.count() - 1)

        where = QLabel(f"Saved to {prefs_path()}", self)
        where.setStyleSheet("color: #888888;")
        where.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel, self)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(where)
        lay.addWidget(bb)

    def _apply(self, sec: str, values: dict):
        for k, w in self._fields.get(sec, {}).items():
            if k in values:
                w.set_value(values[k])

    def _from_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pipeline JSON", "", "JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path) as fh:
                text = re.sub(r"(?m)^\s*//.*$", "", fh.read())
            sec = json.loads(text).get("roi_filter")
        except Exception as exc:
            QMessageBox.warning(self, "Preferences", f"Cannot read {path}:\n{exc}")
            return
        if not isinstance(sec, dict):
            QMessageBox.information(self, "Preferences",
                                    f"{os.path.basename(path)} has no roi_filter section.")
            return
        self._apply("roi_filter", sec)

    def section_values(self, sec: str) -> dict:
        return {k: w.value() for k, w in self._fields.get(sec, {}).items()}

    def values(self) -> dict:
        """All preferences: edited sections merged over what was loaded
        (keys the editor does not show, e.g. roi_filter.enabled, are kept)."""
        out = dict(self._prefs)
        for sec in self._fields:
            cur = dict(out.get(sec) or {})
            cur.update(self.section_values(sec))
            out[sec] = cur
        return out


class RunFilterDialog(QDialog):
    """Analysis ▸ Run ROI filter — shows the resolved thresholds (preferences
    + the run's CaImAn params), what will be skipped, and whether Incl is
    reset to the verdict."""

    def __init__(self, window: 'InspectorWindow', store: 'ComponentStore'):
        super().__init__(window)
        self.setWindowTitle("Run ROI filter")
        self.window_, self.store = window, store
        lay = QVBoxLayout(self)
        intro = (f"Evaluate the {store.n} shown components with the ROI-filter "
                 "preferences.  ")
        intro += ("This replaces the file's current ROI-filter fields (undoable)."
                  if store.qc is not None else
                  "This file has no ROI-filter fields yet.")
        intro += "  Save to keep the result."
        lbl = QLabel(intro, self)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        self.summary = QLabel(self)
        self.summary.setStyleSheet("font-family: monospace;")
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.summary)
        self.warn = QLabel(self)
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #e0a030;")
        lay.addWidget(self.warn)
        self.chk_incl = QCheckBox("Set Incl from the new QC verdict", self)
        self.chk_incl.setChecked(not store.has_approval)
        self.chk_incl.setToolTip(
            "Replace the include flags with `passed`."
            + ("  Replaces event approvals." if store.events is not None else ""))
        lay.addWidget(self.chk_incl)
        bb = QDialogButtonBox(self)
        b_run = bb.addButton("Run", QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QDialogButtonBox.StandardButton.Cancel)
        b_edit = bb.addButton("Edit parameters…", QDialogButtonBox.ButtonRole.ActionRole)
        b_edit.clicked.connect(self._edit)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        b_run.setDefault(True)
        lay.addWidget(bb)
        self._refresh()

    @property
    def set_incl(self) -> bool:
        return self.chk_incl.isChecked()

    def _edit(self):
        if self.window_._edit_prefs("roi_filter"):
            self._refresh()

    def _refresh(self):
        s, rf = self.store, _rf()
        p = s.resolve_filter_params(roi_filter_prefs())
        lo, hi = rf.area_limits(p)
        n = _fmt_num
        area = ("off" if lo is None and hi is None else
                f"{n(round(lo)) if lo is not None else '0'}–{n(round(hi)) if hi is not None else '∞'} px")
        def lim(key, op):
            return "off" if p.get(key) is None else f"{op} {n(p[key])}"
        lines = [
            f"Footprint   mask ≥ {n(p.get('footprint_thr'))}×peak, area {area}"
            + (f" (gSig {n(p.get('gSig'))})" if p.get("gSig") is not None else ""),
            f"            solidity {lim('min_solidity', '≥')}, eccentricity {lim('max_eccentricity', '≤')}",
            f"Edge        centroid {lim('edge_px', '≥')} px from the border",
        ]
        have_q = any(a is not None for a in (s.SNR_comp, s.r_values, s.cnn_preds))
        if not p.get("use_caiman_quality", True):
            lines.append("Quality     off")
        else:
            lines.append(
                f"Quality     SNR ≥ {n(p.get('min_SNR'))} (floor {n(p.get('SNR_lowest'))}), "
                f"r ≥ {n(p.get('rval_thr'))} (floor {n(p.get('rval_lowest'))}), "
                + (f"CNN ≥ {n(p.get('min_cnn_thr'))} (floor {n(p.get('cnn_lowest'))})"
                   if p.get("use_cnn") else "CNN off"))
        acts = [f"{lab} ≥ {n(p[k])}" for k, lab in (
            ("min_events", "events"), ("min_event_rate", "rate"),
            ("min_peak_dff", "peak ΔF/F"), ("min_skewness", "skew")) if p.get(k) is not None]
        lines.append(f"Activity    {', '.join(acts) or 'no thresholds'}; cutoff "
                     f"{n(p.get('event_threshold'))}, fr {n(p.get('fr'))} Hz, "
                     f"decay {n(p.get('decay_time'))} s")
        lines.append(f"QC          {'requires' if p.get('require_activity') else 'ignores'} activity")
        self.summary.setText("\n".join(lines))
        warns = []
        if lo is None and hi is None and (p.get("min_area_frac") is not None
                                          or p.get("max_area_frac") is not None):
            warns.append("No gSig known: set it in the preferences or the area check is skipped.")
        if p.get("use_caiman_quality", True) and not have_q:
            warns.append("No SNR / r / CNN scores in this file: CaImAn quality is skipped.")
        if s.dff is None:
            warns.append("No ΔF/F traces in this file: activity fields are skipped.")
        elif not (p.get("fr") and p.get("decay_time")):
            warns.append("Frame rate or decay time unknown: activity fields are skipped.")
        self.warn.setText("\n".join(warns))
        self.warn.setVisible(bool(warns))


class InspectorWindow(QMainWindow):

    def __init__(self, store: ComponentStore):
        super().__init__()
        self.store = store
        self.setWindowTitle("CaImAn Component Inspector")
        self.resize(1760, 980)
        self._pair  = None
        self._blind = None      # blinded-review session state (dict) or None

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
        a = QAction("Open &movie (.tif / .msr)…", self)
        a.setShortcut(QKeySequence("Ctrl+M"))
        a.setToolTip("Play / scrub a raw stack behind the ROI overlay.")
        a.triggered.connect(lambda: self.movie_bar.open_dialog())
        vm.addAction(a)
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
        vm.addSeparator()
        a = QAction("Session &info…", self)
        a.setToolTip("SNR range, recording length, approval counts, "
                     "imaging frequency, decay time, analysis version.")
        a.triggered.connect(self._session_info)
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
        em.addSeparator()
        a = QAction("&Preferences…", self)
        a.setShortcut(QKeySequence("Ctrl+,"))
        a.setMenuRole(QAction.MenuRole.PreferencesRole)
        a.setToolTip("ROI-filter thresholds and other inspector settings")
        a.triggered.connect(lambda: self._edit_prefs())
        em.addAction(a)

        am = mb.addMenu("&Analysis")
        self.act_detect = QAction("&Detect events…", self)
        self.act_detect.setShortcut(QKeySequence("Ctrl+D"))
        self.act_detect.setToolTip(
            "SLab calcium-event detection on ΔF/F for every component;\n"
            "sets the include flags to the automatic approval.")
        self.act_detect.triggered.connect(self._detect_events_dialog)
        am.addAction(self.act_detect)
        self.act_export = QAction("&Export event stats…", self)
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export.setToolTip(
            "Write <session>_PeakStats and <session>_ROIStats for the included "
            "components (.xlsx, or .csv without openpyxl).")
        self.act_export.triggered.connect(self._export_events)
        am.addAction(self.act_export)
        am.addSeparator()
        self.act_filter = QAction("Run &ROI filter…", self)
        self.act_filter.setShortcut(QKeySequence("Ctrl+Shift+F"))
        self.act_filter.setToolTip(
            "Footprint / edge / CaImAn-quality / activity filter with the\n"
            "preference thresholds; adds or replaces the ROI-filter columns.")
        self.act_filter.triggered.connect(self._run_filter_dialog)
        am.addAction(self.act_filter)
        am.addSeparator()
        a = QAction("&Batch detect events in folder…", self)
        a.setToolTip("Run event detection on every results file under a folder "
                     "and write stroh_results into each (SLab Auto Mode).")
        a.triggered.connect(self._batch_detect)
        am.addAction(a)
        am.addSeparator()
        self.act_blind = QAction("&Blinded review…", self)
        self.act_blind.setShortcut(QKeySequence("Ctrl+R"))
        self.act_blind.setToolTip(
            "Shuffled, file-name-blind include/exclude rating of a folder of "
            "results, saved per rater (SLab Randomization Mode).")
        self.act_blind.triggered.connect(self._start_blinded)
        am.addAction(self.act_blind)
        self.act_blind_end = QAction("&End blinded review", self)
        self.act_blind_end.setEnabled(False)
        self.act_blind_end.triggered.connect(self._end_blinded)
        am.addAction(self.act_blind_end)

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

        self.act_include = QAction("✓  Include / exclude", self)
        self.act_include.setToolTip(
            "Toggle the include flag of the selected component(s) (I).\n"
            "Non-destructive curation, saved as SLab manually_approved;\n"
            "excluded components are drawn red and left out of exports.")
        self.act_include.setShortcut(QKeySequence("I"))
        self.act_include.setEnabled(False)
        self.act_include.triggered.connect(self._toggle_include)
        tb.addAction(self.act_include)
        self.addAction(self.act_include)

        self.act_blind_next = QAction("💾  Save rating && next ▶", self)
        self.act_blind_next.setToolTip(
            "Blinded review: save your include/exclude ratings for this file "
            "and open the next one (Ctrl+Return).")
        self.act_blind_next.setShortcut(QKeySequence("Ctrl+Return"))
        self.act_blind_next.triggered.connect(self._blind_save_next)
        self.act_blind_next.setVisible(False)
        tb.addAction(self.act_blind_next)

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

        self.act_hide_excl = QAction("⊘  Hide excluded", self)
        self.act_hide_excl.setToolTip(
            "Hide excluded (not included) components from the all-ROIs layer.")
        self.act_hide_excl.setCheckable(True)
        self.act_hide_excl.triggered.connect(
            lambda checked: self.cell_view.set_hide_excluded(checked))
        tb.addAction(self.act_hide_excl)

        tb.addSeparator()
        tb.addWidget(QLabel("  BG: "))
        self.bg_combo = QComboBox(self)
        self.bg_combo.setToolTip(
            "Background image shown behind the footprints.\n"
            "Use View ▸ Load background image… to add a data projection.")
        self.bg_combo.setMinimumWidth(140)
        self.bg_combo.currentTextChanged.connect(self._on_bg_changed)
        # Picking a static background leaves movie mode
        self.bg_combo.activated.connect(
            lambda _i: self.movie_bar.set_active(False))
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
        self.event_view = EventViewer(s)
        self.corr_view  = CorrMatrix(s)
        self.event_view.setXLink(self.trace_view)

        # Cell viewer with the movie playback bar underneath
        self.movie_bar = MovieBar(self.cell_view, self.trace_view, s)
        cell_box = QWidget()
        cbl = QVBoxLayout(cell_box)
        cbl.setContentsMargins(0, 0, 0, 0)
        cbl.setSpacing(0)
        cbl.addWidget(self.cell_view, 1)
        cbl.addWidget(self.movie_bar)
        self.cell_view.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        for key, slot in (("Space",       self.movie_bar.toggle_play),
                          ("Right",       lambda: self.movie_bar.step(1)),
                          ("Left",        lambda: self.movie_bar.step(-1)),
                          ("Shift+Right", lambda: self.movie_bar.step(10)),
                          ("Shift+Left",  lambda: self.movie_bar.step(-10)),
                          ("Home",        lambda: self.movie_bar.seek(0)),
                          ("End",         self.movie_bar.seek_end)):
            sc = QShortcut(QKeySequence(key), cell_box)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

        # Top row: cell viewer (left) | distance matrix (right)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(cell_box)
        top.addWidget(self.dist_view)
        top.setSizes([1, 1])

        # Bottom row: trace (left) | corr (right)
        traces = QSplitter(Qt.Orientation.Vertical)
        traces.addWidget(self.trace_view)
        traces.addWidget(self.event_view)
        traces.setSizes([3, 2])
        bot = QSplitter(Qt.Orientation.Horizontal)
        bot.addWidget(traces)
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

    def closeEvent(self, ev):
        self.movie_bar.shutdown()
        super().closeEvent(ev)

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
            sel = self.table.selected_comps()
            self.cell_view.set_selection(sel)
            self.trace_view.set_selection(sel)
            self.event_view.set_selection(sel)
            self.table.tint_selection(sel)
            n = len(sel)
            edit = self._blind is None     # blinded review: rate only
            self.act_merge.setEnabled(edit and n >= 2)
            self.act_delete.setEnabled(edit and n >= 1)
            self.act_toggle_vis.setEnabled(n >= 1)
            self.act_include.setEnabled(n >= 1)
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
        self.table.select_comps([c for c in (i, j) if 0 <= c < self.store.n])

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
        cur = self.table.selected_comps()
        if ctrl:
            if idx in cur:
                new_sel = [r for r in cur if r != idx]
            else:
                new_sel = sorted(cur + [idx])
        else:
            new_sel = [idx]

        # Apply to table selection model without triggering the signal
        # (we call _on_table_sel manually below)
        # Scroll to the last-selected row so it is visible
        self.table.select_comps(new_sel,
                                scroll_to=new_sel[-1] if new_sel else None)

        # Propagate to all views
        self._on_table_sel()
        self.statusBar().showMessage(
            f"Selected  {self.store.labels[idx]}  "
            f"({'+ ' if ctrl and len(new_sel) > 1 else ''}"
            f"{len(new_sel)} component{'s' if len(new_sel) != 1 else ''})")

    # ── Merge / Delete ────────────────────────────────────────────────────────

    def _do_merge(self):
        sel = self.table.selected_comps()
        if len(sel) < 2:
            return
        n_sel   = len(sel)
        new_idx = self.store.merge(sel)
        self._full_refresh()
        self.table.select_comps([new_idx], scroll_to=new_idx)
        self._on_table_sel()
        self.statusBar().showMessage(
            f"Merged {n_sel} → {self.store.labels[new_idx]}  "
            f"({self.store.n} components)")

    def _do_delete(self):
        sel = self.table.selected_comps()
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

    # ── Preferences / ROI filter ──────────────────────────────────────────────

    def _edit_prefs(self, section: str = None) -> bool:
        """Open the preference editor; True if saved."""
        s = self.store
        try:
            hints = s.resolve_filter_params({})       # the run's own values
        except Exception:
            hints = {}
        dlg = PreferencesDialog(load_prefs(), hints,
                                s.qc_params if s.qc is not None else None,
                                section, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        try:
            path = save_prefs(dlg.values())
        except OSError as exc:
            QMessageBox.critical(self, "Preferences", f"Could not save:\n{exc}")
            return False
        self.statusBar().showMessage(f"Preferences saved to {path}", 6000)
        return True

    def _run_filter_dialog(self):
        if self._blind_block():
            return
        s = self.store
        dlg = RunFilterDialog(self, s)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        sel = self._selected_rows()
        try:
            summ = s.run_roi_filter(roi_filter_prefs(), set_incl=dlg.set_incl)
        except Exception as exc:
            QMessageBox.critical(self, "Run ROI filter", str(exc))
            return
        self._full_refresh()
        self._reselect(sel)
        msg = f"ROI filter: {summ.get('n_passed', 0)}/{summ['n_total']} passed"
        if "n_active" in summ:
            msg += f", {summ['n_active']} active"
        fails = {k: v for k, v in summ.get("n_fail", {}).items() if v}
        if fails:
            msg += "  (fails: " + ", ".join(f"{k} {v}" for k, v in fails.items()) + ")"
        self.statusBar().showMessage(msg + "  —  Save to keep")

    def _blind_block(self) -> bool:
        if self._blind is not None:
            self.statusBar().showMessage(
                "Blinded review in progress — Analysis ▸ End blinded review first")
            return True
        return False

    def _open_file(self):
        if self._blind_block():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CaImAn results", "",
            "Results or stack (*.hdf5 *.h5 *.tif *.tiff);;"
            "HDF5 files (*.hdf5 *.h5);;TIFF stacks (*.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            store = load_any(path)
            self._swap_store(store)
            self.statusBar().showMessage(
                f"Loaded {store.n} components from {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            import traceback; traceback.print_exc()

    def _save_file(self):
        if self._blind_block():
            return
        if self.store.fpath:
            path = ComponentStore.curated_path(self.store.fpath)
            self._do_save(path)
        else:
            self._save_as_file()

    def _save_as_file(self):
        if self._blind_block():
            return
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
        self.movie_bar.sync_cursor()          # trace length may differ
        self.statusBar().showMessage(f"Trace → {name}")

    # ── Visibility ────────────────────────────────────────────────────────────

    def _toggle_visibility(self):
        sel = self.table.selected_comps()
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
                  self.trace_view, self.event_view, self.corr_view,
                  self.dist_view):
            w.store = new_store
        self._refresh_bg_combo()
        self._refresh_trace_combo()
        self.movie_bar.set_store(new_store)
        self.cell_view.refresh_background()
        self._full_refresh()


    # ── Events / approval ─────────────────────────────────────────────────────

    def _selected_rows(self) -> list:
        """Selected components (store indices, not table rows)."""
        return self.table.selected_comps()

    def _reselect(self, rows):
        rows = [r for r in rows if 0 <= r < self.store.n]
        if not rows:
            return
        self.table.select_comps(rows, clear=False)
        self._on_table_sel()

    def _toggle_include(self):
        s, sel = self.store, self._selected_rows()
        if not sel:
            return
        if not s.has_approval:
            self.statusBar().showMessage(
                "Include flags come from event results or the ROI filter — "
                "run Analysis ▸ Detect events first")
            return
        s.toggle_approved(sel)
        self.table.mark_hidden()
        self.cell_view.refresh_rois()
        self.event_view.refresh()
        self.statusBar().showMessage(f"{s.n_approved}/{s.n} included")

    def run_detection(self, fr, decay_time, progress=None, **kw):
        """Detect events on the current store and refresh, keeping the
        selection.  Raises on failure (callers report it)."""
        sel = self._selected_rows()
        self.store.detect_events(fr, decay_time, progress=progress, **kw)
        self._full_refresh()
        self._reselect(sel)
        ev = self.store.events
        self.statusBar().showMessage(
            f"Events: {sum(len(p) for p in ev.peaks)} in {ev.n} components; "
            f"{int(ev.auto_approved.sum())} auto-approved / included")

    def _detect_events_dialog(self):
        s = self.store
        if not s.can_detect:
            QMessageBox.warning(
                self, "Detect events",
                "This file has no ΔF/F traces (estimates.F_dff).\n"
                "Run cnm.estimates.detrend_df_f() in the pipeline first.")
            return
        if s.has_approval and QMessageBox.question(
                self, "Detect events",
                ("Existing events and include flags will be replaced."
                 if s.events is not None else
                 "Include flags (seeded from the ROI filter) will be replaced "
                 "by event auto-approval.") + "\nContinue?"
                ) != QMessageBox.StandardButton.Yes:
            return
        ev, prm = s.events, (s.events.params if s.events else {})
        dlg  = QDialog(self)
        dlg.setWindowTitle("Detect events")
        form = QFormLayout(dlg)

        def spin(val, lo, hi, step, dec, suffix=""):
            w = QDoubleSpinBox(dlg)
            w.setRange(lo, hi); w.setDecimals(dec); w.setSingleStep(step)
            w.setValue(val); w.setSuffix(suffix)
            return w
        sp_fr  = spin(s.fr or 30.0, 0.1, 1000.0, 0.1, 2, " Hz")
        sp_dec = spin(s.decay_time or 0.4, 0.0, 5.0, 0.01, 3, " s")
        sp_thr = spin(ev.threshold if ev else 0.2, 0.01, 2.0, 0.01, 3)
        sp_snr = spin(prm.get("min_snr", 3.0), 0.0, 100.0, 0.5, 2)
        sp_cnn = spin(prm.get("min_cnn", 0.15), 0.0, 1.0, 0.05, 2)
        sp_lp  = spin(prm.get("lowpass_hz") or 0.0, 0.0, 500.0, 0.5, 2, " Hz")
        sp_lp.setSpecialValueText("SLab (0.974·Nyquist)")
        for label, w, tip in (
                ("Imaging frequency", sp_fr, "Frame rate; sets refractory "
                 "distance and events/min."),
                ("Indicator decay time", sp_dec, "Minimum peak spacing = "
                 "decay time × frame rate."),
                ("Max threshold (ΔF/F)", sp_thr, "Cap on the noise-derived "
                 "threshold min(10·noise, this)."),
                ("Min SNR", sp_snr, "Components below are skipped and "
                 "excluded."),
                ("Min CNN", sp_cnn, "Auto-exclude below (ignored when the "
                 "file has no CNN scores)."),
                ("Low-pass cutoff", sp_lp, "0 = SLab's normalised cutoff "
                 "(bit-identical to SLabAnalysisApp).")):
            w.setToolTip(tip)
            form.addRow(label, w)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        prog = QProgressDialog("Detecting events…", None, 0, s.n, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(300)

        def cb(i, n):
            prog.setMaximum(n); prog.setValue(i)
            QApplication.processEvents()
        try:
            self.run_detection(sp_fr.value(), sp_dec.value(), progress=cb,
                               threshold=sp_thr.value(), min_snr=sp_snr.value(),
                               min_cnn=sp_cnn.value(),
                               lowpass_hz=sp_lp.value() or None)
        except Exception as exc:
            QMessageBox.critical(self, "Detect events", str(exc))
        finally:
            prog.close()

    def _export_events(self):
        s = self.store
        if s.events is None:
            self.statusBar().showMessage("No events to export — detect first")
            return
        folder = _os_dirname(s.fpath) if s.fpath else os.getcwd()
        prefix = _session_prefix(s.fpath) if s.fpath else "inspector"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export event stats (PeakStats; ROIStats is written alongside)",
            os.path.join(folder, f"{prefix}_PeakStats.xlsx"),
            "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        try:
            out = export_event_tables(s, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            return
        self.statusBar().showMessage(
            "Exported " + ", ".join(os.path.basename(p) for p in out)
            + f"  ({s.n_approved} included components)")

    def _session_info(self):
        s, ev = self.store, self.store.events
        snr = (s.SNR_comp[np.isfinite(s.SNR_comp)]
               if s.SNR_comp is not None else np.array([]))
        T   = s.C.shape[1]
        fr  = ev.fr if ev is not None else s.fr
        dur = f"{T} frames" + (f" ({T / fr / 60:.1f} min)" if fr else "")
        name = ("(hidden in blinded review)" if self._blind is not None
                else (s.fpath or "(demo)"))
        lines = [
            f"File:  {name}",
            f"Components:  {s.n}   (hidden {s.n_hidden})",
            f"Recording length:  {dur}",
            "SNR min / mean / max:  " + (
                f"{snr.min():.2f} / {snr.mean():.2f} / {snr.max():.2f}"
                if snr.size else "n/a"),
            f"Imaging frequency:  {fr if fr else 'n/a'}",
            f"Indicator decay time:  "
            f"{(ev.decay_time if ev is not None else s.decay_time) or 'n/a'}",
        ]
        if s.qc is not None:
            qc = s.qc
            line = "ROI filter:  "
            if "passed" in qc:
                line += f"{int(qc['passed'].sum())}/{s.n} passed"
            if "active" in qc:
                line += f",  {int(qc['active'].sum())} active"
            lines.append(line)
            if "fail_reasons" in qc:
                rf = _rf()
                fails = {n: int(((qc["fail_reasons"] & b) != 0).sum())
                         for n, b in rf.REASONS.items()}
                fails = {n: c for n, c in fails.items() if c}
                if fails:
                    lines.append("   fails:  " + ",  ".join(
                        f"{n} {c}" for n, c in fails.items()))
            if ev is None:
                lines.append(f"Included:  {s.n_approved}")
        if ev is not None:
            lines += [
                f"Auto-approved:  {int(ev.auto_approved.sum())}",
                f"Included (manual):  {s.n_approved}",
                f"Events:  {sum(len(p) for p in ev.peaks)}",
                f"Detection threshold cap:  {ev.threshold}",
                f"Analysis version:  {ev.version}",
            ]
        else:
            lines.append("Analysis version:  not analysed")
        QMessageBox.information(self, "Session info", "\n".join(lines))

    def _batch_detect(self):
        if self._blind_block():
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder of results files")
        if not folder:
            return
        ed    = _ed()
        files = ed.find_results_files(folder)
        if not files:
            QMessageBox.information(self, "Batch detect", "No .hdf5/.h5 files found.")
            return
        if len(files) <= 10:
            listing = "\n".join(files)
        else:
            from collections import Counter
            listing = "\n".join(f"{d}: {n} files" for d, n in
                                 Counter(os.path.dirname(f) for f in files).items())
        ans = QMessageBox.question(
            self, "Batch detect events",
            f"{len(files)} files:\n\n{listing}\n\n"
            "stroh_results is written into each file (estimates untouched).\n"
            "Overwrite files that already have results?\n"
            "Yes = overwrite · No = skip those · Cancel = abort",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Cancel:
            return
        prog = QProgressDialog("Detecting events…", "Cancel", 0, len(files), self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        log = []
        for k, f in enumerate(files):
            if prog.wasCanceled():
                break
            prog.setLabelText(f"{os.path.basename(f)}  ({k + 1}/{len(files)})")
            QApplication.processEvents()
            try:
                msg = ed.detect_events_in_hdf5(
                    f, overwrite=(ans == QMessageBox.StandardButton.Yes))
            except Exception as exc:
                msg = f"ERROR {exc}"
            log.append(f"{os.path.relpath(f, folder)}: {msg}")
            prog.setValue(k + 1)
        prog.close()
        box = QMessageBox(self)
        box.setWindowTitle("Batch detect events")
        box.setText(f"Processed {len(log)}/{len(files)} files.")
        box.setDetailedText("\n".join(log))
        if self.store.fpath and os.path.abspath(self.store.fpath) in map(os.path.abspath, files):
            box.setInformativeText("The open file was processed on disk — "
                                   "reopen it to see its new events.")
        box.exec()

    # ── Blinded review (SLab Randomization Mode) ──────────────────────────────

    def start_blinded(self, rater: str, files: list):
        """Begin a blinded review of ``files`` (already filtered/shuffled)."""
        self._blind = dict(rater=rater, files=list(files), i=0)
        self.act_blind_next.setVisible(True)
        self.act_blind_end.setEnabled(True)
        self.act_blind.setEnabled(False)
        self._blind_open(0)

    def _start_blinded(self):
        import random
        import re as _re
        rater, ok = QInputDialog.getText(self, "Blinded review", "Rater name:")
        rater = _re.sub(r"[^A-Z0-9_-]", "", (rater or "").strip().upper())
        if not ok or not rater:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder to review")
        if not folder:
            return
        unrated, rated, skipped = blind_plan(_ed().find_results_files(folder), rater)
        files = list(unrated)
        if rated and QMessageBox.question(
                self, "Blinded review",
                f"{len(rated)} file(s) were already rated by {rater}.\n"
                "Rate them again as well?") == QMessageBox.StandardButton.Yes:
            files += rated
        if not files:
            QMessageBox.information(
                self, "Blinded review",
                "Nothing to review." + (
                    f"\n{len(skipped)} file(s) have no events yet — run "
                    "Analysis ▸ Batch detect events first." if skipped else ""))
            return
        random.shuffle(files)
        if skipped:
            self.statusBar().showMessage(
                f"{len(skipped)} file(s) without events skipped")
        self.start_blinded(rater, files)

    def _blind_open(self, i: int):
        b = self._blind
        while i < len(b["files"]):
            try:
                store = load_from_hdf5(b["files"][i], all_components=True)
                if store.events is not None:
                    break
                err = "events could not be mapped"
            except Exception as exc:
                err = str(exc)
            print(f"[blind] skipping file {i + 1}: {err}", file=sys.stderr)
            i += 1
        else:
            self._finish_blinded()
            return
        b["i"] = i
        self._swap_store(store)
        self.setWindowTitle(f"Blinded review — file {i + 1} of "
                            f"{len(b['files'])} — rater {b['rater']}")
        self.statusBar().showMessage(
            f"File {i + 1}/{len(b['files'])}: {store.n} components, "
            f"{store.n_approved} included — I toggles, Ctrl+Return saves")

    def _blind_save_next(self):
        b = self._blind
        if b is None:
            return
        try:
            write_blind_rating(b["files"][b["i"]], b["rater"], self.store, b["files"])
        except Exception as exc:
            QMessageBox.critical(self, "Save rating", str(exc))
            return
        if b["i"] + 1 >= len(b["files"]):
            self._finish_blinded()
        else:
            self._blind_open(b["i"] + 1)

    def _finish_blinded(self):
        n = len(self._blind["files"]) if self._blind else 0
        self._end_blinded()
        QMessageBox.information(self, "Blinded review",
                                f"Finished — {n} file(s) in this session.")

    def _end_blinded(self):
        self._blind = None
        self.act_blind_next.setVisible(False)
        self.act_blind_end.setEnabled(False)
        self.act_blind.setEnabled(True)
        self.setWindowTitle("CaImAn Component Inspector")
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
        self.event_view.set_selection([])
        self.corr_view.refresh()
        self.dist_view.refresh()
        self.act_merge.setEnabled(False)
        self.act_delete.setEnabled(False)
        self.act_include.setEnabled(False)
        self.act_export.setEnabled(self.store.events is not None)
        self._sel_lbl.setText(f"  0 selected  |  {self.store.n} total")
        self.statusBar().showMessage(
            f"{self.store.n} components"
            + ("  —  no ROI-filter fields: Analysis ▸ Run ROI filter… adds them"
               if self.store.qc is None and self._blind is None else ""))
        self._update_history_actions()
        if self._blind is not None:          # undo/redo/save are edit ops
            for a in (self.act_undo, self.act_undo_tb,
                      self.act_redo, self.act_redo_tb):
                a.setEnabled(False)


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

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="caiman-inspector",
        description="CaImAn Component Inspector (Qt6 / PyQt6)")
    parser.add_argument("results", nargs="?",
                        help="CaImAn .hdf5 results file, a raw .tif/.tiff stack "
                             "(its <stem>_results.hdf5 is found), or a session "
                             "folder.  Default: the current directory — opens "
                             "<dir>_results.hdf5 and <dir>.tif, where <dir> is "
                             "the folder's name")
    parser.add_argument("--results", dest="results_override", default=None,
                        help="Results file to use with a .tif stack or "
                             "session folder")
    parser.add_argument("--demo", action="store_true",
                        help="Open synthetic demo data")
    parser.add_argument("--background", "--bg", dest="background", default=None,
                        help="Image to show behind the footprints "
                             "(.npy/.npz/.tif/.png; a 3-D stack is max-projected)")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)
    _apply_dark_palette()

    if args.demo:
        store = _make_demo_store()
    else:
        target = args.results or os.getcwd()
        try:
            store = load_any(target, args.results_override)
            print(f"Loaded {store.n} components from {store.fpath}")
        except Exception as exc:
            print(f"Error loading '{target}': {exc}", file=sys.stderr)
            if not args.results:
                print("Run inside a session folder, pass a results file / "
                      "stack / folder, or use --demo.", file=sys.stderr)
            sys.exit(1)

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
