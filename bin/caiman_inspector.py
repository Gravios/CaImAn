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
    Table        — click to select; Ctrl/Shift for multi-select
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
        self.invalidate_corr()

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

        # Events + approvals as an SLab stroh_results group, rows = saved rows
        if self.events is not None:
            import h5py
            with h5py.File(path, "a") as f:
                _ed().write_stroh_results(f, self.events, self.approved)


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
    _attach_stroh_results(store, path)
    _attach_sibling_backgrounds(store, path)
    return store


def _opt_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


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
    use_appr = store.events is not None   # approval only meaningful with events

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
        if self.store.events is not None:
            cols += ["Ev/min", "Incl"]
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
            if s.events is not None:
                vals.append(f"{s.events.frequencies[i]:.2f}")
                vals.append("✓" if s.approved[i] else "✗")
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setBackground(QBrush(dark))
                self.setItem(i, c, item)
        self.blockSignals(False)
        self.mark_hidden()

    def mark_hidden(self):
        """Dim hidden rows; tint excluded rows red (once events exist) and
        keep the Incl column in sync with the store."""
        grey   = QColor(105, 105, 105)
        normal = QColor(220, 220, 220)
        excl   = QColor(255, 110, 110)
        s      = self.store
        has_ev = s.events is not None and self.columnCount() and \
            self.horizontalHeaderItem(self.columnCount() - 1).text() == "Incl"
        for row in range(s.n):
            if s.is_hidden(row):
                fg = grey
            elif has_ev and not s.approved[row]:
                fg = excl
            else:
                fg = normal
            if has_ev:
                it = self.item(row, self.columnCount() - 1)
                if it:
                    it.setText("✓" if s.approved[row] else "✗")
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

        # Top row: cell viewer (left) | distance matrix (right)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.cell_view)
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
                  self.trace_view, self.event_view, self.corr_view,
                  self.dist_view):
            w.store = new_store
        self._refresh_bg_combo()
        self._refresh_trace_combo()
        self.cell_view.refresh_background()
        self._full_refresh()


    # ── Events / approval ─────────────────────────────────────────────────────

    def _selected_rows(self) -> list:
        return sorted({idx.row() for idx in self.table.selectedIndexes()})

    def _reselect(self, rows):
        rows = [r for r in rows if 0 <= r < self.store.n]
        if not rows:
            return
        sm, flags = self.table.selectionModel(), (
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows)
        self.table.blockSignals(True)
        for r in rows:
            sm.select(self.table.model().index(r, 0), flags)
        self.table.blockSignals(False)
        self._on_table_sel()

    def _toggle_include(self):
        s, sel = self.store, self._selected_rows()
        if not sel:
            return
        if s.events is None:
            self.statusBar().showMessage(
                "Include flags belong to the event results — "
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
        if s.events is not None and QMessageBox.question(
                self, "Detect events",
                "Existing events and include flags will be replaced.\n"
                "Continue?") != QMessageBox.StandardButton.Yes:
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
        self.statusBar().showMessage(f"{self.store.n} components")
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
