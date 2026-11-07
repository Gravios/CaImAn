"""Preferences, Run ROI filter and Help dialogs."""

import json
import os
import re
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .theme import UI, _default_font_pt
from .deps import _rf
from .prefs import PREF_SECTIONS, prefs_path, roi_filter_prefs

if TYPE_CHECKING:                      # annotations only
    from .store import ComponentStore
    from .window import InspectorWindow


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
        elif kind == "choice":                 # rng = [(value, label), …]
            self.combo = QComboBox(self)
            for val, label in rng:
                self.combo.addItem(label, val)
            lay.addWidget(self.combo)
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
                self.hint.setStyleSheet(f"color: {UI['hint']};")
                self.chk.toggled.connect(self._sync)
                lay.addWidget(self.chk)
            lay.addWidget(self.spin)
            if self.hint is not None:
                lay.addWidget(self.hint)
            h = hint[0] if isinstance(hint, (list, tuple)) and hint else hint
            if h is not None and not isinstance(h, bool):
                try:                            # QSpinBox needs an int
                    self.spin.setValue(int(round(float(h))) if kind == "int" else float(h))
                except (TypeError, ValueError):
                    pass
        lay.addStretch(1)

    def _sync(self):
        on = self.chk.isChecked()
        self.spin.setEnabled(on)
        self.hint.setVisible(not on)

    def value(self):
        if self.box is not None:
            return bool(self.box.isChecked())
        if self.combo is not None and self.kind == "choice":
            return self.combo.currentData()
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
        if self.combo is not None and self.kind == "choice":
            i = self.combo.findData(v)
            self.combo.setCurrentIndex(i if i >= 0 else 0)
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
                hint = ((run_hints or {}).get(key) if null_text == "from run" else
                        _default_font_pt() if key == "font_size" else None)
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
        where.setStyleSheet(f"color: {UI['hint']};")
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
        win = self.parent()
        path, _ = QFileDialog.getOpenFileName(
            self, "Pipeline JSON",
            win._start_dir() if hasattr(win, "_start_dir") else "",
            "JSON (*.json);;All files (*)")
        if not path:
            return
        if hasattr(win, "_remember_dir"):
            win._remember_dir(path)
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
        self.warn.setStyleSheet(f"color: {UI['warn']};")
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


# ── Help ──────────────────────────────────────────────────────────────────────

def _clean_action_text(text: str) -> str:
    t = text.replace("&&", "\0").replace("&", "").replace("\0", "&")
    return re.sub(r"^[^\w(]+", "", t).strip()


# Descriptions for actions whose tooltip is only their name or changes at
# run time (undo / redo show the pending step).
_HELP_DESCRIPTIONS = {
    "Open…":    "Open a results file, stack or session folder",
    "Save":     "Write <name>_curated.hdf5 next to the source file",
    "Save As…": "Save the curated results under another name",
    "Quit":     "Close the inspector (layout is remembered)",
    "Refresh":  "Rebuild the table and redraw every view",
    "Undo":     "Undo the last merge, delete, ROI-filter run or event detection",
    "Redo":     "Redo the last undone step",
}

# Context-dependent input that is not a menu action:  (where, keys, what)
_HELP_EXTRA_KEYS = (
    ("Cell viewer", "Click", "Select the component under the cursor"),
    ("Cell viewer", "Ctrl+Click", "Add / remove it from the selection"),
    ("Cell viewer", "Wheel · drag", "Zoom · pan (right-click → View all)"),
    ("Cell viewer", "Shift+Wheel", "Step movie frames (while a movie is shown)"),
    ("Movie", "Space", "Play / pause"),
    ("Movie", "← / →", "Previous / next frame"),
    ("Movie", "Shift+← / Shift+→", "10 frames back / forward"),
    ("Movie", "Home / End", "First / last frame"),
    ("Table", "Click · Ctrl/Shift+Click", "Select · extend the selection"),
    ("Table", "Click a header", "Sort by that column (again: reverse)"),
    ("Table", "Right-click a header", "Show / hide columns"),
    ("Matrices", "Click a cell", "Select that pair (orange / magenta)"),
    ("Trace view", "Drag the yellow line", "Seek the movie to that frame"),
)

_HELP_USAGE = """
<h3>Opening</h3>
<p>Run <code>caiman-inspector</code> in a session folder (it opens
<code>&lt;dir&gt;_results.hdf5</code> and the movie <code>&lt;dir&gt;.tif</code>), or pass a
results file, stack or folder; <code>--demo</code> opens synthetic data.
File ▸ Open… starts in the last folder used.</p>

<h3>Reviewing</h3>
<p>Every component of the file is loaded and saved.  The Eval column shows
CaImAn's verdict (✓ accepted / census active, ✗ rejected / census inactive);
View ▸ Hide rejected / inactive ROIs hides the rejected ones without
dropping them.  Select components in the table, the cell viewer or the correlation /
distance matrices (Ctrl+click a cell selects that pair); each selected component keeps one colour in the table,
the footprint overlay and the traces.  The trace dropdown switches between
denoised C, ΔF/F, C+residual and deconvolved S (display only); the BG
dropdown and View ▸ Load background image… choose the image behind the
footprints.</p>

<h3>Curating</h3>
<p><b>Merge</b> (≥ 2 selected: footprint and trace refitted to the parts'
summed signal; SNR / r / CNN become unknown) and
<b>Delete</b> change the components; <b>Hide / show</b> (V) only affects
the display; <b>Include / exclude</b> (I) is the non-destructive curation flag
(excluded footprints are red; “Hide excluded” hides them).  Undo / redo keep
the last 5 merges, deletes, ROI-filter runs or event detections.  <b>Save</b> writes <code>&lt;name&gt;_curated.hdf5</code> next to
the original, with events, include flags and ROI-filter fields.</p>

<h3>Events</h3>
<p>Analysis ▸ Detect events… runs SLab peak detection on ΔF/F (include flags
become the automatic approval); the event panel shows filtered ΔF/F,
threshold, peaks and binary trace.  Export event stats writes PeakStats and
ROIStats for the included components (with each row's source component and,
when known, CaImAn's verdict); Batch detect and Blinded review work on a whole
folder.  A blinded rater starts from the automatic approval and does not see
the Eval or ROI-filter columns.</p>

<h3>Event kinetics</h3>
<p>CNMF's denoised trace C has a fast rise and exponential decay by
construction, so every event is also checked on the raw trace (C + residual,
rescaled to ΔF/F; ΔF/F when the file has no residual): fit of the expected
transient (shape r), fitted decay τ, raw rise time and amplitude over the
raw noise.  Columns Ev ok / Ev ok % / Shape r / τ fit, the event panel (blue
fitted templates; dashed peaks fail) and the export (kin_*) show it;
Preferences ▸ Event kinetics sets τ and the criteria.  Approval is not
changed.  If the file has no residual and its ΔF/F is just C rescaled, the
check would be circular and the status bar says so.</p>

<h3>ROI filter</h3>
<p>Results processed by the pipeline carry a <code>roi_filter</code> group
(footprint shape, edge distance, CaImAn quality, activity); its fields
appear as table columns and Incl starts from the QC verdict.  Analysis ▸
Run ROI filter… adds or recomputes them (e.g. for older sessions) with the
thresholds in Preferences; hover a Fail cell for the full reasons.</p>

<h3>Movie</h3>
<p>The bar under the cell viewer plays the raw stack behind the footprints
(fps defaults to the recording rate; “avg N” shows a running mean).  It is
found next to the results, or opened with View ▸ Open movie….  Picking a BG
image returns to the static background.</p>

<h3>Preferences &amp; layout</h3>
<p>Edit ▸ Preferences… (P) sets the theme, plot background, font size,
ROI overlay opacity (selected / all / excluded) and ROI-filter thresholds.  Window and pane sizes, hidden columns, sort, view
toggles, background / trace choice and the last folder are remembered;
View ▸ Reset layout restores the defaults.  Everything is stored in
<code>{prefs}</code>.</p>
"""


class HelpDialog(QDialog):
    """Help ▸ Shortcuts & usage (H / F1).  The shortcut table is built from
    the window's actual actions, so it always matches the menus."""

    def __init__(self, window: 'QMainWindow'):
        super().__init__(window)
        self.setWindowTitle("Inspector help — shortcuts & usage")
        self.resize(760, 820)
        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(False)
        self.browser.setHtml(self.build_html(window))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.browser, 1)
        lay.addWidget(bb)

    @staticmethod
    def action_rows(window) -> list:
        """(menu path, keys, tooltip) for every action with a shortcut."""
        rows, seen = [], set()

        def keys(a):
            return "  /  ".join(k.toString(QKeySequence.SequenceFormat.NativeText)
                                for k in a.shortcuts() if not k.isEmpty())

        def desc(a):
            name = _clean_action_text(a.text())
            if name in _HELP_DESCRIPTIONS:
                return _HELP_DESCRIPTIONS[name]
            tip = " ".join((a.toolTip() or "").split())
            return "" if _clean_action_text(tip) == name else tip

        def walk(menu, path):
            for a in menu.actions():
                if a.menu() is not None:
                    walk(a.menu(), path + [_clean_action_text(a.menu().title())])
                elif not a.isSeparator() and keys(a) and id(a) not in seen:
                    seen.add(id(a))
                    rows.append((" ▸ ".join(path + [_clean_action_text(a.text())]),
                                 keys(a), desc(a)))
        for top in window.menuBar().actions():
            if top.menu() is not None:
                walk(top.menu(), [_clean_action_text(top.menu().title())])
        extra = list(window.actions())
        for tb in window.findChildren(QToolBar):
            extra += tb.actions()
        for a in extra:
            if not a.isSeparator() and keys(a) and id(a) not in seen:
                seen.add(id(a))
                rows.append(("Toolbar ▸ " + _clean_action_text(a.text()),
                             keys(a), desc(a)))
        return rows

    @classmethod
    def build_html(cls, window) -> str:
        esc = lambda t: (str(t).replace("&", "&amp;").replace("<", "&lt;")
                         .replace(">", "&gt;"))
        th = f"style='text-align:left;padding:3px 10px;border-bottom:1px solid {UI['hint']}'"
        td = "style='padding:2px 10px;vertical-align:top'"
        kd = "style='padding:2px 10px;vertical-align:top;white-space:nowrap'"
        out = ["<h2>Keyboard &amp; mouse</h2>",
               f"<table cellspacing='0'><tr><th {th}>Keys</th><th {th}>Command</th>"
               f"<th {th}>What it does</th></tr>"]
        for path, k, tip in cls.action_rows(window):
            out.append(f"<tr><td {kd}><b>{esc(k)}</b></td><td {td}>{esc(path)}</td>"
                       f"<td {td}>{esc(tip)}</td></tr>")
        out.append("</table>")
        out.append(f"<h3>In context</h3><table cellspacing='0'><tr><th {th}>Where</th>"
                   f"<th {th}>Input</th><th {th}>What it does</th></tr>")
        for where, k, what in _HELP_EXTRA_KEYS:
            out.append(f"<tr><td {td}>{esc(where)}</td><td {kd}><b>{esc(k)}</b></td>"
                       f"<td {td}>{esc(what)}</td></tr>")
        out.append("</table>")
        out.append("<p>Single-letter keys (V, I, P, H) work anywhere in the window "
                   "except while typing in a text field.</p>")
        out.append(_HELP_USAGE.replace("{prefs}", esc(prefs_path())))
        return "\n".join(out)


__all__ = ['_fmt_num', '_PrefField', 'PreferencesDialog', 'RunFilterDialog', '_clean_action_text', '_HELP_DESCRIPTIONS', '_HELP_EXTRA_KEYS', '_HELP_USAGE', 'HelpDialog']
