"""InspectorWindow — the main window wiring every pane together."""

import os
import sys

import numpy as np

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .constants import OVERLAY_ALPHA
from .theme import PT, UI, set_theme
from .deps import _ed, _ek, _rf
from .prefs import kinetics_prefs, load_prefs, prefs_path, roi_filter_prefs, save_prefs
from .store import ComponentStore
from .loading import (
    _load_image_file,
    _os_dirname,
    _session_prefix,
    load_any,
    load_from_hdf5,
)
from .review import blind_plan, export_event_tables, write_blind_rating
from .views import (
    CellViewer,
    CorrMatrix,
    DistMatrix,
    EventViewer,
    MovieBar,
    TraceViewer,
)
from .table import ComponentTable
from .dialogs import HelpDialog, PreferencesDialog, RunFilterDialog


class InspectorWindow(QMainWindow):

    def __init__(self, store: ComponentStore):
        super().__init__()
        self.store = store
        store.set_kinetics_params(kinetics_prefs())
        self.setWindowTitle("CaImAn Component Inspector")
        self.resize(1760, 980)
        self._pair  = None
        self._blind = None      # blinded-review session state (dict) or None
        # Remembered layout / view / last folder (prefs "ui" section)
        ui = load_prefs().get("ui")
        self._ui = dict(ui) if isinstance(ui, dict) else {}
        if store.fpath:
            self._ui["last_dir"] = _os_dirname(store.fpath)

        set_theme()                            # saved appearance (before any widget)
        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._wire_signals()
        self._full_refresh()
        self._restore_ui()

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
        self.act_hide_rejected = QAction("Hide &rejected / inactive ROIs", self)
        self.act_hide_rejected.setCheckable(True)
        self.act_hide_rejected.setToolTip(
            "Hide components outside idx_components (CaImAn-rejected; census: "
            "inactive) from the table and the cell viewer.  They stay in the "
            "data and are always saved.")
        self.act_hide_rejected.triggered.connect(self._set_hide_rejected)
        vm.addAction(self.act_hide_rejected)
        mm = vm.addMenu("&Event markers")
        mm.setToolTipsVisible(True)
        grp = QActionGroup(self)
        self.act_markers = {}
        for mode, text, tip in (
                ("off", "&Off", "No event markers on the traces"),
                ("all", "&All events", "A line at every detected event peak"),
                ("valid", "&Kinetically valid only",
                 "Only events whose raw trace has the expected fast rise and "
                 "exponential decay (Preferences ▸ Event kinetics)")):
            a = QAction(text, self, checkable=True)
            a.setToolTip(tip)
            a.triggered.connect(lambda _c, m=mode: self._set_event_markers(m))
            grp.addAction(a); mm.addAction(a)
            self.act_markers[mode] = a
        self.act_markers["off"].setChecked(True)
        self._columns_menu = vm.addMenu("&Columns")
        self._columns_menu.setToolTip("Show / hide table columns "
                                      "(also: right-click the table header)")
        self._columns_menu.aboutToShow.connect(
            lambda: self.table.fill_column_menu(self._columns_menu))
        a = QAction("&Reset layout", self)
        a.setToolTip("Default pane sizes, all columns shown, unsorted.")
        a.triggered.connect(self._reset_layout)
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
        self.act_prefs = a = QAction("&Preferences…", self)
        a.setShortcuts([QKeySequence("P"), QKeySequence("Ctrl+,")])
        a.setMenuRole(QAction.MenuRole.PreferencesRole)
        a.setToolTip("Theme, ROI-filter thresholds and other inspector settings")
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

        hm = mb.addMenu("&Help")
        self.act_help = a = QAction("&Shortcuts && usage…", self)
        a.setShortcuts([QKeySequence("H"), QKeySequence("F1")])
        a.setToolTip("Keyboard / mouse shortcuts and how to use the inspector")
        a.triggered.connect(self._show_help)
        hm.addAction(a)
        a = QAction("&About", self)
        a.setMenuRole(QAction.MenuRole.AboutRole)
        a.triggered.connect(lambda: QMessageBox.about(
            self, "About CaImAn Component Inspector",
            "<b>CaImAn Component Inspector</b><br>Review, curate and evaluate "
            "CNMF components (Qt6 / PyQt6).<br><br>Preferences and layout: "
            f"<code>{prefs_path()}</code>"))
        hm.addAction(a)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setObjectName("main_toolbar")     # needed for saveState / restoreState
        tb.setMovable(False)

        self.act_merge = QAction("⊕  Merge selected", self)
        self.act_merge.setToolTip(
            "Merge ≥ 2 selected components into one, appended at the end.\n"
            "Footprint and trace: rank-1 fit of the parts' summed signal.\n"
            "ΔF/F: merged signal over the merged baseline.\n"
            "SNR / r-value / CNN become unknown (—).")
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
        self._sel_lbl.setStyleSheet(f"color: {UI['muted']}; font-size: 9pt;")
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
        self._splitters = {"root": root, "right": right, "top": top,
                           "bottom": bot, "traces": traces}
        self._splitter_defaults = {"root": [1, 5], "right": [3, 2], "top": [1, 1],
                                   "bottom": [1, 1], "traces": [3, 2]}

        # Link corr and dist ViewBoxes so zoom/pan stays in sync
        self.corr_view.vb.setXLink(self.dist_view.vb)
        self.corr_view.vb.setYLink(self.dist_view.vb)

        self.setCentralWidget(root)
        self.statusBar().showMessage(f"Loaded {s.n} components")

    def closeEvent(self, ev):
        self._save_ui()
        self.movie_bar.shutdown()
        super().closeEvent(ev)

    # ── Remembered layout / last folder ───────────────────────────────────────

    def _start_dir(self, fallback: str = "") -> str:
        d = self._ui.get("last_dir")
        return d if d and os.path.isdir(d) else fallback

    def _remember_dir(self, path: str) -> None:
        if not path:
            return
        d = path if os.path.isdir(path) else _os_dirname(path)
        if d:
            self._ui["last_dir"] = d
            self._save_ui()

    def _collect_ui(self) -> dict:
        b64 = lambda ba: bytes(ba.toBase64()).decode("ascii")
        ui = dict(self._ui)
        ui["geometry"]  = b64(self.saveGeometry())
        ui["state"]     = b64(self.saveState())
        ui["splitters"] = {k: b64(sp.saveState()) for k, sp in self._splitters.items()}
        ui["hidden_columns"] = self.table.hidden_columns()
        st = self.table.sort_state()
        ui["sort"] = {"column": st[0], "order": st[1]} if st else None
        ui["view"] = {"hide_rejected": self.act_hide_rejected.isChecked(),
                      "invert_bg":     self.act_invert_bg.isChecked(),
                      "show_rois":     self.act_show_rois.isChecked(),
                      "hide_excluded": self.act_hide_excl.isChecked(),
                      "dist_log":      self.act_dist_log.isChecked(),
                      "event_markers": self.trace_view.event_markers}
        return ui

    def _save_ui(self) -> None:
        """Write the "ui" section, re-reading the file so other sections
        (e.g. roi_filter saved from another window) are kept."""
        try:
            self._ui = self._collect_ui()
            prefs = load_prefs()
            prefs["ui"] = self._ui
            save_prefs(prefs)
        except Exception as exc:              # never block closing
            print(f"[prefs] could not save layout: {exc}", file=sys.stderr)

    def _restore_ui(self) -> None:
        ui = self._ui
        ba = lambda s: QByteArray.fromBase64(str(s).encode("ascii"))
        try:
            if ui.get("geometry"):
                self.restoreGeometry(ba(ui["geometry"]))
            if ui.get("state"):
                self.restoreState(ba(ui["state"]))
            for k, st in (ui.get("splitters") or {}).items():
                if k in self._splitters and st:
                    self._splitters[k].restoreState(ba(st))
        except Exception as exc:
            print(f"[prefs] ignoring saved layout: {exc}", file=sys.stderr)
        self.table.set_hidden_columns(ui.get("hidden_columns") or ())
        srt = ui.get("sort")
        if isinstance(srt, dict) and srt.get("column"):
            self.table.sort_by_name(srt["column"], srt.get("order", "asc"))
        view = ui.get("view") or {}
        for key, act in (("hide_rejected", self.act_hide_rejected),
                         ("invert_bg", self.act_invert_bg),
                         ("show_rois", self.act_show_rois),
                         ("hide_excluded", self.act_hide_excl),
                         ("dist_log", self.act_dist_log)):
            want = view.get(key)
            if want is not None and bool(want) != act.isChecked():
                act.trigger()                 # runs the toggle's handler
        if view.get("event_markers") in self.act_markers:
            self._set_event_markers(view["event_markers"])
        self._apply_view_prefs()

    def _set_event_markers(self, mode: str) -> None:
        self.act_markers[mode].setChecked(True)
        self.trace_view.set_event_markers(mode)

    def _set_hide_rejected(self, flag: bool) -> None:
        self.store.hide_rejected = bool(flag)
        self.table.mark_hidden()
        self.cell_view.refresh_visibility()
        self.corr_view.redraw()
        self.dist_view.redraw()
        s = self.store
        if s.eval_known:
            self.statusBar().showMessage(
                f"{s.n - s.n_accepted} rejected / inactive ROIs "
                + ("hidden (still saved)" if flag else "shown"))

    def _apply_view_prefs(self) -> None:
        """Re-select the remembered background / trace if this file has them."""
        bg = self._ui.get("background")
        if bg and bg != self.store.bg_key and bg in self.store.background_names():
            self.bg_combo.setCurrentText(bg)
        tr = self._ui.get("trace")
        if tr and tr != self.store.disp_key and tr in self.store.trace_names():
            self.trace_combo.setCurrentText(tr)

    def _reset_layout(self) -> None:
        for k, sp in self._splitters.items():
            sp.setSizes(self._splitter_defaults[k])
        self.table.show_all_columns()
        self.table.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.table.refresh()
        self.statusBar().showMessage("Layout reset: panes, columns and sort")

    # ── Signals ───────────────────────────────────────────────────────────────

    def _wire_signals(self):
        self.table.itemSelectionChanged.connect(self._on_table_sel)
        self.corr_view.pair_clicked.connect(self._on_corr_click)
        self.dist_view.pair_clicked.connect(self._on_corr_click)
        self.cell_view.component_clicked.connect(self._on_cell_click)
        self.table.columns_changed.connect(self._save_ui)

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
                self.corr_view.clear_highlight()
                self.dist_view.clear_highlight()
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
        self.apply_theme()
        if self.store.set_kinetics_params(kinetics_prefs()):   # new criteria
            sel = self._selected_rows()
            self.table.refresh()
            self._reselect(sel)
            self.event_view.refresh()
            self.trace_view.refresh()          # kinetically valid markers
        self.statusBar().showMessage(f"Preferences saved to {path}", 6000)
        return True

    def apply_theme(self, appearance: dict = None) -> None:
        """Activate the (saved) appearance and re-style every live view."""
        before = (dict(UI), dict(PT), dict(OVERLAY_ALPHA))
        set_theme(appearance)
        self.movie_bar.lbl.setStyleSheet(f"color: {UI['muted']}; font-family: monospace;")
        self._sel_lbl.setStyleSheet(f"color: {UI['muted']}; font-size: 9pt;")
        if before[:2] == (UI, PT):
            if before[2] != OVERLAY_ALPHA:     # only the overlay opacity changed
                self.cell_view.refresh_rois()
                self.cell_view._redraw()
            return
        sel = self._selected_rows()
        for v in (self.cell_view, self.trace_view, self.event_view,
                  self.corr_view, self.dist_view):
            v.apply_theme()
        self._full_refresh()
        self._reselect(sel)

    def _show_help(self) -> None:
        HelpDialog(self).exec()

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
            self, "Open CaImAn results", self._start_dir(),
            "Results or stack (*.hdf5 *.h5 *.tif *.tiff);;"
            "HDF5 files (*.hdf5 *.h5);;TIFF stacks (*.tif *.tiff);;All files (*)")
        if not path:
            return
        self._remember_dir(path)
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
                   if self.store.fpath else
                   os.path.join(self._start_dir(), "curated_results.hdf5"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save curated CaImAn results", default,
            "HDF5 files (*.hdf5 *.h5)")
        if path:
            self._remember_dir(path)
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
        self._ui["background"] = name
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
        self._ui["trace"] = name
        self.statusBar().showMessage(f"Trace → {name}")

    # ── Visibility ────────────────────────────────────────────────────────────

    def _toggle_visibility(self):
        sel = self.table.selected_comps()
        if not sel:
            self.statusBar().showMessage("Select component(s) to hide / show")
            return
        self.store.toggle_hidden(sel)
        self.cell_view.refresh_visibility()
        self.corr_view.redraw()
        self.dist_view.redraw()
        self.table.mark_hidden()
        self.statusBar().showMessage(
            f"{self.store.n_hidden} hidden  |  {self.store.n} total")

    def _show_all_hidden(self):
        if self.store.n_hidden == 0:
            self.statusBar().showMessage("No hidden components")
            return
        self.store.show_all()
        self.cell_view.refresh_visibility()
        self.corr_view.redraw()
        self.dist_view.redraw()
        self.table.mark_hidden()
        self.statusBar().showMessage(f"All {self.store.n} components visible")

    def _load_background(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load background image",
            self._start_dir(_os_dirname(self.store.fpath) if self.store.fpath else ""),
            "Images (*.npy *.npz *.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if not path:
            return
        self._remember_dir(path)
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
        new_store.set_kinetics_params(kinetics_prefs())
        self.store = new_store
        for w in (self.table, self.cell_view,
                  self.trace_view, self.event_view, self.corr_view,
                  self.dist_view):
            w.store = new_store
        self._refresh_bg_combo()
        self._refresh_trace_combo()
        self._apply_view_prefs()
        # a blinded rater must see (and rate) every row
        new_store.hide_rejected = (self.act_hide_rejected.isChecked()
                                   and self._blind is None)
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
        rule = ""
        if ev.auto_slab is not None:
            rule = f" (SLab {int(ev.auto_slab.sum())}; kinetics rule)"
        elif self.store.kinetics_rule_skipped:
            rule = (" — ⚠ kinetics rule not applied: no residual and ΔF/F ≈ C, "
                    "the check would be circular")
        self.statusBar().showMessage(
            f"Events: {sum(len(p) for p in ev.peaks)} in {ev.n} components; "
            f"{int(ev.auto_approved.sum())} auto-approved / included{rule}")

    def _detect_events_dialog(self):
        if self._blind_block():                # the rating must use the file's events
            return
        s = self.store
        if not s.can_detect:
            QMessageBox.warning(
                self, "Detect events",
                "This file has no ΔF/F traces (estimates.F_dff).\n"
                "Run cnm.estimates.detrend_df_f() in the pipeline first.")
            return
        if s.has_approval and QMessageBox.question(
                self, "Detect events",
                ("Existing events and include flags will be replaced "
                 "(Edit ▸ Undo restores them)."
                 if s.events is not None else
                 "Include flags (seeded from the ROI filter) will be replaced "
                 "by event auto-approval (Edit ▸ Undo restores them).")
                + "\nContinue?"
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
        kin_cfg = kinetics_prefs()
        cb_kin = QCheckBox("Require valid kinetics", dlg)
        cb_kin.setChecked(bool(kin_cfg.get("approve_on_kinetics")))
        cb_kin.setToolTip(
            "Auto-approve only components SLab approves that also have "
            f"≥ {kin_cfg.get('min_valid_events', 1)} kinetically valid event(s) on "
            "the raw trace (criteria: Preferences ▸ Event kinetics).  Off = SLab.")
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
        form.addRow("Approval", cb_kin)
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
                               lowpass_hz=sp_lp.value() or None,
                               kinetics=dict(kin_cfg, approve_on_kinetics=cb_kin.isChecked()))
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
            ("CaImAn evaluation:  "
             f"{s.n_accepted} accepted / {s.n - s.n_accepted} rejected "
             "(census: active / inactive)") if s.eval_known and self._blind is None
            else "CaImAn evaluation:  " + ("hidden" if self._blind else "n/a"),
            f"Recording length:  {dur}",
            "SNR min / mean / max:  " + (
                f"{snr.min():.2f} / {snr.mean():.2f} / {snr.max():.2f}"
                if snr.size else "n/a"),
            f"Imaging frequency:  {fr if fr else 'n/a'}",
            f"Indicator decay time:  "
            f"{(ev.decay_time if ev is not None else s.decay_time) or 'n/a'}",
        ]
        if s.qc is not None and self._blind is None:
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
        if ev is not None and s.dff is not None:
            lines.append("Kinetics trace:  " + (
                "C + residual (raw), on the ΔF/F scale" if s.kinetics_source == "raw"
                else "ΔF/F ≈ C (no residual) — circular, every event passes"
                if s.kinetics_circular else "ΔF/F (file has no residual YrA)"))
        if ev is not None:
            lines.append("Approval rule:  " + _ek().describe_rule(ev.params))
            if ev.auto_slab is not None:
                lines.append(f"SLab auto-approved:  {int(ev.auto_slab.sum())}")
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
        folder = QFileDialog.getExistingDirectory(self, "Folder of results files",
                                                  self._start_dir())
        if not folder:
            return
        self._remember_dir(folder)
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
            + ("Auto-approval requires valid kinetics (Preferences ▸ Event "
               "kinetics).\n" if kinetics_prefs().get("approve_on_kinetics") else "")
            +
            "Overwrite files that already have results?\n"
            "Yes = overwrite · No = skip those · Cancel = abort",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Cancel:
            return
        kin = kinetics_prefs()
        kin = kin if kin.get("approve_on_kinetics") else None
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
                    f, overwrite=(ans == QMessageBox.StandardButton.Yes), kinetics=kin)
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
        self.act_hide_rejected.setEnabled(False)   # every row is rated
        self.table.blinded = True                  # no QC / Eval / kinetics
        self.event_view.blinded = True
        self.trace_view.blinded = True
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
        folder = QFileDialog.getExistingDirectory(self, "Folder to review",
                                                  self._start_dir())
        if not folder:
            return
        self._remember_dir(folder)
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
                    # Start from the algorithm's verdict, not whatever manual
                    # curation the file carries (manually_approved).
                    store.approved = store.events.auto_approved.copy()
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
        self.table.blinded = False
        self.event_view.blinded = False
        self.trace_view.blinded = False
        self.act_hide_rejected.setEnabled(True)
        self.store.hide_rejected = self.act_hide_rejected.isChecked()
        self.corr_view.redraw(); self.dist_view.redraw()
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
        s = self.store
        self.statusBar().showMessage(
            f"{s.n} components"
            + (f" ({s.n_accepted} accepted / {s.n - s.n_accepted} rejected)"
               if s.eval_known and self._blind is None else "")
            + ("  —  no ROI-filter fields: Analysis ▸ Run ROI filter… adds them"
               if s.qc is None and self._blind is None else "")
            + ("  —  ⚠ kinetics: no residual in this file and ΔF/F ≈ C, so the "
               "check can't tell real transients from the model's"
               if s.events is not None and s.kinetics_circular else ""))
        self._update_history_actions()
        if self._blind is not None:          # undo/redo/save are edit ops
            for a in (self.act_undo, self.act_undo_tb,
                      self.act_redo, self.act_redo_tb):
                a.setEnabled(False)


__all__ = ['InspectorWindow']
