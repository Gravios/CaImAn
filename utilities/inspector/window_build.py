"""InspectorWindow: menus, toolbar and the pane layout."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .theme import UI
from .prefs import prefs_path
from .views import (
    CellViewer,
    CorrMatrix,
    DistMatrix,
    EventViewer,
    MovieBar,
    TraceViewer,
)
from .table import ComponentTable


class _BuildMixin:
    """Methods of :class:`~utilities.inspector.window.InspectorWindow`."""

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


__all__ = ['_BuildMixin']
