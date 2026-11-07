"""InspectorWindow: open / save, remembered layout, preferences, theme, help."""

import os
import sys

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox

from .constants import OVERLAY_ALPHA
from .theme import PT, UI, set_theme
from .prefs import kinetics_prefs, load_prefs, save_prefs
from .store import ComponentStore
from .loading import _load_image_file, _os_dirname, load_any
from .dialogs import HelpDialog, PreferencesDialog


class _SessionMixin:
    """Methods of :class:`~utilities.inspector.window.InspectorWindow`."""

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

    def _reset_layout(self) -> None:
        for k, sp in self._splitters.items():
            sp.setSizes(self._splitter_defaults[k])
        self.table.show_all_columns()
        self.table.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.table.refresh()
        self.statusBar().showMessage("Layout reset: panes, columns and sort")

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


__all__ = ['_SessionMixin']
