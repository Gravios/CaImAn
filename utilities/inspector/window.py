"""InspectorWindow — the main window wiring every pane together."""

from PyQt6.QtWidgets import QMainWindow

from .theme import set_theme
from .prefs import kinetics_prefs, load_prefs
from .store import ComponentStore
from .loading import _os_dirname
from .window_build import _BuildMixin
from .window_session import _SessionMixin
from .window_analysis import _AnalysisMixin

class InspectorWindow(_BuildMixin, _SessionMixin, _AnalysisMixin,
                      QMainWindow):
    """The main window.  Construction, selection, editing, view toggles and
    refresh live here; menus / toolbar / panes in ``window_build``, open /
    save / layout / preferences in ``window_session``, ROI filter /
    detection / export / blinded review in ``window_analysis``."""

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

    def closeEvent(self, ev):
        self._save_ui()
        self.movie_bar.shutdown()
        super().closeEvent(ev)

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
