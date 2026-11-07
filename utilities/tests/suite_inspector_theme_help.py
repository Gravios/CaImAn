"""Offscreen test: appearance preferences, P / H shortcuts, help dialog."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
import importlib.util, json, os, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np

root = Path(tempfile.mkdtemp())
PREFS = root / "cfg" / "inspector_prefs.json"
os.environ["CAIMAN_INSPECTOR_PREFS"] = str(PREFS)
insp = load_inspector()
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence, QPalette
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog
app = QApplication.instance() or QApplication([])

opened = []
insp.PreferencesDialog.exec = lambda self: (opened.append(("prefs", self.tabs.tabText(self.tabs.currentIndex()))),
                                            QDialog.DialogCode.Rejected)[1]
insp.HelpDialog.exec = lambda self: (opened.append(("help", self)), QDialog.DialogCode.Rejected)[1]


def bg(widget): return widget.backgroundBrush().color().name()
def win_light(): return app.palette().color(QPalette.ColorRole.Window).lightness()


def new_window():
    w = insp.InspectorWindow(insp._make_demo_store()); w.resize(780, 700); w.show(); app.processEvents()
    w.activateWindow(); app.processEvents()
    return w


try:
    default_pt = app.font().pointSizeF()
    w = new_window()
    assert win_light() < 60 and bg(w.trace_view) == "#111111", "default stays dark"
    assert insp.PALETTE_HEX == insp.PALETTE_DARK_BG
    ok("default appearance unchanged: dark UI, dark plots, bright palette")

    # ── P / H / F1
    for focus in (w.table, w.cell_view, w.trace_view):
        focus.setFocus(); app.processEvents()
        opened.clear()
        QTest.keyClick(focus, Qt.Key.Key_P); app.processEvents()
        QTest.keyClick(focus, Qt.Key.Key_H); app.processEvents()
        QTest.keyClick(focus, Qt.Key.Key_F1); app.processEvents()
        kinds = [k for k, _ in opened]
        assert kinds == ["prefs", "help", "help"], (type(focus).__name__, kinds)
    assert w.act_prefs.shortcuts()[0].toString() == "P" and "Ctrl+," in [k.toString() for k in w.act_prefs.shortcuts()]
    ok("P opens Preferences, H and F1 open Help — from the table, cell viewer and trace view")

    # ── help content
    html = insp.HelpDialog.build_html(w)
    keyed = [a for a in w.findChildren(QAction) if any(not k.isEmpty() for k in a.shortcuts())]
    missing = [(a.text(), [k.toString() for k in a.shortcuts()]) for a in keyed
               if not all(k.toString(QKeySequence.SequenceFormat.NativeText) in html
                          for k in a.shortcuts() if not k.isEmpty())]
    assert not missing, missing
    rows = insp.HelpDialog.action_rows(w)
    paths = [r[0] for r in rows]
    assert "Edit ▸ Preferences…" in paths and "Help ▸ Shortcuts & usage…" in paths, paths
    assert "Analysis ▸ Run ROI filter…" in paths and any(p.endswith("Include / exclude") for p in paths)
    assert len(paths) == len(set(paths)), "no duplicates"
    for needle in ("Shift+Wheel", "Space", "Home / End", "Right-click a header", "_curated.hdf5",
                   "Run ROI filter", str(PREFS)):
        assert needle in html, needle
    desc = {r[0]: r[2] for r in rows}
    assert desc["Edit ▸ Undo"].startswith("Undo the last") and desc["File ▸ Save"].startswith("Write")
    assert not desc["File ▸ Open…"].startswith("Open…") and desc["File ▸ Quit"]
    assert desc["Analysis ▸ Run ROI filter…"].endswith("columns."), desc["Analysis ▸ Run ROI filter…"]
    dlg = insp.HelpDialog(w)
    assert "Keyboard" in dlg.browser.toPlainText() and "Ctrl+Shift+F" in dlg.browser.toPlainText()
    ok(f"help lists all {len(keyed)} action shortcuts (from the menus, no duplicates) + context keys + usage")

    # ── preference dialog: Appearance tab
    d = insp.PreferencesDialog(insp.load_prefs(), {}, None, "appearance", w)
    assert [d.tabs.tabText(i) for i in range(d.tabs.count())] == ["Appearance", "ROI filter"]
    f = d._fields["appearance"]
    assert f["theme"].value() == "dark" and f["plot_background"].value() == "auto" and f["font_size"].value() is None
    assert f["font_size"].hint.text() == f"default: {round(default_pt)}" and f["font_size"].spin.value() == round(default_pt)
    f["theme"].set_value("light"); f["font_size"].set_value(13)
    v = d.values()
    assert v["appearance"] == {"theme": "light", "plot_background": "auto", "font_size": 13,
                               "alpha_selected": 55, "alpha_all": 12, "alpha_excluded": 22}
    assert "roi_filter" in v
    ok("Appearance tab first: theme / plot background choices and nullable font size")

    # ── live switch to light (through the real save path)
    w.table.setFocus()
    w._on_cell_click(1, False); w._on_cell_click(3, True); app.processEvents()
    def to_light(dlg):
        dlg._fields["appearance"]["theme"].set_value("light")
        dlg._fields["appearance"]["font_size"].set_value(13)
    insp.PreferencesDialog.exec = lambda self: (to_light(self), QDialog.DialogCode.Accepted)[1]
    assert w._edit_prefs("appearance")
    app.processEvents()
    assert json.loads(PREFS.read_text())["appearance"]["theme"] == "light"
    assert win_light() > 200
    for v in (w.trace_view, w.event_view, w.cell_view, w.corr_view, w.dist_view):
        assert bg(v) == "#ffffff", (type(v).__name__, bg(v))
    assert insp.PALETTE_HEX == insp.PALETTE_LIGHT_BG
    t = w.table
    unsel = next(c for c in range(w.store.n) if c not in (1, 3))
    assert t.item(t.row_of(unsel), 0).background().color().getRgb()[:3] == (255, 255, 255)
    assert t.item(t.row_of(unsel), 0).foreground().color().getRgb()[:3] == (25, 25, 25)
    assert t.selected_comps() == [1, 3], "selection kept across the theme switch"
    sel_bg = t.item(t.row_of(1), 0).background().color()
    assert sel_bg.name().upper() == insp.PALETTE_LIGHT_BG[0].upper(), sel_bg.name()
    assert "#555555" in w.movie_bar.lbl.styleSheet() and "#555555" in w._sel_lbl.styleSheet()
    assert app.font().pointSize() == 13
    assert w.trace_view._cursor.pen.color().name() == insp.PLOT_THEMES["light"]["cursor"]
    ok("saving Light re-themes live: palette, every plot, table cells, selection colours, labels, font")

    # light UI, dark plots
    w.apply_theme({"theme": "light", "plot_background": "dark", "font_size": None}); app.processEvents()
    assert win_light() > 200 and bg(w.trace_view) == "#111111" and insp.PALETTE_HEX == insp.PALETTE_DARK_BG
    assert abs(app.font().pointSizeF() - default_pt) < 0.01, "font back to default"
    ok("light interface with dark plots; font size back to default")

    # system
    ui, plot = insp.resolve_theme({"theme": "system", "plot_background": "auto"})
    assert ui in ("dark", "light") and plot == ui
    insp.theme._system_scheme = lambda: "dark"
    assert insp.resolve_theme({"theme": "system"}) == ("dark", "dark")
    insp.theme._system_scheme = lambda: "light"
    assert insp.resolve_theme({"theme": "system", "plot_background": "dark"}) == ("light", "dark")
    ok("System follows the desktop scheme; plot background can override it")

    # event view / trace view draw cleanly in light
    w.apply_theme({"theme": "light"}); app.processEvents()
    w.store.detect_events(30.0, 0.4) if w.store.can_detect else None
    w._full_refresh(); w._on_cell_click(0, False); app.processEvents()
    w.close(); app.processEvents()

    # ── startup with the saved light theme builds light widgets directly
    PREFS.write_text(json.dumps({"appearance": {"theme": "light"}}))
    w = new_window()
    assert win_light() > 200 and bg(w.trace_view) == "#ffffff" and bg(w.cell_view) == "#ffffff"
    w.close()
    PREFS.write_text(json.dumps({"appearance": {"theme": "nonsense", "plot_background": 5}}))
    w = new_window()
    assert win_light() < 60 and bg(w.trace_view) == "#111111", "invalid values → dark"
    w.close()
    ok("startup uses the saved theme; invalid values fall back to dark")
    print("ALL OK")
finally:
    shutil.rmtree(root)
