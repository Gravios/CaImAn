"""Offscreen test: remembered layout / last folder + column show/hide."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
require('h5py')
import importlib.util, json, os, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np
import h5py

root = Path(tempfile.mkdtemp())
PREFS = root / "cfg" / "inspector_prefs.json"
os.environ["CAIMAN_INSPECTOR_PREFS"] = str(PREFS)
insp = load_inspector()
from caiman.utils import roi_filter as rf
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFileDialog, QMenu
app = QApplication.instance() or QApplication([])

d1, d2, T = 40, 50, 300
dims = (d1, d2)
rng = np.random.default_rng(0)
yy, xx = np.mgrid[:d1, :d2]
K = 6
A = np.column_stack([np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8).ravel(order="F")
                     for cy, cx in rng.uniform(5, 35, (K, 2))]).astype(np.float32)
C = rng.random((K, T)).astype(np.float32)


def store(folder, name, qc=False, rvals=False):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_results.hdf5"
    with h5py.File(path, "w") as h:
        h.create_dataset("estimates/C", data=C)
        if qc:
            p = rf.resolve_params({"gSig": [2, 2], "min_SNR": 2.0, "use_cnn": False})
            rf.write_group(h, rf.evaluate(A, dims, SNR_comp=np.linspace(1, 6, K), params=p), p)
    st = insp.ComponentStore(A, C, rng.random(dims), dims, SNR_comp=np.linspace(1, 6, K),
                             r_values=np.linspace(0, 1, K) if rvals else None,
                             traces={insp.TRACE_DENOISED: C, insp.TRACE_DFF: C * 0.1},
                             fpath=str(path))
    insp._attach_roi_filter(st, str(path))
    return st


def headers(t): return t.column_names()
def visible(t): return [n for c, n in enumerate(t.column_names()) if not t.isColumnHidden(c)]


try:
    PREFS.parent.mkdir(parents=True)
    PREFS.write_text(json.dumps({"roi_filter": {"edge_px": 25}}))

    # ── first session: change everything, close
    a = store(root / "dayA", "A", qc=True)
    w = insp.InspectorWindow(a); w.resize(780, 700); w.show(); app.processEvents()
    assert w._start_dir() == str(root / "dayA"), "folder of the opened file is the last folder"
    w._splitters["root"].setSizes([300, 480]); app.processEvents()
    root_sizes = w._splitters["root"].sizes()
    t = w.table
    t.set_column_visible("SNR", False); t.set_column_visible("Solid", False)
    t.set_column_visible("Label", False)                         # not hideable
    assert "SNR" not in visible(t) and "Solid" not in visible(t) and "Label" in visible(t)
    saved = json.loads(PREFS.read_text())
    assert sorted(saved["ui"]["hidden_columns"]) == ["SNR", "Solid"], "saved immediately"
    assert saved["roi_filter"] == {"edge_px": 25}, "other sections kept"
    t.sort_by_name("QC", "desc"); app.processEvents()
    w.act_invert_bg.trigger(); w.act_show_rois.trigger(); w.act_dist_log.trigger()
    w.bg_combo.setCurrentText("footprint max"); w.trace_combo.setCurrentText(insp.TRACE_DFF)
    assert a.bg_key == "footprint max" and a.disp_key == insp.TRACE_DFF
    w.close(); app.processEvents()
    ui = json.loads(PREFS.read_text())["ui"]
    assert ui["sort"] == {"column": "QC", "order": "desc"}
    assert ui["view"] == {"hide_rejected": False, "invert_bg": True, "show_rois": False, "hide_excluded": False, "dist_log": True, "event_markers": "off"}
    assert ui["background"] == "footprint max" and ui["trace"] == insp.TRACE_DFF
    assert ui["last_dir"] == str(root / "dayA") and ui["geometry"] and len(ui["splitters"]) == 5
    ok("close saves geometry, panes, hidden columns, sort, toggles, BG/trace, last folder")

    # ── reopen on a file with a different column set (r_val shifts QC right)
    b = store(root / "dayB", "B", qc=True, rvals=True)
    w = insp.InspectorWindow(b); w.show(); app.processEvents()
    t = w.table
    assert w.width() == 780 and w.height() == 700, (w.width(), w.height())
    rs = w._splitters["root"].sizes()
    assert abs(rs[0] - root_sizes[0]) <= 2, (rs, root_sizes)
    assert "SNR" not in visible(t) and "Solid" not in visible(t) and "r_val" in visible(t)
    assert t.sort_state() == ("QC", "desc"), t.sort_state()
    qc = [b.qc["passed"][t.comp_of(r)] for r in range(t.rowCount())]
    assert qc == sorted(qc, key=lambda v: 0 if v else 1, reverse=True), qc
    assert w.act_invert_bg.isChecked() and w.cell_view._invert_bg
    assert not w.act_show_rois.isChecked() and not w.cell_view._show_all
    assert w.act_dist_log.isChecked()
    assert b.bg_key == "footprint max" and b.disp_key == insp.TRACE_DFF
    ok("reopen restores all of it; sort follows the column *name* (QC moved to another index)")

    # ── last folder in dialogs
    seen = []
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (seen.append(a[2]), ("", ""))[1])
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: (seen.append(a[2]), "")[1])
    w._open_file(); w._load_background(); w._batch_detect()
    assert seen == [str(root / "dayB")] * 3, seen
    c_path = root / "dayC" / "C_results.hdf5"; c_path.parent.mkdir()
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(c_path), ""))
    insp.window_session.load_any = lambda p: store(root / "dayC", "C")
    w._open_file(); app.processEvents()
    assert w._start_dir() == str(root / "dayC")
    assert json.loads(PREFS.read_text())["ui"]["last_dir"] == str(root / "dayC"), "saved right away"
    ok("file dialogs start in the last folder; opening a file updates it immediately")

    # swap to a file without QC: sort column missing → unsorted, BG/trace still applied
    t = w.table
    assert "QC" not in headers(t) and t.sort_state() is None
    assert w.store.bg_key == "footprint max" and w.store.disp_key == insp.TRACE_DFF
    ok("file without the sorted column: unsorted; remembered BG/trace applied to the new file")

    # ── Columns menu (header right-click and View ▸ Columns)
    m = QMenu(); t.fill_column_menu(m)
    acts = {a.text(): a for a in m.actions() if a.text()}
    assert list(acts)[:len(headers(t))] == headers(t)
    assert not acts["Label"].isEnabled() and not acts["SNR"].isChecked() and acts["Max A"].isChecked()
    assert acts["Show all columns"].isEnabled()
    acts["Max A"].setChecked(False)
    assert "Max A" not in visible(t)
    assert "Max A" in json.loads(PREFS.read_text())["ui"]["hidden_columns"]
    w._columns_menu.aboutToShow.emit()
    assert [a.text() for a in w._columns_menu.actions() if a.text()][:len(headers(t))] == headers(t)
    acts["Show all columns"].trigger()
    assert visible(t) == headers(t)
    assert json.loads(PREFS.read_text())["ui"]["hidden_columns"] == []
    ok("Columns menu: per-column toggles (Label locked), Show all; changes saved immediately")

    # hidden columns survive a refresh (merge)
    t.set_column_visible("Peak ΔF", False)
    w._on_cell_click(0, False); w._on_cell_click(1, True); w._do_merge(); app.processEvents()
    assert "Peak ΔF" not in visible(t)
    ok("hidden columns stay hidden across table rebuilds")

    # reset layout
    t.sort_by_name("Max A", "desc"); w._splitters["root"].setSizes([700, 600])
    w._reset_layout(); app.processEvents()
    assert visible(t) == headers(t) and t.sort_state() is None
    rs = w._splitters["root"].sizes()
    os.environ["CAIMAN_INSPECTOR_PREFS"] = str(root / "fresh.json")      # a default window
    fresh = insp.InspectorWindow(insp._make_demo_store()); fresh.resize(w.size()); fresh.show(); app.processEvents()
    fs = fresh._splitters["root"].sizes(); fresh.close()
    os.environ["CAIMAN_INSPECTOR_PREFS"] = str(PREFS)
    assert abs(rs[0] - fs[0]) <= 3 and rs[0] < 600, (rs, fs)
    ok("View ▸ Reset layout: default panes, all columns, unsorted")
    w.close()

    # ── garbage in the ui section never breaks startup
    PREFS.write_text(json.dumps({"ui": {"geometry": "%%%", "splitters": {"root": "@@", "nope": "x"},
                                        "sort": "bad", "view": {"invert_bg": True},
                                        "hidden_columns": ["Label", "SNR"], "last_dir": "/does/not/exist"}}))
    w = insp.InspectorWindow(store(root / "dayD", "D")); w.show(); app.processEvents()
    assert "Label" in visible(w.table) and "SNR" not in visible(w.table) and w.act_invert_bg.isChecked()
    assert w._start_dir("fb") == str(root / "dayD")
    w.close()
    PREFS.write_text("not json")
    w = insp.InspectorWindow(insp._make_demo_store()); w.show(); app.processEvents()
    assert w._start_dir("fb") == "fb" and w.act_show_rois.isChecked()
    w.close()
    assert "ui" in json.loads(PREFS.read_text()), "rewritten cleanly on close"
    ok("corrupt or missing values ignored; defaults used; file rewritten on close")
    print("ALL OK")
finally:
    shutil.rmtree(root)
