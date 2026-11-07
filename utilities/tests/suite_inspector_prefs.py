"""Offscreen test: Run ROI filter + preference editor in caiman_inspector."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
require('h5py')
import importlib.util, json, os, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np
import h5py

root = Path(tempfile.mkdtemp())
os.environ["CAIMAN_INSPECTOR_PREFS"] = str(root / "cfg" / "inspector_prefs.json")
insp = load_inspector()
from caiman.utils import roi_filter as rf
from caiman.utils import event_detection as ed
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
app = QApplication.instance() or QApplication([])
msgs = []
QMessageBox.information = staticmethod(lambda *a, **k: msgs.append(a[2]))
QMessageBox.warning = staticmethod(lambda *a, **k: msgs.append(a[2]))
QMessageBox.critical = staticmethod(lambda *a, **k: msgs.append(("CRIT", a[2])))

d1, d2, T, fr = 60, 80, 1800, 30.0
dims = (d1, d2)
yy, xx = np.mgrid[:d1, :d2]
def disk(cy, cx, r): return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
foot = [disk(30, 40, 3), disk(30, 15, 0.6), disk(25, 60, 12), disk(3, 40, 3),
        disk(45, 20, 3), disk(20, 15, 3), disk(15, 30, 3), disk(16, 55, 3)]
K = len(foot)
A = np.column_stack([f.ravel(order="F") for f in foot]).astype(np.float32); A[A < 1e-3] = 0
rng = np.random.default_rng(0)
F = 0.01 * rng.standard_normal((K, T)); C = np.abs(0.001 * rng.standard_normal((K, T)))
kern = np.exp(-np.arange(60) / 15.0)
for k in (0, 4, 5, 6):
    for t0 in rng.choice(T - 80, 4, replace=False):
        F[k, t0:t0 + 60] += 0.6 * kern; C[k, t0:t0 + 60] += 0.6 * kern
F = F.astype(np.float32); C = C.astype(np.float32)
SNR = np.array([5, 3, 3, 3, 5, 5, 0.2, 3.0])
RUN = types.SimpleNamespace(quality={"min_SNR": 1.5, "SNR_lowest": 0.5, "rval_thr": 0.6,
                                     "rval_lowest": -1.0, "use_cnn": True, "min_cnn_thr": 0.6,
                                     "cnn_lowest": 0.1},
                            init={"gSig": (3, 3)}, data={"fr": fr, "decay_time": 0.4})


class FakeCNM:
    def __init__(self): self.estimates = types.SimpleNamespace(); self.provenance = []; self.params = RUN
    def save(self, path):
        with h5py.File(path, "w") as h: h.create_dataset("estimates/C", data=np.asarray(self.estimates.C))


def make_store(path, dff=True, cnm=True):
    tr = {insp.TRACE_DENOISED: C}
    if dff: tr[insp.TRACE_DFF] = F
    st = insp.ComponentStore(A, C, rng.random(dims), dims, SNR_comp=SNR, traces=tr, fr=fr,
                             decay_time=0.4, cnm_obj=FakeCNM() if cnm else None, fpath=str(path))
    insp._attach_roi_filter(st, str(path)); insp._attach_stroh_results(st, str(path))
    return st


def headers(t): return [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
def accept_with(cls, fn=None):
    """Make cls.exec run fn(dialog) then accept."""
    def _exec(self):
        if fn: fn(self)
        return QDialog.DialogCode.Accepted
    cls.exec = _exec

try:
    # ── prefs storage
    assert insp.load_prefs() == {}
    d = insp.roi_filter_prefs()
    assert {k: v for k, v in d.items() if k != "gSig"} == rf.DEFAULTS and d["gSig"] is None
    Path(insp.prefs_path()).parent.mkdir(parents=True); Path(insp.prefs_path()).write_text("{not json")
    assert insp.load_prefs() == {}, "unreadable prefs → empty, no crash"
    os.remove(insp.prefs_path())
    ok("prefs: per-user JSON path, defaults when missing or unreadable")

    # ── older session: file without roi_filter fields
    old = root / "old_results.hdf5"
    with h5py.File(old, "w") as h: h.create_dataset("estimates/C", data=C)
    st = make_store(old)
    w = insp.InspectorWindow(st); w.resize(1500, 800); w.show(); app.processEvents()
    assert "QC" not in headers(w.table) and st.qc is None
    assert "Run ROI filter" in w.statusBar().currentMessage(), w.statusBar().currentMessage()
    ok("older session: no filter columns; status bar points to Run ROI filter")

    # run dialog contents
    dlg = insp.RunFilterDialog(w, st)
    txt = dlg.summary.text()
    assert "SNR ≥ 1.5 (floor 0.5)" in txt and "CNN ≥ 0.6" in txt and "gSig 3×3" in txt, txt
    assert "fr 30 Hz" in txt and "decay 0.4 s" in txt and "events ≥ 1" in txt
    assert dlg.chk_incl.isChecked() and not dlg.warn.isVisibleTo(dlg)
    ok("run dialog: thresholds resolved from prefs + the run's CaImAn params; Incl reset defaults on")

    # run it
    accept_with(insp.RunFilterDialog)
    w._on_cell_click(2, False)
    w._run_filter_dialog(); app.processEvents()
    assert st.qc is not None and st.qc["passed"].tolist() == [True, False, False, False, True, True, False, True]
    H = headers(w.table)
    assert "QC" in H and "Rate" in H and H[-1] == "Incl", H
    assert st.approved.tolist() == st.qc["passed"].tolist()
    assert w.statusBar().currentMessage().startswith("ROI filter: 4/8 passed"), w.statusBar().currentMessage()
    assert w.table.selected_comps() == [2], "selection kept"
    assert st.qc_params["gSig"] == [3, 3] and st.qc_params["min_SNR"] == 1.5
    ok("Run ROI filter on an older session adds the fields, columns and Incl")

    # undo / redo
    w._do_undo(); app.processEvents()
    assert st.qc is None and "QC" not in headers(w.table) and not st.has_approval
    w._do_redo(); app.processEvents()
    assert st.qc is not None and "QC" in headers(w.table)
    ok("undo removes the fields again; redo restores them")

    # ── preference editor
    dlg = insp.PreferencesDialog(insp.load_prefs(), st.resolve_filter_params({}),
                                 st.qc_params, "roi_filter", w)
    f = dlg._fields["roi_filter"]
    assert f["min_SNR"].value() is None and f["min_SNR"].hint.text() == "from run: 1.5"
    assert f["gSig"].hint.text() == "from run: 3×3"
    assert f["use_cnn"].combo.currentText() == "from run (on)"
    assert f["edge_px"].value() == 10 and f["min_solidity"].value() == 0.7
    assert dlg.tabs.tabText(dlg.tabs.currentIndex()) == "ROI filter"
    f["edge_px"].set_value(25); f["min_solidity"].set_value(None); f["use_cnn"].combo.setCurrentIndex(2)
    f["gSig"].chk.setChecked(True); f["gSig"].spin.setValue(4.0)
    f["min_SNR"].chk.setChecked(True); f["min_SNR"].spin.setValue(2.5)
    v = dlg.values()["roi_filter"]
    assert v["edge_px"] == 25 and v["min_solidity"] is None and v["use_cnn"] is False
    assert v["gSig"] == 4.0 and v["min_SNR"] == 2.5 and v["rval_thr"] is None
    assert "enabled" not in v or v["enabled"] is True
    ok("editor: nullable fields show the run's value; edits read back with null = off/from run")

    # From pipeline JSON (with // comment lines) and From this file
    pj = root / "S_pipeline.json"
    pj.write_text('{\n  // comment\n  "roi_filter": {"edge_px": 33, "min_events": 2, "enabled": true}\n}\n')
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(pj), ""))
    dlg._from_json()
    assert f["edge_px"].value() == 33 and f["min_events"].value() == 2 and f["gSig"].value() == 4.0
    pj.write_text('{"cnmf": {}}'); n = len(msgs); dlg._from_json()
    assert len(msgs) == n + 1 and "no roi_filter section" in msgs[-1]
    dlg.findChildren(type(dlg.tabs.widget(0)))  # noqa
    dlg._apply("roi_filter", st.qc_params)
    assert f["edge_px"].value() == 10 and f["min_SNR"].value() == 1.5
    dlg._apply("roi_filter", insp.PREF_SECTIONS["roi_filter"][2]())
    assert f["min_SNR"].value() is None and f["edge_px"].value() == 10
    ok("editor: Defaults / From pipeline JSON (// comments) / From this file")

    # save through the window, keeping unrelated keys
    insp.save_prefs({"roi_filter": {"enabled": False}, "other": {"x": 1}})
    def edit(d):
        ff = d._fields["roi_filter"]
        ff["edge_px"].set_value(25); ff["gSig"].set_value(4.0)
    accept_with(insp.PreferencesDialog, edit)
    assert w._edit_prefs("roi_filter")
    saved = json.loads(Path(insp.prefs_path()).read_text())
    assert saved["other"] == {"x": 1} and saved["roi_filter"]["enabled"] is False
    assert saved["roi_filter"]["edge_px"] == 25 and saved["roi_filter"]["gSig"] == 4.0
    assert insp.roi_filter_prefs()["edge_px"] == 25
    ok("Preferences… saves to the JSON file; unrelated keys and hidden keys kept")

    # rerun with the new prefs, keep manual Incl
    st.approved[:] = True
    accept_with(insp.RunFilterDialog, lambda d: d.chk_incl.setChecked(False))
    w._run_filter_dialog(); app.processEvents()
    assert st.qc_params["edge_px"] == 25 and st.qc_params["gSig"] == [4.0, 4.0]
    assert int(((st.qc["fail_reasons"] & rf.REASONS["edge"]) != 0).sum()) >= 3
    assert st.approved.all(), "Incl kept when unticked"
    ok("rerun uses the saved preferences; unticked → include flags kept")

    # save writes the group; reload matches
    out = root / "old_curated_results.hdf5"
    st.save(str(out))
    st2 = insp.ComponentStore(A, C, st.Cn, dims, fpath=str(out)); insp._attach_roi_filter(st2, str(out))
    assert st2.qc_params["edge_px"] == 25 and st2.approved.all()
    assert all(np.array_equal(st2.qc[k], st.qc[k], equal_nan=True) for k in st.qc)
    ok("Save stores the rerun fields + parameters in the results file")
    w.close()

    # ── warnings: no ΔF/F, no gSig source
    st3 = make_store(old, dff=False, cnm=False)
    insp.save_prefs({})
    w3 = insp.InspectorWindow(st3); w3.show(); app.processEvents()
    dlg = insp.RunFilterDialog(w3, st3)
    wt = dlg.warn.text()
    assert "No ΔF/F" in wt and "No gSig" in wt, wt
    accept_with(insp.RunFilterDialog)
    w3._run_filter_dialog(); app.processEvents()
    assert st3.qc is not None and "active" not in st3.qc and "Rate" not in headers(w3.table)
    ok("missing ΔF/F / gSig: warned, run anyway, activity columns omitted")

    # events present: Incl reset defaults off and approvals survive
    both = root / "ev_results.hdf5"
    ev = ed.detect_events(F, C, fr, 0.4, SNR, None); manual = np.zeros(K, bool); manual[[1, 2]] = True
    with h5py.File(both, "w") as h:
        h.create_dataset("estimates/C", data=C); ed.write_stroh_results(h, ev, manual)
    st4 = make_store(both)
    w4 = insp.InspectorWindow(st4); w4.show(); app.processEvents()
    dlg = insp.RunFilterDialog(w4, st4)
    assert not dlg.chk_incl.isChecked() and "event approvals" in dlg.chk_incl.toolTip()
    accept_with(insp.RunFilterDialog, lambda d: None)
    w4._run_filter_dialog(); app.processEvents()
    assert st4.approved.tolist() == manual.tolist() and st4.qc is not None
    H = headers(w4.table); assert H[-2:] == ["Ev/min", "Incl"] and "QC" in H
    ok("with event approvals: Incl reset is off by default and approvals survive")

    # blinded review blocks the action
    w4._blind = {"files": [], "i": 0, "rater": "x"}
    called = []
    insp.RunFilterDialog.exec = lambda self: called.append(1) or QDialog.DialogCode.Rejected
    w4._run_filter_dialog()
    assert not called and "Blinded review" in w4.statusBar().currentMessage()
    w4._blind = None
    ok("blocked during blinded review")
    w3.close(); w4.close()
    print("ALL OK")
finally:
    shutil.rmtree(root)
