"""Offscreen test: roi_filter fields in caiman_inspector."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
require('h5py')
import csv, importlib.util, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np
import h5py

insp = load_inspector()
from caiman.utils import roi_filter as rf
from caiman.utils import event_detection as ed

from PyQt6.QtWidgets import QApplication, QMessageBox
app = QApplication.instance() or QApplication([])
info = []
QMessageBox.information = staticmethod(lambda *a, **k: info.append(a[2]))
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)

d1, d2, T, fr = 60, 80, 1800, 30.0
dims = (d1, d2)
yy, xx = np.mgrid[:d1, :d2]
def disk(cy, cx, r): return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
foot = [disk(30, 40, 3), disk(30, 15, 0.6), disk(25, 60, 12), disk(3, 40, 3),
        disk(45, 20, 3), disk(45, 30, 3), disk(15, 25, 3), disk(15, 55, 3)]
K = len(foot)
A = np.column_stack([f.ravel(order="F") for f in foot]).astype(np.float32)
A[A < 1e-3] = 0
rng = np.random.default_rng(0)
F = 0.01 * rng.standard_normal((K, T)); C = np.abs(0.001 * rng.standard_normal((K, T)))
kern = np.exp(-np.arange(60) / 15.0)
for k in (0, 4, 5, 6):                     # rows 1,2,3,7 silent
    for t0 in rng.choice(T - 80, 4, replace=False):
        F[k, t0:t0 + 60] += 0.6 * kern; C[k, t0:t0 + 60] += 0.6 * kern
F = F.astype(np.float32); C = C.astype(np.float32)
SNR = np.array([5, 3, 3, 3, 5, 5, 0.2, 3.0])
params = rf.resolve_params({"gSig": [3, 3], "fr": fr, "decay_time": 0.4, "min_SNR": 2.5,
                            "SNR_lowest": 0.5, "rval_thr": None, "use_cnn": False})
fields = rf.evaluate(A, dims, SNR_comp=SNR, F_dff=F, C=C, params=params)
# expected verdicts
exp_pass = fields["passed"]
assert exp_pass.tolist() == [True, False, False, False, True, True, False, True], exp_pass
assert fields["active"].tolist() == [True, False, False, False, True, True, True, False]


class FakeCNM:
    """Stands in for the CaImAn object: save() rewrites the file from scratch
    (mode 'w'), as caiman's save_dict_to_hdf5 does."""
    def __init__(self):
        self.estimates = types.SimpleNamespace()
        self.provenance = []
    def save(self, path):
        with h5py.File(path, "w") as h:
            h.create_dataset("estimates/C", data=np.asarray(self.estimates.C))


def make_store(path, idx=None, with_cnm=True):
    idx = np.arange(K) if idx is None else np.asarray(idx)
    st = insp.ComponentStore(A[:, idx], C[idx], rng.random(dims), dims, SNR_comp=SNR[idx],
                             traces={insp.TRACE_DENOISED: C[idx], insp.TRACE_DFF: F[idx]},
                             orig_idx=idx, fr=fr, decay_time=0.4,
                             cnm_obj=FakeCNM() if with_cnm else None, fpath=str(path))
    insp._attach_roi_filter(st, str(path))
    insp._attach_stroh_results(st, str(path))
    return st


def headers(t):
    return [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]


root = Path(tempfile.mkdtemp())
try:
    # ── no roi_filter group: inspector unchanged
    plain = root / "plain_results.hdf5"
    with h5py.File(plain, "w") as h: h.create_dataset("estimates/C", data=C)
    st = make_store(plain)
    assert st.qc is None and not st.has_approval and st.approved.all()
    w = insp.InspectorWindow(st); w.show(); app.processEvents()
    assert headers(w.table) == ["Label", "Max A", "Peak ΔF", "SNR"], headers(w.table)
    w.close()
    ok("file without roi_filter: no filter columns, no Incl (unchanged)")

    # ── full group; inspector shows idx_components subset (row 3 excluded)
    full = root / "S_results.hdf5"
    with h5py.File(full, "w") as h:
        h.create_dataset("estimates/C", data=C)
        rf.write_group(h, fields, params)
    shown = [0, 1, 2, 4, 5, 6, 7]
    st = make_store(full, shown)
    assert st.qc is not None and st.qc["area_px"].tolist() == fields["area_px"][shown].tolist()
    assert st.approved.tolist() == exp_pass[shown].tolist(), "Incl seeded from passed"
    w = insp.InspectorWindow(st); w.resize(1500, 800); w.show(); app.processEvents()
    t = w.table
    H = headers(t)
    assert H == ["Label", "Max A", "Peak ΔF", "SNR", "QC", "Active", "Fail", "Area", "Solid",
                 "Ecc", "Edge", "Rate", "Pk ΔF/F", "Skew", "Act %", "Incl"], H
    col = {h: i for i, h in enumerate(H)}
    r2 = t.row_of(2)                                         # shown row 2 = file row 2 (large)
    assert t.item(r2, col["QC"]).text() == "✗" and t.item(r2, col["Fail"]).text() == "area>,inact"
    assert "area_large" in t.item(r2, col["Fail"]).toolTip()
    assert t.item(t.row_of(0), col["Fail"]).text() == "—"
    assert t.item(t.row_of(0), col["Incl"]).text() == "✓" and t.item(r2, col["Incl"]).text() == "✗"
    assert "verdict" in t.horizontalHeaderItem(col["QC"]).toolTip() and "QC verdict" in t.horizontalHeaderItem(col["Incl"]).toolTip()
    ok("roi_filter rows mapped through idx_components; columns + Incl seeded from QC")

    # overlay: excluded (failed) footprints drawn red
    ov = insp._build_roi_overlay(st)
    red = np.array(insp._rgb(insp.EXCLUDED)) if hasattr(insp, "_rgb") else None
    m1 = st.footprint(1) > 0.5
    m0 = st.footprint(0) > 0.5
    assert ov[m1][:, 0].mean() > ov[m1][:, 2].mean(), "failed ROI tinted red"
    assert ov[m0][:, 2].mean() > ov[m0][:, 0].mean(), "passed ROI cyan"
    ok("overlay tints QC-failed ROIs as excluded")

    # sorting the new columns
    from PyQt6.QtCore import Qt
    t.sortByColumn(col["Area"], Qt.SortOrder.AscendingOrder); app.processEvents()
    areas = [st.qc["area_px"][t.comp_of(r)] for r in range(t.rowCount())]
    assert areas == sorted(areas)
    t.sortByColumn(col["Fail"], Qt.SortOrder.AscendingOrder); app.processEvents()
    nfail = [bin(int(st.qc["fail_reasons"][t.comp_of(r)])).count("1") for r in range(t.rowCount())]
    assert nfail == sorted(nfail)
    t.sortByColumn(-1, Qt.SortOrder.AscendingOrder); t.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    ok("new columns sort numerically (Area) and by number of failed criteria (Fail)")

    # include toggle works without events
    w._on_cell_click(1, False); w._toggle_include(); app.processEvents()
    assert st.approved[1] and t.item(t.row_of(1), col["Incl"]).text() == "✓"
    ok("I toggles include without event results")

    # merge recomputes the merged row; undo restores
    n0 = st.n
    w._on_cell_click(4, False); w._on_cell_click(5, True); w._do_merge(); app.processEvents()
    new = st.n - 1
    assert st.n == n0 - 1 and all(v.size == st.n for v in st.qc.values())
    expect = rf.evaluate(st._A[:, new:new + 1], dims, SNR_comp=st.SNR_comp[new:new + 1],
                         F_dff=st.dff[new:new + 1], C=st.C[new:new + 1], params=params)
    # store rows 4,5 = file rows 5,6: two separate blobs → the mask keeps the
    # peak's connected region, so the merged area equals one blob's area
    assert st.qc["area_px"][new] == expect["area_px"][0], (st.qc["area_px"][new], expect["area_px"][0])
    assert st.qc["solidity"][new] == expect["solidity"][0]
    assert st.qc["n_events"][new] == expect["n_events"][0]
    assert st.qc["passed"].dtype == np.bool_
    w._do_undo(); app.processEvents()
    assert st.n == n0 and st.qc["area_px"].tolist() == fields["area_px"][shown].tolist()
    w._do_redo(); app.processEvents()
    assert st.n == n0 - 1
    ok("merge re-evaluates the merged ROI; undo/redo restore the fields")

    # delete
    gone = st.labels.index("C001")
    w._on_cell_click(gone, False); w._do_delete(); app.processEvents()
    assert "C001" not in st.labels and all(v.size == st.n for v in st.qc.values())
    ok("delete drops the rows' fields")

    # session info
    w._session_info()
    assert "ROI filter:" in info[-1] and "passed" in info[-1] and "fails:" in info[-1], info[-1]
    ok("session info summarises the filter")

    # save → group rewritten for saved rows with the include flags; reload
    out = root / "S_curated_results.hdf5"
    st.save(str(out))
    with h5py.File(out, "r") as h:
        assert "estimates" in h and "roi_filter" in h
        f2, p2, appr2 = rf.read_group(h["roi_filter"])
        assert f2["area_px"].size == st.n and appr2.tolist() == st.approved.tolist()
        assert p2["min_SNR"] == 2.5
        assert h["roi_filter/idx_passed"][()].tolist() == np.flatnonzero(st.qc["passed"]).tolist()
    st2 = make_store(out, None) if False else None
    re_st = insp.ComponentStore(st._A, st.C, st.Cn, dims, orig_idx=np.arange(st.n), fpath=str(out))
    insp._attach_roi_filter(re_st, str(out))
    assert re_st.approved.tolist() == st.approved.tolist(), "manual approvals win over passed"
    assert all(np.array_equal(re_st.qc[k], st.qc[k], equal_nan=True) for k in st.qc)
    ok("save writes the group for the curated rows + manual approvals; reload matches")
    w.close()

    # ── activity fields absent (no ΔF/F at filter time) → those columns hidden
    noact = root / "N_results.hdf5"
    f_noact = rf.evaluate(A, dims, SNR_comp=SNR, params=params)
    with h5py.File(noact, "w") as h:
        h.create_dataset("estimates/C", data=C); rf.write_group(h, f_noact, params)
    st = make_store(noact)
    w = insp.InspectorWindow(st); w.show(); app.processEvents()
    H = headers(w.table)
    assert "QC" in H and "Area" in H and "Active" not in H and "Rate" not in H and "Skew" not in H, H
    w.close()
    ok("columns follow field existence (no activity fields → no activity columns)")

    # ── stroh_results approvals take precedence over the filter seed
    both = root / "B_results.hdf5"
    ev = ed.detect_events(F, C, fr, 0.4, SNR, None)
    manual = np.zeros(K, bool); manual[[1, 2]] = True
    with h5py.File(both, "w") as h:
        h.create_dataset("estimates/C", data=C)
        rf.write_group(h, fields, params)
        ed.write_stroh_results(h, ev, manual)
    st = make_store(both)
    assert st.events is not None and st.approved.tolist() == manual.tolist()
    w = insp.InspectorWindow(st); w.show(); app.processEvents()
    H = headers(w.table)
    assert H[-2:] == ["Ev/min", "Incl"] and "QC" in H
    # export: ROIStats carries qc_ columns for the included ROIs
    paths = insp.export_event_tables(st, str(root / "B_PeakStats.csv"))
    roi_csv = [p for p in paths if "ROIStats" in p][0]
    with open(roi_csv) as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["ROI"]) for r in rows] == [1, 2]
    assert rows[1]["qc_area_px"] == str(fields["area_px"][2]) and rows[1]["qc_passed"] == "0"
    assert "area_large" in rows[1]["qc_fail"]
    w.close()
    ok("event approvals win; export ROIStats gains qc_* columns")
    print("ALL OK")
finally:
    shutil.rmtree(root)
