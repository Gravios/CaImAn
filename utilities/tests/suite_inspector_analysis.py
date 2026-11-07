"""Analysis side of the inspector: merge refit, unknown merged metrics,
undoable event detection, export columns, blinded-review starting state."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
require("h5py")
import csv, os, pickle, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np
import scipy.sparse

root = Path(tempfile.mkdtemp())
os.environ["CAIMAN_INSPECTOR_PREFS"] = str(root / "prefs.json")
stub_caiman()


class Est: pass
class CNM:
    def save(self, path):
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("pickle", data=np.void(pickle.dumps(self)))
def load_CNMF(path):
    import h5py
    with h5py.File(path, "r") as f:
        return pickle.loads(f["pickle"][()].tobytes())
for name in ("caiman.source_extraction", "caiman.source_extraction.cnmf"):
    m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m
mc = types.ModuleType("caiman.source_extraction.cnmf.cnmf"); mc.load_CNMF = load_CNMF
sys.modules["caiman.source_extraction.cnmf.cnmf"] = mc

insp = load_inspector()
import h5py
from caiman.utils import event_detection as ed
from PyQt6.QtWidgets import QApplication, QMessageBox
app = QApplication.instance() or QApplication([])


def pg_color(pen):
    import pyqtgraph as pg
    return pg.mkPen(pen).color().name()

rng = np.random.default_rng(7)
d1, d2, T, FR = 40, 60, 1800, 30.0
yy, xx = np.mgrid[:d1, :d2]
B0 = 2.0                                           # flat background / pixel


def blob(cy, cx, s=6.0):
    a = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / s).ravel(order="F")
    a[a < 0.02] = 0
    return a


def calcium(rate=0.02):
    spikes = (rng.random(T) < rate) * rng.uniform(1, 3, T)
    k = np.exp(-np.arange(60) / (0.4 * FR))
    return np.convolve(spikes, k)[:T]


def F0(a): return B0 * a.sum() / (a @ a)           # CaImAn-style baseline projection


cells = [blob(10, 10), blob(10, 30), blob(28, 15), blob(28, 45)]
traces = [calcium() for _ in cells]
# component 4 + 5: one cell (cells[3]-like) split into unequal-scale halves
half = (xx.ravel(order="F") < 45)
big = blob(10, 48, 10)
left, right = big * half * 0.25, big * ~half * 2.0
cbig = calcium(0.03)
A = np.column_stack(cells + [left, right]).astype(np.float32)
C = np.vstack(traces + [4 * cbig, 0.5 * cbig]).astype(np.float32)
K = A.shape[1]
noise = 0.01 * rng.standard_normal((K, T))
noise[4:] = 0                                      # merged parts: exact ΔF/F check
Fdff = np.vstack([C[i] / F0(A[:, i]) for i in range(K)]) + noise
YrA = (0.05 * rng.standard_normal((K, T))).astype(np.float32)
SNR = np.array([8, 6, 5, 9, 7, 4], float); RV = np.linspace(.6, .9, K); CNN = np.full(K, .9)

try:
    s = insp.ComponentStore(A, C, np.zeros((d1, d2)), (d1, d2), SNR_comp=SNR, r_values=RV,
                            cnn_preds=CNN, fr=FR, decay_time=0.4,
                            traces={insp.TRACE_DENOISED: C, insp.TRACE_DFF: Fdff.astype(np.float32),
                                    "C + residual (raw)": C + YrA})

    # ── merge refit ──────────────────────────────────────────────────────────
    s.detect_events(FR, 0.4)
    s.run_roi_filter(insp.roi_filter_prefs(insp.load_prefs()))
    Y = A[:, 4:6].astype(float) @ C[4:6].astype(float)
    new = s.merge([4, 5])
    a_new, c_new = s._A[:, new].toarray().ravel(), s.C[new].astype(float)
    err = np.linalg.norm(np.outer(a_new, c_new) - Y) / np.linalg.norm(Y)
    old = np.outer(A[:, 4:6].max(1), C[4:6].mean(0))
    err_old = np.linalg.norm(old - Y) / np.linalg.norm(Y)
    assert err < 1e-5 and err_old > 1.0, (err, err_old)
    assert abs(np.corrcoef(a_new, big)[0, 1] - 1) < 1e-6
    assert abs(np.linalg.norm(a_new) - np.linalg.norm(A[:, 4:6], axis=0).max()) < 1e-3
    dff_true = c_new / F0(a_new)
    dff = s.dff[new].astype(float)
    assert np.abs(dff - dff_true).max() < 1e-4 * dff_true.max(), np.abs(dff - dff_true).max()
    raw_expected = s.traces["C + residual (raw)"][new]       # β-weighted
    assert np.allclose(raw_expected - s.C[new], 0, atol=0.2)
    ok(f"merge = rank-1 refit of Σ aᵢcᵢ: reconstruction error {err:.1e} (old pixel-max / mean "
       f"{err_old:.2f}); ΔF/F on the merged baseline")

    for attr in ("SNR_comp", "r_values", "cnn_preds"):
        assert np.isnan(getattr(s, attr)[new]), attr
    assert s.events.n_events(new) > 0 and s.events.auto_approved[new], \
        "NaN SNR / CNN does not auto-reject the merged row"
    assert "caiman_quality" not in insp.loading._rf().decode_reasons(int(s.qc["fail_reasons"][new]))
    ok("merged SNR / r / CNN unknown (NaN); detection and ROI filter skip them, don't reject")

    w = insp.InspectorWindow(s); w.show(); app.processEvents()
    t = w.table
    hdr = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    for col in ("SNR", "r_val", "CNN"):
        assert t.item(t.row_of(new), hdr.index(col)).text() == "—"
    ok("table shows unknown metrics as —")

    # ── detection undo ───────────────────────────────────────────────────────
    s.approved[:] = False; s.approved[0] = True
    ev_before, appr_before = s.events, s.approved.copy()
    QMessageBox.question = lambda *a, **k: QMessageBox.StandardButton.Yes
    s.detect_events(FR, 0.4, threshold=0.1)
    assert s.events is not ev_before and s.approved.sum() > 1
    s.undo()
    assert s.events is ev_before and np.array_equal(s.approved, appr_before)
    s.redo(); s.undo()
    ok("Detect events is undoable (events and include flags restored)")

    # ── export columns ───────────────────────────────────────────────────────
    s.approved[:] = True
    out = insp.export_event_tables(s, str(root / "x_PeakStats.csv"))
    rows = list(csv.DictReader(open(out[1])))
    comp = [int(r["component"]) for r in rows]
    assert comp == [0, 1, 2, 3, -1], comp
    peaks = list(csv.DictReader(open(out[0])))
    assert {int(r["component"]) for r in peaks} <= {0, 1, 2, 3, -1}
    assert all(int(r["component"]) == s.orig_idx[int(r["roi"])] for r in peaks)
    assert "eval_accepted" not in rows[0]                        # no idx_components here
    ok("PeakStats / ROIStats carry the source component (-1 merged)")
    w.close()

    # ── event kinetics (raw-ΔF/F check) in the inspector ─────────────────────
    from caiman.utils import event_kinetics as ek
    from PyQt6.QtWidgets import QDialog
    s2 = insp.ComponentStore(A, C, np.zeros((d1, d2)), (d1, d2), SNR_comp=SNR, r_values=RV,
                             cnn_preds=CNN, fr=FR, decay_time=0.4,
                             traces={insp.TRACE_DENOISED: C, insp.TRACE_DFF: Fdff.astype(np.float32)})
    s2.detect_events(FR, 0.4)
    w = insp.InspectorWindow(s2); w.show(); app.processEvents()
    t = w.table
    H = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    for col in ("Ev ok", "Ev ok %", "Shape r", "τ fit (s)"):
        assert col in H, H
    ref = ek.summary_table(ek.validate(s2.dff, s2.events, insp.kinetics_prefs()))
    got = [int(t.item(t.row_of(i), H.index("Ev ok")).text()) for i in range(s2.n)]
    assert got == ref["n_valid"].tolist() and sum(got) > 0, (got, ref["n_valid"])
    ok(f"table: Ev ok / Ev ok % / Shape r / τ fit columns ({sum(got)} valid events)")

    calls = []
    orig_validate = ek.validate
    ek.validate = lambda *a, **k: (calls.append(1), orig_validate(*a, **k))[1]
    _ = s2.kinetics
    new2 = s2.merge([0, 1]); _ = s2.kinetics
    assert not calls, "merge updates kinetics row-wise"
    fresh = orig_validate(s2.dff, s2.events, s2.kinetics_params)
    for a_, b_ in zip(s2.kinetics, fresh):
        assert np.array_equal(a_["valid"], b_["valid"])
        assert np.allclose(a_["shape_r"], b_["shape_r"], equal_nan=True)
    before = s2.kinetics
    s2.undo(); s2.redo()
    assert s2.kinetics is not None and not calls
    ek.validate = orig_validate
    ok("merge computes only the new row; undo / redo restore the cached kinetics")

    def strict(dlg):
        dlg._fields["kinetics"]["min_shape_r"].set_value(0.99)
    insp.PreferencesDialog.exec = lambda self: (strict(self), QDialog.DialogCode.Accepted)[1]
    w._full_refresh(); app.processEvents()
    assert w._edit_prefs("kinetics"); app.processEvents()
    assert s2.kinetics_params["min_shape_r"] == 0.99
    H = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    got2 = [int(t.item(t.row_of(i), H.index("Ev ok")).text()) for i in range(s2.n)]
    ref2 = ek.summary_table(ek.validate(s2.dff, s2.events, dict(insp.kinetics_prefs())))
    assert got2 == ref2["n_valid"].tolist() and sum(got2) < sum(got), (got2, got)
    ok("Preferences ▸ Event kinetics re-evaluates and refreshes the table")

    comp = int(np.argmax([len(p) for p in s2.events.peaks]))
    w._on_cell_click(comp, False); app.processEvents()
    fit_pen = insp.PT["fit"]
    fits = [it for it in w.event_view.listDataItems()
            if it.opts["pen"] is not None and pg_color(it.opts["pen"]) == fit_pen]
    n_test = ek.summarize_row(s2.kinetics[comp])["n_testable"]
    assert len(fits) == n_test, (len(fits), n_test)
    assert "kinetics:" in w.event_view.plotItem.titleLabel.text
    ok(f"event panel draws {len(fits)} fitted templates and counts valid events in its title")

    s2.approved[:] = True
    out = insp.export_event_tables(s2, str(root / "k_PeakStats.csv"))
    pk_rows = list(csv.DictReader(open(out[0])))
    roi_rows = list(csv.DictReader(open(out[1])))
    assert {"kin_shape_r", "kin_tau_s", "kin_rise_s", "kin_amp_snr", "kin_valid", "kin_fail"} <= set(pk_rows[0])
    assert sum(int(r["kin_valid"]) for r in pk_rows) == sum(int(r["kin_n_valid"]) for r in roi_rows)
    w.close()
    ok("PeakStats kin_* per event, ROIStats kin_* per ROI (consistent)")

    # ── blinded review from a file ───────────────────────────────────────────
    e = Est()
    e.A = scipy.sparse.csc_matrix(A); e.C = C; e.dims = (d1, d2)
    e.SNR_comp, e.r_values, e.cnn_preds = SNR, RV, CNN
    e.F_dff = Fdff; e.YrA = None; e.S = None; e.Cn = None
    e.idx_components = np.array([0, 1, 2, 3]); e.idx_components_bad = np.array([4, 5])
    c = CNM(); c.estimates = e; c.dims = (d1, d2); c.provenance = []
    c.params = types.SimpleNamespace(data={"fr": FR, "decay_time": 0.4})
    src = root / "sess_results.hdf5"; c.save(src)
    res = ed.detect_events(Fdff, C, FR, 0.4, SNR, CNN)
    manual = ~res.auto_approved                       # someone's curation, opposite of auto
    with h5py.File(src, "a") as f:
        ed.write_stroh_results(f, res, manual)
    st = insp.load_from_hdf5(str(src))
    st.run_roi_filter(insp.roi_filter_prefs(insp.load_prefs()))
    from caiman.utils import roi_filter as rf
    with h5py.File(src, "a") as f:
        rf.write_group(f, st.qc, st.qc_params)
    w = insp.InspectorWindow(insp.load_from_hdf5(str(src))); w.show(); app.processEvents()
    t = w.table
    hdr = lambda: [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    assert "Eval" in hdr() and np.array_equal(w.store.approved, manual), "normal open: file's curation"
    qc_cols = [h for h, *_ in t._QC_COLS]
    had_qc = [h for h in hdr() if h in qc_cols]
    assert had_qc, "file has roi_filter columns before blinding"
    w.store.approved[:] = True
    out = insp.export_event_tables(w.store, str(root / "f_PeakStats.csv"))
    rows = list(csv.DictReader(open(out[1])))
    assert [int(r["eval_accepted"]) for r in rows] == [1, 1, 1, 1, 0, 0]
    assert [int(r["component"]) for r in rows] == list(range(K))
    w.store.approved[:] = manual
    w.start_blinded("R1", [str(src)]); app.processEvents()
    assert np.array_equal(w.store.approved, res.auto_approved), "rater starts from auto_approved"
    assert "Eval" not in hdr() and not [h for h in hdr() if h in qc_cols], hdr()
    assert "Ev ok" not in hdr() and w.event_view.blinded
    assert "accepted" not in w.statusBar().currentMessage()
    shown = []
    QMessageBox.information = lambda *a, **k: shown.append(a[2])
    w._session_info()
    assert "hidden" in shown[-1] and "ROI filter" not in shown[-1], shown[-1]
    ran = []
    w.run_detection = lambda *a, **k: ran.append(1)
    w._detect_events_dialog(); assert not ran, "detection blocked in a blinded review"
    w._blind_save_next(); app.processEvents()
    with h5py.File(src, "r") as f:
        g = f["SLabRandomization/R1"]
        assert np.array_equal(np.sort(g["manually_approved"][()]), np.flatnonzero(res.auto_approved))
    assert "Eval" in hdr() and [h for h in hdr() if h in qc_cols] == had_qc
    assert "Ev ok" in hdr() and not w.event_view.blinded
    w.close()
    ok("export carries eval_accepted (1 = idx_components / census active)")

    # ── kinetics source: raw C + YrA when loaded, warning when circular ──────
    RAW = insp.ComponentStore.RAW_TRACE
    s3 = insp.ComponentStore(A, C, np.zeros((d1, d2)), (d1, d2), SNR_comp=SNR, fr=FR,
                             decay_time=0.4, traces={insp.TRACE_DENOISED: C,
                             insp.TRACE_DFF: Fdff.astype(np.float32), RAW: C + YrA})
    s3.detect_events(FR, 0.4)
    assert s3.kinetics_source == "raw" and not s3.kinetics_circular
    exp = ek.validate(ek.raw_dff(C, YrA, Fdff.astype(np.float32)), s3.events, s3.kinetics_params)
    assert all(np.array_equal(a_["valid"], b_["valid"]) for a_, b_ in zip(s3.kinetics, exp))
    s4 = insp.ComponentStore(A, C, np.zeros((d1, d2)), (d1, d2), SNR_comp=SNR, fr=FR,
                             decay_time=0.4, traces={insp.TRACE_DENOISED: C,
                             insp.TRACE_DFF: (C / 3.0).astype(np.float32)})
    s4.detect_events(FR, 0.4)
    assert s4.kinetics_source == "dff" and s4.kinetics_circular
    w = insp.InspectorWindow(s4); w.show(); app.processEvents()
    assert "⚠ kinetics" in w.statusBar().currentMessage()
    w._session_info(); assert "circular" in shown[-1], shown[-1]
    w.close()
    ok("kinetics run on raw C + YrA when loaded; ΔF/F ≈ C without residual is flagged")

    # Save keeps the residual (merged row: β-weighted), so C + YrA survives
    e2 = Est()
    e2.A = scipy.sparse.csc_matrix(A); e2.C = C; e2.dims = (d1, d2)
    e2.SNR_comp, e2.r_values, e2.cnn_preds = SNR, RV, CNN
    e2.F_dff = Fdff; e2.YrA = YrA; e2.S = None; e2.Cn = None
    e2.idx_components = None; e2.idx_components_bad = None
    c2 = CNM(); c2.estimates = e2; c2.dims = (d1, d2); c2.provenance = []
    c2.params = types.SimpleNamespace(data={"fr": FR, "decay_time": 0.4})
    p2 = root / "yra_results.hdf5"; c2.save(p2)
    st = insp.load_from_hdf5(str(p2))
    assert st.kinetics_source == "raw"
    st.merge([4, 5])
    out2 = root / "yra_curated.hdf5"; st.save(str(out2))
    e3 = load_CNMF(out2).estimates
    assert e3.YrA is not None and e3.YrA.shape == (K - 1, T)
    assert np.allclose(e3.YrA[:K - 2], YrA[:4], atol=1e-5)
    assert np.allclose(e3.YrA[-1], st.traces[RAW][-1] - st.C[-1], atol=1e-5)
    assert insp.load_from_hdf5(str(out2)).kinetics_source == "raw"
    ok("Save writes YrA (merged row β-weighted); curated files keep the raw trace")
    ok("blinded review: starts from auto_approved, hides Eval / QC (table, status, session "
       "info), blocks re-detection; columns return afterwards")
    print("ALL OK")
finally:
    shutil.rmtree(root)
