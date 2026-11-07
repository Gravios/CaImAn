"""Offscreen test: every row loaded/saved; Eval column; hide rejected."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
require('h5py')
import importlib.util, json, os, pickle, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np
import scipy.sparse

root = Path(tempfile.mkdtemp())
os.environ["CAIMAN_INSPECTOR_PREFS"] = str(root / "cfg" / "prefs.json")
stub_caiman()


# fake CNMF object: save()/load_CNMF via pickle — the store only uses
# estimates attributes, params.data, dims and save(path)
class Est: pass
class CNM:
    def save(self, path):
        import h5py
        with h5py.File(path, "w") as f:
            f.attrs["pickle"] = np.void(pickle.dumps(self))
def load_CNMF(path):
    import h5py
    with h5py.File(path, "r") as f:
        return pickle.loads(f.attrs["pickle"].tobytes())
for name in ("caiman.source_extraction", "caiman.source_extraction.cnmf"):
    m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m
mc = types.ModuleType("caiman.source_extraction.cnmf.cnmf"); mc.load_CNMF = load_CNMF
sys.modules["caiman.source_extraction.cnmf.cnmf"] = mc

insp = load_inspector()
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

rng = np.random.default_rng(1)
dims, K, T = (40, 40), 10, 300
A = np.zeros((dims[0] * dims[1], K), np.float32)
for k in range(K):
    img = np.zeros(dims, np.float32); cy, cx = 5 + 8 * (k // 5), 5 + 8 * (k % 5)
    img[cy - 2:cy + 3, cx - 2:cx + 3] = 1.0
    A[:, k] = img.ravel(order="F")
C = rng.random((K, T)).astype(np.float32)
active = np.array([0, 2, 3, 5, 8]); inactive = np.setdiff1d(np.arange(K), active)


def make(census=True, idx=True):
    e = Est()
    e.A = scipy.sparse.csc_matrix(A); e.C = C.copy(); e.dims = dims
    e.SNR_comp = np.arange(K, dtype=float); e.r_values = np.linspace(0, 1, K); e.cnn_preds = None
    e.F_dff = C.copy(); e.YrA = None; e.S = None; e.Cn = None
    e.idx_components = active.copy() if idx else None
    e.idx_components_bad = inactive.copy() if idx else None
    if census:
        e.census = {"n_total": K, "n_active": 5, "n_inactive": 5,
                    "idx_active": active.tolist(), "idx_inactive": inactive.tolist()}
    c = CNM(); c.estimates = e; c.dims = dims; c.provenance = []
    c.params = types.SimpleNamespace(data={"fr": 30.0, "decay_time": 0.4})
    return c


try:
    src = root / "res.hdf5"; make().save(src)
    s = insp.load_from_hdf5(str(src))
    assert s.n == K and s.eval_known and s.accepted.tolist() == np.isin(np.arange(K), active).tolist()
    assert np.array_equal(s.orig_idx, np.arange(K)) and np.allclose(s.C, C)
    ok("census file: all 10 rows loaded (5 active, 5 inactive flagged)")

    # untouched save keeps every row and the split
    out = root / "res_curated.hdf5"; s.save(str(out))
    e = load_CNMF(out).estimates
    assert e.A.shape[1] == K and e.idx_components.tolist() == active.tolist()
    assert e.idx_components_bad.tolist() == inactive.tolist()
    assert e.census["n_active"] == 5 and e.census["idx_inactive"] == inactive.tolist()
    ok("save writes all rows; idx_components / _bad / census unchanged")

    # delete one inactive (4) and merge active 2 with inactive 7 → accepted
    s = insp.load_from_hdf5(str(src))
    s.delete([4]); i2, i7 = 2, 6                      # 7 shifts to 6 after deleting 4
    s.merge([i2, i7])
    exp = np.isin(np.arange(K), active)
    exp = np.delete(exp, 4); m = bool(exp[[i2, i7]].any()); exp = np.append(np.delete(exp, [i2, i7]), m)
    assert s.n == K - 2 and s.accepted.tolist() == exp.tolist() and s.accepted[-1]
    s.undo(); assert s.n == K - 1 and s.accepted.tolist() == np.delete(np.isin(np.arange(K), active), 4).tolist()
    s.merge([i2, i7])
    s.save(str(out)); e = load_CNMF(out).estimates
    assert e.A.shape[1] == K - 2 and e.idx_components.tolist() == np.flatnonzero(exp).tolist()
    assert e.census == {"n_total": K - 2, "n_active": int(exp.sum()), "n_inactive": int((~exp).sum()),
                        "idx_active": np.flatnonzero(exp).tolist(),
                        "idx_inactive": np.flatnonzero(~exp).tolist()}
    assert load_CNMF(out).provenance[-1]["n_accepted"] == int(exp.sum())
    ok("delete / merge (any active → active) / undo track accepted; census rewritten on save")

    # no idx_components → eval unknown, everything accepted, no Eval column
    p2 = root / "plain.hdf5"; make(census=False, idx=False).save(p2)
    s2 = insp.load_from_hdf5(str(p2))
    assert not s2.eval_known and s2.accepted.all()
    s2.save(str(root / "plain_c.hdf5")); e2 = load_CNMF(root / "plain_c.hdf5").estimates
    assert e2.idx_components.tolist() == list(range(K)) and e2.idx_components_bad.tolist() == []
    assert not hasattr(e2, "census")
    ok("file without idx_components: all accepted, no census written")

    # GUI: Eval column, hide toggle, overlay, click
    w = insp.InspectorWindow(insp.load_from_hdf5(str(src))); w.resize(780, 700); w.show(); app.processEvents()
    t = w.table
    hdr = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    assert "Eval" in hdr
    ce = hdr.index("Eval")
    assert [t.item(t.row_of(i), ce).text() for i in range(K)] == \
        [("✓" if i in active else "✗") for i in range(K)], [t.item(t.row_of(i), ce).text() for i in range(K)]
    assert not any(t.isRowHidden(r) for r in range(t.rowCount()))
    assert "5 accepted / 5 rejected" in w.statusBar().currentMessage(), w.statusBar().currentMessage()
    ok("Eval column ✓/✗; all rows visible by default; status shows counts")

    w.act_hide_rejected.trigger(); app.processEvents()
    assert w.store.hide_rejected
    assert [t.isRowHidden(t.row_of(i)) for i in range(K)] == [i not in active for i in range(K)]
    w.cell_view.refresh_rois(); app.processEvents()
    img = np.asarray(w.cell_view._img_all.image)
    def alpha_at(i):
        fp = w.store.footprint(i); pk = np.unravel_index(np.argmax(fp), fp.shape); return float(img[pk][3])
    assert all(alpha_at(i) == 0 for i in inactive) and all(alpha_at(i) > 0 for i in active)
    assert not w.store.is_visible(int(inactive[0])) and w.store.is_visible(int(active[0]))
    for v in (w.corr_view, w.dist_view):
        shown = np.asarray(v._img.image)
        mid = sum(v._img.getLevels()) / 2
        for i in inactive:
            assert np.allclose(shown[i, :], mid) and np.allclose(shown[:, i], mid), type(v).__name__
        pairs = []
        v.pair_clicked.connect(lambda i, j: pairs.append((i, j)))
        class Ev:
            def __init__(self, i, j): self.p = (i, j)
            def button(self): return Qt.MouseButton.LeftButton
            def modifiers(self): return Qt.KeyboardModifier.ControlModifier
            def scenePos(self): return self.p
        v.vb.mapSceneToView = lambda p: types.SimpleNamespace(x=lambda: p[1] + 0.5, y=lambda: p[0] + 0.5)
        v._on_scene_click(Ev(int(inactive[0]), int(active[1])))
        v._on_scene_click(Ev(int(active[0]), int(active[1])))
        assert pairs == [(int(active[0]), int(active[1]))], pairs
    ok("Hide rejected: rows hidden, footprints dropped from the overlay, not clickable; "
       "matrix rows neutral and Ctrl+click ignores them")

    w.close(); app.processEvents()
    prefs = json.loads(Path(os.environ["CAIMAN_INSPECTOR_PREFS"]).read_text())
    assert prefs["ui"]["view"]["hide_rejected"] is True
    w = insp.InspectorWindow(insp.load_from_hdf5(str(src))); w.show(); app.processEvents()
    assert w.act_hide_rejected.isChecked() and w.store.hide_rejected
    out3 = root / "hidden_saved.hdf5"; w.store.save(str(out3))
    assert load_CNMF(out3).estimates.A.shape[1] == K, "hidden rejected ROIs are still saved"
    w.close()
    ok("toggle remembered; hidden rejected ROIs still saved")
    print("ALL OK")
finally:
    shutil.rmtree(root)
