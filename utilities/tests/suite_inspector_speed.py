"""Offscreen test: sparse footprints, cached / incremental derived data and
sparse overlays equal brute-force dense references."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
import importlib.util, os, sys, tempfile, types
from pathlib import Path
import numpy as np

os.environ["CAIMAN_INSPECTOR_PREFS"] = str(Path(tempfile.mkdtemp()) / "p.json")
insp = load_inspector()
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

rng = np.random.default_rng(3)
dims = (48, 64); d1, d2 = dims; K, T = 40, 400
A = np.zeros((d1 * d2, K), np.float32)
yy, xx = np.mgrid[-5:6, -5:6]
for k in range(K):                                   # overlapping blobs
    cy, cx = rng.integers(5, d1 - 6), rng.integers(5, d2 - 6)
    blob = np.exp(-(yy ** 2 + xx ** 2) / rng.uniform(4, 12)) * rng.uniform(0.5, 2)
    blob[blob < 0.05] = 0
    A[np.ravel_multi_index(((yy + cy).ravel(), (xx + cx).ravel()), dims, order="F"), k] = blob.ravel()
A[:, 7] = 0                                          # one empty footprint
C = np.maximum(rng.standard_normal((K, T)).astype(np.float32) - 0.8, 0)
C[12] = C[4] + 0.02 * rng.random(T).astype(np.float32)


def ref_corr(C, active_thr=0.0):
    K, T = C.shape
    mat = np.full((K, K), np.nan, dtype=np.float32)
    if K == 0 or T < 3:
        return mat
    active = C > active_thr
    for i in range(K):
        for j in range(i + 1, K):
            m = active[i] | active[j]
            if int(m.sum()) < 3:
                continue
            a = C[i][m].astype(np.float64); b = C[j][m].astype(np.float64)
            a -= a.mean(); b -= b.mean()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                continue
            mat[i, j] = mat[j, i] = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
    return mat


def dense_fp(Ad, i): return Ad[:, i].reshape(dims, order="F")


def ref_overlay(Ad, items):                          # [(i, rgb, alpha)]
    out = np.zeros((d1, d2, 4), np.float32)
    for i, rgb, a in items:
        fp = dense_fp(Ad, i); pk = fp.max()
        if pk < 1e-9:
            continue
        am = fp / pk * a
        for ch, v in enumerate((rgb[0] * am, rgb[1] * am, rgb[2] * am, am)):
            out[..., ch] = np.maximum(out[..., ch], v)
    return np.clip(out, 0, 1)


def ref_centroids(Ad):
    p = np.arange(d1 * d2); out = []
    for i in range(Ad.shape[1]):
        t = Ad[:, i].sum()
        out.append(((Ad[:, i] * (p % d1)).sum() / t, (Ad[:, i] * (p // d1)).sum() / t)
                   if t > 1e-9 else (d1 / 2, d2 / 2))
    return np.array(out)


def ref_click(Ad, s, row, col):
    best, bv = None, 0.0
    for i in range(Ad.shape[1]):
        if not s.is_visible(i):
            continue
        fp = dense_fp(Ad, i); pk = fp.max()
        if pk < 1e-9:
            continue
        v = fp[row, col]
        if v >= pk * 0.1 and v > bv:
            best, bv = i, v
    return best


def check(s, Ad, Cd, tag):
    assert s.n == Ad.shape[1]
    assert np.array_equal(s._A.toarray(), Ad), tag
    for i in range(s.n):
        assert np.array_equal(s.footprint(i), dense_fp(Ad, i))
    full = ref_corr(Cd)                               # the original pair loop
    assert np.array_equal(np.isnan(s.corr), np.isnan(full)), tag
    assert np.allclose(s.corr, full, atol=1e-6, equal_nan=True), tag
    assert np.allclose(s.centroids, ref_centroids(Ad), atol=1e-3), tag
    c = ref_centroids(Ad); dd = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)); np.fill_diagonal(dd, np.nan)
    assert np.allclose(s.dist, dd, atol=1e-3, equal_nan=True), tag
    assert np.allclose(s.peaks, Ad.max(axis=0)), tag


try:
    s = insp.ComponentStore(A, C, np.zeros(dims), dims)
    assert s._A.format == "csc" and s._A.dtype == np.float32 and s._A.nnz == np.count_nonzero(A)
    check(s, A, C, "initial")
    ok("sparse footprints, corr, centroids, distances, peaks equal dense references")

    # overlays
    s.approved[:] = True; s.approved[[3, 9]] = False
    s.qc = {"passed": s.approved.copy()}; s.hidden[5] = True
    cyan, exc = insp._hex_rgb(insp.CYAN), insp._hex_rgb(insp.EXCLUDED)
    a_all, a_ex, a_sel = (insp.OVERLAY_ALPHA[k] for k in ("all", "excluded", "selected"))
    items = [(i, exc if not s.approved[i] else cyan, a_ex if not s.approved[i] else a_all)
             for i in range(K) if i != 5]
    assert np.allclose(insp._build_roi_overlay(s), ref_overlay(A, items), atol=1e-6)
    items_h = [it for it in items if s.approved[it[0]]]
    assert np.allclose(insp._build_roi_overlay(s, hide_excluded=True), ref_overlay(A, items_h), atol=1e-6)
    sel, pair = [2, 11, 5, 30], (11, 20)
    pal = insp.PALETTE_HEX
    items = [(2, insp._hex_rgb(pal[0]), a_sel), (11, insp._hex_rgb(insp.CORR_A), a_sel),
             (30, insp._hex_rgb(pal[3]), a_sel), (20, insp._hex_rgb(insp.CORR_B), a_sel)]
    assert np.allclose(insp._build_overlay(s, sel, pair), ref_overlay(A, items), atol=1e-6)
    ok("all-ROIs / selection overlays (hidden, excluded, pair colours) equal dense compositing")

    # click hit-test at every pixel
    s.accepted[:] = True; s.accepted[6] = False; s.hide_rejected = True
    w = insp.InspectorWindow(s); w.show(); app.processEvents()
    hits = []
    w.cell_view.component_clicked.connect(lambda i, c: hits.append(i))
    class Ev:
        def __init__(self, r, c): self.r, self.c = r, c
        def double(self): return False
        def button(self): return Qt.MouseButton.LeftButton
        def modifiers(self): return Qt.KeyboardModifier.NoModifier
        def scenePos(self): return (self.r, self.c)
    w.cell_view.vb.mapSceneToView = lambda p: types.SimpleNamespace(x=lambda: p[1], y=lambda: p[0])
    n_bad = 0
    for r in range(d1):
        for c in range(d2):
            hits.clear(); w.cell_view._on_scene_click(Ev(r, c))
            exp = ref_click(A, s, r, c)
            n_bad += (hits[0] if hits else None) != exp
    assert n_bad == 0, n_bad
    ok(f"click hit-test equals brute force at all {d1 * d2} pixels (hidden / rejected skipped)")

    # merge / delete keep caches incremental and exact; undo restores them
    Ad, Cd = A.copy(), C.copy()
    _ = s.corr, s.centroids, s.peaks
    calls = []
    orig = insp._active_pearson_corr
    insp.store._active_pearson_corr = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    s.delete([0, 17])
    keep = [j for j in range(K) if j not in (0, 17)]; Ad, Cd = Ad[:, keep], Cd[keep]
    check_c = s.corr                                # no full recompute
    m = [3, 10, 25]
    s.merge(m)
    keep = [j for j in range(Ad.shape[1]) if j not in m]
    Ad = np.column_stack([Ad[:, keep], Ad[:, m].max(axis=1)]); Cd = np.vstack([Cd[keep], Cd[m].mean(0)])
    _ = s.corr
    assert not calls, "corr updated incrementally (no K×K recompute)"
    insp.store._active_pearson_corr = orig
    check(s, Ad, Cd, "after delete + merge")
    snap_corr = s.corr
    s.undo(); s.undo()
    assert s._derived.get("corr") is not None, "undo restores the cached matrix"
    check(s, A, C, "after undo ×2")
    s.redo()
    check(s, A[:, [j for j in range(K) if j not in (0, 17)]], C[[j for j in range(K) if j not in (0, 17)]], "redo")
    ok("delete / merge update corr, centroids, peaks incrementally; undo / redo restore them exactly")

    # undo snapshot shares arrays
    snap = s._capture()
    assert snap["A"] is s._A and snap["traces"][insp.TRACE_DENOISED] is s.C
    assert snap["approved"] is not s.approved and snap["hidden"] is not s.hidden
    ok("undo snapshots share footprints / traces; copy only the in-place flags")

    # matrix highlight changes don't recompute
    renders = []
    for v in (w.corr_view, w.dist_view):
        o = v._matrix; v._matrix = (lambda o: lambda: (renders.append(1), o())[1])(o)
    w._full_refresh(); app.processEvents(); renders.clear()
    w._on_cell_click(1, False); w._on_cell_click(2, True); app.processEvents()
    w._on_cell_click(4, False); app.processEvents()
    assert not renders, renders
    assert w.corr_view._highlight is None and not w.corr_view._hl_items
    ok("selecting / clearing a pair redraws only the highlight (matrices not recomputed)")

    # distance matrix, log scale: levels span log10 of the distances
    dv = w.dist_view
    dv._log = True; dv._render()
    lo, hi = dv._img.getLevels()
    dd = s.dist[~np.isnan(s.dist)]
    dd = np.log10(dd[dd > 0])
    assert abs(lo - dd.min()) < 1e-4 and abs(hi - np.percentile(dd, 99)) < 1e-4, (lo, hi, dd.min())
    w.close()
    ok("log distance levels are log10(distance), not log10(log10(distance))")

    # blocked correlation equals one block and the pair loop
    Cb = np.maximum(rng.standard_normal((70, 300)).astype(np.float32) - 0.5, 0)
    one = insp.store._masked_pearson(Cb, Cb)
    many = insp.store._masked_pearson(Cb, Cb, block_bytes=1)       # 16-row blocks
    assert np.array_equal(np.isnan(one), np.isnan(many)) and np.allclose(one, many, equal_nan=True, atol=1e-6)
    assert np.allclose(many, ref_corr(Cb), equal_nan=True, atol=1e-6)
    ok("blocked correlation (16-row blocks) equals one block and the pair loop")
    print("ALL OK")
finally:
    pass
