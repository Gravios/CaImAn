"""Unit tests for caiman/utils/roi_filter.py."""
from _harness import REPO, ok, require, stub_caiman  # noqa: F401
require('numpy', 'scipy', 'h5py')
stub_caiman()
import importlib.util, json, sys, tempfile, types
from pathlib import Path
import numpy as np
import scipy.sparse
import h5py

from caiman.utils import roi_filter as rf

d1, d2 = 60, 80                     # non-square FOV
yy, xx = np.mgrid[:d1, :d2]


def disk(cy, cx, r):
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))


def col(img):
    return img.ravel(order="F")


gSig = [3, 3]                        # expected area π·9 ≈ 28 px
comps = {
    "good":    disk(30, 40, 3),
    "small":   disk(30, 15, 0.6),
    "large":   disk(25, 60, 12),
    "line":    np.exp(-((yy - 45) ** 2) / 2) * ((xx > 20) & (xx < 60)),
    "ring":    np.exp(-((np.hypot(yy - 15, xx - 40) - 6) ** 2) / 1.5),
    "edge":    disk(3, 40, 3),
    "empty":   np.zeros((d1, d2)),
    "twoblob": disk(50, 10, 2.5) + 0.9 * disk(50, 30, 2.5),   # mask keeps the peak blob only
}
names = list(comps)
A = np.column_stack([col(comps[n]) for n in names]).astype(np.float32)
A[A < 1e-3] = 0
K = len(names)

# ── footprint metrics
f = rf.footprint_metrics(scipy.sparse.csc_matrix(A), (d1, d2), 0.2)
g = names.index("good")
assert abs(f["centroid_y"][g] - 30) < 0.5 and abs(f["centroid_x"][g] - 40) < 0.5, (f["centroid_y"][g], f["centroid_x"][g])
assert f["eccentricity"][g] < 0.3 and f["solidity"][g] > 0.85, (f["eccentricity"][g], f["solidity"][g])
assert f["eccentricity"][names.index("line")] > 0.95
assert f["solidity"][names.index("ring")] < 0.7, f["solidity"][names.index("ring")]
assert f["area_px"][names.index("empty")] == 0 and np.isnan(f["centroid_y"][names.index("empty")])
tb = names.index("twoblob")
assert abs(f["centroid_x"][tb] - 10) < 1, f"connected-region mask: {f['centroid_x'][tb]}"
dense = rf.footprint_metrics(A, (d1, d2), 0.2)
assert all(np.allclose(dense[k], f[k], equal_nan=True) for k in f)
ok("footprint metrics: centroid (Fortran order, non-square), eccentricity, solidity, peak region, dense==sparse")

# ── full evaluation, footprint + edge
p = rf.resolve_params({"gSig": gSig, "min_SNR": None, "rval_thr": None, "SNR_lowest": None,
                       "rval_lowest": None, "use_cnn": False})
fld = rf.evaluate(A, (d1, d2), params=p)
R = {n: rf.decode_reasons(fld["fail_reasons"][i]) for i, n in enumerate(names)}
assert fld["passed"][g] and R["good"] == [], R["good"]
assert "area_small" in R["small"] and "area_large" in R["large"]
assert "eccentricity" in R["line"] and "solidity" in R["ring"]
assert R["edge"] == ["edge"] and not fld["edge_ok"][names.index("edge")]
assert R["empty"] == ["empty"] and not fld["passed"][names.index("empty")]
assert "quality_ok" not in fld and "active" not in fld, "absent inputs → absent fields"
ok(f"criteria fire one per test footprint: { {n: R[n] for n in names} }")

# ── CaImAn quality rule
q = rf.caiman_quality(np.array([3, 1, 1, 0.2, np.nan]), np.array([0.1, 0.9, 0.1, 0.95, 0.1]),
                      None, dict(min_SNR=2.5, SNR_lowest=0.5, rval_thr=0.8, rval_lowest=-1))
assert q.tolist() == [True, True, False, False, False], q
qc = rf.caiman_quality(np.array([1.0, 1.0]), None, np.array([0.95, 0.05]),
                       dict(min_SNR=2.5, use_cnn=True, min_cnn_thr=0.9, cnn_lowest=0.1))
assert qc.tolist() == [True, False]
assert rf.caiman_quality(np.array([1.0]), None, np.array([0.95]),
                         dict(min_SNR=2.5, use_cnn=False, min_cnn_thr=0.9)).tolist() == [False]
ok("CaImAn acceptance rule: any upper threshold, no floor, CNN only with use_cnn")

# ── activity
fr, T = 30.0, 1800
rng = np.random.default_rng(0)
F = 0.01 * rng.standard_normal((K, T))
C = np.abs(0.001 * rng.standard_normal((K, T)))
kern = np.exp(-np.arange(60) / 15.0)
for t0 in (200, 700, 1300):
    F[g, t0:t0 + 60] += 0.6 * kern
    C[g, t0:t0 + 60] += 0.6 * kern
p2 = rf.resolve_params({"gSig": gSig, "fr": fr, "decay_time": 0.4, "min_SNR": None,
                        "rval_thr": None, "use_cnn": False, "SNR_lowest": None, "rval_lowest": None})
fld2 = rf.evaluate(A, (d1, d2), F_dff=F, C=C, params=p2)
assert fld2["n_events"][g] == 3 and fld2["active"][g], fld2["n_events"]
assert abs(fld2["event_rate"][g] - 3 / (T / fr / 60)) < 1e-4
assert fld2["peak_dff"][g] > 0.3 and fld2["skewness"][g] > 3 and fld2["active_frac"][g] > 0
quiet = names.index("small")
assert fld2["n_events"][quiet] == 0 and not fld2["active"][quiet]
assert "inactive" in rf.decode_reasons(fld2["fail_reasons"][quiet])
assert fld2["passed"][g]
p3 = dict(p2, require_activity=True)
fld3 = rf.evaluate(A, (d1, d2), F_dff=F, C=C, params=p3)
assert fld3["passed"].sum() == 1 and fld3["passed"][g]
p4 = dict(p2, min_events=1, min_skewness=50.0)
assert not rf.evaluate(A, (d1, d2), F_dff=F, C=C, params=p4)["active"][g], "skewness threshold applied"
ok("activity: events/rate/peak/skew/active_frac; inactive reason; require_activity")

# ── resolve_params from CaImAn-style params
cnm_params = types.SimpleNamespace(quality={"min_SNR": 1.5, "rval_thr": 0.6, "use_cnn": True,
                                            "min_cnn_thr": 0.6, "cnn_lowest": 0.1,
                                            "SNR_lowest": 0.5, "rval_lowest": -1},
                                   init={"gSig": (5, 5)}, data={"fr": 28.67, "decay_time": 0.4})
rp = rf.resolve_params({"min_SNR": 3.0, "_comment": "x"}, cnm_params)
assert rp["min_SNR"] == 3.0 and rp["rval_thr"] == 0.6 and rp["gSig"] == [5, 5]
assert rp["fr"] == 28.67 and "_comment" not in rp
ok("params: JSON overrides win; nulls filled from the run's quality / init / data")

# ── HDF5 round trip + apply_to_results on an Estimates-like object
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "S_results.hdf5"
    with h5py.File(path, "w") as h:
        h.create_group("estimates").create_dataset("C", data=C)
    est = types.SimpleNamespace(A=scipy.sparse.csc_matrix(A), dims=(d1, d2), C=C, F_dff=F,
                                SNR_comp=np.r_[[3.0] * (K - 1), 0.1], r_values=None, cnn_preds=None)
    params = types.SimpleNamespace(quality={"min_SNR": 2.5, "SNR_lowest": 0.5}, init={"gSig": gSig},
                                   data={"fr": fr, "decay_time": 0.4})
    s = rf.apply_to_results(path, est, params, {"enabled": True})
    assert s["n_total"] == K and s["n_passed"] >= 1 and "n_active" in s
    with h5py.File(path, "r") as h:
        assert "estimates" in h, "existing content kept"
        grp = h["roi_filter"]
        fields, prm, appr = rf.read_group(grp)
        assert appr is None and prm["min_SNR"] == 2.5 and prm["gSig"] == gSig
        assert set(fields) >= {"passed", "active", "quality_ok", "area_px", "event_rate"}
        assert grp["idx_passed"][()].tolist() == np.flatnonzero(fields["passed"]).tolist()
        assert sorted(grp["idx_passed"][()].tolist() + grp["idx_failed"][()].tolist()) == list(range(K))
        assert not fields["quality_ok"][K - 1], "SNR 0.1 below SNR_lowest"
        summ = json.loads(grp.attrs["summary"]); assert summ["n_fail"]["edge"] == 2   # "edge" + "twoblob" (9 px from the bottom)
        assert json.loads(grp.attrs["reasons"])["inactive"] == 128
    with h5py.File(path, "a") as h:                       # rewrite with approvals
        appr_in = np.zeros(K, bool); appr_in[[0, 2]] = True
        rf.write_group(h, rf.take(fields, range(K)), prm, manually_approved=appr_in)
    with h5py.File(path, "r") as h:
        _, _, appr = rf.read_group(h["roi_filter"])
        assert appr.tolist() == appr_in.tolist()
ok("HDF5: group written beside existing data, index sets, attrs, manual approvals round-trip")

# ── take / append (inspector merge/delete)
sub = rf.take(fields, [3, 1])
assert sub["area_px"].tolist() == [fields["area_px"][3], fields["area_px"][1]]
both = rf.append(sub, rf.take(fields, [0]))
assert both["passed"].dtype == np.bool_ and both["area_px"].size == 3
ok("take / append keep dtypes")
print("ALL OK")
