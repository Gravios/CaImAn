"""
caiman/utils/roi_filter.py
==========================
Automatic per-ROI quality and activity filter — **label only**.

Every component is kept (census mode: an ROI that fails is flagged, never
removed).  The filter computes per-ROI metrics, applies thresholds, and
writes the result as a ``roi_filter`` group next to CaImAn's estimates in
``<session>_results.hdf5``.  ``estimates`` and ``idx_components`` are not
touched.  Rows are aligned to the file's estimates (all components).

Criteria (each can be switched off with ``null`` thresholds)
-------------------------------------------------------------
footprint   area of the thresholded footprint vs the expected cell area
            π·gSig_y·gSig_x, solidity (area / convex-hull area) and
            eccentricity of the pixel mask
edge        centroid distance from the FOV border (``edge_px``)
quality     CaImAn's own acceptance rule on SNR_comp / r_values /
            cnn_preds, thresholds defaulting to the run's ``quality`` params
activity    SLab event detection on ΔF/F (no SNR gating): event count and
            rate, peak smoothed ΔF/F, skewness, fraction of frames in events

``passed``   = footprint ∧ edge ∧ quality (∧ active if ``require_activity``)
``active``   = activity criteria met; reported separately because silent
               cells are real cells in census mode
``fail_reasons`` = bitmask of every failed criterion (see ``REASONS``),
               including ``inactive`` even when activity is not required

HDF5 layout (``roi_filter`` group)
----------------------------------
datasets  one (K,) array per field in ``FIELDS`` that could be computed;
          idx_passed / idx_failed / idx_active / idx_inactive (int64);
          manually_approved (int64 rows; written by caiman-inspector)
attrs     version, generator, created, n_components, params (JSON, the
          resolved thresholds), reasons (JSON bit map), summary (JSON counts
          and medians)

Usage
-----
    from caiman.utils.roi_filter import apply_to_results
    summary = apply_to_results("<session>_results.hdf5",
                               cnm.estimates, cnm.params, cfg_dict)
"""
from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np

VERSION   = 1
GROUP     = "roi_filter"
GENERATOR = "caiman.utils.roi_filter"

DEFAULTS: dict = dict(
    enabled=True,
    # ── footprint ────────────────────────────────────────────────────────────
    footprint_thr=0.2,       # mask = pixels ≥ thr × the footprint's max
    min_area_frac=0.5,       # × expected area π·gSig_y·gSig_x   (null = off)
    max_area_frac=10.0,
    min_area_px=None,        # absolute limits; override the fractions
    max_area_px=None,
    min_solidity=0.7,        # area / convex-hull area           (null = off)
    max_eccentricity=0.95,   # 0 = circle, → 1 = line            (null = off)
    # ── edge ─────────────────────────────────────────────────────────────────
    edge_px=10,              # min centroid distance to the border (null = off)
    # ── CaImAn quality (null → value from the run's params.quality) ─────────
    use_caiman_quality=True,
    min_SNR=None, SNR_lowest=None,
    rval_thr=None, rval_lowest=None,
    use_cnn=None, min_cnn_thr=None, cnn_lowest=None,
    # ── activity (ΔF/F) ─────────────────────────────────────────────────────
    event_threshold=0.2,     # SLab detection cutoff (ΔF/F)
    lowpass_hz=None,         # null = SLab-normalised low-pass
    min_events=1,            # active ⇔ all set thresholds are met
    min_event_rate=None,     # events / min
    min_peak_dff=None,       # peak of the smoothed ΔF/F
    min_skewness=None,
    require_activity=False,  # True: inactive ROIs also fail `passed`
)

REASONS: dict = {
    "empty":          1 << 0,
    "area_small":     1 << 1,
    "area_large":     1 << 2,
    "solidity":       1 << 3,
    "eccentricity":   1 << 4,
    "edge":           1 << 5,
    "caiman_quality": 1 << 6,
    "inactive":       1 << 7,
}

# field → dtype, in display / file order
FIELDS: dict = {
    "passed":        np.bool_,
    "active":        np.bool_,
    "fail_reasons":  np.int32,
    "footprint_ok":  np.bool_,
    "edge_ok":       np.bool_,
    "quality_ok":    np.bool_,
    "area_px":       np.int32,
    "solidity":      np.float32,
    "eccentricity":  np.float32,
    "centroid_y":    np.float32,
    "centroid_x":    np.float32,
    "edge_dist_px":  np.float32,
    "n_events":      np.int32,
    "event_rate":    np.float32,
    "peak_dff":      np.float32,
    "skewness":      np.float32,
    "active_frac":   np.float32,
}

_INDEX_SETS = (("idx_passed", "passed", True), ("idx_failed", "passed", False),
               ("idx_active", "active", True), ("idx_inactive", "active", False))


# ── parameters ───────────────────────────────────────────────────────────────

def _pget(cnm_params, group: str, key: str, default=None):
    """Read ``cnm_params.<group>[key]`` from a CNMFParams or nested dict."""
    if cnm_params is None:
        return default
    g = getattr(cnm_params, group, None)
    if g is None and isinstance(cnm_params, dict):
        g = cnm_params.get(group)
    if isinstance(g, dict):
        v = g.get(key, default)
        return default if v is None else v
    return default


def resolve_params(cfg: Optional[dict] = None, cnm_params=None) -> dict:
    """DEFAULTS ← cfg, with nulls filled from the CaImAn run's params:
    quality thresholds, gSig, fr and decay_time."""
    p = dict(DEFAULTS)
    for k, v in (cfg or {}).items():
        if not str(k).startswith("_"):
            p[k] = v
    for k in ("min_SNR", "SNR_lowest", "rval_thr", "rval_lowest",
              "use_cnn", "min_cnn_thr", "cnn_lowest"):
        if p.get(k) is None:
            p[k] = _pget(cnm_params, "quality", k)
    if p.get("gSig") is None:
        g = _pget(cnm_params, "init", "gSig")
        p["gSig"] = list(g) if g is not None else None
    for k in ("fr", "decay_time"):
        if p.get(k) is None:
            p[k] = _pget(cnm_params, "data", k)
    return p


def expected_area(p: dict) -> Optional[float]:
    g = p.get("gSig")
    if g is None:
        return None
    g = list(np.atleast_1d(g).astype(float))
    gy, gx = (g[0], g[1]) if len(g) > 1 else (g[0], g[0])
    return float(np.pi * gy * gx) if gy > 0 and gx > 0 else None


def area_limits(p: dict) -> tuple:
    """(min, max) footprint area in px; None = not applied."""
    exp = expected_area(p)
    lo = p.get("min_area_px")
    hi = p.get("max_area_px")
    if lo is None and exp is not None and p.get("min_area_frac") is not None:
        lo = float(p["min_area_frac"]) * exp
    if hi is None and exp is not None and p.get("max_area_frac") is not None:
        hi = float(p["max_area_frac"]) * exp
    return lo, hi


# ── metrics ──────────────────────────────────────────────────────────────────

def footprint_metrics(A, dims, thr: float = 0.2) -> dict:
    """Area, solidity, eccentricity and weighted centroid of each footprint.

    ``A`` is (d, K) dense or sparse, pixels in Fortran order of ``dims``.
    The mask is the connected (8-neighbour) region of pixels ≥ thr × max
    that contains the peak pixel.
    """
    import scipy.sparse
    from scipy import ndimage
    from scipy.spatial import ConvexHull, QhullError

    A = scipy.sparse.csc_matrix(A)
    d1, d2 = int(dims[0]), int(dims[1])
    K = A.shape[1]
    out = {"area_px": np.zeros(K, np.int32)}
    for f in ("solidity", "eccentricity", "centroid_y", "centroid_x"):
        out[f] = np.full(K, np.nan, np.float32)
    corners = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    st8 = np.ones((3, 3), bool)

    for k in range(K):
        lo, hi = A.indptr[k], A.indptr[k + 1]
        pix, val = A.indices[lo:hi], np.asarray(A.data[lo:hi], float)
        keep = val > 0
        pix, val = pix[keep], val[keep]
        if not pix.size:
            continue
        y, x = np.unravel_index(pix, (d1, d2), order="F")
        m = val >= thr * val.max()
        y, x, val = y[m], x[m], val[m]
        # largest-peak connected region, on the bounding-box crop
        y0, x0 = y.min(), x.min()
        img = np.zeros((y.max() - y0 + 1, x.max() - x0 + 1), bool)
        img[y - y0, x - x0] = True
        lab, _ = ndimage.label(img, structure=st8)
        ipk = int(np.argmax(val))
        sel = lab[y - y0, x - x0] == lab[y[ipk] - y0, x[ipk] - x0]
        y, x, val = y[sel], x[sel], val[sel]

        n = y.size
        out["area_px"][k] = n
        w = val / val.sum()
        out["centroid_y"][k] = float((w * y).sum())
        out["centroid_x"][k] = float((w * x).sum())
        if n >= 3:
            cov = np.cov(np.vstack([y, x]).astype(float))
            ev = np.sort(np.linalg.eigvalsh(cov))
            if ev[1] > 0:
                out["eccentricity"][k] = float(np.sqrt(max(0.0, 1.0 - ev[0] / ev[1])))
        pts = (np.column_stack([y, x])[:, None, :] + corners[None]).reshape(-1, 2)
        try:
            out["solidity"][k] = float(n / ConvexHull(pts).volume)
        except (QhullError, ValueError):
            out["solidity"][k] = 1.0
    return out


def edge_distance(cy, cx, dims) -> np.ndarray:
    """Distance (px) of each centroid from the nearest FOV border."""
    cy, cx = np.asarray(cy, float), np.asarray(cx, float)
    d1, d2 = int(dims[0]), int(dims[1])
    return np.minimum.reduce([cy, cx, (d1 - 1) - cy, (d2 - 1) - cx]).astype(np.float32)


def caiman_quality(SNR_comp, r_values, cnn_preds, p: dict) -> Optional[np.ndarray]:
    """CaImAn's acceptance rule (``select_components_from_metrics``): pass any
    upper threshold and fail no ``*_lowest`` floor.  Metrics that are
    missing or whose threshold is null are skipped; None if none apply."""
    def _arr(a):
        return None if a is None or np.size(a) == 0 else np.asarray(a, float).ravel()
    snr, rv, cnn = _arr(SNR_comp), _arr(r_values), _arr(cnn_preds)
    use_cnn = bool(p.get("use_cnn")) and cnn is not None
    K = next((a.size for a in (snr, rv, cnn) if a is not None), None)
    if K is None:
        return None
    upper, floor = [], []
    for arr, thr, low, on in ((snr, p.get("min_SNR"), p.get("SNR_lowest"), True),
                              (rv,  p.get("rval_thr"), p.get("rval_lowest"), True),
                              (cnn, p.get("min_cnn_thr"), p.get("cnn_lowest"), use_cnn)):
        if arr is None or not on:
            continue
        with np.errstate(invalid="ignore"):
            if thr is not None:
                upper.append(arr >= float(thr))
            if low is not None:
                floor.append(~(arr < float(low)))       # NaN does not hit a floor
    if not upper and not floor:
        return None
    ok = np.logical_or.reduce(upper) if upper else np.ones(K, bool)
    if floor:
        ok &= np.logical_and.reduce(floor)
    return ok


def _event_detection():
    """``caiman.utils.event_detection``; falls back to the sibling file when
    the ``caiman`` package can't be imported (inspector demo / light envs)."""
    import sys
    mod = sys.modules.get("_inspector_event_detection")
    if mod is not None:
        return mod
    try:
        from caiman.utils import event_detection as mod
        return mod
    except Exception:
        import importlib.util
        from pathlib import Path
        name = "_inspector_event_detection"
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("event_detection.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod             # @dataclass resolves its module here
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[name]
            raise
        return mod


def activity_metrics(F_dff, C, fr: float, decay_time: float, p: dict) -> dict:
    """Event-based activity on ΔF/F, without SNR / CNN gating."""
    from scipy.stats import skew
    detect_events = _event_detection().detect_events
    F = np.asarray(F_dff, np.float64)
    res = detect_events(F, C, fr, decay_time, None, None,
                        threshold=float(p.get("event_threshold") or 0.2),
                        lowpass_hz=p.get("lowpass_hz"))
    K = F.shape[0]
    with np.errstate(invalid="ignore"):
        sk = skew(F, axis=1, nan_policy="omit")
    return {
        "n_events":    np.array([res.n_events(i) for i in range(K)], np.int32),
        "event_rate":  res.frequencies.astype(np.float32),
        "peak_dff":    res.filtered.max(axis=1).astype(np.float32),
        "skewness":    np.asarray(sk, np.float32),
        "active_frac": (res.binary > 0).mean(axis=1).astype(np.float32),
    }


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(A, dims, *, SNR_comp=None, r_values=None, cnn_preds=None,
             F_dff=None, C=None, params: Optional[dict] = None) -> dict:
    """Compute every field that the inputs allow.  ``params`` must already be
    resolved (:func:`resolve_params`).  Returns ``{field: (K,) array}``."""
    p = params if params is not None else resolve_params()
    f = footprint_metrics(A, dims, float(p.get("footprint_thr") or 0.2))
    K = f["area_px"].size
    reasons = np.zeros(K, np.int32)

    empty = f["area_px"] == 0
    reasons[empty] |= REASONS["empty"]
    lo, hi = area_limits(p)
    fp_ok = ~empty
    with np.errstate(invalid="ignore"):
        for bad, why in (
                (f["area_px"] < lo if lo is not None else None, "area_small"),
                (f["area_px"] > hi if hi is not None else None, "area_large"),
                (f["solidity"] < p["min_solidity"]
                 if p.get("min_solidity") is not None else None, "solidity"),
                (f["eccentricity"] > p["max_eccentricity"]
                 if p.get("max_eccentricity") is not None else None, "eccentricity")):
            if bad is not None:
                bad = bad & ~empty
                reasons[bad] |= REASONS[why]
                fp_ok &= ~bad
    f["footprint_ok"] = fp_ok

    f["edge_dist_px"] = edge_distance(f["centroid_y"], f["centroid_x"], dims)
    edge_ok = ~empty
    if p.get("edge_px") is not None:
        with np.errstate(invalid="ignore"):
            bad = (f["edge_dist_px"] < float(p["edge_px"])) & ~empty
        reasons[bad] |= REASONS["edge"]
        edge_ok &= ~bad
    f["edge_ok"] = edge_ok

    passed = fp_ok & edge_ok
    if p.get("use_caiman_quality", True):
        q = caiman_quality(SNR_comp, r_values, cnn_preds, p)
        if q is not None:
            f["quality_ok"] = q
            reasons[~q] |= REASONS["caiman_quality"]
            passed &= q

    fr, dt = p.get("fr"), p.get("decay_time")
    if F_dff is not None and C is not None and fr and dt \
            and np.shape(F_dff) == np.shape(C) and np.shape(F_dff)[0] == K:
        f.update(activity_metrics(F_dff, C, float(fr), float(dt), p))
        active = np.ones(K, bool)
        with np.errstate(invalid="ignore"):
            for field, key in (("n_events", "min_events"), ("event_rate", "min_event_rate"),
                               ("peak_dff", "min_peak_dff"), ("skewness", "min_skewness")):
                if p.get(key) is not None:
                    active &= f[field] >= float(p[key])
        f["active"] = active
        reasons[~active] |= REASONS["inactive"]
        if p.get("require_activity"):
            passed &= active

    f["passed"] = passed
    f["fail_reasons"] = reasons
    return {k: np.asarray(f[k], dtype=FIELDS[k]) for k in FIELDS if k in f}


def evaluate_estimates(est, cnm_params=None, cfg: Optional[dict] = None):
    """Run :func:`evaluate` on a CaImAn ``Estimates`` object (all rows).
    Returns ``(fields, resolved_params)``."""
    p = resolve_params(cfg, cnm_params)
    dims = getattr(est, "dims", None)
    if dims is None:
        dims = _pget(cnm_params, "data", "dims")
    fields = evaluate(est.A, dims,
                      SNR_comp=getattr(est, "SNR_comp", None),
                      r_values=getattr(est, "r_values", None),
                      cnn_preds=getattr(est, "cnn_preds", None),
                      F_dff=getattr(est, "F_dff", None),
                      C=getattr(est, "C", None), params=p)
    return fields, p


# ── row bookkeeping (inspector merge / delete) ───────────────────────────────

def take(fields: dict, rows) -> dict:
    rows = np.asarray(list(rows), dtype=np.int64)
    return {k: np.asarray(v)[rows] for k, v in fields.items()}


def append(fields: dict, other: dict) -> dict:
    return {k: np.concatenate([np.asarray(v), np.asarray(other[k], dtype=np.asarray(v).dtype)])
            for k, v in fields.items() if k in other}


def decode_reasons(mask: int) -> list:
    return [name for name, bit in REASONS.items() if int(mask) & bit]


# ── summary / HDF5 ───────────────────────────────────────────────────────────

def summarize(fields: dict) -> dict:
    """Counts per verdict and fail reason; medians of the metrics over all
    ROIs and over passed ROIs."""
    K = int(next(iter(fields.values())).size) if fields else 0
    s: dict = {"n_total": K}
    for flag in ("passed", "active", "footprint_ok", "edge_ok", "quality_ok"):
        if flag in fields:
            s[f"n_{flag}"] = int(np.asarray(fields[flag], bool).sum())
    if "fail_reasons" in fields:
        fr = np.asarray(fields["fail_reasons"])
        s["n_fail"] = {name: int(((fr & bit) != 0).sum()) for name, bit in REASONS.items()}
    passed = np.asarray(fields["passed"], bool) if "passed" in fields else None
    med, med_p = {}, {}
    for k in ("area_px", "solidity", "eccentricity", "edge_dist_px", "n_events",
              "event_rate", "peak_dff", "skewness", "active_frac"):
        if k in fields:
            v = np.asarray(fields[k], float)
            if np.isfinite(v).any():
                med[k] = float(np.nanmedian(v))
            if passed is not None and passed.any() and np.isfinite(v[passed]).any():
                med_p[k] = float(np.nanmedian(v[passed]))
    s["median"], s["median_passed"] = med, med_p
    return s


def _json_safe(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (tuple, list, np.ndarray)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return v


def write_group(parent, fields: dict, params: dict, manually_approved=None,
                group: str = GROUP):
    """(Re)write ``parent[group]`` (an open h5py File/Group)."""
    if group in parent:
        del parent[group]
    g = parent.create_group(group)
    for k, v in fields.items():
        g.create_dataset(k, data=np.asarray(v, dtype=FIELDS.get(k, None)))
    for name, flag, val in _INDEX_SETS:
        if flag in fields:
            g.create_dataset(name, data=np.flatnonzero(
                np.asarray(fields[flag], bool) == val).astype(np.int64))
    if manually_approved is not None:
        g.create_dataset("manually_approved", data=np.flatnonzero(
            np.asarray(manually_approved, bool)).astype(np.int64))
    K = int(next(iter(fields.values())).size) if fields else 0
    g.attrs["version"]      = VERSION
    g.attrs["generator"]    = GENERATOR
    g.attrs["created"]      = time.strftime("%Y-%m-%dT%H:%M:%S")
    g.attrs["n_components"] = K
    g.attrs["params"]       = json.dumps(_json_safe(params))
    g.attrs["reasons"]      = json.dumps(REASONS)
    g.attrs["summary"]      = json.dumps(summarize(fields))
    return g


def read_group(g):
    """Parse an h5py ``roi_filter`` group →
    ``(fields, params, manually_approved_mask_or_None)``."""
    fields = {k: np.asarray(g[k][()], dtype=FIELDS[k]) for k in FIELDS if k in g}
    K = int(g.attrs.get("n_components",
                        next(iter(fields.values())).size if fields else 0))
    try:
        params = json.loads(g.attrs.get("params", "{}"))
    except (TypeError, ValueError):
        params = {}
    appr = None
    if "manually_approved" in g:
        appr = np.zeros(K, bool)
        idx = np.asarray(g["manually_approved"][()], np.int64)
        appr[idx[(idx >= 0) & (idx < K)]] = True
    return fields, params, appr


def apply_to_results(path, est, cnm_params=None, cfg: Optional[dict] = None) -> dict:
    """Evaluate ``est`` and write the ``roi_filter`` group into ``path``
    (an existing results HDF5; never created here).  Returns the summary."""
    import h5py
    fields, p = evaluate_estimates(est, cnm_params, cfg)
    with h5py.File(str(path), "r+") as f:      # "a" would create a stray file
        write_group(f, fields, p)
    return summarize(fields)
