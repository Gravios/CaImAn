#!/usr/bin/env python3
"""
group-compare
=============
Compare calcium activity between groups of sessions: how often cells fire,
the rhythm of population activity, and how strongly cells are coupled.

    group-compare --groups strohA-zb_groups.csv --root strohA-zb/strohA-zb-000000 \\
                  --compare alz alz+MS125 --out zb-compare

--groups is a CSV with at least the columns ``session`` and ``group``
(other columns are ignored; e.g. the strohA-zb_groups.csv list).  Sessions
whose group is not in --compare are still measured and shown, but not
tested.  ``kind == zstack`` rows (or sessions with no results) are skipped.

Which cells (per session)
-------------------------
The session's results file is found below --root
(``<session>/<session>-C<NN>-fc<N>/…_results.hdf5``):

* ``…_results_curated.hdf5`` (saved by the inspector) if present: CNMF-
  accepted (idx_components) AND inspector-approved cells
  (stroh_results/manually_approved), events from its stroh_results;
* else ``…_results.hdf5``: CNMF-accepted cells; events from its
  stroh_results if present, otherwise detected here with the same SLab
  detector (caiman.utils.event_detection, defaults) — in memory, the file
  is never written.

The sessions table says which applied (roi_source, event_source).

Metrics
-------
Frequency
  rate_per_min       events / minute per cell (0 for silent cells)
  active_frac        fraction of cells with >= 1 event
  pop_peak_slow_hz   peak of the population spectrum in 0.1–1 Hz, above the
                     aperiodic (1/f) background (power law fitted in log-log,
                     0.05 Hz – min(Nyquist, 5 Hz))
  pop_peak_slow_db   how far that peak stands above the background (dB; near
                     0 = no rhythm, the frequency is then meaningless)
  aperiodic_slope    slope of that power-law fit
  relpow_<band>      power in a band / total power (0.01 Hz – Nyquist);
                     bands as caiman.utils.oscillation.BANDS, cut at Nyquist
  The population signal is the mean of the cells' z-scored ΔF/F; spectra
  are Thomson multitaper (NW = 4), the pipeline's own estimator.

Connectivity
  mean_r             mean pairwise Pearson r of ΔF/F
  sig_pair_frac      fraction of pairs correlated above chance: a pair's r
                     against its own --n-surrogates circular shifts of the
                     traces (shift 10–90 % of the session), one-sided
                     empirical p <= --pair-alpha (0.01: r above all 100
                     shifts).  Independent cells give ~0.01: the chance
                     level, the same in every session and group
  degree / degree_frac  significant partners per cell (absolute / / (n-1))
  clustering         mean local clustering coefficient of that network
  sttc               mean spike-time tiling coefficient over pairs of cells
                     with events (Cutts & Eglen 2014; window ±--sttc-dt s):
                     co-firing that does not grow with firing rate

Statistics
----------
Sessions of one day are not independent (same animal, same preparation), and
cells of one session even less, so the day is the unit:

* day means (mean over that day's sessions) per group;
* the difference between the two groups' means with a 95 % hierarchical
  bootstrap CI (resample days, then sessions in a day, then cells in a
  session; --n-boot);
* with statsmodels installed, a linear mixed model per metric (session
  values ~ group, random intercept per day): estimate and p.

With few days per group (strohA-zb: 3 vs 2) the CI is wide and the p-values
are not trustworthy: read them as descriptive.

Output (--out)
--------------
  cells.csv  sessions.csv  days.csv  stats.csv  excluded.csv
  fig_<metric>.png   one strip plot per metric (sessions, coloured by day)
  report.html        tables + figures, self-contained
"""
from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import io
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

SESSION_METRICS = ("rate_per_min", "active_frac", "pop_peak_slow_hz", "pop_peak_slow_db",
                   "aperiodic_slope", "relpow_infra-slow",
                   "relpow_slow", "relpow_delta", "mean_r", "sig_pair_frac",
                   "degree_frac", "clustering", "sttc")
CELL_METRICS = ("rate_per_min", "mean_r", "degree_frac", "sttc")
_DATE_RE = re.compile(r"^[^-]+-[^-]+-[^-]+-(\d{8})-")
_trapz = getattr(np, "trapezoid", None) or np.trapz     # NumPy 2 renamed it


# ── caiman modules without importing the caiman package ─────────────────────

def _module(name: str):
    """caiman.utils.<name>, or the file itself when caiman can't be imported."""
    key = f"_gc_{name}"
    if key in sys.modules:
        return sys.modules[key]
    try:
        mod = importlib.import_module(f"caiman.utils.{name}")
    except Exception:
        spec = importlib.util.spec_from_file_location(key, _REPO / "caiman" / "utils" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod                  # dataclasses look themselves up here
        spec.loader.exec_module(mod)
    sys.modules[key] = mod
    return mod


# ── loading one session ──────────────────────────────────────────────────────

def find_results(root: Path, session: str, channel: str = "00") -> Path | None:
    """The session's results file: *_results_curated.hdf5 before *_results.hdf5."""
    sdir = next((p for p in root.rglob(session) if p.is_dir()), None)
    if sdir is None:
        return None
    files = sorted(sdir.rglob("*_results*.hdf5"))
    files = [f for f in files if f"-C{channel}-" in f.name or f"-C{channel}" in f.parent.name]
    for suffix in ("_results_curated.hdf5", "_results.hdf5"):
        hit = [f for f in files if f.name.endswith(suffix)]
        if hit:
            return hit[0]
    return None


def _get(f, path):
    if path not in f or not hasattr(f[path], "shape"):
        return None
    v = f[path][()]
    if isinstance(v, (bytes, str)) and v in (b"NoneType", "NoneType"):
        return None
    return v


def load_session(path: Path) -> dict:
    """ΔF/F, C, selected cells and events of one results file."""
    import h5py
    ed = _module("event_detection")
    with h5py.File(path, "r") as f:
        F = _get(f, "estimates/F_dff")
        C = _get(f, "estimates/C")
        if F is None or C is None:
            raise ValueError("no estimates/F_dff or estimates/C")
        F, C = np.asarray(F, float), np.asarray(C, float)
        K, T = F.shape
        fr = _get(f, "params/data/fr")
        decay = _get(f, "params/data/decay_time")
        idx = _get(f, "estimates/idx_components")
        accepted = np.zeros(K, bool)
        if idx is None:
            accepted[:] = True
        else:
            idx = np.asarray(idx, int).ravel()
            accepted[idx[(idx >= 0) & (idx < K)]] = True
        snr, cnn = _get(f, "estimates/SNR_comp"), _get(f, "estimates/cnn_preds")
        res, approved = (ed.read_stroh_results(f["stroh_results"]) if "stroh_results" in f
                         else (None, None))
    curated = path.name.endswith("_results_curated.hdf5")
    if fr is None:
        raise ValueError("no params/data/fr")
    fr = float(fr)
    if res is None:
        if decay is None:
            raise ValueError("no stroh_results and no params/data/decay_time")
        res = ed.detect_events(F, C, fr, float(decay), snr, cnn)
        event_source = "detected here"
    else:
        event_source = "stroh_results (curated)" if curated else "stroh_results"
    if curated and approved is not None:
        sel = accepted & approved
        roi_source = "curated: accepted & approved"
    else:
        sel = accepted
        roi_source = "CNMF accepted"
    rows = np.flatnonzero(sel)
    peaks = [np.asarray(res.peaks[r])[:, 0].astype(int) if len(res.peaks[r]) else
             np.zeros(0, int) for r in rows]
    return dict(path=path, F=F[rows], fr=fr, T=T, rows=rows, peaks=peaks,
                n_total=K, roi_source=roi_source, event_source=event_source)


# ── metrics ──────────────────────────────────────────────────────────────────

def _zscore(F):
    sd = F.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    return (F - F.mean(axis=1, keepdims=True)) / sd


def spectrum_metrics(F: np.ndarray, fr: float) -> dict:
    """Population spectrum: peak in the slow band and relative band power."""
    osc = _module("oscillation")
    pop = _zscore(F).mean(axis=0)
    f, p = osc._multitaper_psd(pop, fr, NW=4.0, fmax=fr / 2, adaptive=True, xp=np)
    out = {}
    tot = (f >= 0.01) & (f <= fr / 2)
    total = _trapz(p[tot], f[tot]) if tot.sum() > 1 else np.nan
    for band, (lo, hi) in osc.BANDS.items():
        m = (f >= lo) & (f < min(hi, fr / 2))
        out[f"relpow_{band}"] = (float(_trapz(p[m], f[m]) / total)
                                 if m.sum() > 1 and total > 0 else np.nan)
    # rhythm above the aperiodic (1/f) background: a power law fitted in
    # log-log over 0.05 Hz – min(Nyquist, 5 Hz); the peak of the residual in
    # 0.1–1 Hz and its height (dB above the fit).  The raw spectrum's
    # maximum would sit at the band's lower edge whenever the rhythm is weak.
    fit = (f >= 0.05) & (f <= min(fr / 2, 5.0)) & (p > 0)
    m = (f >= 0.1) & (f <= 1.0) & (p > 0)
    if fit.sum() > 3 and m.any():
        slope, icpt = np.polyfit(np.log10(f[fit]), np.log10(p[fit]), 1)
        resid = np.log10(p[m]) - (icpt + slope * np.log10(f[m]))
        k = int(np.argmax(resid))
        out["pop_peak_slow_hz"] = float(f[m][k])
        out["pop_peak_slow_db"] = float(10 * resid[k])
        out["aperiodic_slope"] = float(slope)
    else:
        out["pop_peak_slow_hz"] = out["pop_peak_slow_db"] = out["aperiodic_slope"] = np.nan
    out["_psd"] = (f, p)
    return out


def correlation_metrics(F: np.ndarray, n_surr: int = 100, seed: int = 0,
                        alpha: float = 0.01) -> dict:
    """Pairwise r, significance against circular shifts, network measures."""
    K, T = F.shape
    if K < 2:
        nan = np.full(K, np.nan)
        return dict(r=np.full((K, K), np.nan), sig=np.zeros((K, K), bool),
                    cell_mean_r=nan, degree=np.zeros(K), mean_r=np.nan,
                    sig_pair_frac=np.nan, clustering=np.nan)
    Z = _zscore(F)
    r = Z @ Z.T / T
    iu = np.triu_indices(K, 1)
    exceed = np.zeros(iu[0].size)
    rng = np.random.default_rng(seed)
    lo, hi = max(1, T // 10), max(2, 9 * T // 10)
    idx = np.arange(T)
    for _ in range(n_surr):
        shifts = rng.integers(lo, hi, size=K)
        Zs = Z[np.arange(K)[:, None], (idx[None, :] - shifts[:, None]) % T]
        rs = Zs @ Z.T / T                      # shifted i vs original j
        # one shift per pair (i < j: i shifted).  Averaging with "j shifted vs
        # i" would mix two surrogates, halve the null's variance and make
        # chance correlations look significant.
        exceed += rs[iu] >= r[iu]
    # per pair, empirical one-sided p against its own shifts; a pair counts
    # at p <= alpha, so independent cells give ~alpha of pairs (the known
    # chance level).  (Parametric or pooled nulls were too liberal for
    # sparse event traces: a few chance coincidences give heavy tails.)
    p = (1 + exceed) / (1 + n_surr)
    disc = (p <= alpha) & (r[iu] > 0)
    sig = np.zeros((K, K), bool)
    sig[iu] = disc
    sig |= sig.T
    deg = sig.sum(1).astype(float)
    # local clustering: triangles / possible
    A = sig.astype(float)
    tri = np.einsum("ij,jk,ki->i", A, A, A) / 2
    poss = deg * (deg - 1) / 2
    cc = np.where(poss > 0, tri / np.where(poss > 0, poss, 1), np.nan)
    rr = r.copy()
    np.fill_diagonal(rr, np.nan)
    return dict(r=r, sig=sig, cell_mean_r=np.nanmean(rr, axis=1), degree=deg,
                mean_r=float(r[iu].mean()), sig_pair_frac=float(disc.mean()),
                clustering=float(np.nanmean(cc)) if np.isfinite(cc).any() else np.nan)


def sttc_matrix(peaks: list, T: int, fr: float, dt_s: float = 0.5) -> np.ndarray:
    """Spike-time tiling coefficient for every pair (NaN where a cell has no
    events).  Cutts & Eglen (2014), J Neurosci 34:14288."""
    K = len(peaks)
    w = max(1, int(round(dt_s * fr)))
    E = np.zeros((K, T), np.float32)
    for i, pk in enumerate(peaks):
        pk = pk[(pk >= 0) & (pk < T)]
        E[i, pk] = 1
    n = E.sum(1)
    # cover: within ±w frames of an event
    cs = np.concatenate([np.zeros((K, 1), np.float32), np.cumsum(E, axis=1)], axis=1)
    lo = np.clip(np.arange(T) - w, 0, T)
    hi = np.clip(np.arange(T) + w + 1, 0, T)
    cover = ((cs[:, hi] - cs[:, lo]) > 0).astype(np.float32)
    Tfrac = cover.mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = (E @ cover.T) / n[:, None]            # P[a, b]: a's events near b's
        Tb = Tfrac[None, :]
        Ta = Tfrac[:, None]
        Pab, Pba = P, P.T
        s = 0.5 * ((Pab - Tb) / (1 - Pab * Tb) + (Pba - Ta) / (1 - Pba * Ta))
    bad = (n[:, None] == 0) | (n[None, :] == 0)
    s[bad] = np.nan
    np.fill_diagonal(s, np.nan)
    return s


def session_metrics(d: dict, n_surr: int, sttc_dt: float, seed: int,
                    alpha: float = 0.01) -> tuple:
    """(session row, list of cell rows, psd)"""
    F, fr, T = d["F"], d["fr"], d["T"]
    K = F.shape[0]
    minutes = T / fr / 60
    n_ev = np.array([len(p) for p in d["peaks"]], float)
    rate = n_ev / minutes
    spec = spectrum_metrics(F, fr) if K else {}
    cor = correlation_metrics(F, n_surr=n_surr, seed=seed, alpha=alpha)
    st = sttc_matrix(d["peaks"], T, fr, sttc_dt) if K else np.zeros((0, 0))
    iu = np.triu_indices(K, 1)
    st_pairs = st[iu] if K > 1 else np.array([])
    with np.errstate(invalid="ignore"):
        cell_sttc = np.nanmean(np.where(np.isfinite(st), st, np.nan), axis=1) if K > 1 else np.full(K, np.nan)
    row = dict(n_cells=K, n_total=d["n_total"], fr_hz=round(fr, 3), n_frames=T,
               duration_s=round(T / fr, 1), roi_source=d["roi_source"],
               event_source=d["event_source"],
               rate_per_min=float(rate.mean()) if K else np.nan,
               active_frac=float((n_ev > 0).mean()) if K else np.nan,
               mean_r=cor["mean_r"], sig_pair_frac=cor["sig_pair_frac"],
               degree_frac=float((cor["degree"] / max(1, K - 1)).mean()) if K else np.nan,
               clustering=cor["clustering"],
               sttc=float(np.nanmean(st_pairs)) if np.isfinite(st_pairs).any() else np.nan)
    row.update({k: v for k, v in spec.items() if not k.startswith("_")})
    cells = [dict(cell=int(d["rows"][i]), n_events=int(n_ev[i]), rate_per_min=float(rate[i]),
                  mean_r=float(cor["cell_mean_r"][i]), degree=int(cor["degree"][i]),
                  degree_frac=float(cor["degree"][i] / max(1, K - 1)),
                  sttc=float(cell_sttc[i])) for i in range(K)]
    return row, cells, spec.get("_psd")


# ── statistics ───────────────────────────────────────────────────────────────

def _nanmean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan


def hier_boot(groups: dict, n_boot: int, seed: int) -> tuple:
    """groups: {name: {day: [session values or arrays of cell values]}} for two
    groups.  Returns (difference B - A, 2.5 %, 97.5 %) of group means, with
    days, then sessions, then (if arrays) cells resampled with replacement."""
    rng = np.random.default_rng(seed)
    names = list(groups)

    def gmean(g, boot):
        days = list(g)
        if not days:
            return np.nan
        pick = rng.choice(len(days), len(days)) if boot else range(len(days))
        vals = []
        for di in pick:
            ss = g[days[di]]
            spick = rng.choice(len(ss), len(ss)) if boot else range(len(ss))
            for si in spick:
                v = ss[si]
                if isinstance(v, np.ndarray):
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        continue
                    vals.append(float(v[rng.choice(v.size, v.size)].mean() if boot else v.mean()))
                elif np.isfinite(v):
                    vals.append(float(v))
        return float(np.mean(vals)) if vals else np.nan

    a, b = groups[names[0]], groups[names[1]]
    obs = gmean(b, False) - gmean(a, False)
    bs = np.array([gmean(b, True) - gmean(a, True) for _ in range(n_boot)])
    bs = bs[np.isfinite(bs)]
    if bs.size < 10:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mixed_model(rows: list, metric: str, ref: str, other: str):
    """(estimate, p) of group in metric ~ group + (1 | day), or (nan, nan)."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        import warnings
    except ImportError:
        return np.nan, np.nan, "statsmodels not installed"
    df = pd.DataFrame([dict(y=r[metric], g=int(r["group"] == other), day=r["day"])
                       for r in rows if r["group"] in (ref, other) and np.isfinite(r[metric])])
    if df.day.nunique() < 3 or df.g.nunique() < 2:
        return np.nan, np.nan, "too few days"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm("y ~ g", df, groups=df["day"]).fit(reml=True)
        return float(fit.params["g"]), float(fit.pvalues["g"]), ""
    except Exception as exc:                       # noqa: BLE001
        return np.nan, np.nan, f"fit failed: {type(exc).__name__}"


# ── output ───────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _fig(metric: str, rows: list, order: list, compare: list) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(order), 3.2))
    days = sorted({r["day"] for r in rows})
    cmap = plt.get_cmap("tab10")
    for gi, g in enumerate(order):
        gr = [r for r in rows if r["group"] == g and np.isfinite(r.get(metric, np.nan))]
        for r in gr:
            di = days.index(r["day"])
            jitter = (hash(r["session"]) % 1000 / 1000 - 0.5) * 0.35
            ax.scatter(gi + jitter, r[metric], s=14, color=cmap(di % 10), alpha=0.8,
                       edgecolor="none")
        dm = [np.mean([r[metric] for r in gr if r["day"] == d]) for d in days
              if any(r["day"] == d for r in gr)]
        if dm:
            ax.hlines(np.mean(dm), gi - 0.3, gi + 0.3, color="k", lw=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([g + ("" if g in compare else "\n(not tested)") for g in order], fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: sessions (colour = day), bar = mean of day means", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def _html_table(rows: list, keys: list) -> str:
    def fmt(k, v):
        if k == "mixed_p" and isinstance(v, float) and 0 <= v < 1e-4:
            return "&lt;1e-4"
        return f"{v:.3g}" if isinstance(v, float) else str(v)
    head = "".join(f"<th>{k}</th>" for k in keys)
    body = "".join("<tr>" + "".join(f"<td>{fmt(k, r.get(k, ''))}</td>" for k in keys) + "</tr>"
                   for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


# ── main ─────────────────────────────────────────────────────────────────────

def read_groups(path: Path) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"session", "group"} <= set(rows[0]):
        raise SystemExit(f"group-compare: {path} needs columns session and group")
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="group-compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=Path, required=True, help="CSV with session, group")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="folder the sessions are below (default .)")
    ap.add_argument("--compare", nargs=2, metavar=("REF", "OTHER"), required=True,
                    help="the two groups to test (difference = OTHER - REF)")
    ap.add_argument("--out", type=Path, default=Path("group-compare"))
    ap.add_argument("--channel", default="00")
    ap.add_argument("--min-duration", type=float, default=60.0, metavar="S",
                    help="skip sessions shorter than this (default 60 s)")
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--n-surrogates", type=int, default=100)
    ap.add_argument("--pair-alpha", type=float, default=0.01,
                    help="per-pair significance vs shifts (needs n-surrogates >= 1/alpha - 1)")
    ap.add_argument("--sttc-dt", type=float, default=0.5, metavar="S")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if (1 + 0) / (1 + a.n_surrogates) > a.pair_alpha:
        raise SystemExit(f"group-compare: --n-surrogates {a.n_surrogates} can't reach "
                         f"p <= {a.pair_alpha}; use at least {int(round(1 / a.pair_alpha)) - 1}")
    ref, other = a.compare
    groups = read_groups(a.groups)
    known = {r["group"] for r in groups}
    for g in (ref, other):
        if g not in known:
            raise SystemExit(f"group-compare: no group {g!r} in {a.groups} "
                             f"(has: {', '.join(sorted(known))})")
    a.out.mkdir(parents=True, exist_ok=True)
    srows, crows, excluded = [], [], []
    for i, g in enumerate(groups):
        name, grp = g["session"], g["group"]
        if g.get("kind", "") == "zstack" or name.endswith("-zstack"):
            excluded.append(dict(session=name, group=grp, reason="z-stack"))
            continue
        m = _DATE_RE.match(name)
        day = m.group(1) if m else ""
        path = find_results(a.root, name, a.channel)
        if path is None:
            excluded.append(dict(session=name, group=grp, reason="no results file"))
            continue
        try:
            d = load_session(path)
        except Exception as exc:                   # noqa: BLE001
            excluded.append(dict(session=name, group=grp, reason=f"unreadable: {exc}"))
            continue
        dur = d["T"] / d["fr"]
        if dur < a.min_duration:
            excluded.append(dict(session=name, group=grp,
                                 reason=f"{dur:.0f} s < {a.min_duration:g} s"))
            continue
        if d["F"].shape[0] < a.min_cells:
            excluded.append(dict(session=name, group=grp,
                                 reason=f"{d['F'].shape[0]} cells < {a.min_cells}"))
            continue
        print(f"  {name}: {d['F'].shape[0]} cells, {dur:.0f} s ({d['roi_source']})", flush=True)
        row, cells, _psd = session_metrics(d, a.n_surrogates, a.sttc_dt, a.seed + i,
                                           a.pair_alpha)
        srows.append(dict(session=name, group=grp, day=day, results=str(path), **row))
        crows += [dict(session=name, group=grp, day=day, **c) for c in cells]
    if not srows:
        print("group-compare: no usable sessions")
        _write_csv(a.out / "excluded.csv", excluded)
        return 1

    # day means
    drows = []
    for (grp, day) in sorted({(r["group"], r["day"]) for r in srows}):
        rs = [r for r in srows if r["group"] == grp and r["day"] == day]
        drows.append(dict(group=grp, day=day, n_sessions=len(rs),
                          n_cells=int(sum(r["n_cells"] for r in rs)),
                          **{k: _nanmean([r[k] for r in rs]) for k in SESSION_METRICS}))

    # statistics
    stats = []
    for metric in SESSION_METRICS:
        level = "cell" if metric in CELL_METRICS else "session"
        nest = {}
        for grp in (ref, other):
            nest[grp] = {}
            for r in srows:
                if r["group"] != grp:
                    continue
                if level == "cell":
                    v = np.array([c[metric] for c in crows if c["session"] == r["session"]], float)
                else:
                    v = float(r[metric])
                nest[grp].setdefault(r["day"], []).append(v)
        diff, lo, hi = hier_boot(nest, a.n_boot, a.seed)
        est, p, note = mixed_model(srows, metric, ref, other)
        dm = {grp: [d[metric] for d in drows if d["group"] == grp] for grp in (ref, other)}
        if np.isfinite(p) and min(len(dm[ref]), len(dm[other])) < 5:
            note = "< 5 days/group: p too optimistic"
        stats.append(dict(metric=metric, bootstrap_level=f"day>session>{level}" if level == "cell"
                          else "day>session",
                          **{f"mean_{grp}": _nanmean(dm[grp]) for grp in (ref, other)},
                          **{f"days_{grp}": len(dm[grp]) for grp in (ref, other)},
                          diff=diff, ci95_lo=lo, ci95_hi=hi,
                          ci_excludes_0=bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                          mixed_est=est, mixed_p=p, note=note))

    _write_csv(a.out / "sessions.csv", srows)
    _write_csv(a.out / "cells.csv", crows)
    _write_csv(a.out / "days.csv", drows)
    _write_csv(a.out / "stats.csv", stats)
    _write_csv(a.out / "excluded.csv", excluded)

    order = [ref, other] + sorted({r["group"] for r in srows} - {ref, other})
    figs = []
    for metric in SESSION_METRICS:
        png = _fig(metric, srows, order, [ref, other])
        (a.out / f"fig_{metric}.png").write_bytes(png)
        figs.append((metric, base64.b64encode(png).decode()))
    n_days = {g: len({r["day"] for r in srows if r["group"] == g}) for g in order}
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Group comparison</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0 20px}}td,th{{border:1px solid #ccc;padding:3px 6px;text-align:right}}
th{{background:#f3f3f3}}.figs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:8px}}
.figs img{{width:100%}}.warn{{background:#fff4d6;padding:8px 12px;border-left:4px solid #e0a800}}</style></head><body>
<h1>{other} vs {ref}</h1>
<p>{len(srows)} sessions, {len(crows)} cells; days per group: {", ".join(f"{g} {n}" for g, n in n_days.items())}.
Difference = {other} − {ref}; 95 % CI from a hierarchical bootstrap (days → sessions → cells).
{("Also shown, not tested: " + ", ".join(g for g in order if g not in (ref, other)) + ".") if len(order) > 2 else ""}</p>
<p class="warn">The unit is the recording day. With {n_days.get(ref, 0)} vs {n_days.get(other, 0)} days the
intervals are wide and p-values are not reliable: treat this as descriptive. Sessions of one animal
are not independent replicates.</p>
<h2>Statistics</h2>{_html_table(stats, list(stats[0]))}
<h2>Day means</h2>{_html_table(drows, list(drows[0]))}
<h2>Figures</h2><div class="figs">{"".join(f'<img alt="{m}" src="data:image/png;base64,{b}">' for m, b in figs)}</div>
<h2>Sessions</h2>{_html_table(srows, [k for k in srows[0] if k != "results"])}
<h2>Excluded</h2>{_html_table(excluded, ["session", "group", "reason"]) if excluded else "<p>none</p>"}
</body></html>"""
    (a.out / "report.html").write_text(html)
    print(f"\n{len(srows)} sessions, {len(crows)} cells, {len(excluded)} excluded -> {a.out}")
    for s in stats:
        mark = " *" if s["ci_excludes_0"] else ""
        print(f"  {s['metric']:20s} {s[f'mean_{ref}']:9.4g} -> {s[f'mean_{other}']:9.4g}   "
              f"diff {s['diff']:+.3g} [{s['ci95_lo']:+.3g}, {s['ci95_hi']:+.3g}]{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
