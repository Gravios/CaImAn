"""group-compare: planted group differences are recovered; cell selection
(curated / accepted), exclusions and outputs."""
from _harness import REPO, ok, require
require("h5py", "scipy", "matplotlib")
import contextlib, csv, io, shutil, sys, tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(REPO))
from utilities import group_compare as gc  # noqa: E402

FR, T, K = 15.0, 4000, 40
root = Path(tempfile.mkdtemp()) / "strohA-zb-000000"


def simulate(rate_per_min, drive, f_osc, seed, rhythm_amp=0.6):
    """Poisson events modulated by a weak shared rhythm (f_osc) and by
    irregular synchronous population bursts (gain `drive`: the coupling);
    ΔF/F = events convolved with a 1 s decay + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FR
    rhythm = 1 + rhythm_amp * np.sin(2 * np.pi * f_osc * t + rng.uniform(0, 6.28))
    bursts = np.zeros(T)
    bursts[rng.random(T) < 0.1 / FR] = 1                 # ~0.1 bursts / s
    bursts = np.convolve(bursts, np.ones(int(0.4 * FR)))[:T]
    pop = rhythm * (1 + drive * 10 * bursts)
    lam = rate_per_min / 60 / FR * pop / pop.mean()
    S = rng.random((K, T)) < lam[None, :]
    kern = np.exp(-np.arange(int(5 * FR)) / FR)
    C = np.array([np.convolve(s.astype(float), kern)[:T] for s in S])
    F = C + rng.normal(0, 0.05, C.shape)
    return F, C


def write_results(path, F, C, idx=None, stroh=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["estimates/F_dff"] = F
        f["estimates/C"] = C
        f["estimates/SNR_comp"] = np.full(F.shape[0], 10.0)
        f["estimates/cnn_preds"] = np.full(F.shape[0], 0.9)
        f["estimates/idx_components"] = (np.arange(F.shape[0]) if idx is None else idx)
        f["params/data/fr"] = FR
        f["params/data/decay_time"] = 1.0
        if stroh is not None:
            ed = gc._module("event_detection")
            res = ed.detect_events(F, C, FR, 1.0, np.full(F.shape[0], 10.0))
            ed.write_stroh_results(f, res, stroh)


def session(day, tl, group_kind="spont"):
    return f"strohA-zb-000000-{day}-TL{tl:03d}_120000-00x-{group_kind}"


def make(name, F, C, **kw):
    d = root / f"strohA-zb-000000-{name.split('-')[3]}" / name / f"{name}-C00-fc{T:06d}"
    write_results(d / f"{name}-C00-fc{T:06d}_results.hdf5", F, C, **kw)
    return d


groups = []
seed = 0
plan = {"alz": ["20140211", "20140228", "20140310"], "alz+MS125": ["20140221", "20140222"],
        "MS125 only": ["20140318"]}
truth = {"alz": (3.0, 0.1, 0.25), "alz+MS125": (6.0, 1.0, 0.5), "MS125 only": (4.0, 0.5, 0.25)}
try:
    for grp, days in plan.items():
        rate, drive, fo = truth[grp]
        for day in days:
            for tl in range(1, 5):
                seed += 1
                F, C = simulate(rate, drive, fo, seed)
                name = session(day, tl)
                kw = {}
                if (grp, day, tl) == ("alz", "20140211", 2):
                    kw = dict(idx=np.arange(30))                  # 30 accepted
                if (grp, day, tl) == ("alz+MS125", "20140221", 3):
                    kw = dict(stroh=np.ones(K, bool))             # events in the file
                d = make(name, F, C, **kw)
                if (grp, day, tl) == ("alz", "20140228", 1):      # curated copy
                    appr = np.zeros(K, bool); appr[:25] = True
                    write_results(d / f"{name}-C00-fc{T:06d}_results_curated.hdf5", F, C,
                                  idx=np.arange(35), stroh=appr)
                groups.append(dict(group=grp, kind="spont", session=name))
    # exclusions: short session, z-stack, no results
    F, C = simulate(3.0, 0.3, 0.25, 999)
    short = session("20140310", 9)
    d = root / "strohA-zb-000000-20140310" / short / f"{short}-C00-fc000500"
    write_results(d / f"{short}-C00-fc000500_results.hdf5", F[:, :500], C[:, :500])
    groups += [dict(group="alz", kind="spont", session=short),
               dict(group="alz", kind="zstack", session=session("20140228", 1, "zstack")),
               dict(group="alz", kind="spont", session=session("20140310", 8))]
    gcsv = root.parent / "groups.csv"
    with open(gcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "kind", "session"]); w.writeheader()
        w.writerows(groups)

    out = root.parent / "out"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gc.main(["--groups", str(gcsv), "--root", str(root), "--compare", "alz", "alz+MS125",
                      "--out", str(out), "--n-surrogates", "99", "--n-boot", "300"])
    assert rc == 0, buf.getvalue()
    for fn in ("cells.csv", "sessions.csv", "days.csv", "stats.csv", "excluded.csv", "report.html",
               "fig_rate_per_min.png", "fig_sttc.png"):
        assert (out / fn).stat().st_size > 0, fn
    read = lambda fn: list(csv.DictReader(open(out / fn)))
    S = {r["session"]: r for r in read("sessions.csv")}
    assert len(S) == 24, len(S)
    ok("24 sessions measured (3 groups); CSVs, figures and report.html written")

    ex = {r["session"]: r["reason"] for r in read("excluded.csv")}
    assert ex[short].startswith("33 s < 60") and ex[session("20140228", 1, "zstack")] == "z-stack"
    assert ex[session("20140310", 8)] == "no results file"
    ok("excluded and listed: z-stack, no results file, session shorter than 60 s")

    cur = S[session("20140228", 1)]
    assert cur["roi_source"].startswith("curated") and cur["n_cells"] == "25", cur
    assert cur["results"].endswith("_results_curated.hdf5")
    assert S[session("20140211", 2)]["n_cells"] == "30" and \
        S[session("20140211", 2)]["roi_source"] == "CNMF accepted"
    assert S[session("20140221", 3)]["event_source"] == "stroh_results"
    assert S[session("20140222", 1)]["event_source"] == "detected here"
    ok("cells: curated file → accepted ∩ approved (25 of 35 accepted); else CNMF-accepted "
       "(30); events from stroh_results when present, else detected in memory")

    # planted differences recovered
    rate = {g: np.mean([float(r["rate_per_min"]) for r in S.values() if r["group"] == g])
            for g in plan}
    assert 2.3 < rate["alz"] < 3.7 and 5.0 < rate["alz+MS125"] < 7.0, rate
    peak = {g: np.median([float(r["pop_peak_slow_hz"]) for r in S.values() if r["group"] == g])
            for g in plan}
    assert abs(peak["alz"] - 0.25) < 0.05 and abs(peak["alz+MS125"] - 0.5) < 0.05, peak
    st = {r["metric"]: r for r in read("stats.csv")}
    for m in ("rate_per_min", "mean_r", "sig_pair_frac", "sttc", "pop_peak_slow_hz"):
        assert float(st[m]["diff"]) > 0 and st[m]["ci_excludes_0"] == "True", (m, st[m])
    assert float(st["sig_pair_frac"]["mean_alz+MS125"]) > float(st["sig_pair_frac"]["mean_alz"])
    assert st["rate_per_min"]["days_alz"] == "3" and st["rate_per_min"]["days_alz+MS125"] == "2"
    ok("planted differences recovered: rate ×2, stronger burst coupling (r, significant pairs, STTC), "
       "population peak 0.25 → 0.5 Hz; CI excludes 0; days are the unit (3 vs 2)")

    # null: same parameters in both groups → CI should mostly include 0
    F0 = [simulate(3.0, 0.0, 0.25, 500 + i, rhythm_amp=0.0)[0] for i in range(6)]
    rows = [gc.correlation_metrics(F[:, :2000], n_surr=99, seed=i)["sig_pair_frac"]
            for i, F in enumerate(F0)]
    assert np.mean(rows) < 0.02, rows
    ok(f"independent cells: {100 * np.mean(rows):.1f} % of pairs significant "
       "(chance level 1 % at p <= 0.01 vs circular shifts)")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            gc.main(["--groups", str(gcsv), "--root", str(root), "--compare", "alz", "alz+MS125",
                     "--out", str(out), "--n-surrogates", "20"])
            raise AssertionError("too few surrogates accepted")
        except SystemExit as e:
            assert "use at least 99" in str(e)
    ok("--n-surrogates too small for --pair-alpha is refused")

    s = gc.sttc_matrix([np.array([10, 100, 200]), np.array([10, 100, 200]),
                        np.array([3000]), np.array([], int)], 4000, 15.0, 0.5)
    assert abs(s[0, 1] - 1) < 1e-6 and s[0, 2] < 0.05 and np.isnan(s[0, 3])
    ok("STTC: identical trains 1, unrelated ~0, silent cell NaN")
    assert "not tested" in (out / "report.html").read_text()
    print("ALL OK")
finally:
    shutil.rmtree(root.parent, ignore_errors=True)
