"""caiman/utils/event_kinetics.py — kinetic check of events on raw ΔF/F."""
from _harness import REPO, ok, require, stub_caiman  # noqa: F401
require("numpy", "scipy")
stub_caiman()
import numpy as np

from caiman.utils import event_detection as ed
from caiman.utils import event_kinetics as ek

FR, DEC, T = 30.0, 0.4, 9000
rng = np.random.default_rng(1)


def kernel(tau_s, rise_frames=0, n=None):
    n = n or int(8 * tau_s * FR)
    d = np.exp(-np.arange(n) / (tau_s * FR))
    if rise_frames:
        d = np.concatenate([np.linspace(0, 1, rise_frames + 1)[1:-1], d])
    return d


def train(rate=0.01):
    return (rng.random(T) < rate) * rng.uniform(0.5, 2, T)


def detect(y, c):
    return ed.detect_events(y[None], c[None], FR, DEC)


# ── 1. real transients vs. what C fabricates on a silent cell ──────────────
c = np.convolve(train(), kernel(DEC))[:T]
y_real = c + 0.05 * rng.standard_normal(T)
drift = np.convolve(rng.standard_normal(T), np.ones(150) / 150, "same") * 3
y_fake = drift + 0.05 * rng.standard_normal(T)
c_fake = np.convolve(train(), kernel(DEC))[:T]          # AR-shaped C, unrelated to y
res = ed.detect_events(np.vstack([y_real, y_fake]), np.vstack([np.maximum(y_real, 0), c_fake]),
                       FR, DEC)
rows = ek.validate(np.vstack([y_real, y_fake]), res)
sm = ek.summary_table(rows)
assert sm["frac_valid"][0] > 0.75 and sm["frac_valid"][1] == 0, sm["frac_valid"]
assert abs(sm["median_tau_s"][0] - DEC) < 0.05 and sm["median_tau_s"][1] > 2 * DEC
assert sm["median_shape_r"][0] > 0.9 and sm["median_shape_r"][1] < 0.4
assert abs(sm["noise_sigma"][0] - 0.05) < 0.005
ok(f"real transients {sm['frac_valid'][0]:.0%} valid (τ {sm['median_tau_s'][0]:.2f} s, "
   f"r {sm['median_shape_r'][0]:.2f}); drift under a fabricated C 0 % "
   f"(τ {sm['median_tau_s'][1]:.1f} s, r {sm['median_shape_r'][1]:.2f})")


# ── 2. one isolated event per case: each criterion trips its own bit ───────
def one_event(shape, amp=1.0, noise=0.03, at=3000):
    y = noise * rng.standard_normal(T)
    y[at:at + shape.size] += amp * shape[:T - at]
    pk = at + int(np.argmax(shape))                 # C: AR-shaped, peaking with y
    cc = np.zeros(T); cc[pk:pk + 240] = amp * kernel(DEC, n=240)
    r = detect(y, cc)
    assert r.n_events(0) == 1, r.n_events(0)
    row = ek.validate_row(y, r.peaks[0], FR, DEC)
    return row, ek.decode_fail(row["fail"][0])


row, f = one_event(kernel(DEC))
assert f == [] and row["valid"][0] and abs(row["tau_s"][0] - DEC) < 0.04
row, f = one_event(kernel(1.6))                      # 4× too slow
assert "tau" in f and row["tau_s"][0] > 2 * DEC, (f, row["tau_s"])
row, f = one_event(kernel(DEC, rise_frames=15))      # 0.5 s ramp
assert "rise" in f and row["rise_s"][0] > 0.3, (f, row["rise_s"])
sq = np.r_[np.ones(40), np.zeros(10)]                # 1.3 s plateau (≲ 1.7 τ passes)
row, f = one_event(sq)
assert "shape" in f and "tau" in f, (f, row["shape_r"])
y = 0.12 * rng.standard_normal(T); y[3000:3000 + 96] += 0.25 * kernel(DEC)[:96]
row = ek.validate_row(y, np.array([[3000, 0.25, 2999, 3010]], float), FR, DEC)
f = ek.decode_fail(row["fail"][0])          # (too small for detection itself)
assert "amplitude" in f and row["amp_snr"][0] < 3, (f, row["amp_snr"])
ok("valid event passes; slow decay → tau, slow ramp → rise, plateau → shape + tau, "
   "small vs noise → amplitude")

# fitted τ follows the indicator
for tau in (0.2, 0.8, 1.2):
    y = 0.02 * rng.standard_normal(T); y[4000:4000 + int(8 * tau * FR)] += kernel(tau)
    cc = np.zeros(T); cc[4000:4000 + int(8 * tau * FR)] = kernel(tau)
    r = ed.detect_events(y[None], cc[None], FR, tau)
    row = ek.validate_row(y, r.peaks[0], FR, tau)
    assert abs(row["tau_s"][0] - tau) / tau < 0.1 and row["valid"][0], (tau, row["tau_s"])
ok("fitted τ recovers 0.2 / 0.8 / 1.2 s within 10 %")

# ── 3. params, short windows, edge cases ───────────────────────────────────
p = dict(ek.DEFAULTS, tau_max_frac=None)
assert ek.validate_row(0.03 * rng.standard_normal(T), np.zeros((0, 4)), FR, DEC)["valid"].size == 0
y = 0.03 * rng.standard_normal(T); y[3000:3000 + 96] += kernel(1.6)[:96]
cc = np.zeros(T); cc[3000:3240] = kernel(DEC, n=240)
r = detect(y, cc)
assert "tau" in ek.decode_fail(ek.validate_row(y, r.peaks[0], FR, DEC)["fail"][0])
assert ek.validate_row(y, r.peaks[0], FR, DEC, p)["valid"][0], "criterion off → passes"
assert ek.validate_row(y, r.peaks[0], FR, DEC, {"decay_time": 1.6})["valid"][0], \
    "decay_time parameter overrides the events' τ"
# two events 3 frames apart: the first is untestable (short)
pk = np.array([[3000, 1, 2999, 3005], [3003, 1, 3002, 3010]], float)
yy = 0.03 * rng.standard_normal(T); yy[3000:3240] += kernel(DEC, n=240); yy[3003:3243] += kernel(DEC, n=240)
row = ek.validate_row(yy, pk, FR, DEC)
assert row["fail"][0] == ek.FAIL["short"] and not row["valid"][0]
s = ek.summarize_row(row)
assert s["n_events"] == 2 and s["n_testable"] == 1
bad = ek.validate_row(yy, pk, FR, float("nan"))
assert (bad["fail"] == ek.FAIL["short"]).all()
ok("criteria can be switched off; decay_time override; cut-short events are untestable; "
   "unknown τ → untestable")

# ── 4. tables line up with event_detection's peak table ───────────────────
pt = res.peak_table([1, 0])
et = ek.event_table(rows, [1, 0])
assert et["shape_r"].size == pt["peak"].size
k0 = res.n_events(1)
assert np.allclose(et["shape_r"][:k0], rows[1]["shape_r"], equal_nan=True)
assert np.allclose(et["tau_s"][k0:], rows[0]["tau_s"], equal_nan=True)
tx, ty = ek.template_curve(rows[0], 0, FR, DEC, res.peaks[0])
assert tx.size and abs(ty.max() - (rows[0]["baseline"][0] + rows[0]["amp"][0])) < 1e-9
ok("event_table rows align with EventResults.peak_table; template_curve peaks at b + A")

# ── 5. which trace: the raw C + YrA, not a ΔF/F that is C rescaled ────────
F0 = 4.0
yr = drift + 0.05 * rng.standard_normal(T)              # raw: no transients at all
YrA = yr - c_fake                                       # C fabricates events on it
dff_resid = yr / F0                                     # use_residuals=True
dff_noresid = c_fake / F0                               # use_residuals=False: C rescaled
assert ek.dff_tracks_C(dff_noresid[None], c_fake[None]) > 0.999
assert ek.dff_tracks_C(dff_resid[None], c_fake[None]) < 0.5
r5 = ed.detect_events(dff_noresid[None], c_fake[None], FR, DEC)
no_amp = dict(ek.DEFAULTS, min_amp_snr=None)          # C has no noise to scale by
circ = ek.summarize_row(ek.validate_row(dff_noresid, r5.peaks[0], FR, DEC, no_amp))
raw = ek.raw_dff(c_fake[None], YrA[None], dff_noresid[None])[0]
assert np.allclose(raw, yr / F0), "raw trace rescaled with the ΔF/F's own F0"
honest = ek.summarize_row(ek.validate_row(raw, r5.peaks[0], FR, DEC, no_amp))
assert circ["frac_valid"] > 0.6 and honest["frac_valid"] < 0.1, (circ, honest)
ok(f"fabricated C: on ΔF/F without residual {circ['frac_valid']:.0%} 'valid' (circular, "
   f"flagged by dff_tracks_C); on raw C + YrA {honest['frac_valid']:.0%}")

# ── 6. opt-in approval rule ─────────────────────────────────────────────────
import os, tempfile, h5py
Y6 = np.vstack([y_real, yr])                            # raw ΔF/F: real cell, silent cell (drift ≫ 0.2)
C6 = np.vstack([np.maximum(y_real, 0), c_fake])         # C: real fit, fabricated
YrA6 = Y6 - C6                                          # (F0 = 1 for row 0)
plain = ed.detect_events(Y6, C6, FR, DEC)
assert plain.auto_approved.tolist() == [True, True], "SLab approves the fabricated cell"
off = ed.detect_events(Y6, C6, FR, DEC)
assert ek.apply_approval(off, Y6, None) is None and off.auto_slab is None
assert np.array_equal(off.auto_approved, plain.auto_approved)
on = ed.detect_events(Y6, C6, FR, DEC)
kin = ek.apply_approval(on, Y6, {"approve_on_kinetics": True})
assert on.auto_approved.tolist() == [True, False] and on.auto_slab.tolist() == [True, True]
assert on.params["approval_rule"] == "slab+kinetics" and on.params["kinetics"]["min_valid_events"] == 1
strict = ed.detect_events(Y6, C6, FR, DEC)
ek.apply_approval(strict, Y6, {"approve_on_kinetics": True, "min_valid_frac": 0.99})
assert strict.auto_approved.tolist() == [False, False]
assert "≥ 1 kinetically valid" in ek.describe_rule(on.params) and \
    ek.describe_rule(plain.params).startswith("SLab (")
sub = on.take([1, 0]).append(plain.take([0]))
assert sub.auto_slab.tolist() == [True, True, True] and sub.auto_approved.tolist() == [False, True, True]
ok("rule off: auto-approval untouched; on: real cell kept, fabricated one rejected "
   "(SLab approved both); min share; take / append keep SLab's verdict aligned")

tmp = tempfile.mkdtemp()
fp = os.path.join(tmp, "r.hdf5")
with h5py.File(fp, "w") as f:
    ed.write_stroh_results(f, on, on.auto_approved)
    ed.write_stroh_results(f, plain, plain.auto_approved, group="plain")
with h5py.File(fp, "r") as f:
    g = f["stroh_results"]
    assert g.attrs["approval_rule"] == "slab+kinetics" and list(g["auto_approved"][()]) == [0]
    assert list(g["auto_approved_slab"][()]) == [0, 1] and "kinetics_params" in g.attrs
    assert f["plain"].attrs["approval_rule"] == "slab" and "auto_approved_slab" not in f["plain"]
    back, _ = ed.read_stroh_results(g)
    back_plain, _ = ed.read_stroh_results(f["plain"])
assert back.auto_slab.tolist() == [True, True] and back.params["approval_rule"] == "slab+kinetics"
assert back_plain.auto_slab is None and "approval_rule" not in back_plain.params
ok("stroh_results records the rule, its criteria and SLab's own verdict; read back")


def results_file(path, F, C, YrA=None):
    with h5py.File(path, "w") as f:
        f["estimates/F_dff"] = F; f["estimates/C"] = C
        if YrA is not None:
            f["estimates/YrA"] = YrA
        f["params/data/fr"] = FR; f["params/data/decay_time"] = DEC


p1 = os.path.join(tmp, "a.hdf5"); results_file(p1, Y6, C6, YrA6)
msg = ed.detect_events_in_hdf5(p1, kinetics={"approve_on_kinetics": True})
assert msg.startswith("1/2 auto-approved (SLab 2; kinetics on C + YrA)"), msg
p2 = os.path.join(tmp, "b.hdf5"); results_file(p2, C6 / 2.0, C6)        # ΔF/F = C rescaled
msg2 = ed.detect_events_in_hdf5(p2, kinetics={"approve_on_kinetics": True})
assert "kinetics rule skipped" in msg2 and msg2.startswith("2/2"), msg2
p3 = os.path.join(tmp, "c.hdf5"); results_file(p3, Y6, C6, YrA6)
import contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ed.main([p3, "--require-kinetics"])
assert "1/2 auto-approved" in buf.getvalue(), buf.getvalue()
with h5py.File(p3, "r") as f:
    assert f["stroh_results"].attrs["approval_rule"] == "slab+kinetics"
with contextlib.redirect_stdout(io.StringIO()):
    ed.main([p3, "--overwrite"])
with h5py.File(p3, "r") as f:
    assert f["stroh_results"].attrs["approval_rule"] == "slab"
ok("headless / CLI: --require-kinetics uses C + YrA; skipped (and SLab kept) when "
   "ΔF/F ≈ C without residual; default stays SLab")
print("ALL OK")
