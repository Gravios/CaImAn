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
print("ALL OK")
