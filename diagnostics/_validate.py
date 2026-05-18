"""Validation harness for noise_diagnostics.

Builds a baseline synthetic 2P stack (Poisson signal + Gaussian read noise +
~25 simulated cell ROIs), then injects each noise source one at a time and
verifies the corresponding test/source detects it.

Pass criterion per source: detected as 'moderate' or 'high' with the
correct evidence trend.  Clean baseline must produce 'negligible' for all
fault sources and high shot_noise_dominated.
"""
import sys, os, time
import numpy as np
from scipy.ndimage import shift as nd_shift

sys.path.insert(0, "/tmp/build/diagnostics")
from noise_diagnostics import run_diagnostics  # noqa: E402

OUT = "/tmp/build/validate_out"
os.makedirs(OUT, exist_ok=True)


def make_clean_stack(T=200, H=256, W=256, gain=2.0, read_sigma=1.5,
                     rng_seed=42, n_cells=25):
    rng = np.random.default_rng(rng_seed)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    base = 30 + 0.04 * yy
    cells = []
    for _ in range(n_cells):
        cy, cx, r = int(rng.uniform(20, H - 20)), int(rng.uniform(20, W - 20)), rng.uniform(5, 10)
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        cells.append((g / g.max()).astype(np.float32))

    stack = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        sig = base.copy()
        for k, c in enumerate(cells):
            amp = 30 * (np.sin(2 * np.pi * t / (30 + 5 * (k % 6))) + 1)
            sig += amp * c
        photons = sig / gain
        noisy = rng.poisson(photons).astype(np.float32) * gain
        noisy += rng.normal(0, read_sigma, noisy.shape).astype(np.float32)
        stack[t] = noisy
    return np.clip(stack, 0, 65535).astype(np.uint16)


def report_for(stack, label):
    sub = f"{OUT}/{label}"
    os.makedirs(sub, exist_ok=True)
    return run_diagnostics(stack, out_dir=sub, n_frames=200, save_panel=False,
                           save_json=False, write_summary=False)


def src(report, name):
    return report["sources"][name]


def level_at_or_above(d, threshold):
    levels = ("negligible", "low", "moderate", "high")
    return levels.index(d["level"]) >= levels.index(threshold)


results = []


def check(name, ok, detail=""):
    flag = "PASS" if ok else "FAIL"
    print(f"  {flag}  {name}{('  ' + detail) if detail else ''}")
    results.append((ok, name, detail))


print("=" * 70)
print("BASELINE — clean stack, only shot noise should fire")
print("=" * 70)
clean = make_clean_stack()
r_clean = report_for(clean, "00_clean")

# shot_noise_dominated should be moderate or higher
shot = src(r_clean, "shot_noise_dominated")
check("baseline: shot_noise_dominated ≥ moderate",
      level_at_or_above(shot, "moderate"),
      f"score={shot['score']:.3f}")
# Every fault source should be negligible
fault_sources = [
    "bidirectional_phase_offset", "horizontal_banding_fixed",
    "fast_axis_periodic", "galvo_flyback_edge", "hot_dead_pixels",
    "photobleaching", "saturation_clipping", "frame_discontinuity",
]
for s in fault_sources:
    d = src(r_clean, s)
    check(f"baseline: {s} == negligible", d["level"] == "negligible",
          f"score={d['score']:.3f}")


print()
print("=" * 70)
print("INJECT — bidirectional offset (+0.6 px on even rows)")
print("=" * 70)
s = clean.astype(np.float32).copy()
for t in range(s.shape[0]):
    f = s[t].copy()
    f[0::2] = nd_shift(f[0::2], shift=(0, 0.6), order=1, mode="nearest")
    s[t] = f
s = np.clip(s, 0, 65535).astype(np.uint16)
r = report_for(s, "01_bidir")
d = src(r, "bidirectional_phase_offset")
check("bidir injected → bidirectional_phase_offset ≥ moderate",
      level_at_or_above(d, "moderate"),
      f"shift={d['evidence']['bidir_shift_px']:.2f}px score={d['score']:.3f}")


print()
print("=" * 70)
print("INJECT — horizontal banding (fixed pattern, 0.10 cyc/px)")
print("=" * 70)
s = clean.astype(np.float32).copy()
H = s.shape[1]
banding = 8.0 * np.sin(2 * np.pi * 0.10 * np.arange(H)).astype(np.float32)
s += banding[None, :, None]
s = np.clip(s, 0, 65535).astype(np.uint16)
r = report_for(s, "02_banding")
d = src(r, "horizontal_banding_fixed")
check("banding injected → horizontal_banding_fixed ≥ moderate",
      level_at_or_above(d, "moderate"),
      f"peak_db={d['evidence']['peak_power_slow_db']:.1f} score={d['score']:.3f}")


print()
print("=" * 70)
print("INJECT — galvo flyback edge (8 dead columns on left)")
print("=" * 70)
s = clean.copy()
s[:, :, :8] = 0
r = report_for(s, "03_edge")
d = src(r, "galvo_flyback_edge")
check("edge injected → galvo_flyback_edge ≥ moderate",
      level_at_or_above(d, "moderate"),
      f"dead_left={d['evidence']['dead_left']} score={d['score']:.3f}")


print()
print("=" * 70)
print("INJECT — hot pixels (3 hot, 1 dead)")
print("=" * 70)
s = clean.copy()
for (y, x) in [(50, 30), (130, 200), (220, 110)]:
    s[:, y, x] = 60000  # stuck high
s[:, 80, 90] = 0         # stuck low
r = report_for(s, "04_hot")
d = src(r, "hot_dead_pixels")
check("hot pixels injected → hot_dead_pixels ≥ moderate",
      level_at_or_above(d, "moderate"),
      f"hot={d['evidence']['hot_count']} dead={d['evidence']['dead_count']} "
      f"score={d['score']:.3f}")


print()
print("=" * 70)
print("INJECT — photobleaching (τ=80 frames, T=200)")
print("=" * 70)
s = clean.astype(np.float32)
decay = np.exp(-np.arange(s.shape[0]) / 80.0).astype(np.float32)
s = s * decay[:, None, None] + 10
s = np.clip(s, 0, 65535).astype(np.uint16)
r = report_for(s, "05_bleach")
d = src(r, "photobleaching")
check("bleach injected → photobleaching ≥ moderate",
      level_at_or_above(d, "moderate"),
      f"tau={d['evidence']['decay_tau_frames']:.0f} "
      f"amp={d['evidence']['decay_amp_frac']:.2f} score={d['score']:.3f}")


print()
print("=" * 70)
print("INJECT — saturation clipping (1% of pixels at uint16 max)")
print("=" * 70)
s = clean.copy()
rng = np.random.default_rng(7)
mask = rng.random(s.shape) < 0.01
s[mask] = 65535
r = report_for(s, "06_sat")
d = src(r, "saturation_clipping")
check("saturation injected → saturation_clipping ≥ moderate",
      level_at_or_above(d, "moderate"),
      f"sat_frac={d['evidence']['sat_fraction']:.4f} score={d['score']:.3f}")


print()
print("=" * 70)
print("INJECT — frame discontinuity (4 frames with big jumps)")
print("=" * 70)
s = clean.astype(np.float32)
for ti in [40, 80, 120, 160]:
    s[ti] = s[ti] * 0.3   # large negative jump
s = np.clip(s, 0, 65535).astype(np.uint16)
r = report_for(s, "07_glitch")
d = src(r, "frame_discontinuity")
check("glitch injected → frame_discontinuity ≥ moderate",
      level_at_or_above(d, "moderate"),
      f"glitches={d['evidence']['discontinuity_count']} score={d['score']:.3f}")


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
passed = sum(1 for ok, _, _ in results if ok)
failed = [(n, det) for ok, n, det in results if not ok]
print(f"{passed}/{len(results)} checks passed")
if failed:
    for n, det in failed:
        print(f"  FAIL  {n}  {det}")
sys.exit(0 if not failed else 1)
