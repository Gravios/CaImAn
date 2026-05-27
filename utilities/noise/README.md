# utilities/noise — 2P noise characterization and correction

Two cooperating modules:

- **`noise_diagnostics.py`** (read-only): runs a battery of tests on a sampled
  subset of frames and maps the metrics to a calibrated likelihood
  (`negligible | low | moderate | high`) for each of ~13 known noise sources,
  with a one-line recommended fix per source.
- **`noise_correction.py`** (writes): pure-function primitives that target the
  fixable diagnostic flags. The two modules share a vocabulary —
  `recommend_corrections(report)` reads a diagnostic JSON and returns an
  ordered correction recipe.

## noise_diagnostics — what it tests

| # | Test                  | Metrics                                              |
|---|-----------------------|------------------------------------------------------|
| 1 | photon_transfer       | gain (DN/e-), read noise (DN), shot/read ratio, R²   |
| 2 | bidirectional         | sub-pixel even-row shift, every-other-row alternation |
| 3 | spectral              | narrowband peaks on fast & slow axes, stationarity   |
| 4 | edge_artifacts        | dead-column count at each of 4 edges                 |
| 5 | hot_pixels            | hot/dead pixel fraction via local-z + low temp var   |
| 6 | drift                 | linear % change, exponential τ estimate              |
| 7 | fixed_pattern         | high-pass(temporal mean) variance vs shot floor      |
| 8 | saturation            | ADC ceiling/floor clip fraction, dynamic-range usage |
| 9 | frame_discontinuity   | outlier z-score on frame-mean jumps                  |

## Sources scored

```
shot_noise_dominated              (informational — high is healthy)
bidirectional_phase_offset
horizontal_banding_fixed          → row-pedestal subtraction
horizontal_banding_drifting       → 1D notch filter
fast_axis_periodic                → pixel-clock / sample-hold issue
galvo_flyback_edge                → crop edge columns
hot_dead_pixels                   → median replace
photobleaching                    → detrend pixel traces
illumination_drift_increase       → drop warm-up frames
fixed_pattern_noise               → dark-frame subtract
saturation_clipping               → reduce PMT gain
quantization_loss                 → increase PMT gain
frame_discontinuity               → drop / interpolate bad frames
```

Each comes with a `level`, a continuous `score`, an `evidence` dict, and a
one-line recommendation.

## Usage

```python
from utilities.noise.noise_diagnostics import run_diagnostics

# From a tif/bigtiff/npy file (memory-efficient — only sampled pages decoded):
rep = run_diagnostics("/data/src/stroh-sa/sess/recording.tif",
                     out_dir="diag_out", n_frames=500)

# From an in-memory (T, H, W) array:
rep = run_diagnostics(stack, out_dir="diag_out", n_frames=500)
```

CLI:
```
python utilities/noise/noise_diagnostics.py /data/.../recording.tif --out diag_out --n_frames 500
```

## Outputs

- `diag_out/diagnostic_report.json`  — every metric, every source, every recommendation
- `diag_out/diagnostic_panel.png`    — 3×3 visual summary
- `diag_out/summary.txt`             — printable top-issue ranking

## Calibration notes

Thresholds for level assignment are conservative; tune in `score_sources()`
once you have a few reference recordings.  Notable behaviours:

- **`shot_noise_dominated`** is reported as "high" when shot variance dominates
  read variance by ≥10× — that's the *healthy* regime, not an alarm.
- **PTC gain** is order-of-magnitude correct from diff-based variance; for
  precision use a still / dark-and-flat acquisition rather than session frames.
- The split-half **stationarity** check needs ≥4 sampled frames in each half;
  with `n_frames=500` this is comfortable.
- **`fixed_pattern_noise`** can flare with strong horizontal banding (the
  banding *is* a fixed pattern); look at `horizontal_banding_fixed` first.

## Dependencies

`numpy`, `scipy`, `matplotlib`, and `tifffile` (only required if you pass a
`.tif` / `.tiff` / `.btf` path).  No CaImAn import required — works standalone.

## Integration with `pipeline_p2.py`

Run as a pre-flight check before NoRMCorre on a new dataset:

```python
from utilities.noise.noise_diagnostics import run_diagnostics
rep = run_diagnostics(input_tif, out_dir=str(session_dir / "diag"), n_frames=500)
# fail-fast on serious issues
high = {n for n, d in rep["sources"].items() if d["level"] == "high"
        and n not in ("shot_noise_dominated",)}
if high:
    log.warning("preflight noise diagnostics flagged: %s", ", ".join(high))
```

The recommendations field gives you the right hook to drive parameter changes
(e.g. setting `bord_px` to `dead_cols_left + dead_cols_right + 4` when
`galvo_flyback_edge` is at least moderate).

---

## noise_correction — primitives

| function | addresses flag | technique |
|---|---|---|
| `replace_hot_pixels` | hot/dead pixels | per-frame 3×3 spatial-median substitution; pixels flagged by combined high local-z + low variance/mean ratio |
| `correct_bidirectional` | `bidirectional_phase_offset` | **sub-pixel** even-row shift via `scipy.ndimage.shift(order=1)`. Complements the existing **integer-pixel** `caiman.utils.xcorr_correction.correct_line_scan` |
| `subtract_row_pedestal` | `horizontal_banding_fixed` / `_drifting` | per-row temporal-median or per-frame-median offset subtraction (rank-1 banding removal) |
| `subtract_column_pedestal` | `fast_axis_periodic` | per-column temporal-median or per-frame-median offset subtraction — removes FOV-uniform vertical-stripe artifacts from resonant-scanner velocity non-linearity or column-clocked detector structure |
| `regress_common_mode` | `periodic_temporal_global` | OLS projection of the centred frame-mean (or a user-supplied trace) out of every pixel |
| `notch_temporal` | known mains/aliased line | per-pixel `scipy.signal.iirnotch` + `filtfilt`, chunked across pixels |

Plus glue:
- `recommend_corrections(report, stack=None)` — read a diagnostic report dict
  and return `[(callable, kwargs), ...]` in priority order.
- `apply_corrections(stack, ops)` — chain runner.

### Provenance
All five primitives are textbook techniques. Implementations are written from
scratch to avoid GPL-3-vs-GPL-2 license mixing with suite2p (whose
bidirectional-shift algorithm is closest in spirit to `correct_bidirectional`).
The algorithm families are cited per-function in the docstrings: suite2p
(Pachitariu 2017), NoRMCorre (Pnevmatikakis 2017), CompCor (Behzadi 2007).

### Usage

```python
from utilities.noise.noise_diagnostics import run_diagnostics
from utilities.noise.noise_correction import recommend_corrections, apply_corrections
import numpy as np

# Diagnose
rep = run_diagnostics(stack, out_dir="diag", n_frames=3000, fs_hz=30,
                      sampling_mode="contiguous")

# Build recipe (pass stack so the stricter var/mean hot-pixel detector runs)
ops = recommend_corrections(rep, min_level="moderate", stack=stack.astype(np.float32))
# e.g. ops == [(replace_hot_pixels, {"mask": ...}),
#              (correct_bidirectional, {"shift_px": -0.51}),
#              (subtract_row_pedestal, {"mode": "temporal_median"}),
#              (regress_common_mode, {})]

# Apply in order, returns a corrected (T, H, W) float32 stack
out = apply_corrections(stack, ops)

# Or apply individually with custom args
from utilities.noise.noise_correction import notch_temporal
out = notch_temporal(stack, fs_hz=30, freq_hz=10.0, Q=30)  # surgical mains removal
```

### Streaming file-based wrapper

For datasets too large to hold as float32 in RAM, ``correct_stack_file``
runs the same recipe on a TIFF input in two or three streaming passes
(depending on whether ``regress_common_mode`` is in the recipe), writing
to a sibling ``<stem>_Ncorrected.tif``. Peak memory is bounded by the
chunk size (default ~0.5 GB at 500 frames) regardless of stack length.

```python
from utilities.noise.noise_correction import correct_stack_file

# From a diagnostic report:
out_path = correct_stack_file(
    "/data/.../session.tif",
    report=rep,                  # rep from run_diagnostics
    chunk_frames=500,            # ~0.5 GB peak per chunk
    out_dtype="same",            # preserve uint16; warns if clipped
)

# Or with an explicit recipe:
from utilities.noise.noise_correction import (correct_bidirectional,
                                                subtract_row_pedestal,
                                                regress_common_mode)
out_path = correct_stack_file(
    "/data/.../session.tif",
    ops=[(correct_bidirectional, {"shift_px": -0.5}),
         (subtract_row_pedestal, {"mode": "temporal_median"}),
         (regress_common_mode, {})],
    out_dtype="float32",
)
```

Pass structure:
- **Pass 1**: stream raw input, accumulate per-frame stats (frame means,
  row means) and per-pixel stats (temporal mean, variance) needed by the
  recipe. Derive: bidirectional shift, row-pedestal offsets, hot-pixel mask,
  centred common-mode trace `c`.
- **Pass 2** (only with `regress_common_mode`): stream again to accumulate
  per-pixel `x · c`; divide by `c · c` → per-pixel β.
- **Pass 3**: stream and apply all corrections to each chunk using the
  cached state, write to BigTIFF via atomic-rename through `.tmp`.

The output is bit-stable for a given input + recipe (statistics derived
solely from the raw input). It differs from the in-memory chain by ~1 %
relative (median |diff| ≈ 0.04 DN on a uint16 source) because in-memory
re-derives statistics from each step's partially-corrected output —
neither is strictly "more correct"; the streamed version is more
deterministic.

`notch_temporal` is not supported in the streamed path (`filtfilt` has
acausal lookback over the full temporal axis). Pre-apply it in-memory if
required, or use `regress_common_mode` (the default for
`periodic_temporal_global`) which handles all globally-coherent
oscillations regardless of frequency.

### Signal-level validation

On a synthetic 1024×256×256 stack with 30 cells and four injected artifacts
(0.6 px bidir shift, 4-DN banding at 0.10 cyc/px, 3 hot pixels at +200 DN,
5-DN 10 Hz mains-alias line), the auto-built recipe restores all four metrics
to near-reference:

| metric                        | pre        | post       | reduction |
|-------------------------------|-----------:|-----------:|----------:|
| slow-axis PSD at banding freq | 5.3e+04    | 1.1e+01    | 37 dB     |
| adjacent-row \|diff\|         | 1.75       | 0.62       | back below ref 0.68 |
| hot-pixel z-score (mean)      | 53.6       | -0.0       | back to noise floor |
| frame-mean PSD at 10 Hz       | 1.4e+06    | 4.5e-04    | 95 dB     |

### Caveats

- The diagnostic's `test_hot_pixels` uses an absolute-variance threshold and
  may miss hot pixels on bright FOVs where shot noise raises the variance
  floor. The correction module's `detect_hot_pixels` uses variance/mean (the
  Poisson gain estimate) and is more reliable. Pass `stack=...` to
  `recommend_corrections` to use it.
- `correct_bidirectional` runs `scipy.ndimage.shift(order=1)` over the entire
  stack — O(T·H·W). On a 14 GB recording that's a few seconds; chunked
  variants for multi-session batch processing are a follow-up.
- `notch_temporal` is opt-in. The default for `periodic_temporal_global` is
  `regress_common_mode`, which removes whatever is globally coherent without
  notching biology. Use notch when you have explicit confidence in a frequency
  (e.g. a known 50 Hz mains line aliasing to 10 Hz at 30 fps).
