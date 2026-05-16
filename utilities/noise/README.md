# noise_diagnostics — 2P imaging noise characterization

A standalone diagnostic suite that runs a battery of tests on a sampled subset
of frames from a 2P calcium imaging recording and maps the metrics to a
calibrated likelihood (`negligible | low | moderate | high`) for each of ~13
known noise sources, with a one-line recommended fix per source.

## What it tests

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
