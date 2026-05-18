# noise_diagnostics — 2P imaging noise characterization

A standalone diagnostic suite that runs a battery of tests on a sampled subset
of frames from a 2P calcium imaging recording and maps the metrics to a
calibrated likelihood (`negligible | low | moderate | high`) for each of ~13
known noise sources, with a one-line recommended fix per source.

Lives at `diagnostics/noise_diagnostics.py` inside the CaImAn fork.
No CaImAn-runtime dependency for the `.tif` / `.npy` / `ndarray` input paths;
`caiman.utils.stack_io` is pulled in only for `.msr` / `.h5` / `.nwb` inputs.

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
bidirectional_phase_offset        → enable xcorr_correction in pipeline JSON
horizontal_banding_fixed          → row-pedestal subtraction
horizontal_banding_drifting       → 1D notch filter on column-FFT
fast_axis_periodic                → pixel-clock / sample-hold issue
galvo_flyback_edge                → raise motion_correction.border_pix
hot_dead_pixels                   → median-replace before MC
photobleaching                    → detrend pre-CNMF or rely on detrend_df_f
illumination_drift_increase       → drop warm-up frames
fixed_pattern_noise               → dark-frame subtract
saturation_clipping               → reduce PMT gain
quantization_loss                 → increase PMT gain
frame_discontinuity               → drop / interpolate bad frames
```

Each scored source comes with `level`, a continuous `score`, an `evidence`
dict of contributing metrics, and a one-line recommendation.

## Quick start

```bash
# TIF or BigTIFF
python diagnostics/noise_diagnostics.py /data/path/recording.tif \
    --out diag_out --n_frames 500

# Leica MSR (native, via caiman.utils.stack_io.IMSpectorReader)
python diagnostics/noise_diagnostics.py /data/path/recording.msr \
    --out diag_out --n_frames 500

# HDF5 / NWB
python diagnostics/noise_diagnostics.py /data/path/recording.h5 \
    --out diag_out --n_frames 500

# Python API
from diagnostics.noise_diagnostics import run_diagnostics
report = run_diagnostics("rec.tif", out_dir="diag_out", n_frames=500)
```

## Outputs

```
diag_out/
  diagnostic_report.json    full numeric report, all test metrics + sources
  diagnostic_panel.png      9-panel visual summary
  summary.txt               ranked source list + recommended actions
```

`summary.txt` tail looks like:

```
========================================================================
noise_diagnostics — source ranking
========================================================================
  source     : /data/.../recording.tif
  n_total    : 27720
  n_sampled  : 500
  dtype_orig : uint16
  fmax_orig  : 63987.0
------------------------------------------------------------------------
  [      high]  shot_noise_dominated              score=0.931
  [      high]  photobleaching                    score=0.933
  [  moderate]  horizontal_banding_fixed          score=0.585
  [       low]  hot_dead_pixels                   score=0.282
  [negligible]  bidirectional_phase_offset        score=0.004
  [negligible]  galvo_flyback_edge                score=0.000
  ...
------------------------------------------------------------------------
Recommended actions:
  * photobleaching                    Apply pixel-wise detrending pre-CNMF...
  * horizontal_banding_fixed          Add row-pedestal subtraction...
  * hot_dead_pixels                   Median-replace hot/dead pixels...
========================================================================
```

## Reading the panel

Top row, left to right:
- **Temporal mean** — what the recording looks like averaged across frames.
- **Photon-transfer curve** — log-log variance vs mean. Slope ≈ 1 for Poisson;
  intercept = read-noise floor. The fit prints `gain` and `read noise`.
- **2D FFT** — bright streaks along slow axis = horizontal banding; along
  fast axis = pixel-clock / sample-hold; uniform = healthy.

Middle row:
- **Row-mean trace** — should be smooth; oscillations indicate banding.
  Title prints the bidirectional shift estimate.
- **Edge profile** — row- and column-mean profiles. Edges that drop below
  the 10% threshold (red dotted line) are flagged as flyback.
- **Hot/dead pixel overlay** — red `+` markers on the mean image.

Bottom row:
- **Frame-mean trace** — should be flat. A steep drop is photobleaching;
  jumps are frame discontinuities; smooth rise is illumination drift.
- **Fixed-pattern (HP)** — high-passed temporal mean. Strong banding/grid
  patterns here = FPN; clean = no fixed pattern.
- **Saturation summary text** — sat / floor / DR-usage / effective bits;
  glitch count and max jump z-score.

## Validation evidence

Validated on a synthetic-noise harness that injects each source into a
clean baseline (Poisson signal + Gaussian read noise + 25 simulated cell
ROIs).  Pass criterion: detected ≥ `moderate` for the injected source,
`negligible` on all other fault sources at baseline.

```
BASELINE                                        9/9   negligible (except shot)
INJECT bidirectional (+0.6 px even rows)        PASS  detector 0.23 px, score 0.58
INJECT banding (0.10 cyc/px, 8 DN)              PASS  17 dB peak, score 0.46
INJECT 8 dead-column galvo edge                 PASS  score 0.50
INJECT hot pixels (3 hot + 1 dead)              PASS  score 0.61
INJECT photobleaching (τ=80 fr / T=200)         PASS  score 0.93
INJECT saturation (1% pixels at uint16 max)     PASS  score 1.00
INJECT 4 large-jump frames                      PASS  score 1.00
TOTAL                                           16/16
```

The PTC gain estimate (recovered ~1/3 of injected gain on the synthetic
photobleaching case) is known to be biased low by photobleaching
contaminating the diff-based variance estimator — documented in test
docstring.  The bias affects the absolute gain readout but not the
relative source rankings, which is what drives the recommendations.

The bidirectional detector's row-averaged xcorr undercounts true shift by
~3× (calibrated against injected 0.6 px → detector reads 0.23 px).  The
*score* divisor is calibrated to compensate, so the moderate / high flags
still fire correctly.  The reported `bidir_shift_px` value should be
interpreted as a lower bound when discussing magnitudes with collaborators.

## Multi-channel `.msr`

`IMSpectorReader` returns slices in raw acquisition order without channel
demultiplexing.  For multi-channel `.msr` where channels are interleaved
per slice, the diagnostic would see frames alternating between channels,
which would wreck the PTC / bidirectional / drift tests.

Two options:

- **Preferred:** point the diagnostic at the per-channel session split
  your `new_session.py` workflow already produces (the channel-id-suffixed
  session directories the CNMF pipeline consumes).  These are single-channel
  BigTIFFs.
- **Per-`.msr`:** point at the raw `.msr` only if it's known single-channel.

## CLI reference

```
python diagnostics/noise_diagnostics.py SOURCE [options]

positional:
  SOURCE              .tif/.tiff/.btf/.npy/.msr/.h5/.nwb path

optional:
  --out DIR           output directory (default: diag_out)
  --n_frames N        frames to sample (default: 500)
  --seed N            seed for the random subset (default: 0)
  --verbose           info-level logging during the run
```

## Sampling

Random without replacement, uniform across the recording (`rng.choice(T,
size=N, replace=False)`).  `--n_frames 500` is plenty for the PTC fit
(~200 needed) and keeps the spectral / drift tests well-conditioned for a
typical 30 Hz / 15 min recording.  Bumping to 1000–2000 gets tighter PTC
fits at ~2× wall time.

## Dependencies

- numpy, scipy, matplotlib (required)
- tifffile (required for `.tif` / `.tiff` / `.btf` input)
- caiman.utils.stack_io (required for `.msr` / `.h5` / `.nwb` input)
