# caiman/utils/oscillation.py — Technical Reference

**Module:** `caiman/utils/oscillation.py`  
**Context:** Two-photon calcium imaging — CaImAn pipeline output  
**Sampling rate target:** 30 Hz (usable range 0–15 Hz)  
**Backend:** NumPy (CPU) or CuPy (GPU), auto-detected

---

## Contents

1. [Overview](#1-overview)
2. [Installation & Dependencies](#2-installation--dependencies)
3. [Quick Start](#3-quick-start)
4. [Signals — What Gets Analysed](#4-signals--what-gets-analysed)
5. [Spectral Theory](#5-spectral-theory)
6. [Frequency Bands](#6-frequency-bands)
7. [API Reference — OscillationAnalyzer](#7-api-reference--oscillationanalyzer)
8. [NPZ File Layout](#8-npz-file-layout)
9. [load_npz Helper](#9-load_npz-helper)
10. [CLI Reference](#10-cli-reference)
11. [Parameter Selection Guide](#11-parameter-selection-guide)
12. [GPU Acceleration](#12-gpu-acceleration)
13. [Implementation Notes](#13-implementation-notes)
14. [Known Limitations & Caveats](#14-known-limitations--caveats)

---

## 1. Overview

`caiman/utils/oscillation.py` computes oscillatory network activity from CaImAn pipeline output using **Thomson's multitaper method** with **discrete prolate spheroidal sequences (DPSS / Slepian functions)** throughout. It produces three complementary measures:

- **Welch-style multitaper PSD** — dominant frequencies over the whole session, with adaptive taper weighting to minimise spectral leakage
- **Sliding-window multitaper spectrogram** — time-resolved power, showing when oscillatory epochs occur (e.g. UP/DOWN state transitions, anaesthetic depth changes)
- **Hilbert band-power time series** — instantaneous power in defined frequency bands via FFT-domain analytic signal construction, suitable for correlation with behaviour or stimuli

The **Thomson F-test** is applied at each frequency bin to distinguish genuine narrowband oscillations (sinusoidal components) from broadband coloured noise.

Two orthogonal performance layers are applied:

- **Batched FFT** (always active): all spectrogram frames are extracted and processed in a single 3-D `rfft` call, eliminating the Python loop and enabling NumPy's internal multi-threading
- **CuPy GPU acceleration** (when available): the same code paths run on GPU via the drop-in CuPy array API

---

## 2. Installation & Dependencies

### Required

```
numpy >= 1.24
scipy >= 1.10        # signal.windows.dpss, stats.f
matplotlib >= 3.7
```

### Optional — GPU acceleration

```bash
# Match your CUDA version (check with: nvidia-smi)
pip install cupy-cuda12x     # CUDA 12.x
pip install cupy-cuda11x     # CUDA 11.x
# or via conda:
conda install -c conda-forge cupy
```

If CuPy is not installed or fails to initialise, the module falls back to NumPy silently with a `WARNING` log message.

### Optional — PC1 population signal

```
scikit-learn    # only needed if pop_method='pc1'
```

### For CLI use

```
h5py            # to read CaImAn .hdf5 result files
```

---

## 3. Quick Start

### From a live CaImAn session

```python
from caiman.utils.oscillation import OscillationAnalyzer

# estimates is a CaImAn Estimates object with .f, .F_dff/.C, .S
osc = OscillationAnalyzer(estimates, fs=30.0, NW=4, use_gpu=True)

# Compute everything, save .npz + figures
osc.run_all(
    output_dir="./osc_output",
    session_id="stroh-sa-2966-20251222",
    win_s=4.0,       # spectrogram window (s)
    overlap_s=2.0,   # 50% overlap (default)
)
```

### From a saved CaImAn HDF5 file (CLI)

```bash
python -m caiman.utils.oscillation results.hdf5 \
    --fs 30 --NW 4 --win 4 --overlap 2 \
    --session stroh-sa-2966-20251222 \
    --outdir ./osc_output \
    --gpu
```

### Reload results without re-running CaImAn

```python
from caiman.utils.oscillation import load_npz

data = load_npz("./osc_output/stroh-sa-2966-20251222_oscillations.npz")

# Spectrogram
t   = data["background_sg_t"]       # (W,) window-centre timestamps, seconds
f   = data["background_sg_freqs"]   # (F,) Hz
Sdb = data["background_sg_Sxx_db"]  # (F, W) dB

# Whole-session PSD
f_psd = data["background_psd_freqs"]
psd   = data["background_psd"]

# Instantaneous delta-band power
delta = data["background_bp_delta"]  # (T,) same length as raw signal
t_raw = data["background_t_ax"]
```

---

## 4. Signals — What Gets Analysed

Three 1-D time series are extracted from the CaImAn `Estimates` object. The module uses whichever are available; at minimum `estimates.f` or `estimates.F_dff`/`estimates.C` must be present.

### `background` — `estimates.f` (neuropil temporal components)

The background temporal components from CaImAn's neuropil model. These are **not convolved with the GCaMP indicator kinetics** and therefore preserve signal content up to the Nyquist frequency (15 Hz at 30 Hz sampling). They reflect a mixture of:

- Haemodynamic fluctuations (vascular dilation/constriction)
- Scattered light from neuropil
- Slow global network oscillations

**This is the preferred channel for 5–15 Hz analysis.** If `estimates.f` has multiple components, they are averaged to a single trace.

### `neural_pop` — `estimates.F_dff` or `estimates.C`

The population signal derived from the accepted neural components. `F_dff` (dF/F) is used if available; otherwise the denoised calcium trace `C` is used. Multiple cells are reduced to a single trace via the `pop_method` parameter.

**Important bandwidth caveat:** GCaMP6/7/8 has a decay time constant τ ≈ 400 ms–1 s. The indicator acts as a low-pass filter with a −3 dB point near 1–3 Hz. Calcium-derived oscillatory content is:

- Reliable: 0.01–3 Hz (slow oscillations, UP/DOWN state envelope)  
- Attenuated and phase-shifted: 3–8 Hz  
- Unreliable: >8 Hz (use `background` instead)

### `spike_rate` — `estimates.S`

The mean deconvolved spike rate across all accepted cells. Less attenuated than dF/F for fast events, but the deconvolution algorithm itself introduces assumptions. Useful for >3 Hz analysis if deconvolution quality is high.

### Population signal reduction (`pop_method`)

| Method | Formula | Use when |
|--------|---------|----------|
| `'mean'` | Simple mean across cells | Default; interpretable |
| `'pc1'` | First principal component | Cells have very different amplitudes |
| `'norm'` | L2-normalised mean | Robustness to outlier cells |

---

## 5. Spectral Theory

### Why multitaper?

A single windowed periodogram (e.g. Hann-windowed FFT) suffers from a fundamental trade-off: reducing spectral leakage requires increasing the taper width, which reduces frequency resolution. More importantly, a single periodogram has very high variance — the estimate at each frequency bin has a chi-squared distribution with 2 degrees of freedom regardless of how long the signal is.

**Thomson's multitaper method** (Thomson 1982) resolves this by averaging K orthogonal eigenspectra computed from K DPSS tapers. The result:

- Variance reduced by a factor of K
- Leakage controlled by the half-bandwidth W = NW / (N · Δt)
- No additional frequency resolution loss beyond the concentration guarantee

### DPSS (Slepian) tapers

The Discrete Prolate Spheroidal Sequences are the optimal set of tapers for concentrating spectral energy within a half-bandwidth W while being mutually orthogonal. They are parameterised by:

- **N** — signal length in samples
- **NW** — time-bandwidth product (half-bandwidth in units of 1/N)

The number of usable tapers is **K = 2·NW − 1**. The first K tapers have spectral concentration ratios close to 1; higher-order tapers begin to leak.

### Adaptive taper weighting

By default, the module uses adaptive weighting (Percival & Walden §7.7). Instead of averaging all K eigenspectra equally, each taper's contribution at each frequency is weighted by its estimated signal-to-leakage ratio. This is beneficial when the spectrum has a large dynamic range (e.g. a strong slow oscillation peak many orders of magnitude above the high-frequency floor), because low-order tapers with high concentration receive more weight in frequency regions where leakage would otherwise be significant.

### Thomson F-test for spectral lines

The F-test fits a complex sinusoid at each frequency bin and computes the F(2, 2K−2) statistic comparing the variance explained by the sinusoid against the residual across-taper power. A significant result (p < threshold) indicates a genuine narrowband oscillation rather than a broad spectral peak from coloured noise.

The test is applied to the **background** signal in `plot_psd` and overlaid as green dashed vertical lines. Full F-statistic and p-value arrays are saved in the `.npz`.

### One-sided PSD correction

For real signals, the two-sided power spectrum is folded to a one-sided spectrum by doubling all bins except DC (bin 0) and Nyquist (bin N/2 for even N). This doubling is applied to the **full rfft output before frequency clipping** — not after — to ensure the last kept bin is handled correctly when `fmax` is less than the Nyquist frequency.

---

## 6. Frequency Bands

The following bands are defined in the module-level `BANDS` dictionary and apply to mouse cortex two-photon calcium imaging context:

| Band | Range (Hz) | Physiological correlate | Best signal source |
|------|-----------|------------------------|--------------------|
| `infra-slow` | 0.01–0.1 | Haemodynamic drift, BOLD-like fluctuations | background |
| `slow` | 0.1–1.0 | Slow cortical oscillations, cortical up-states | background, neural_pop |
| `delta` | 1.0–4.0 | Delta / UP–DOWN state alternation | background, neural_pop |
| `theta` | 4.0–8.0 | Theta rhythm | background |
| `alpha-beta` | 8.0–15.0 | Alpha / low beta (near Nyquist at 30 Hz) | background only |

To modify bands, edit the `BANDS` dict directly before instantiating `OscillationAnalyzer`.

Band power is computed via an FFT-domain analytic signal (equivalent to Hilbert transform of the bandpass-filtered signal). The Hilbert envelope squared gives instantaneous power.

---

## 7. API Reference — OscillationAnalyzer

### Constructor

```python
OscillationAnalyzer(
    estimates,
    fs: float = 30.0,
    fmax: float = 15.0,
    NW: float = 4.0,
    adaptive: bool = True,
    pop_method: str = "mean",
    use_gpu: bool = True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimates` | CaImAn `Estimates` | — | Must have at least one of `.f`, `.F_dff`, `.C`. `.S` is optional. |
| `fs` | float | 30.0 | Sampling rate in Hz |
| `fmax` | float | 15.0 | Upper frequency limit. For 30 Hz sampling, max is 15 Hz (Nyquist). |
| `NW` | float | 4.0 | DPSS time-bandwidth product. Controls leakage vs resolution trade-off. See §11 for guidance. |
| `adaptive` | bool | True | Use adaptive taper weighting. Recommended when spectrum has large dynamic range. Set `False` for equal-weight averaging (faster, less accurate). |
| `pop_method` | str | `'mean'` | How to reduce the (n_cells, T) neural matrix to a 1-D signal. Options: `'mean'`, `'pc1'`, `'norm'`. |
| `use_gpu` | bool | True | Try CuPy GPU acceleration. Falls back to NumPy silently if CuPy is unavailable. |

**Instance attributes (after construction):**

| Attribute | Description |
|-----------|-------------|
| `self.bg` | Background signal, shape (T,), float64. `None` if `estimates.f` absent. |
| `self.neural_pop` | Population signal, shape (T,). `None` if no neural traces. |
| `self.spike_rate` | Mean spike rate, shape (T,). `None` if `estimates.S` absent. |
| `self.t_ax` | Time axis in seconds, shape (T,). `arange(T) / fs`. |
| `self.fs` | Sampling rate (Hz) |
| `self.NW` | Time-bandwidth product |
| `self.K` | Number of tapers = `2·NW − 1` |
| `self.fmax` | Upper frequency limit (Hz) |
| `self.adaptive` | Whether adaptive weighting is active |
| `self.use_gpu` | Whether CuPy GPU is active |
| `self.xp` | Array module in use (`numpy` or `cupy`) |

---

### `run_all` — Standard batch output

```python
osc.run_all(
    output_dir: str = ".",
    session_id: str = "session",
    dpi: int = 150,
    win_s: float = 4.0,
    overlap_s: float = None,
    p_threshold: float = 0.01,
) -> dict
```

The primary entry point. Runs all analyses in order, saves numerical data to `.npz` first (so figures can be regenerated without recomputation), then generates all figures. Spectrograms are rendered from the saved `.npz` rather than being recomputed.

**Output files written to `output_dir/`:**

| File | Contents |
|------|---------|
| `{session_id}_oscillations.npz` | All numerical arrays — see §8 |
| `{session_id}_mt_psd.png` | Multitaper PSD for all signals with F-test line overlay |
| `{session_id}_f_test_background.png` | F-statistic and p-value spectra for background |
| `{session_id}_f_test_neural_pop.png` | F-statistic and p-value spectra for neural population |
| `{session_id}_mt_spectrogram_background.png` | Sliding-window spectrogram, background signal |
| `{session_id}_mt_spectrogram_neural_pop.png` | Sliding-window spectrogram, neural population |
| `{session_id}_bandpower_background.png` | Stacked band-power time series, background |
| `{session_id}_bandpower_neural_pop.png` | Stacked band-power time series, neural population |

**Returns:** `dict` — `{signal_key: {band_name: mean_power}}` summary.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `win_s` | 4.0 | Spectrogram window length in seconds. Sets frequency resolution: Δf ≈ 2·NW / win_s |
| `overlap_s` | `None` | Window overlap in seconds. `None` → win_s/2 (50%). Must be in [0, win_s). Step = win_s − overlap_s. |
| `p_threshold` | 0.01 | F-test significance threshold for spectral line detection |
| `dpi` | 150 | Figure output resolution |

---

### `compute_psd` — Whole-session PSD

```python
osc.compute_psd() -> dict[str, tuple[np.ndarray, np.ndarray]]
```

Returns `{signal_key: (freqs, psd)}` for each available signal. `freqs` and `psd` are 1-D CPU numpy arrays. Units of `psd` are signal_units²/Hz.

---

### `plot_psd` — PSD figure

```python
osc.plot_psd(
    ax=None,
    show_bands: bool = True,
    show_f_lines: bool = True,
    p_threshold: float = 0.01,
) -> plt.Figure
```

Plots log-scale PSD for all available signals on a single axes. Band regions are shaded. Significant F-test spectral lines from the background signal are overlaid as green dashed verticals.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ax` | `None` | Existing Axes to plot into. If `None`, a new Figure is created. |
| `show_bands` | True | Shade frequency band regions |
| `show_f_lines` | True | Overlay significant spectral lines from background F-test |
| `p_threshold` | 0.01 | F-test threshold for line display |

---

### `plot_f_test` — F-test detail figure

```python
osc.plot_f_test(
    signal_key: str = "background",
    p_threshold: float = 0.01,
) -> plt.Figure
```

Two-panel figure showing the Thomson F-statistic (top) and p-value (bottom) across frequency. The critical value line and significant frequencies (green verticals) are shown. Also prints significant frequencies to stdout.

**Parameters:**

| Parameter | Options | Description |
|-----------|---------|-------------|
| `signal_key` | `'background'`, `'neural_pop'`, `'spike_rate'` | Which signal to test |
| `p_threshold` | 0.01 | Significance threshold |

---

### `plot_spectrogram` — Sliding-window multitaper spectrogram

```python
osc.plot_spectrogram(
    signal_key: str = "background",
    win_s: float = 4.0,
    overlap_s: float = None,
    overlay_signal: bool = True,
    db_range: float = 40.0,
) -> plt.Figure
```

Computes and plots the multitaper spectrogram. Uses the batched FFT implementation (and GPU if enabled). The spectrogram is rendered with an `inferno` colourmap clipped to ±`db_range/2` dB around the median power.

If `overlay_signal=True`, the z-scored raw signal is shown in the upper panel.

**Timestamp convention:** The time axis represents **window centres**, i.e. `t[i] = (start_frame_i + win_n/2) / fs`. The first timestamp is therefore `win_s/2` seconds into the recording.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `signal_key` | `'background'` | Signal to analyse |
| `win_s` | 4.0 | Window length (s). Frequency resolution ≈ 2·NW / win_s |
| `overlap_s` | `None` | Overlap (s). None → win_s/2. Step = win_s − overlap_s |
| `overlay_signal` | True | Show z-scored raw signal in upper panel |
| `db_range` | 40.0 | Colour dynamic range in dB |

---

### `compute_band_power` — Band-power time series

```python
osc.compute_band_power(
    signal_key: str = "background",
) -> dict[str, np.ndarray]
```

Returns `{band_name: instantaneous_power_array}` for each band in `BANDS`. Power arrays have shape (T,) — same length as the raw signal. Computed via FFT-domain analytic signal construction (equivalent to Hilbert transform of the bandpass-filtered signal).

---

### `plot_band_power` — Stacked band-power figure

```python
osc.plot_band_power(
    signal_key: str = "background",
    smooth_s: float = 1.0,
) -> plt.Figure
```

Stacked time series showing instantaneous Hilbert envelope power for each frequency band.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `signal_key` | `'background'` | Signal to analyse |
| `smooth_s` | 1.0 | Gaussian smoothing kernel width in seconds (σ). Set to 0 to disable. |

---

### `band_power_summary` — Scalar summary per band

```python
osc.band_power_summary() -> dict
```

Returns `{signal_key: {band_name: mean_power}}` for all available signals and all bands. Mean is taken over the full session. Useful for logging or cross-session comparisons.

---

### `dominant_frequency` — Peak frequency

```python
osc.dominant_frequency(
    signal_key: str = "background",
    band: tuple = None,
) -> float
```

Returns the frequency (Hz) of peak PSD power. If `band=(flo, fhi)` is provided, the search is restricted to that frequency range.

---

### `save_data` — Save numerical arrays

```python
osc.save_data(
    output_dir: str = ".",
    session_id: str = "session",
    win_s: float = 4.0,
    overlap_s: float = None,
    p_threshold: float = 0.01,
) -> Path
```

Computes all analyses and saves results to a compressed `.npz` file. Returns the `Path` to the saved file. All GPU arrays are transferred back to CPU numpy before saving.

This is called automatically by `run_all`; call it directly if you only want the numerical data without figures.

---

## 8. NPZ File Layout

All arrays in the `.npz` are CPU numpy. `sig` is one of `background`, `neural_pop`, `spike_rate`.

### Metadata (0-d arrays)

| Key | dtype | Description |
|-----|-------|-------------|
| `meta_fs` | float64 | Sampling rate (Hz) |
| `meta_NW` | float64 | DPSS time-bandwidth product |
| `meta_K` | int32 | Number of tapers |
| `meta_adaptive` | bool | Adaptive weighting active |
| `meta_fmax` | float64 | Upper frequency limit (Hz) |
| `meta_win_s` | float64 | Spectrogram window length (s) |
| `meta_overlap_s` | float64 | Spectrogram overlap (s) |
| `meta_p_threshold` | float64 | F-test significance threshold |
| `meta_gpu` | bool | GPU was used for computation |

### Per-signal arrays

**Raw signal and time axis:**

| Key | Shape | Description |
|-----|-------|-------------|
| `{sig}_t_ax` | (T,) | Time axis in seconds: `arange(T) / fs` |
| `{sig}_raw` | (T,) | The 1-D signal itself |

**Whole-session PSD:**

| Key | Shape | Description |
|-----|-------|-------------|
| `{sig}_psd_freqs` | (F,) | Frequency axis in Hz |
| `{sig}_psd` | (F,) | Power spectral density (units²/Hz) |

**Thomson F-test:**

| Key | Shape | Description |
|-----|-------|-------------|
| `{sig}_ftest_freqs` | (F,) | Frequency axis in Hz |
| `{sig}_ftest_fstat` | (F,) | F-statistic at each frequency |
| `{sig}_ftest_pval` | (F,) | p-value at each frequency |
| `{sig}_ftest_sigf` | (M,) | Frequencies of significant spectral lines (Hz) |

**Multitaper spectrogram:**

| Key | Shape | Description |
|-----|-------|-------------|
| `{sig}_sg_freqs` | (F,) | Frequency axis in Hz |
| `{sig}_sg_t` | (W,) | **Window-centre timestamps** in seconds (half-window shifted) |
| `{sig}_sg_Sxx` | (F, W) | Power in units²/Hz — linear scale |
| `{sig}_sg_Sxx_db` | (F, W) | Power in dB: `10·log10(Sxx + 1e-30)` |

**Spectrogram timestamp note:** `sg_t[i] = (start_frame_i + win_n/2) / fs`. The first timestamp equals `meta_win_s / 2` seconds into the recording. All timestamps point to the midpoint of the analysis window, not the leading edge.

**Band power (Hilbert envelope, same time axis as raw signal):**

| Key | Shape | Description |
|-----|-------|-------------|
| `{sig}_bp_t` | (T,) | Time axis — identical to `{sig}_t_ax` |
| `{sig}_bp_infra_slow` | (T,) | Instantaneous power, 0.01–0.1 Hz |
| `{sig}_bp_slow` | (T,) | Instantaneous power, 0.1–1.0 Hz |
| `{sig}_bp_delta` | (T,) | Instantaneous power, 1.0–4.0 Hz |
| `{sig}_bp_theta` | (T,) | Instantaneous power, 4.0–8.0 Hz |
| `{sig}_bp_alpha_beta` | (T,) | Instantaneous power, 8.0–15.0 Hz |

---

## 9. `load_npz` Helper

```python
from caiman.utils.oscillation import load_npz

data = load_npz("path/to/session_oscillations.npz")
# Returns a plain dict; all values are numpy arrays
```

Returns a plain Python `dict` by materialising the `NpzFile` object. No CaImAn import required. Convenient for replotting or downstream statistics without re-running the pipeline.

**Example — replot spectrogram from saved data:**

```python
import matplotlib.pyplot as plt
import numpy as np
from caiman.utils.oscillation import load_npz

data = load_npz("./osc_output/stroh-sa-2966-20251222_oscillations.npz")

fig, ax = plt.subplots(figsize=(13, 4))
ax.pcolormesh(
    data["background_sg_t"],
    data["background_sg_freqs"],
    data["background_sg_Sxx_db"],
    cmap="inferno", shading="gouraud",
)
ax.set_xlabel("Time (s) — window centres (half-window shifted)")
ax.set_ylabel("Frequency (Hz)")
plt.colorbar(ax.collections[0], label="dB")
plt.tight_layout()
```

---

## 10. CLI Reference

```
python -m caiman.utils.oscillation <hdf5> [options]
```

Reads a CaImAn `.hdf5` result file and runs `run_all`.

**Positional:**

| Argument | Description |
|----------|-------------|
| `hdf5` | Path to CaImAn `.hdf5` results file |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--fs` | 30.0 | Sampling rate in Hz |
| `--fmax` | 15.0 | Upper frequency limit (Hz) |
| `--NW` | 4.0 | DPSS time-bandwidth product |
| `--no-adaptive` | (off) | Use equal-weight taper averaging instead of adaptive |
| `--gpu` | (on) | Use CuPy GPU (default) |
| `--no-gpu` | (off) | Force NumPy CPU |
| `--win` | 4.0 | Spectrogram window length (s) |
| `--overlap` | None | Spectrogram overlap (s); None = win/2 (50%) |
| `--outdir` | `./osc_output` | Output directory |
| `--session` | `session` | Session ID string used in filenames |
| `--method` | `mean` | Population signal method: `mean`, `pc1`, `norm` |

**Examples:**

```bash
# Default run — GPU, NW=4, 50% overlap, 4 s windows
python -m caiman.utils.oscillation results.hdf5 --session my_session

# High time resolution — 75% overlap, 2 s windows
python -m caiman.utils.oscillation results.hdf5 --win 2 --overlap 1.5

# Fine frequency resolution — NW=2 (fewer tapers, higher variance)
python -m caiman.utils.oscillation results.hdf5 --NW 2

# Force CPU for debugging
python -m caiman.utils.oscillation results.hdf5 --no-gpu
```

---

## 11. Parameter Selection Guide

### Time-bandwidth product `NW`

| NW | K tapers | Half-bandwidth at 30 Hz / 924 s | Use when |
|----|----------|----------------------------------|----------|
| 2 | 3 | ≈ 0.065 Hz | Resolving closely spaced slow oscillations (< 1 Hz separation) |
| 3 | 5 | ≈ 0.097 Hz | Good compromise |
| **4** | **7** | **≈ 0.130 Hz** | **Default — lowest variance, appropriate for most sessions** |

The half-bandwidth W = NW / (N · Δt) where N is the signal length. For a 924 s recording (27,720 frames at 30 Hz), NW=4 gives W ≈ 0.13 Hz — adequate to separate infra-slow, slow, and delta bands.

Increasing NW reduces variance (more tapers) but widens the spectral smoothing kernel. For detecting a sharp UP/DOWN state peak at 0.5 Hz, NW=2 or 3 gives better frequency precision. For characterising broad band power, NW=4 is preferred.

### Spectrogram window length `win_s`

The window length controls the frequency resolution of each time-slice: **Δf ≈ 2·NW / win_s**.

| win_s | Δf (NW=4) | Time resolution | Good for |
|-------|-----------|-----------------|----------|
| 2 s | 4.0 Hz | step_s seconds | Detecting rapid changes; coarse frequency |
| **4 s** | **2.0 Hz** | step_s seconds | **Default — good for UP/DOWN states** |
| 8 s | 1.0 Hz | step_s seconds | Resolving slow vs delta bands cleanly |
| 16 s | 0.5 Hz | step_s seconds | Infra-slow and haemodynamic characterisation |

### Overlap `overlap_s`

Controls the time resolution of the spectrogram: step = win_s − overlap_s.

| overlap_s | Step | Use when |
|-----------|------|----------|
| `None` (default) | win_s / 2 | Standard; 50% overlap gives good temporal smoothing |
| `0` | win_s | Non-overlapping; maximum speed |
| `win_s × 0.75` | win_s / 4 | Smooth time series; longer computation |

Higher overlap does **not** improve statistical independence of frames — adjacent frames share most of their data. It does improve visual continuity of the spectrogram.

---

## 12. GPU Acceleration

### Backend detection

At construction, `_get_xp(use_gpu=True)` attempts to import CuPy, allocates a small test array, and logs the device name and free memory. If this fails for any reason, NumPy is used silently.

### What runs on GPU

All computationally intensive array operations:

- DPSS taper upload
- Signal upload
- Frame extraction via advanced indexing
- Batched rfft (spectrogram) and rfft (PSD, F-test)
- Power computation and taper averaging
- FFT-domain bandpass + analytic signal (band power)
- Adaptive weight iteration

### What stays on CPU

- DPSS taper computation (scipy, always CPU)
- F-distribution CDF for F-test p-values (scipy, CPU)
- All plotting (matplotlib, CPU)
- `.npz` file I/O

### Memory estimate

For a 924 s × 30 Hz recording (27,720 frames):

| Object | Shape | Size |
|--------|-------|------|
| Signal upload (float64) | (27,720,) | ~0.2 MB |
| Spectrogram frames (4 s win, 50% ov) | (13,849, 120) | ~13 MB |
| Tapered frames | (13,849, 7, 120) | ~88 MB |
| rfft output (complex128) | (13,849, 7, 61) | ~90 MB |
| **Total GPU peak** | | **~200 MB** |

Well within the 16 GB GPU budget of the lab workstation.

### Expected speedup

Benchmarks on RTX 3090 class hardware vs CPU loop:

| Operation | CPU loop | Batched CPU | GPU |
|-----------|---------|-------------|-----|
| Spectrogram (4 s / 50%) | ~15 s | ~2 s | ~0.5 s |
| PSD | ~0.1 s | ~0.1 s | ~0.05 s |
| Band power (5 bands) | ~0.5 s | ~0.5 s | ~0.1 s |
| **Full `run_all`** | **~90 s** | **~20 s** | **~8 s** |

Note: plotting time (~10 s) dominates at the GPU end and is CPU-bound.

---

## 13. Implementation Notes

### Batched spectrogram FFT

All spectrogram frames are extracted in a single operation via advanced indexing (no Python loop):

```python
starts = xp.arange(0, N - win_n + 1, step_n)          # (n_frames,)
idx    = starts[:, None] + xp.arange(win_n)[None, :]   # (n_frames, win_n)
frames = x_xp[idx]                                      # (n_frames, win_n)
```

Taper application and FFT are then batched over all frames simultaneously:

```python
xk = frames[:, None, :] * tapers[None, :, :]            # (n_frames, K, win_n)
Xk = xp.fft.rfft(xk, n=win_n, axis=-1)                 # (n_frames, K, M)
```

### FFT-domain analytic signal (band power)

Rather than using `scipy.signal.filtfilt` + `scipy.signal.hilbert` (which don't run on CuPy), band power is computed entirely in the frequency domain:

1. Full complex FFT: `X = fft(x)`
2. Build analytic mask: positive bins in `[flo, fhi]` set to 2.0; all others remain 0
3. Multiply and inverse FFT: `z = ifft(X * mask)`
4. Instantaneous power: `|z|²`

This is mathematically equivalent to `filtfilt` + `hilbert` but avoids Butterworth edge effects on long signals and runs natively on GPU.

### One-sided correction before clipping

The one-sided PSD correction (×2 for all bins except DC and Nyquist) is applied to the **full rfft output** before frequency clipping to `fmax`. Applying it after clipping would fail to double the last kept bin whenever `fmax < fs/2`.

### F-test integer slice

The frequency clipping in `_f_test_lines` uses an integer count `M_clip = int(f_mask.sum())` rather than a boolean mask to slice the GPU array. This avoids an implicit CPU→GPU boolean mask transfer that CuPy handles but which is fragile and undocumented behaviour.

---

## 14. Known Limitations & Caveats

### GCaMP bandwidth

GCaMP6/7/8 calcium indicators have decay τ ≈ 400 ms–1 s, acting as a single-pole low-pass filter. The −3 dB point is approximately:

- GCaMP6f: ~3 Hz
- GCaMP7f: ~2 Hz  
- GCaMP8f: ~4 Hz

Neural traces (dF/F, denoised C) above ~5 Hz are attenuated and phase-shifted relative to the true firing. **Use `background` (neuropil temporal components) for reliable 5–15 Hz analysis.**

### Minimum signal length

The multitaper PSD requires at least `2·NW` samples to have any useful spectral estimate. In practice, the entire session is used for the PSD. For the spectrogram, the window length `win_n = int(win_s * fs)` must satisfy `win_n >= 2·K` where K = 2·NW − 1.

### Deconvolution quality for spike_rate

The `spike_rate` signal is only meaningful for oscillation detection if the deconvolution (OASIS, FOOPSI, etc.) correctly captures sub-second spike timing. For slow oscillations (< 1 Hz), this is generally adequate. For theta/alpha-beta, interpret with caution.

### Adaptive weight convergence

The adaptive weighting iteration runs up to 150 iterations with tolerance 1e-6. For signals with extreme spectral dynamic range (>60 dB), convergence may be slow. In pathological cases (max_iter=0 in testing, or instant convergence), `psd_new` is initialised to the equal-weight estimate as a fallback.

### Nyquist for alpha-beta band

The `alpha-beta` band (8–15 Hz) extends to the Nyquist at 30 Hz sampling. Use `estimates.f` (background) rather than `F_dff` or `C` for this band, as the indicator kinetics attenuate calcium signals well before 8 Hz.

### Cross-session comparisons

Band power values in the `.npz` are in raw units (the square of whatever units your fluorescence data is in). For cross-session comparisons, normalise by baseline or use the ratio of band power to total power in the `psd` arrays.
