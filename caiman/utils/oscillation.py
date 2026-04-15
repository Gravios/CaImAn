"""
caiman/utils/oscillation.py
============================
Multitaper (DPSS / Slepian) oscillation analysis for CaImAn pipeline output.

Computes power spectra, time-frequency spectrograms, band-power time series,
and Thomson F-test spectral line detection from CaImAn ``Estimates`` objects.
All heavy computation uses Thomson's multitaper method throughout, with optional
GPU acceleration via CuPy (falls back to NumPy silently if CuPy is unavailable).

Public API
----------
OscillationAnalyzer(estimates, fs, NW, ...)
    Main analysis class.  Construct once, call ``run_all`` or individual
    ``compute_*`` / ``plot_*`` methods.

load_npz(path)
    Reload a previously saved ``.npz`` results file into a plain dict without
    requiring CaImAn or re-running the pipeline.

Typical usage::

    from caiman.utils.oscillation import OscillationAnalyzer

    osc = OscillationAnalyzer(cnm.estimates, fs=30.0, NW=4, use_gpu=True)
    osc.run_all(output_dir="./osc_output", session_id="stroh-sa-2966-20251222")

    # Reload later
    from caiman.utils.oscillation import load_npz
    data = load_npz("./osc_output/stroh-sa-2966-20251222_oscillations.npz")

GPU acceleration
----------------
Two independent layers:

* **Batched FFT** (always active): all spectrogram frames are extracted and
  processed in a single 3-D ``rfft`` call — no Python loop, uses NumPy's
  internal threading on CPU, a single kernel launch on GPU.

* **CuPy** (when available): DPSS eigenspectra, PSD, spectrogram, and Hilbert
  band-power all run on the GPU.  Results are pulled back to CPU numpy before
  saving or plotting.  Expected ~12× speedup for the spectrogram step on
  RTX 3090-class hardware.

See Also
--------
:doc:`oscillation_analysis`
    Full reference documentation including parameter selection guidance,
    NPZ file layout, and implementation notes.
"""


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import f as f_dist, zscore
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import logging
import time

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  GPU / CPU backend selection
# ═══════════════════════════════════════════════════════════════════════════

def _get_xp(use_gpu: bool = True):
    """
    Return the array module to use: cupy if available and requested,
    numpy otherwise.  Logs which backend was selected.
    """
    if use_gpu:
        try:
            import cupy as cp
            # Quick smoke-test — allocates a tiny array on the default device
            _ = cp.array([1.0])
            dev   = cp.cuda.Device()
            name  = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
            mem   = dev.mem_info[1] / 1e9
            log.info(f"GPU backend: CuPy — device {dev.id}: {name} "
                     f"({mem:.1f} GB free)")
            return cp
        except Exception as e:
            log.warning(f"CuPy not available ({e}); falling back to NumPy CPU")
    log.info("Array backend: NumPy (CPU)")
    return np


def _to_numpy(arr, xp) -> np.ndarray:
    """Move array to CPU numpy regardless of whether xp is cupy or numpy."""
    if xp is np:
        return np.asarray(arr)
    return xp.asnumpy(arr)


def _to_xp(arr: np.ndarray, xp):
    """Upload a numpy array to xp device (no-op if xp is numpy)."""
    if xp is np:
        return arr
    return xp.asarray(arr)


# ═══════════════════════════════════════════════════════════════════════════
#  DPSS tapers  (always computed on CPU via scipy, then optionally uploaded)
# ═══════════════════════════════════════════════════════════════════════════

def _dpss_tapers(N: int, NW: float):
    """
    Return CPU (tapers, ratios) — shapes (K, N) and (K,).
    K = 2·NW − 1.  Always computed on CPU (scipy); caller uploads to GPU.
    """
    K = int(2 * NW) - 1
    tapers, ratios = signal.windows.dpss(N, NW, Kmax=K, return_ratios=True)
    return tapers.astype(np.float64), ratios.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
#  Multitaper PSD
# ═══════════════════════════════════════════════════════════════════════════

def _multitaper_psd(x: np.ndarray, fs: float, NW: float = 4.0,
                    fmax: float = 15.0, adaptive: bool = True,
                    xp=np):
    """
    Thomson multitaper PSD.  Runs on xp (numpy or cupy).

    Returns CPU numpy arrays (freqs, psd).
    """
    N = len(x)
    tapers_cpu, ratios_cpu = _dpss_tapers(N, NW)
    tapers = _to_xp(tapers_cpu, xp)
    ratios = _to_xp(ratios_cpu, xp)
    x_xp   = _to_xp(np.asarray(x), xp)

    xk = tapers * (x_xp - x_xp.mean())       # (K, N)
    Xk = xp.fft.rfft(xk, n=N, axis=-1)       # (K, M)
    Sk = (xp.abs(Xk) ** 2) / fs
    Sk[:, 1:-1] *= 2                          # one-sided

    psd = _adaptive_weights(Sk, ratios, xp) if adaptive else Sk.mean(axis=0)

    freqs_cpu = np.fft.rfftfreq(N, d=1.0 / fs)
    mask      = freqs_cpu <= fmax
    return freqs_cpu[mask], _to_numpy(psd[mask], xp)


def _adaptive_weights(Sk, ratios, xp=np,
                      max_iter: int = 150, tol: float = 1e-6):
    """
    Adaptive taper weighting (Thomson 1982; Percival & Walden §7.7).
    All ops run on xp device.
    """
    psd     = Sk.mean(axis=0)
    psd_new = psd                 # fallback if max_iter=0 or instant convergence
    for _ in range(max_iter):
        sig  = psd[xp.newaxis, :]
        leak = psd.mean()
        b    = sig / (ratios[:, xp.newaxis] * sig
                      + (1.0 - ratios[:, xp.newaxis]) * leak)
        w       = b ** 2 * ratios[:, xp.newaxis]
        psd_new = (w * Sk).sum(axis=0) / w.sum(axis=0)
        if float(xp.max(xp.abs(psd_new - psd) / (psd + 1e-30))) < tol:
            break
        psd = psd_new
    return psd_new


# ═══════════════════════════════════════════════════════════════════════════
#  Thomson F-test for spectral lines
# ═══════════════════════════════════════════════════════════════════════════

def _f_test_lines(x: np.ndarray, fs: float, NW: float = 4.0,
                  p_threshold: float = 0.01, fmax: float = 15.0,
                  xp=np):
    """
    Thomson F-test for sinusoidal components.  GPU-accelerated eigenspectra;
    F-distribution CDF evaluated on CPU (scipy).

    Returns CPU numpy arrays.
    """
    N = len(x)
    K = int(2 * NW) - 1
    tapers_cpu, _ = _dpss_tapers(N, NW)
    tapers = _to_xp(tapers_cpu, xp)
    x_xp   = _to_xp(np.asarray(x), xp)

    xk    = tapers * (x_xp - x_xp.mean())
    Xk    = xp.fft.rfft(xk, n=N, axis=-1)     # (K, M)

    freqs_cpu = np.fft.rfftfreq(N, d=1.0 / fs)
    f_mask    = freqs_cpu <= fmax
    # Slice using an integer index to avoid implicit CPU→GPU mask transfer
    M_clip    = int(f_mask.sum())
    Xk        = Xk[:, :M_clip]
    freqs     = freqs_cpu[f_mask]

    mu    = Xk.mean(axis=0)                    # (M,)
    num   = K * xp.abs(mu) ** 2
    denom = (xp.abs(Xk - mu[xp.newaxis, :]) ** 2).sum(axis=0) / (K - 1)
    denom = xp.clip(denom, 1e-30, None)

    f_stat_np = _to_numpy(num / denom, xp)
    p_val     = 1.0 - f_dist.cdf(f_stat_np, dfn=2, dfd=2 * (K - 1))
    sig_f     = freqs[p_val < p_threshold]
    return sig_f, f_stat_np, p_val, freqs


# ═══════════════════════════════════════════════════════════════════════════
#  Batched multitaper spectrogram  ← primary speedup target
# ═══════════════════════════════════════════════════════════════════════════

def _multitaper_spectrogram(x: np.ndarray, fs: float, NW: float = 4.0,
                             win_s: float = 4.0, overlap_s: float = None,
                             fmax: float = 15.0, xp=np):
    """
    Sliding-window multitaper spectrogram — fully batched, no Python loop.

    Algorithm
    ---------
    1. Compute window start indices.
    2. Extract all frames at once:  frames[i] = x[start_i : start_i + win_n]
       Shape: (n_frames, win_n)  — done with fancy indexing, no copy loop.
    3. Apply K DPSS tapers via broadcasting:
       xk = frames[:, None, :] * tapers[None, :, :]   → (n_frames, K, win_n)
    4. Single rfft call over last axis               → (n_frames, K, M)
    5. |·|² / fs, one-sided correction, mean over K → (n_frames, F_clip)
    6. Transpose                                     → (F_clip, n_frames)

    Steps 3–6 run on xp (GPU if available), with a single kernel launch
    for the rfft.

    Parameters
    ----------
    overlap_s : overlap in seconds; None → win_s/2 (50 %).
                step = win_s − overlap_s.

    Returns CPU numpy arrays
    ------------------------
    freqs : (F,)    Hz
    t     : (W,)    window-centre timestamps (s) — half-window shifted
    Sxx   : (F, W)  power (units²/Hz, linear)
    """
    win_n = int(win_s * fs)

    if overlap_s is None:
        overlap_s = win_s / 2.0
    if not (0.0 <= overlap_s < win_s):
        raise ValueError(
            f"overlap_s={overlap_s:.3f} must be in [0, win_s={win_s})")
    step_n = max(1, int((win_s - overlap_s) * fs))

    # ── Pre-compute DPSS tapers on CPU, then upload ──────────────────────
    tapers_cpu, _ = _dpss_tapers(win_n, NW)       # (K, win_n), float64, CPU
    tapers = _to_xp(tapers_cpu, xp)               # → GPU if available

    # ── Upload signal ────────────────────────────────────────────────────
    x_xp = _to_xp(np.asarray(x, dtype=np.float64), xp)

    # ── Frame extraction via advanced indexing (no Python loop) ──────────
    N      = len(x_xp)
    starts = xp.arange(0, N - win_n + 1, step_n)  # (n_frames,)
    idx    = starts[:, xp.newaxis] + xp.arange(win_n)[xp.newaxis, :]
    # idx shape: (n_frames, win_n)
    frames = x_xp[idx]                             # (n_frames, win_n)
    frames = frames - frames.mean(axis=1, keepdims=True)   # demean per frame

    # ── Batched taper application + FFT ─────────────────────────────────
    # frames[:, None, :] * tapers[None, :, :] → (n_frames, K, win_n)
    xk = frames[:, xp.newaxis, :] * tapers[xp.newaxis, :, :]

    # Single rfft over last axis → (n_frames, K, M)
    Xk = xp.fft.rfft(xk, n=win_n, axis=-1)

    # ── Power, frequency clipping, taper average ─────────────────────────
    freqs_cpu = np.fft.rfftfreq(win_n, d=1.0 / fs)
    f_mask    = freqs_cpu <= fmax                  # CPU boolean mask
    M_clip    = int(f_mask.sum())

    # One-sided correction must be applied to the FULL rfft spectrum
    # before frequency clipping, so the last kept bin is correctly doubled
    # regardless of whether fmax equals Nyquist or not.
    Sk_full = (xp.abs(Xk) ** 2) / fs              # (n_frames, K, M_full)
    Sk_full[:, :, 1:-1] *= 2                       # DC=×1, Nyquist=×1, rest=×2
    Sxx_T = Sk_full[:, :, :M_clip].mean(axis=1)   # clip then average (n_frames, F_clip)

    # ── Pull results back to CPU ─────────────────────────────────────────
    Sxx        = _to_numpy(Sxx_T, xp).T            # (F_clip, n_frames)
    starts_cpu = _to_numpy(starts, xp)
    t          = (starts_cpu + win_n / 2.0) / fs   # window-centre timestamps

    return freqs_cpu[f_mask], t, Sxx


# ═══════════════════════════════════════════════════════════════════════════
#  Bandpass + Hilbert envelope — FFT-domain (GPU-compatible)
# ═══════════════════════════════════════════════════════════════════════════

def _bandpass_envelope(x: np.ndarray, fs: float,
                       flo: float, fhi: float,
                       xp=np) -> np.ndarray:
    """
    Frequency-domain bandpass + Hilbert → instantaneous power.

    Implemented entirely via FFT (not filtfilt) so it runs on xp (GPU).
    Equivalent to: rfft → zero freqs outside [flo,fhi] → irfft → Hilbert
    → |·|²  but done in a single complex FFT pass:

      1. FFT the real signal → one-sided spectrum X[k]
      2. Zero bins outside [flo, fhi]
      3. Double the kept positive bins (analytic signal construction)
      4. IFFT → complex analytic signal z(t)
      5. |z(t)|² = instantaneous power

    Returns CPU numpy array of shape (T,).
    """
    N    = len(x)
    x_xp = _to_xp(np.asarray(x, dtype=np.float64), xp)

    # Full FFT (not rfft) for analytic signal construction
    X = xp.fft.fft(x_xp)                     # (N,) complex

    freqs = np.fft.fftfreq(N, d=1.0 / fs)    # CPU, not uploaded

    # Build bandpass + analytic mask on CPU then upload
    mask = np.zeros(N, dtype=np.float64)
    # Positive-frequency bins in [flo, fhi] are doubled to form the
    # analytic signal; all other bins stay zero (mask already zero-init).
    # freqs >= flo implicitly excludes DC (0 Hz) and negative freqs
    # because flo >= 0.01 Hz always.  No extra guard needed.
    pos = (freqs >= flo) & (freqs <= fhi)
    mask[pos] = 2.0
    if N % 2 == 0:
        mask[N // 2] = 0.0                    # zero Nyquist

    mask_xp = _to_xp(mask, xp)
    z       = xp.fft.ifft(X * mask_xp)       # complex analytic signal
    power   = xp.abs(z) ** 2                  # instantaneous power

    return _to_numpy(power, xp)


# ═══════════════════════════════════════════════════════════════════════════
#  Population signal reduction  (CPU — small op)
# ═══════════════════════════════════════════════════════════════════════════

def _population_signal(F: np.ndarray, method: str = "mean") -> np.ndarray:
    if method == "mean":
        return F.mean(axis=0)
    elif method == "pc1":
        from sklearn.decomposition import PCA
        return PCA(n_components=1).fit_transform(F.T).squeeze()
    elif method == "norm":
        norms = np.linalg.norm(F, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (F / norms).mean(axis=0)
    else:
        raise ValueError(f"Unknown pop_method: {method!r}")


# ═══════════════════════════════════════════════════════════════════════════
#  Frequency bands
# ═══════════════════════════════════════════════════════════════════════════

BANDS = {
    "infra-slow": (0.01, 0.1),
    "slow":       (0.1,  1.0),
    "delta":      (1.0,  4.0),
    "theta":      (4.0,  8.0),
    "alpha-beta": (8.0, 15.0),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════════════════

class OscillationAnalyzer:
    """
    Multitaper (DPSS) oscillation analyser — GPU-accelerated.

    Parameters
    ----------
    estimates  : CaImAn Estimates object with .f, .F_dff/.C, .S
    fs         : sampling rate (Hz)
    fmax       : upper frequency limit (default 15 Hz)
    NW         : DPSS time-bandwidth product
                   NW=2 → K=3 tapers  (fine resolution)
                   NW=4 → K=7 tapers  (low variance, default)
    adaptive   : adaptive taper weighting (recommended)
    pop_method : 'mean' | 'pc1' | 'norm'
    use_gpu    : True → use CuPy if available, fall back to NumPy silently
                 False → always use NumPy (CPU)
    """

    def __init__(self, estimates, fs: float = 30.0, fmax: float = 15.0,
                 NW: float = 4.0, adaptive: bool = True,
                 pop_method: str = "mean", use_gpu: bool = True):
        self.fs       = fs
        self.fmax     = fmax
        self.NW       = NW
        self.K        = int(2 * NW) - 1
        self.adaptive = adaptive
        self.xp          = _get_xp(use_gpu)
        self.use_gpu     = self.xp is not np
        self._neural_label = "signal"   # overwritten below if traces found

        # ── Background ───────────────────────────────────────────────────
        # estimates.f is the raw temporal coefficient of the global low-rank
        # background spatial components (b), shape (gnb, T).  Unlike neural
        # traces it is never baselined — it carries strong DC offset and slow
        # photobleaching drift that swamp the multitaper spectral estimate.
        # Subtract a per-component running median (window = 10 s) before
        # averaging so only oscillatory structure remains.
        if hasattr(estimates, "f") and estimates.f is not None:
            fm = np.atleast_2d(np.array(estimates.f, dtype=np.float64))
            win = max(3, int(fs * 10) | 1)          # 10-s window, odd
            try:
                from scipy.ndimage import uniform_filter1d as _uf1
                baseline = _uf1(fm, size=win, axis=1, mode="nearest")
                fm = fm - baseline
            except Exception:
                fm = fm - fm.mean(axis=1, keepdims=True)   # DC removal fallback
            self.bg = fm.mean(axis=0)
            log.info(f"Background: {fm.shape[0]} component(s), "
                     f"{fm.shape[1]} frames  (detrended, window={win} samples)")
        else:
            self.bg = None
            log.warning("estimates.f not found — background analysis disabled")

        # ── Neural population ────────────────────────────────────────────
        neural_mat = None
        if hasattr(estimates, "F_dff") and estimates.F_dff is not None:
            neural_mat, self._neural_label = estimates.F_dff, "dF/F"
        elif hasattr(estimates, "C") and estimates.C is not None:
            neural_mat, self._neural_label = estimates.C, "denoised C"

        if neural_mat is not None and neural_mat.shape[0] > 0:
            self.neural_pop = _population_signal(
                neural_mat.astype(np.float64), method=pop_method)
            log.info(f"Neural: {neural_mat.shape[0]} cells, "
                     f"method={pop_method}")
        else:
            self.neural_pop = None
            log.warning("No neural traces — neural analysis disabled")

        # ── Spike rate ───────────────────────────────────────────────────
        if (hasattr(estimates, "S") and estimates.S is not None
                and estimates.S.shape[0] > 0):
            self.spike_rate = estimates.S.mean(axis=0).astype(np.float64)
        else:
            self.spike_rate = None

        ref = next((s for s in (self.bg, self.neural_pop) if s is not None),
                   None)
        if ref is None:
            raise RuntimeError("No usable signal found in estimates.")
        self._T   = ref.shape[0]
        self.t_ax = np.arange(self._T) / fs

        bw = NW / (self._T / fs)
        log.info(f"MT: NW={NW}, K={self.K}, adaptive={adaptive}, "
                 f"half-BW≈{bw:.4f} Hz, N={self._T}, "
                 f"backend={'CuPy/GPU' if self.use_gpu else 'NumPy/CPU'}")

    # ───────────────────────────────────────────────────────────────────────
    #  PSD
    # ───────────────────────────────────────────────────────────────────────

    def compute_psd(self) -> dict:
        return {k: _multitaper_psd(x, self.fs, NW=self.NW, fmax=self.fmax,
                                   adaptive=self.adaptive, xp=self.xp)
                for k, x in self._signal_map().items() if x is not None}

    def plot_psd(self, ax=None, show_bands: bool = True,
                 show_f_lines: bool = True,
                 p_threshold: float = 0.01) -> plt.Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.figure

        colors = {"background": "#2196F3",
                  "neural_pop": "#E91E63",
                  "spike_rate": "#4CAF50"}
        labels = {"background": "Background / neuropil",
                  "neural_pop": f"Population {self._neural_label}",
                  "spike_rate": "Population spike rate"}

        for key, (f, pxx) in self.compute_psd().items():
            ax.semilogy(f, pxx, lw=1.5, color=colors[key], label=labels[key])

        if show_bands:
            tab = plt.cm.tab10.colors
            for i, (band, (flo, fhi)) in enumerate(BANDS.items()):
                if fhi <= self.fmax:
                    ax.axvspan(flo, fhi, alpha=0.07, color=tab[i % len(tab)],
                               label=band)

        if show_f_lines and self.bg is not None:
            sig_f, *_ = _f_test_lines(self.bg, self.fs, NW=self.NW,
                                       p_threshold=p_threshold,
                                       fmax=self.fmax, xp=self.xp)
            for sf in sig_f:
                ax.axvline(sf, color="limegreen", lw=0.9, ls="--", alpha=0.8)
            if len(sig_f):
                ax.axvline(np.nan, color="limegreen", lw=0.9, ls="--",
                           label=f"F-test lines (p<{p_threshold})")

        bw = self.NW / (self._T / self.fs)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power spectral density (units²/Hz)")
        ax.set_xlim(0, self.fmax)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.set_title(
            f"Multitaper PSD  NW={self.NW}  K={self.K}  "
            f"half-BW≈{bw:.3f} Hz  adaptive={self.adaptive}  "
            f"({'GPU' if self.use_gpu else 'CPU'})")
        fig.tight_layout()
        return fig

    # ───────────────────────────────────────────────────────────────────────
    #  F-test
    # ───────────────────────────────────────────────────────────────────────

    def plot_f_test(self, signal_key: str = "background",
                    p_threshold: float = 0.01) -> plt.Figure:
        x = self._signal_map()[signal_key]
        if x is None:
            raise ValueError(f"Signal '{signal_key}' not available")

        sig_f, f_stat, p_val, freqs = _f_test_lines(
            x, self.fs, NW=self.NW, p_threshold=p_threshold,
            fmax=self.fmax, xp=self.xp)
        crit = f_dist.ppf(1 - p_threshold, dfn=2, dfd=2 * (self.K - 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        title_base = (f"{signal_key.replace('_', ' ').title()}  "
                      f"NW={self.NW}  K={self.K}")

        ax1.plot(freqs, f_stat, lw=0.8, color="#1565C0")
        ax1.axhline(crit, color="red", lw=1.0, ls="--",
                    label=f"F crit (p={p_threshold})")
        for sf in sig_f:
            ax1.axvline(sf, color="limegreen", lw=0.8, alpha=0.8)
        ax1.set_ylabel("F statistic")
        ax1.legend(fontsize=8)
        ax1.set_title(f"Thomson F-test — {title_base}")

        ax2.semilogy(freqs, p_val + 1e-15, lw=0.8, color="#6A1B9A")
        ax2.axhline(p_threshold, color="red", lw=1.0, ls="--",
                    label=f"p = {p_threshold}")
        for sf in sig_f:
            ax2.axvline(sf, color="limegreen", lw=0.8, alpha=0.8)
        ax2.set_ylabel("p-value")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_xlim(0, self.fmax)
        ax2.legend(fontsize=8)

        if len(sig_f):
            print(f"Significant lines [{signal_key}]: "
                  + ", ".join(f"{f:.3f} Hz" for f in sig_f))
        else:
            print(f"No significant lines at p<{p_threshold} [{signal_key}]")

        fig.tight_layout()
        return fig

    # ───────────────────────────────────────────────────────────────────────
    #  Spectrogram
    # ───────────────────────────────────────────────────────────────────────

    def _render_spectrogram_fig(self, sig_label: str,
                                Sxx_db: np.ndarray, sg_t: np.ndarray,
                                sg_f: np.ndarray, win_s: float, ov_s: float,
                                x_raw=None, db_range: float = 40.0) -> plt.Figure:
        """
        Shared spectrogram rendering used by plot_spectrogram and run_all.
        Accepts pre-computed arrays so computation is never repeated.
        """
        step_s = win_s - ov_s
        df_res = 2 * self.NW / win_s
        med    = np.median(Sxx_db)
        vmin, vmax = med - db_range / 2, med + db_range / 2

        nrows = 2 if x_raw is not None else 1
        hr    = [1, 3.5] if nrows == 2 else [1]
        fig, axes = plt.subplots(
            nrows, 1,
            figsize=(13, 6 if nrows == 2 else 3.5),
            gridspec_kw={"height_ratios": hr})
        if nrows == 1:
            axes = [axes]

        if x_raw is not None:
            axes[0].plot(self.t_ax, zscore(x_raw), lw=0.5, color="k", alpha=0.7)
            axes[0].set_ylabel("z-score", fontsize=8)
            axes[0].set_title(f"{sig_label} — raw signal")
            axes[0].set_xlim(0, self.t_ax[-1])
            axes[0].margins(y=0.15)

        ax_sg = axes[-1]
        im = ax_sg.pcolormesh(sg_t, sg_f, Sxx_db, shading="gouraud",
                               cmap="inferno", vmin=vmin, vmax=vmax,
                               rasterized=True)
        cb = plt.colorbar(im, ax=ax_sg, pad=0.01)
        cb.set_label("Power (dB re 1 unit\u00b2/Hz)", fontsize=8)

        for band, (flo, fhi) in BANDS.items():
            if fhi <= self.fmax:
                ax_sg.axhline(flo, color="white", lw=0.35, alpha=0.55)
                mid = (flo + min(fhi, self.fmax)) / 2
                ax_sg.text(sg_t[-1] * 0.01, mid, band,
                           color="white", fontsize=6, va="center")

        backend = "GPU" if self.use_gpu else "CPU"
        ax_sg.set_xlabel("Time (s)")
        ax_sg.set_ylabel("Frequency (Hz)")
        ax_sg.set_ylim(0, self.fmax)
        ax_sg.set_title(
            f"{sig_label} — MT spectrogram  "
            f"NW={self.NW}  K={self.K}  win={win_s}s  "
            f"overlap={ov_s}s  step={step_s}s  \u0394f\u2248{df_res:.2f} Hz  "
            f"({backend})")
        fig.tight_layout()
        return fig

    def plot_spectrogram(self, signal_key: str = "background",
                         win_s: float = 4.0, overlap_s: float = None,
                         overlay_signal: bool = True,
                         db_range: float = 40.0) -> plt.Figure:
        """
        Sliding-window multitaper spectrogram (batched + GPU-accelerated).

        Parameters
        ----------
        win_s          : window length (s) — freq resolution ≈ 2·NW/win_s
        overlap_s      : overlap (s); default None → win_s/2 (50 %)
                         step = win_s − overlap_s
        overlay_signal : show z-scored raw signal above spectrogram
        db_range       : colour dynamic range (dB)
        """
        x = self._signal_map()[signal_key]
        if x is None:
            raise ValueError(f"Signal '{signal_key}' not available")

        _ov = win_s / 2.0 if overlap_s is None else overlap_s
        log.info(f"MT spectrogram [{signal_key}]: "
                 f"win={win_s}s overlap={_ov}s step={win_s - _ov}s "
                 f"({'GPU' if self.use_gpu else 'CPU'}) ...")
        t0 = time.perf_counter()
        sg_f, sg_t, Sxx = _multitaper_spectrogram(x, self.fs, NW=self.NW,
                                                   win_s=win_s, overlap_s=overlap_s,
                                                   fmax=self.fmax, xp=self.xp)
        log.info(f"  done in {time.perf_counter()-t0:.2f}s  shape={Sxx.shape}")
        return self._render_spectrogram_fig(
            sig_label = signal_key.replace("_", " ").title(),
            Sxx_db    = 10 * np.log10(Sxx + 1e-30),
            sg_t      = sg_t,
            sg_f      = sg_f,
            win_s     = win_s,
            ov_s      = _ov,
            x_raw     = x if overlay_signal else None,
            db_range  = db_range,
        )

    # ───────────────────────────────────────────────────────────────────────
    #  Band power
    # ───────────────────────────────────────────────────────────────────────

    def compute_band_power(self, signal_key: str = "background") -> dict:
        """Return {band: instantaneous_power_array (CPU numpy)} for each band."""
        x = self._signal_map()[signal_key]
        if x is None:
            raise ValueError(f"Signal '{signal_key}' not available")
        out = {}
        for band, (flo, fhi) in BANDS.items():
            try:
                out[band] = _bandpass_envelope(x, self.fs, flo, fhi,
                                               xp=self.xp)
            except Exception as e:
                log.warning(f"Band '{band}' skipped: {e}")
        return out

    def plot_band_power(self, signal_key: str = "background",
                        smooth_s: float = 1.0) -> plt.Figure:
        bp = self.compute_band_power(signal_key)
        n  = len(bp)
        fig, axes = plt.subplots(n, 1, figsize=(13, 1.8 * n), sharex=True)
        if n == 1:
            axes = [axes]

        colors    = plt.cm.viridis(np.linspace(0, 1, n))
        sig_label = signal_key.replace("_", " ").title()

        for ax, (band, power), color in zip(axes, bp.items(), colors):
            if smooth_s > 0:
                power = gaussian_filter1d(power, sigma=smooth_s * self.fs)
            ax.fill_between(self.t_ax, power, alpha=0.65, color=color)
            ax.set_ylabel(band, fontsize=8, rotation=0,
                          labelpad=65, va="center")
            ax.set_yticks([])
            ax.margins(x=0)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"Instantaneous band power — {sig_label}  "
            f"(Hilbert envelope, smooth={smooth_s}s, "
            f"{'GPU' if self.use_gpu else 'CPU'})",
            fontsize=9, y=1.01)
        fig.tight_layout()
        return fig

    # ───────────────────────────────────────────────────────────────────────
    #  Summary
    # ───────────────────────────────────────────────────────────────────────

    def band_power_summary(self) -> dict:
        out = {}
        for key in ("background", "neural_pop", "spike_rate"):
            try:
                bp = self.compute_band_power(key)
                out[key] = {b: float(np.mean(p)) for b, p in bp.items()}
            except ValueError:
                pass
        return out

    def dominant_frequency(self, signal_key: str = "background",
                           band: tuple = None) -> float:
        x = self._signal_map()[signal_key]
        if x is None:
            raise ValueError(f"Signal '{signal_key}' not available")
        f, pxx = _multitaper_psd(x, self.fs, NW=self.NW, fmax=self.fmax,
                                  adaptive=self.adaptive, xp=self.xp)
        if band is not None:
            mask = (f >= band[0]) & (f <= band[1])
            f, pxx = f[mask], pxx[mask]
        return float(f[np.argmax(pxx)])

    # ───────────────────────────────────────────────────────────────────────
    #  Numerical data export
    # ───────────────────────────────────────────────────────────────────────

    def save_data(self, output_dir: str = ".", session_id: str = "session",
                  win_s: float = 4.0, overlap_s: float = None,
                  p_threshold: float = 0.01) -> Path:
        """
        Save all numerical results to a compressed .npz file.

        All arrays are CPU numpy (GPU results pulled before saving).

        Layout
        ------
        Metadata:
          meta_fs, meta_NW, meta_K, meta_adaptive, meta_fmax,
          meta_win_s, meta_overlap_s, meta_p_threshold, meta_gpu

        For each sig ∈ {background, neural_pop, spike_rate}:
          {sig}_t_ax          — sample time axis (s), shape (T,)
          {sig}_raw           — 1-D signal, shape (T,)
          {sig}_psd_freqs     — Hz
          {sig}_psd           — units²/Hz
          {sig}_ftest_freqs/fstat/pval/sigf
          {sig}_sg_freqs      — Hz, shape (F,)
          {sig}_sg_t          — window-centre times (s), shape (W,)
                                *** half-window shifted: t[i]=(start_i+win_n/2)/fs
          {sig}_sg_Sxx        — linear power, shape (F, W)
          {sig}_sg_Sxx_db     — dB, shape (F, W)
          {sig}_bp_t          — same as t_ax
          {sig}_bp_{band}     — Hilbert envelope power per band
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        npz_path = out / f"{session_id}_oscillations.npz"
        arrays: dict = {}

        _ov = win_s / 2.0 if overlap_s is None else overlap_s

        # metadata
        arrays["meta_fs"]          = np.float64(self.fs)
        arrays["meta_NW"]          = np.float64(self.NW)
        arrays["meta_K"]           = np.int32(self.K)
        arrays["meta_adaptive"]    = np.bool_(self.adaptive)
        arrays["meta_fmax"]        = np.float64(self.fmax)
        arrays["meta_win_s"]       = np.float64(win_s)
        arrays["meta_overlap_s"]   = np.float64(_ov)
        arrays["meta_p_threshold"] = np.float64(p_threshold)
        arrays["meta_gpu"]         = np.bool_(self.use_gpu)

        for sig_key, x in self._signal_map().items():
            if x is None:
                continue
            arrays[f"{sig_key}_t_ax"] = self.t_ax
            arrays[f"{sig_key}_raw"]  = np.asarray(x)

            # PSD
            try:
                f_p, psd = _multitaper_psd(x, self.fs, NW=self.NW,
                                            fmax=self.fmax,
                                            adaptive=self.adaptive,
                                            xp=self.xp)
                arrays[f"{sig_key}_psd_freqs"] = f_p
                arrays[f"{sig_key}_psd"]       = psd
            except Exception as e:
                log.warning(f"  PSD skipped [{sig_key}]: {e}")

            # F-test
            try:
                sig_f, fstat, pval, all_f = _f_test_lines(
                    x, self.fs, NW=self.NW, p_threshold=p_threshold,
                    fmax=self.fmax, xp=self.xp)
                arrays[f"{sig_key}_ftest_freqs"] = all_f
                arrays[f"{sig_key}_ftest_fstat"] = fstat
                arrays[f"{sig_key}_ftest_pval"]  = pval
                arrays[f"{sig_key}_ftest_sigf"]  = sig_f
            except Exception as e:
                log.warning(f"  F-test skipped [{sig_key}]: {e}")

            # Spectrogram
            try:
                t0 = time.perf_counter()
                sg_f, sg_t, Sxx = _multitaper_spectrogram(
                    x, self.fs, NW=self.NW,
                    win_s=win_s, overlap_s=overlap_s,
                    fmax=self.fmax, xp=self.xp)
                Sxx_db = 10 * np.log10(Sxx + 1e-30)
                arrays[f"{sig_key}_sg_freqs"]  = sg_f
                arrays[f"{sig_key}_sg_t"]      = sg_t
                arrays[f"{sig_key}_sg_Sxx"]    = Sxx
                arrays[f"{sig_key}_sg_Sxx_db"] = Sxx_db
                log.info(f"  Spectrogram [{sig_key}]: shape={Sxx.shape}  "
                         f"t[0]={sg_t[0]:.3f}s  overlap={_ov:.2f}s  "
                         f"step={win_s-_ov:.2f}s  "
                         f"({time.perf_counter()-t0:.2f}s)")
            except Exception as e:
                log.warning(f"  Spectrogram skipped [{sig_key}]: {e}")

            # Band power
            try:
                bp = self.compute_band_power(sig_key)
                arrays[f"{sig_key}_bp_t"] = self.t_ax
                for band, power in bp.items():
                    key_safe = band.replace("-", "_").replace(" ", "_")
                    arrays[f"{sig_key}_bp_{key_safe}"] = np.asarray(power)
            except Exception as e:
                log.warning(f"  Band power skipped [{sig_key}]: {e}")

        np.savez_compressed(str(npz_path), **arrays)
        log.info(f"Saved → {npz_path}  "
                 f"({npz_path.stat().st_size / 1e6:.1f} MB)")
        return npz_path

    # ───────────────────────────────────────────────────────────────────────
    #  run_all
    # ───────────────────────────────────────────────────────────────────────

    def run_all(self, output_dir: str = ".", session_id: str = "session",
                dpi: int = 150, win_s: float = 4.0, overlap_s: float = None,
                p_threshold: float = 0.01):
        """
        Save all numerical data (.npz) then generate all figures.

        Output files
        ------------
        {session_id}_oscillations.npz
        {session_id}_mt_psd.png
        {session_id}_f_test_{sig}.png
        {session_id}_mt_spectrogram_{sig}.png
        {session_id}_bandpower_{sig}.png
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        t_total = time.perf_counter()

        npz_path = self.save_data(output_dir=output_dir,
                                   session_id=session_id,
                                   win_s=win_s, overlap_s=overlap_s,
                                   p_threshold=p_threshold)

        fig = self.plot_psd(show_f_lines=True, p_threshold=p_threshold)
        fig.savefig(out / f"{session_id}_mt_psd.png",
                    dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        for key in ("background", "neural_pop"):
            try:
                fig = self.plot_f_test(signal_key=key,
                                       p_threshold=p_threshold)
                fig.savefig(out / f"{session_id}_f_test_{key}.png",
                            dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except ValueError as e:
                log.warning(e)

        # Spectrograms: load pre-computed arrays from the npz rather than
        # recomputing — avoids running the most expensive step twice.
        _npz = load_npz(str(npz_path))
        for key in ("background", "neural_pop"):
            if f"{key}_sg_Sxx_db" not in _npz:
                log.warning(f"Spectrogram not in npz for [{key}], skipping plot")
                continue
            try:
                fig = self._render_spectrogram_fig(
                    sig_label = key.replace("_", " ").title(),
                    Sxx_db    = _npz[f"{key}_sg_Sxx_db"],
                    sg_t      = _npz[f"{key}_sg_t"],
                    sg_f      = _npz[f"{key}_sg_freqs"],
                    win_s     = win_s,
                    ov_s      = float(_npz["meta_overlap_s"]),
                    x_raw     = _npz.get(f"{key}_raw"),
                )
                fig.savefig(out / f"{session_id}_mt_spectrogram_{key}.png",
                            dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                log.warning(f"Spectrogram plot failed for [{key}]: {e}")

        for key in ("background", "neural_pop"):
            try:
                fig = self.plot_band_power(signal_key=key)
                fig.savefig(out / f"{session_id}_bandpower_{key}.png",
                            dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except ValueError as e:
                log.warning(e)

        summary = self.band_power_summary()
        bw = self.NW / (self._T / self.fs)
        print(f"\n{'═'*64}")
        print(f"  Oscillation summary — {session_id}")
        print(f"  NW={self.NW}  K={self.K}  half-BW≈{bw:.4f} Hz  "
              f"adaptive={self.adaptive}  "
              f"backend={'GPU' if self.use_gpu else 'CPU'}")
        print(f"{'═'*64}")
        for sig, bands in summary.items():
            print(f"\n  {sig}:")
            for band, power in bands.items():
                try:
                    dom = self.dominant_frequency(sig, band=BANDS[band])
                    dom_s = f"peak={dom:.3f} Hz"
                except Exception:
                    dom_s = "peak=N/A"
                print(f"    {band:<14}  mean_power={power:.5g}  {dom_s}")
        print(f"\n  Total wall time: {time.perf_counter()-t_total:.1f}s")
        print(f"{'═'*64}\n")
        return summary

    # ───────────────────────────────────────────────────────────────────────

    def _signal_map(self) -> dict:
        return {"background": self.bg,
                "neural_pop": self.neural_pop,
                "spike_rate": self.spike_rate}


# ═══════════════════════════════════════════════════════════════════════════
#  Reload helper
# ═══════════════════════════════════════════════════════════════════════════

def load_npz(path: str) -> dict:
    """
    Load a saved oscillation .npz into a plain dict.

    Usage
    -----
        data = load_npz("./osc_output/session_oscillations.npz")

        # Spectrogram — timestamps are window-centre (half-window shifted)
        t   = data["background_sg_t"]      # (W,) seconds, centre of each window
        f   = data["background_sg_freqs"]  # (F,) Hz
        Sxx = data["background_sg_Sxx"]    # (F, W) units²/Hz linear
        Sdb = data["background_sg_Sxx_db"] # (F, W) dB

        # PSD
        f_psd = data["background_psd_freqs"]
        psd   = data["background_psd"]

        # Band power (same time axis as raw signal)
        t_bp      = data["background_bp_t"]
        delta_pow = data["background_bp_delta"]
        theta_pow = data["background_bp_theta"]

        # F-test significant lines
        sig_lines = data["background_ftest_sigf"]  # Hz

        # Metadata
        fs      = float(data["meta_fs"])
        NW      = float(data["meta_NW"])
        overlap = float(data["meta_overlap_s"])
        gpu     = bool(data["meta_gpu"])

    Timestamp note
    --------------
    sg_t[i] = (start_frame_i + win_n / 2) / fs
    First timestamp = meta_win_s / 2 seconds into the recording.
    All timestamps point to the midpoint of their analysis window.
    """
    return dict(np.load(path, allow_pickle=False))


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, h5py

    class _EstimatesProxy:
        def __init__(self, path):
            with h5py.File(path, "r") as h:
                def _l(k):
                    return h[k][()] if k in h else None
                self.f     = _l("estimates/f")
                self.F_dff = _l("estimates/F_dff")
                self.C     = _l("estimates/C")
                self.S     = _l("estimates/S")

    p = argparse.ArgumentParser(
        description="GPU-accelerated multitaper oscillation analysis")
    p.add_argument("hdf5")
    p.add_argument("--fs",   type=float, default=30.0)
    p.add_argument("--fmax", type=float, default=15.0)
    p.add_argument("--NW",   type=float, default=4.0,
                   help="DPSS time-bandwidth product (2, 3, or 4)")
    p.add_argument("--no-adaptive", dest="adaptive", action="store_false",
                   help="Equal-weight taper averaging")
    p.add_argument("--gpu",    dest="use_gpu", action="store_true",  default=True,
                   help="Use GPU via CuPy (default)")
    p.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                   help="Force CPU / NumPy")
    p.add_argument("--win",     type=float, default=4.0,
                   help="Spectrogram window length (s)")
    p.add_argument("--overlap", type=float, default=None,
                   help="Spectrogram overlap (s); default None = win/2 (50%%)")
    p.add_argument("--outdir",  default="./osc_output")
    p.add_argument("--session", default="session")
    p.add_argument("--method",  default="mean",
                   choices=["mean", "pc1", "norm"])
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    est = _EstimatesProxy(args.hdf5)
    osc = OscillationAnalyzer(est, fs=args.fs, fmax=args.fmax,
                               NW=args.NW, adaptive=args.adaptive,
                               pop_method=args.method, use_gpu=args.use_gpu)
    osc.run_all(output_dir=args.outdir, session_id=args.session,
                win_s=args.win, overlap_s=args.overlap)
