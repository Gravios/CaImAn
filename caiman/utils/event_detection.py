"""
caiman/utils/event_detection.py
===============================
Calcium-event (peak) detection on CNMF ΔF/F traces, ported from the Stroh-lab
SLabAnalysisPackage (``PeakDetection.detect_peaks``, results "version 7.4"),
plus reader / writer for its ``stroh_results`` HDF5 group so files stay
interchangeable with ``SLabAnalysisApp.py``.

Algorithm (per component, unchanged from SLab unless noted under *Fixes*)
------------------------------------------------------------------------
1. Skip (auto-reject) if SNR < ``min_snr`` (3).
2. ``filtered`` = Gaussian(σ=1) of a 5th-order zero-phase Butterworth low-pass
   of ΔF/F.
3. noise = (median − min) / 1.349 over the *positive* samples of ``filtered``;
   threshold = min(10 · noise, ``threshold``).
4. Peaks = ``scipy.signal.find_peaks`` on C rescaled so max(C) == max(ΔF/F),
   with prominence = height = threshold, width ≥ 3, rel_height 0.9 and a
   refractory distance of ``decay_time · fr`` frames.
5. A peak is an *event* if ``filtered`` at the peak is ≥ threshold.  The
   binary trace holds the event amplitude from onset (left 90 %-width ip) to
   the peak — not 0/1, despite the SLab name.
6. events/min = n_events / (T / fr / 60).
7. Auto-reject if ``find_peaks`` returned nothing or CNN < ``min_cnn`` (0.15).

Fixes relative to SLab 7.4
--------------------------
* The low-pass sampling rate was hard-coded (``fs = 30.8``, cutoff 15 Hz), so
  the normalised cutoff was always 15/15.4 = 0.974·Nyquist whatever the real
  frame rate.  ``lowpass_hz=None`` (default) keeps that normalised cutoff, so
  results match SLab bit-for-bit at any ``fr``; pass a value in Hz to filter
  at a physical frequency instead.
* The "Detection threshold" parameter was ignored (``hardCutOff`` commented
  out; 0.2 hard-coded).  It is now the cap in step 3; the default 0.2
  reproduces SLab.
* ``auto_approved`` and ``manually_approved`` were the same list object, so
  manual curation silently rewrote the auto list.  They are independent here.
* No crash on missing ``cnn_preds`` (criterion skipped), ``decay_time·fr < 1``
  (distance clamped to 1), no positive samples, or traces too short to
  filter.  No NumPy-2-removed ``np.NaN``; no unused ``oasis`` import.

Public API
----------
detect_events(F_dff, C, fr, decay_time, SNR_comp=None, cnn_preds=None, ...)
    -> EventResults
EventResults.roi_stats(T)  /  .peak_table()
write_stroh_results(h5group_parent, results, approved)
read_stroh_results(h5group)  -> EventResults | None
detect_events_in_hdf5(path, ...)   # headless, writes stroh_results in place

CLI::

    python -m caiman.utils.event_detection FOLDER_OR_FILES... [--overwrite]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

SLAB_VERSION = 7.4          # results-format version written (SLab reader keys on >= 7.1)
SLAB_FS = 30.8              # SLab's hard-coded low-pass sampling rate …
SLAB_CUTOFF_HZ = 15.0       # … and cutoff → normalised Wn = 0.974
PEAK_COLUMNS = ("id", "roi", "peak", "amp", "onset", "offset", "binary_offset")


@dataclass
class EventResults:
    """Per-component event-detection output; every array is aligned to rows."""
    binary: np.ndarray                 # (K, T) amplitude-valued onset→peak plateaus
    filtered: np.ndarray               # (K, T) low-passed + Gaussian ΔF/F
    thresholds: np.ndarray             # (K,)
    frequencies: np.ndarray            # (K,) events per minute
    peaks: list                        # K arrays, each (n, 4): peak, amp, onset, offset
    auto_approved: np.ndarray          # (K,) bool
    fr: float
    decay_time: float
    threshold: float = 0.2
    params: dict = field(default_factory=dict)
    version: float = SLAB_VERSION

    @property
    def n(self) -> int:
        return len(self.peaks)

    def n_events(self, i: int) -> int:
        return int(len(self.peaks[i]))

    # ── row bookkeeping (used by the inspector's merge / delete) ────────────
    def take(self, rows: Sequence[int]) -> "EventResults":
        rows = list(rows)
        return EventResults(
            binary=self.binary[rows], filtered=self.filtered[rows],
            thresholds=self.thresholds[rows], frequencies=self.frequencies[rows],
            peaks=[self.peaks[r] for r in rows],
            auto_approved=self.auto_approved[rows],
            fr=self.fr, decay_time=self.decay_time, threshold=self.threshold,
            params=dict(self.params), version=self.version)

    def append(self, other: "EventResults") -> "EventResults":
        return EventResults(
            binary=np.vstack([self.binary, other.binary]),
            filtered=np.vstack([self.filtered, other.filtered]),
            thresholds=np.concatenate([self.thresholds, other.thresholds]),
            frequencies=np.concatenate([self.frequencies, other.frequencies]),
            peaks=list(self.peaks) + list(other.peaks),
            auto_approved=np.concatenate([self.auto_approved, other.auto_approved]),
            fr=self.fr, decay_time=self.decay_time, threshold=self.threshold,
            params=dict(self.params), version=self.version)

    def copy(self) -> "EventResults":
        return self.take(range(self.n))

    # ── tables ──────────────────────────────────────────────────────────────
    def peak_table(self, rows: Optional[Sequence[int]] = None) -> dict:
        """SLab ``peak_stats`` columns as a dict of 1-D arrays."""
        rows = range(self.n) if rows is None else rows
        cols = {c: [] for c in PEAK_COLUMNS}
        for r in rows:
            for pid, (pk, amp, on, off) in enumerate(self.peaks[r]):
                cols["id"].append(pid);   cols["roi"].append(r)
                cols["peak"].append(int(pk)); cols["amp"].append(float(amp))
                cols["onset"].append(int(on)); cols["offset"].append(int(off))
                cols["binary_offset"].append(np.nan)   # never filled by SLab
        return {k: np.asarray(v, dtype=float if k in ("amp", "binary_offset")
                              else np.int64) for k, v in cols.items()}


# ── signal helpers ───────────────────────────────────────────────────────────

def _lowpass(x: np.ndarray, fr: float, lowpass_hz: Optional[float],
             order: int = 5) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    if lowpass_hz is None:
        wn = SLAB_CUTOFF_HZ / (0.5 * SLAB_FS)            # SLab-identical
    else:
        wn = float(lowpass_hz) / (0.5 * float(fr))
    if not (0.0 < wn < 1.0):
        return np.asarray(x, dtype=np.float64)             # cutoff ≥ Nyquist → no-op
    b, a = butter(order, wn, btype="low", analog=False)
    if x.size <= 3 * max(len(a), len(b)):                  # filtfilt padlen
        return np.asarray(x, dtype=np.float64)
    return filtfilt(b, a, x)


def detect_events(F_dff, C, fr: float, decay_time: float,
                  SNR_comp=None, cnn_preds=None, *,
                  threshold: float = 0.2,
                  min_snr: float = 3.0,
                  min_cnn: float = 0.15,
                  lowpass_hz: Optional[float] = None,
                  rows: Optional[Sequence[int]] = None,
                  progress=None) -> EventResults:
    """Run SLab event detection on every row of ``F_dff`` / ``C`` (K, T).

    ``rows`` limits the work to a subset (others are left empty and not
    auto-approved).  ``progress(i, K)`` is called after each row if given.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    F = np.asarray(F_dff, dtype=np.float64)
    Cc = np.asarray(C, dtype=np.float64)
    if F.shape != Cc.shape:
        raise ValueError(f"F_dff {F.shape} and C {Cc.shape} differ")
    K, T = F.shape
    fr = float(fr)
    if fr <= 0:
        raise ValueError("imaging frequency (fr) must be > 0")
    snr = None if SNR_comp is None or np.size(SNR_comp) == 0 else np.asarray(SNR_comp, float)
    cnn = None if cnn_preds is None or np.size(cnn_preds) == 0 else np.asarray(cnn_preds, float)

    binary = np.zeros((K, T), dtype=np.float32)
    filtered = np.zeros((K, T), dtype=np.float32)
    thr = np.zeros(K, dtype=np.float32)
    freq = np.zeros(K, dtype=np.float32)
    peaks = [np.zeros((0, 4)) for _ in range(K)]
    auto = np.zeros(K, dtype=bool)
    distance = max(1, int(decay_time * fr))
    minutes = T / fr / 60.0

    todo = range(K) if rows is None else sorted(set(int(r) for r in rows))
    for n_done, r in enumerate(todo, 1):
        if progress is not None:
            progress(n_done, len(todo))
        if snr is not None and not (snr[r] >= min_snr):   # NaN SNR → reject
            continue
        fx = gaussian_filter1d(_lowpass(F[r], fr, lowpass_hz), sigma=1)
        yp = fx[fx > 0]
        if yp.size:
            noise = (np.percentile(yp, 50) - np.percentile(yp, 0)) / 1.349
            t_r = min(noise * 10.0, threshold)
        else:
            t_r = threshold
        thr[r] = t_r
        filtered[r] = fx

        fmax, cmax = F[r].max(), Cc[r].max()
        if not (fmax > 0 and cmax > 0):
            continue                                       # no positive signal
        denoised = Cc[r] / (cmax / fmax)
        pk, props = find_peaks(denoised, prominence=t_r, height=t_r, width=3,
                               rel_height=0.9, distance=distance)
        events = []
        b = binary[r]
        for k, p in enumerate(pk):
            amp = fx[p]
            if amp < t_r:
                continue
            on = int(round(props["left_ips"][k]))
            off = int(round(props["right_ips"][k]))
            b[on:p] = amp
            events.append((p, amp, on, off))
        peaks[r] = np.asarray(events, dtype=float).reshape(-1, 4)
        freq[r] = len(events) / minutes if minutes > 0 else 0.0
        cnn_ok = cnn is None or cnn[r] >= min_cnn
        auto[r] = len(pk) >= 1 and cnn_ok

    return EventResults(binary=binary, filtered=filtered, thresholds=thr,
                        frequencies=freq, peaks=peaks, auto_approved=auto,
                        fr=fr, decay_time=float(decay_time), threshold=float(threshold),
                        params=dict(min_snr=min_snr, min_cnn=min_cnn,
                                    lowpass_hz=lowpass_hz))


def roi_stats(res: EventResults, approved: np.ndarray, T: int,
              labels: Optional[Sequence[str]] = None,
              SNR_comp=None, cnn_preds=None) -> dict:
    """Per-ROI table for every *approved* ROI — including those with 0 events
    (SLab's ROIStats dropped them, biasing rates upward)."""
    rows = np.flatnonzero(np.asarray(approved, bool))
    minutes = T / res.fr / 60.0
    out = {"ROI": rows.astype(np.int64),
           "n_events": np.array([res.n_events(r) for r in rows], dtype=np.int64)}
    out["FiringFrequency (peaks/minute)"] = out["n_events"] / minutes
    if labels is not None:
        out["label"] = np.array([labels[r] for r in rows], dtype=object)
    for name, arr in (("SNR", SNR_comp), ("CNN", cnn_preds)):
        if arr is not None and np.size(arr):
            out[name] = np.asarray(arr, float)[rows]
    return out


# ── stroh_results HDF5 group (SLabAnalysisApp-compatible) ────────────────────

def write_stroh_results(parent, res: EventResults, approved,
                        group: str = "stroh_results", extra_attrs=None):
    """(Re)write ``parent[group]`` in SLab layout.  ``parent`` is an open
    h5py File/Group.  ``approved`` is a (K,) bool mask (manual curation)."""
    if group in parent:
        del parent[group]
    g = parent.create_group(group)
    g.create_dataset("version", data=res.version)
    g.create_dataset("binaryTraces", data=res.binary)
    g.create_dataset("filtered_dff", data=res.filtered)
    g.create_dataset("thersholds", data=res.thresholds)        # sic — SLab key
    g.create_dataset("frequencies", data=res.frequencies)
    g.create_dataset("decay_time", data=res.decay_time)
    g.create_dataset("detection_cutoff", data=res.threshold)
    g.create_dataset("imaging_frequency", data=res.fr)
    g.create_dataset("auto_approved",
                     data=np.flatnonzero(res.auto_approved).astype(np.int64))
    g.create_dataset("manually_approved",
                     data=np.flatnonzero(np.asarray(approved, bool)).astype(np.int64))
    for k, v in res.peak_table().items():
        g.create_dataset(f"peak_stats/{k}", data=v)
    g.attrs["generator"] = "caiman.utils.event_detection"
    g.attrs["lowpass_hz"] = ("slab-normalised" if res.params.get("lowpass_hz") is None
                             else float(res.params["lowpass_hz"]))
    for k, v in (extra_attrs or {}).items():
        g.attrs[k] = v
    return g


def _scalar(g, key, default=None):
    if key not in g:
        return default
    v = g[key][()]
    return v.item() if hasattr(v, "item") else v


def read_stroh_results(g):
    """Parse an h5py ``stroh_results`` group → (EventResults, approved mask).

    Returns ``(None, None)`` for groups too old to carry the fields the
    inspector needs (SLab < 7.1 lacks imaging_frequency / auto_approved).
    """
    need = ("binaryTraces", "filtered_dff", "frequencies", "manually_approved")
    if any(k not in g for k in need):
        return None, None
    binary = np.asarray(g["binaryTraces"][()], dtype=np.float32)
    K = binary.shape[0]
    peaks = [[] for _ in range(K)]
    if "peak_stats" in g and "roi" in g["peak_stats"]:
        ps = g["peak_stats"]
        cols = {c: np.asarray(ps[c][()]) for c in ("roi", "peak", "amp", "onset", "offset")
                if c in ps}
        for j in range(cols["roi"].size):
            r = int(cols["roi"][j])
            if 0 <= r < K:
                peaks[r].append(tuple(float(cols[c][j]) if c in cols else np.nan
                                      for c in ("peak", "amp", "onset", "offset")))
    peaks = [np.asarray(p, dtype=float).reshape(-1, 4) for p in peaks]
    auto = np.zeros(K, dtype=bool)
    if "auto_approved" in g:
        idx = np.asarray(g["auto_approved"][()], dtype=np.int64).ravel()
        auto[idx[(idx >= 0) & (idx < K)]] = True
    appr = np.zeros(K, dtype=bool)
    idx = np.asarray(g["manually_approved"][()], dtype=np.int64).ravel()
    appr[idx[(idx >= 0) & (idx < K)]] = True
    thr = (np.asarray(g["thersholds"][()], dtype=np.float32) if "thersholds" in g
           else np.zeros(K, np.float32))
    res = EventResults(
        binary=binary,
        filtered=np.asarray(g["filtered_dff"][()], dtype=np.float32),
        thresholds=thr,
        frequencies=np.asarray(g["frequencies"][()], dtype=np.float32),
        peaks=peaks, auto_approved=auto,
        fr=float(_scalar(g, "imaging_frequency", np.nan)),
        decay_time=float(_scalar(g, "decay_time", np.nan)),
        threshold=float(_scalar(g, "detection_cutoff", 0.2)),
        version=float(_scalar(g, "version", SLAB_VERSION)))
    return res, appr


# ── headless HDF5 entry point (SLab "Auto Mode") ─────────────────────────────

def _h5_get(f, path):
    if path not in f:
        return None
    obj = f[path]
    if not hasattr(obj, "shape"):
        return None
    v = obj[()]
    if isinstance(v, (bytes, str)) and v in (b"NoneType", "NoneType"):
        return None
    return v


def detect_events_in_hdf5(path: str, *, overwrite: bool = False,
                          threshold: float = 0.2, min_snr: float = 3.0,
                          min_cnn: float = 0.15, lowpass_hz=None,
                          fr: Optional[float] = None,
                          decay_time: Optional[float] = None) -> str:
    """Detect events on every row of ``estimates/F_dff`` in a CaImAn results
    file and write ``stroh_results`` *into that file* (SLab behaviour; nothing
    else in the file is touched).  Rows index the full estimates (not
    ``idx_components``), exactly as SLab does.  Returns a status string."""
    import h5py
    with h5py.File(path, "r") as f:
        if "stroh_results" in f and not overwrite:
            return "skipped (stroh_results exists; use overwrite)"
        F = _h5_get(f, "estimates/F_dff")
        C = _h5_get(f, "estimates/C")
        if F is None or C is None:
            return "skipped (no estimates/F_dff — run detrend_df_f first)"
        snr = _h5_get(f, "estimates/SNR_comp")
        cnn = _h5_get(f, "estimates/cnn_preds")
        fr = fr if fr is not None else _h5_get(f, "params/data/fr")
        decay_time = (decay_time if decay_time is not None
                      else _h5_get(f, "params/data/decay_time"))
    if fr is None or decay_time is None:
        return "skipped (fr / decay_time unknown; pass them explicitly)"
    res = detect_events(F, C, float(fr), float(decay_time), snr, cnn,
                        threshold=threshold, min_snr=min_snr, min_cnn=min_cnn,
                        lowpass_hz=lowpass_hz)
    with h5py.File(path, "a") as f:
        write_stroh_results(f, res, res.auto_approved)
    return (f"{int(res.auto_approved.sum())}/{res.n} auto-approved, "
            f"{sum(len(p) for p in res.peaks)} events")


def find_results_files(root: str) -> list:
    """Every ``*.hdf5`` / ``*.h5`` under ``root`` except ``*_curated`` copies."""
    import glob
    out = []
    for ext in ("hdf5", "h5"):
        out += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
    return sorted(p for p in set(out)
                  if not os.path.splitext(p)[0].endswith("_curated"))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="SLab-compatible calcium-event detection on CaImAn results "
                    "(writes a stroh_results group into each file).")
    ap.add_argument("paths", nargs="+", help="results .hdf5 files or folders")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="ΔF/F cap on the noise-derived threshold (default 0.2)")
    ap.add_argument("--min-snr", type=float, default=3.0)
    ap.add_argument("--min-cnn", type=float, default=0.15)
    ap.add_argument("--lowpass-hz", type=float, default=None,
                    help="physical cutoff; default = SLab's 0.974·Nyquist")
    ap.add_argument("--fr", type=float, default=None)
    ap.add_argument("--decay-time", type=float, default=None)
    a = ap.parse_args(argv)
    files = []
    for p in a.paths:
        files += find_results_files(p) if os.path.isdir(p) else [p]
    for p in files:
        try:
            msg = detect_events_in_hdf5(
                p, overwrite=a.overwrite, threshold=a.threshold,
                min_snr=a.min_snr, min_cnn=a.min_cnn, lowpass_hz=a.lowpass_hz,
                fr=a.fr, decay_time=a.decay_time)
        except Exception as exc:                       # keep the batch going
            msg = f"ERROR {exc}"
        print(f"{p}: {msg}")


if __name__ == "__main__":
    main()
