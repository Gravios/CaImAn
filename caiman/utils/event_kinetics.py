"""
caiman/utils/event_kinetics.py
==============================
Kinetic check of detected calcium events against the raw ΔF/F.

CNMF's denoised trace C is an autoregressive fit, so *every* transient in C
has a fast rise and an exponential decay — including ones fitted to
neuropil, a neighbour's spill-over or noise.  Whether an event is a real
calcium transient therefore has to be judged on a trace the model did not
shape: the raw trace C + YrA rescaled to ΔF/F (:func:`raw_dff`), or
``estimates.F_dff`` when YrA is gone — which is only independent of C if it
was computed with the residual (CaImAn's default ``use_residuals=True``;
:func:`dff_tracks_C` tells).

For every event of :mod:`caiman.utils.event_detection` this module fits,
on raw ΔF/F around the event,

    y(t) ≈ b + A·h(t),  h = 0 before onset, linear rise onset → peak,
                        exp(−(t − peak)/τ) after the peak,

and reports

``shape_r``   Pearson r of y with h at the *expected* τ (the indicator's
              decay time) — "does this look like the transient we expect?"
``tau_s``     best-fitting τ over a log grid (0.1–10 × expected)
``rise_s``    raw rise time, 20 % → 100 % of the raw peak (± 2 frames of
              the detected one) above the trough preceding it
``amp_snr``   A / σ, σ = 1.4826·MAD of the row's ΔF/F outside events
``valid``     every set criterion met (see :data:`DEFAULTS`)

An event whose decay window is cut short (< 4 samples before the next
event or the end of the recording) is *untestable*: ``valid`` is False and
``fail`` carries ``short``; per-component fractions use testable events
only.  Detection and approval are not changed by this module.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULTS = {
    "min_shape_r":   0.7,    # Pearson r with the expected-τ template
    "tau_min_frac":  0.5,    # fitted τ ≥ this × expected decay time
    "tau_max_frac":  2.0,    # fitted τ ≤ this × expected decay time
    "max_rise_s":    0.15,   # raw rise (20 % → peak) in seconds
    "min_amp_snr":   3.0,    # amplitude / robust noise of the raw ΔF/F
    "decay_time":    None,   # s; None → the events' / file's decay time
}
FAIL = {"shape": 1, "tau": 2, "rise": 4, "amplitude": 8, "short": 16}
EVENT_FIELDS = ("shape_r", "tau_s", "rise_s", "amp", "amp_snr", "baseline",
                "start", "end", "fail", "valid")
SUMMARY_FIELDS = ("n_events", "n_testable", "n_valid", "frac_valid",
                  "median_shape_r", "median_tau_s", "noise_sigma")
_MIN_DECAY = 4                       # samples after the peak for a fit
_TAU_GRID = np.geomspace(0.1, 10.0, 31)   # + parabolic refinement


def raw_dff(C, YrA, F_dff) -> np.ndarray:
    """(C + YrA) on each row's ΔF/F scale.

    Per row F0 = Σc² / Σc·ΔF/F (ΔF/F ≈ C / F0 fitted through the origin);
    rows where that fails (no signal, F0 ≤ 0) keep their ``F_dff`` row.
    The local pre-event baseline of :func:`validate_row` absorbs slow
    drifts that CaImAn's detrending would have removed."""
    Cc = np.asarray(C, dtype=np.float64)
    raw = Cc + np.asarray(YrA, dtype=np.float64)
    D = np.asarray(F_dff, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        F0 = (Cc * Cc).sum(1) / (Cc * D).sum(1)
    ok = np.isfinite(F0) & (F0 > 0)
    out = D.copy()
    out[ok] = raw[ok] / F0[ok, None]
    return out


def dff_tracks_C(F_dff, C, r_min: float = 0.99) -> float:
    """Median over rows of corr(ΔF/F, C), or NaN.  Above ``r_min`` the ΔF/F
    is (nearly) C rescaled — computed without the residual — and a kinetic
    check on it is circular (C has the model's kinetics by construction)."""
    F = np.asarray(F_dff, dtype=np.float64)
    Cc = np.asarray(C, dtype=np.float64)
    F = F - F.mean(1, keepdims=True); Cc = Cc - Cc.mean(1, keepdims=True)
    den = np.sqrt((F * F).sum(1) * (Cc * Cc).sum(1))
    ok = den > 0
    if not ok.any():
        return float("nan")
    return float(np.median((F * Cc).sum(1)[ok] / den[ok]))


def resolve_params(cfg: Optional[dict] = None) -> dict:
    p = dict(DEFAULTS)
    for k, v in (cfg or {}).items():
        if k in p:
            p[k] = v
    return p


def decode_fail(mask: int) -> list:
    return [n for n, b in FAIL.items() if int(mask) & b]


def _template(n_pre: int, n_rise: int, n_decay: int, tau_frames) -> np.ndarray:
    """(len(tau), n_pre + n_rise + n_decay) templates; peak value 1 at
    index n_pre + n_rise (first decay sample)."""
    tau = np.atleast_1d(np.asarray(tau_frames, float))[:, None]
    pre = np.zeros((tau.shape[0], n_pre))
    # onset is the last baseline sample: 0 at onset, linear up to the peak
    rise = np.broadcast_to(np.arange(n_rise) / max(n_rise, 1), (tau.shape[0], n_rise))
    dec = np.exp(-np.arange(n_decay)[None, :] / tau)
    return np.hstack([pre, rise, dec])


def _empty_row(n: int) -> dict:
    row = {k: np.full(n, np.nan) for k in EVENT_FIELDS}
    row["start"] = np.zeros(n, np.int64); row["end"] = np.zeros(n, np.int64)
    row["fail"] = np.zeros(n, np.int64); row["valid"] = np.zeros(n, bool)
    return row


def validate_row(dff, peaks, fr: float, decay_time: float,
                 params: Optional[dict] = None) -> dict:
    """Kinetic fields for one component.

    ``dff``    (T,) raw ΔF/F;  ``peaks`` (n, 4) event_detection rows
    (peak, amp, onset, offset).  Returns a dict of (n,) arrays
    (:data:`EVENT_FIELDS`) plus ``noise_sigma`` / ``noise_baseline``
    scalars.
    """
    p = resolve_params(params)
    y = np.asarray(dff, dtype=np.float64).ravel()
    T = y.size
    pk = np.asarray(peaks, float).reshape(-1, 4)
    n = pk.shape[0]
    row = _empty_row(n)
    fr = float(fr)
    dec = float(p["decay_time"] or decay_time or np.nan)
    if T == 0 or not (fr > 0 and np.isfinite(dec) and dec > 0):
        row["noise_sigma"] = row["noise_baseline"] = np.nan
        row["fail"][:] = FAIL["short"]
        return row
    tau_exp = dec * fr                                   # frames
    n_pre = max(3, int(round(0.2 * fr)))
    n_dec_max = int(np.ceil(3 * tau_exp)) + 1
    order = np.argsort(pk[:, 0]) if n else np.zeros(0, int)
    peak_f = np.clip(np.round(pk[:, 0]).astype(np.int64), 0, max(T - 1, 0))
    onset = np.clip(np.round(pk[:, 2]).astype(np.int64), 0, max(T - 1, 0))
    onset = np.minimum(onset, peak_f)

    # robust noise / baseline outside every event window
    quiet = np.ones(T, bool)
    for k in range(n):
        quiet[max(0, onset[k] - n_pre):min(T, peak_f[k] + n_dec_max)] = False
    q = y[quiet] if quiet.sum() >= 20 else y
    med = float(np.median(q))
    sigma = float(1.4826 * np.median(np.abs(q - med)))
    row["noise_sigma"], row["noise_baseline"] = sigma, med

    # neighbours in time: next event's onset, previous event's peak
    next_on = np.full(n, T, np.int64)
    prev_pk = np.full(n, -1, np.int64)
    if n > 1:
        a, b_ = order[:-1], order[1:]
        next_on[a] = np.maximum(onset[b_], peak_f[a] + 1)
        prev_pk[b_] = peak_f[a]
    start = np.maximum(0, onset - n_pre)
    end = np.minimum(np.minimum(T, peak_f + n_dec_max), next_on)
    row["start"][:], row["end"][:] = start, end
    testable = (end - peak_f) >= _MIN_DECAY
    row["fail"][~testable] = FAIL["short"]
    E = np.flatnonzero(testable)
    if E.size == 0:
        return row

    # every testable event's window, padded to one (m, L) block
    on, pf, st, en = onset[E], peak_f[E], start[E], end[E]
    L = int((en - st).max())
    idx = st[:, None] + np.arange(L)[None, :]
    msk = idx < en[:, None]
    seg = y[np.clip(idx, 0, T - 1)]
    pre_m = msk & (idx < on[:, None])
    with np.errstate(all="ignore"), _quiet_nanwarn():
        b = np.nanmedian(np.where(pre_m, seg, np.nan), axis=1)
    b = np.where(pre_m.sum(1) >= 3, b, med)              # local baseline
    r = np.where(msk, seg - b[:, None], 0.0)
    nrise = np.maximum(pf - on, 1)[:, None]
    lin = np.where(idx < on[:, None], 0.0, (idx - on[:, None]) / nrise)  # pre 0, rise
    dk = np.clip(idx - pf[:, None], 0, None).astype(np.float64)
    decay_part = idx >= pf[:, None]

    def templates(tau_frames):
        """(..., m, L) templates for a scalar or (G,) array of τ (frames)."""
        tau = np.asarray(tau_frames, float)[..., None, None]
        h = np.where(decay_part, np.exp(-dk / tau), lin)
        return np.where(msk, h, 0.0)

    # expected-τ template: amplitude and shape r (masked Pearson)
    h = templates(tau_exp)
    hh = (h * h).sum(1)
    with np.errstate(all="ignore"):
        A = np.where(hh > 0, np.maximum(0.0, (h * r).sum(1) / hh), 0.0)
        nm = msk.sum(1)
        ds = np.where(msk, seg - (np.where(msk, seg, 0).sum(1) / nm)[:, None], 0.0)
        dh = np.where(msk, h - (h.sum(1) / nm)[:, None], 0.0)
        den = np.sqrt((ds * ds).sum(1) * (dh * dh).sum(1))
        shape_r = np.where(den > 0, (ds * dh).sum(1) / den, np.nan)

        # best τ on the grid (amplitude free per τ), parabola in log τ
        H = templates(_TAU_GRID * tau_exp)                          # (G, m, L)
        Ag = np.maximum(0.0, (H * r).sum(2) / (H * H).sum(2))
        sse = (((r[None] - Ag[..., None] * H) ** 2) * msk).sum(2)  # (G, m)
    jg = np.argmin(sse, axis=0)
    lt = np.log(_TAU_GRID[jg])
    inner = (jg > 0) & (jg < _TAU_GRID.size - 1)
    cols = np.arange(E.size)
    jj = np.clip(jg, 1, _TAU_GRID.size - 2)
    y0, y1, y2 = sse[jj - 1, cols], sse[jj, cols], sse[jj + 1, cols]
    dd = y0 - 2 * y1 + y2
    step = np.log(_TAU_GRID[1] / _TAU_GRID[0])
    with np.errstate(all="ignore"):
        lt = np.where(inner & (dd > 0), lt + 0.5 * step * (y0 - y2) / dd, lt)
    tau_s = np.exp(lt) * dec

    # raw rise: 20 % → 100 % of the raw peak above the trough that precedes
    # it (≤ 1 s back, after the previous event's peak, so an event riding on
    # an earlier one's tail is measured from its trough)
    lo = np.maximum(on, pf - 2); hi = np.minimum(T, pf + 3)
    w5 = lo[:, None] + np.arange(5)[None, :]
    v5 = np.where(w5 < hi[:, None], y[np.clip(w5, 0, T - 1)], -np.inf)
    rp = lo + np.argmax(v5, axis=1)
    b0 = np.maximum(np.maximum(0, rp - int(np.ceil(fr))), prev_pk[E] + 1)
    Lb = int((rp - b0).max()) + 1
    ib = b0[:, None] + np.arange(Lb)[None, :]
    vb = ib <= rp[:, None]
    back = y[np.clip(ib, 0, T - 1)]
    trough = np.where(vb, back, np.inf).min(1)
    top = y[rp] - trough
    below = vb & (back <= (trough + 0.2 * top)[:, None]) & (top > 0)[:, None]
    last = np.where(below, np.arange(Lb)[None, :], -1).max(1)
    rise_s = np.where(last >= 0, (rp - b0 - last) / fr, np.nan)

    row["shape_r"][E], row["tau_s"][E], row["rise_s"][E] = shape_r, tau_s, rise_s
    row["amp"][E], row["baseline"][E] = A, b
    with np.errstate(all="ignore"):
        amp_snr = A / sigma if sigma > 0 else np.full(E.size, np.nan)
    row["amp_snr"][E] = amp_snr

    fail = np.zeros(E.size, np.int64)
    with np.errstate(invalid="ignore"):
        thr = p.get("min_shape_r")
        if thr is not None:
            fail |= np.where(shape_r >= float(thr), 0, FAIL["shape"])
        lo_f, hi_f = p.get("tau_min_frac"), p.get("tau_max_frac")
        t_rel = tau_s / dec
        if lo_f is not None:
            fail |= np.where(t_rel >= float(lo_f), 0, FAIL["tau"])
        if hi_f is not None:
            fail |= np.where(t_rel <= float(hi_f), 0, FAIL["tau"])
        mr = p.get("max_rise_s")
        if mr is not None:
            fail |= np.where(rise_s <= float(mr), 0, FAIL["rise"])
        ms = p.get("min_amp_snr")
        if ms is not None:
            fail |= np.where(amp_snr >= float(ms), 0, FAIL["amplitude"])
    row["fail"][E] = fail
    row["valid"][E] = fail == 0
    return row


class _quiet_nanwarn:
    """Silence 'All-NaN slice' from nanmedian on events without a pre window."""
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)


def template_curve(row: dict, k: int, fr: float, decay_time: float,
                   peaks) -> tuple:
    """(t, y) of event k's fitted expected-τ template, for plotting."""
    pk = np.asarray(peaks, float).reshape(-1, 4)
    start, end = int(row["start"][k]), int(row["end"][k])
    if end <= start or not np.isfinite(row["amp"][k]):
        return np.zeros(0), np.zeros(0)
    pf, on = int(round(pk[k, 0])), int(round(pk[k, 2]))
    on = min(on, pf)
    h = _template(on - start, pf - on, end - pf, decay_time * fr)[0]
    return np.arange(start, end, dtype=float), row["baseline"][k] + row["amp"][k] * h


def summarize_row(row: dict) -> dict:
    """Per-component summary (:data:`SUMMARY_FIELDS`)."""
    fail = np.asarray(row["fail"])
    testable = (fail & FAIL["short"]) == 0
    n_t = int(testable.sum())
    n_v = int(np.asarray(row["valid"]).sum())
    r = np.asarray(row["shape_r"])[testable]
    tau = np.asarray(row["tau_s"])[testable]
    return {"n_events": int(fail.size), "n_testable": n_t, "n_valid": n_v,
            "frac_valid": n_v / n_t if n_t else np.nan,
            "median_shape_r": float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan,
            "median_tau_s": float(np.nanmedian(tau)) if np.isfinite(tau).any() else np.nan,
            "noise_sigma": float(row.get("noise_sigma", np.nan))}


def validate(F_dff, events, params: Optional[dict] = None,
             rows=None) -> list:
    """:func:`validate_row` for every row (or ``rows``) of an
    ``event_detection.EventResults``; None for rows not computed."""
    p = resolve_params(params)
    F = np.asarray(F_dff)
    todo = range(events.n) if rows is None else rows
    out = [None] * events.n
    for r in todo:
        out[r] = validate_row(F[r], events.peaks[r], events.fr, events.decay_time, p)
    return out


def summary_table(rows: list) -> dict:
    """(K,) arrays of :data:`SUMMARY_FIELDS` over a :func:`validate` list."""
    s = [summarize_row(r) if r is not None else None for r in rows]
    out = {}
    for f in SUMMARY_FIELDS:
        dt = np.int64 if f.startswith("n_") else float
        fill = 0 if dt is np.int64 else np.nan
        out[f] = np.array([x[f] if x is not None else fill for x in s], dtype=dt)
    return out


def event_table(rows: list, which) -> dict:
    """Per-event fields for rows ``which`` concatenated in event order —
    aligned with ``EventResults.peak_table(which)``."""
    cols = {f: [] for f in ("shape_r", "tau_s", "rise_s", "amp_snr", "valid", "fail")}
    for r in which:
        row = rows[r]
        for f in cols:
            cols[f].append(np.asarray(row[f]))
    return {f: (np.concatenate(v) if v else np.zeros(0)) for f, v in cols.items()}
