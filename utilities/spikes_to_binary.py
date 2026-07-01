#!/usr/bin/env python3
"""
spikes_to_binary.py — convert CaImAn deconvolved S into an up-sampled
binary spike matrix and save it as a compressed .npz.

The deconvolved activity ``S`` from CNMF/OASIS is continuous and lives in
fluorescence units, not spike counts. This tool calibrates ``S`` to an
expected per-frame spike count, then draws a binary spike raster at a finer
time resolution using a Poisson model.

Pipeline
--------
1. Calibrate S -> expected spike count per frame:
       lambda[i, t] = S[i, t] / A_unit[i]
   ``A_unit[i]`` (the single-spike amplitude) is estimated per component as a
   percentile of the non-zero S values (default: the median, i.e. a typical
   OASIS event ~= 1 spike). Override with --unit-amp, or pass --global-scale K
   to instead use lambda = K * S with one factor for every component.

2. Up-sample by factor U: each 1/fr frame is split into U fine bins
   (default U = 33, i.e. ~1 ms bins at 30 Hz).

3. Poisson assignment: each frame's expected count lambda is treated as an
   inhomogeneous Poisson process over its U fine bins — every fine bin draws
   ~ Poisson(lambda / U) — and the result is clipped to {0, 1}. With U large
   enough, >1 spike per fine bin is rare; the clipped fraction is reported.
   This is how the non-integer per-frame counts are assigned to time bins.

The output is a single stochastic realization for a fixed --seed; rerun with
different seeds for more draws, or use the stored ``lambda_frame`` for the
deterministic expectation.

Usage
-----
    # output name auto-derived: <base>_results.hdf5 -> <base>_spikes.npz
    python spikes_to_binary.py results.hdf5
    python spikes_to_binary.py results.hdf5 -o custom.npz   # explicit override
    python spikes_to_binary.py results.hdf5 --min-snr 1.5 --upsample 33
    python spikes_to_binary.py results.hdf5 --all --global-scale 1e-3
    python spikes_to_binary.py results.hdf5 --fr 30.0    # if fr not in params

Output .npz
-----------
    spikes        uint8 (N, T*U)   binary raster (1 = spike)
    comp_idx      int   (N,)       component indices (row -> component)
    lambda_frame  float (N, T)     calibrated expected counts per frame
    unit_amp      float (N,)       single-spike amplitude used per component
    n_spikes      int   (N,)       total spikes per component
    mean_rate_hz  float (N,)       mean firing rate per component
    fr_orig, fr_up, dt, upsample, seed
"""

import argparse
import os
import re
import numpy as np
import h5py


# ── I/O ─────────────────────────────────────────────────────────────────────

def default_out_path(hdf5_path):
    """Derive '<base>_spikes.npz' from the input, alongside it.

    A trailing results/cnmf/cnm/estimates/curated tag is stripped so the
    output shares the session base with the input, e.g.
        .../sess_results.hdf5  ->  .../sess_spikes.npz
    """
    folder = os.path.dirname(os.path.abspath(hdf5_path))
    stem   = os.path.splitext(os.path.basename(hdf5_path))[0]
    base   = re.sub(r'[._-]?(results?|cnmf?|cnm|estimates?|curated)$', '',
                    stem, flags=re.I)
    return os.path.join(folder, f"{base}_spikes.npz")


def load_S(path, fr_fallback=None):
    """Read only S, SNR_comp, idx_components and fr from a results .hdf5.

    Targeted reads keep this cheap on large result files (S is not loaded
    alongside A / C / YrA). Returns (S, good_idx, SNR_comp, fr).
    """
    with h5py.File(path, "r") as f:
        est = f["estimates"]
        S = np.asarray(est["S"][()], dtype=np.float64)
        good = (np.asarray(est["idx_components"][()]).astype(int)
                if "idx_components" in est else np.arange(S.shape[0]))
        snr = (np.asarray(est["SNR_comp"][()], dtype=np.float64)
               if "SNR_comp" in est else np.full(S.shape[0], np.nan))
        fr = None
        if "params" in f and "data" in f["params"] and "fr" in f["params/data"]:
            fr = float(f["params/data/fr"][()])
    if fr is None:
        fr = fr_fallback
    if fr is None:
        raise ValueError("frame rate not found in params/data/fr; pass --fr")
    return S, good, snr, fr


# ── Calibration + binarisation ────────────────────────────────────────────────

def unit_amplitudes(S, q, eps_frac=1e-6):
    """Per-component single-spike amplitude = q-th percentile of non-zero S."""
    amps = np.empty(S.shape[0])
    for i in range(S.shape[0]):
        s = S[i]
        nz = s[s > max(1e-12, eps_frac * s.max())]
        amps[i] = np.percentile(nz, q) if nz.size else np.nan
    return amps


def binarize(S, upsample, q=50.0, global_scale=None, seed=0):
    """Return (spikes, lambda_frame, unit_amp, total_spikes, clipped_spikes).

    ``spikes`` is uint8 (N, T*U); a Poisson draw per fine bin, clipped to {0,1}.
    """
    rng = np.random.default_rng(seed)

    if global_scale is not None:
        unit = np.full(S.shape[0], 1.0 / global_scale)
        lam_frame = S * global_scale
    else:
        unit = unit_amplitudes(S, q)
        lam_frame = S / unit[:, None]
    lam_frame = np.clip(lam_frame, 0, None)          # drop tiny negatives

    lam_fine = np.repeat(lam_frame / upsample, upsample, axis=1)
    counts   = rng.poisson(lam_fine)
    spikes   = (counts > 0).astype(np.uint8)

    total   = int(counts.sum())
    clipped = int((counts[counts > 1] - 1).sum())    # spikes merged by clip
    return spikes, lam_frame, unit, total, clipped


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Up-sample CaImAn deconvolved S into a binary spike .npz")
    ap.add_argument("hdf5", help="CaImAn results .hdf5 file")
    ap.add_argument("-o", "--out", default=None,
                    help="output .npz path (default: <input-base>_spikes.npz "
                         "next to the input, with any _results tag stripped)")
    ap.add_argument("--upsample", type=int, default=33,
                    help="fine bins per frame (33 ~ 1 ms at 30 Hz)")
    ap.add_argument("--unit-amp", type=float, default=50.0,
                    help="percentile of non-zero S used as the single-spike "
                         "amplitude (default: 50 = median)")
    ap.add_argument("--global-scale", type=float, default=None,
                    help="use lambda = scale * S instead of a per-component "
                         "unit amplitude")
    ap.add_argument("--min-snr", type=float, default=1.5,
                    help="keep accepted components with SNR_comp >= this")
    ap.add_argument("--all", action="store_true",
                    help="keep every component (ignore idx_components / SNR)")
    ap.add_argument("--fr", type=float, default=None,
                    help="frame rate (Hz) if not stored in params/data/fr")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the Poisson draw")
    a = ap.parse_args()

    out = a.out if a.out is not None else default_out_path(a.hdf5)

    S, good, snr, fr = load_S(a.hdf5, fr_fallback=a.fr)

    if a.all:
        idx = np.arange(S.shape[0])
    else:
        idx = np.array([i for i in good if snr[i] >= a.min_snr], dtype=int)
        idx = idx[np.argsort(-snr[idx])]              # SNR-descending rows
    if idx.size == 0:
        raise SystemExit("no components selected (try --all or lower --min-snr)")

    spikes, lam, unit, total, clipped = binarize(
        S[idx], a.upsample, q=a.unit_amp,
        global_scale=a.global_scale, seed=a.seed)

    fr_up = fr * a.upsample
    dur   = S.shape[1] / fr
    rate  = spikes.sum(1) / dur

    np.savez_compressed(
        out,
        spikes=spikes, comp_idx=idx.astype(int),
        lambda_frame=lam.astype(np.float32), unit_amp=unit.astype(np.float32),
        n_spikes=spikes.sum(1).astype(int), mean_rate_hz=rate.astype(np.float32),
        fr_orig=fr, fr_up=fr_up, dt=1.0 / fr_up, upsample=a.upsample, seed=a.seed)

    print(f"{idx.size} components -> binary {spikes.shape} @ {fr_up:.0f} Hz "
          f"(dt={1e3 / fr_up:.2f} ms)")
    print(f"total spikes={total}  clipped(>1/bin)={clipped} "
          f"({100 * clipped / max(total, 1):.2f}%)")
    print(f"firing rate Hz: median={np.median(rate):.2f} "
          f"range={rate.min():.2f}-{rate.max():.2f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
