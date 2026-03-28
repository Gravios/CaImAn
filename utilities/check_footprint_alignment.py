#!/usr/bin/env python3
"""
Overlay the single brightest footprint on Cn to test whether the
coordinate assembly is placing footprints at the correct FOV position.

Usage:
    python check_footprint_alignment.py <hdf5_estimates_file> <Cn_npy_file>

Example:
    python check_footprint_alignment.py \
        stroh-sa-2966-20251222-record-0001_cnmf.hdf5 \
        stroh-sa-2966-20251222-record-0001_Cn.npy
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import caiman


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    hdf5_path = Path(sys.argv[1])
    cn_path   = Path(sys.argv[2])

    print(f"Loading estimates: {hdf5_path}")
    from caiman.utils.utils import load_dict_from_hdf5
    import types, scipy.sparse

    d = load_dict_from_hdf5(str(hdf5_path))

    # Build a minimal namespace with just what we need
    cnm = types.SimpleNamespace()
    cnm.dims = tuple(int(x) for x in d['dims'])

    estimates = types.SimpleNamespace()
    # A is stored as a sparse matrix dict or dense array
    A_raw = d['estimates']['A']
    if isinstance(A_raw, dict) and 'data' in A_raw:
        estimates.A = scipy.sparse.csc_matrix(
            (A_raw['data'], A_raw['indices'], A_raw['indptr']),
            shape=tuple(int(x) for x in A_raw['shape']))
    elif scipy.sparse.issparse(A_raw):
        estimates.A = A_raw.tocsc()
    else:
        estimates.A = scipy.sparse.csc_matrix(A_raw)
    cnm.estimates = estimates

    print(f"Loading Cn:        {cn_path}")
    Cn = np.load(str(cn_path))

    d1, d2 = cnm.dims
    K      = cnm.estimates.A.shape[1]
    A      = np.asarray(cnm.estimates.A.todense())  # (d1*d2, K)

    print(f"  dims: ({d1}, {d2})   K={K}")
    print(f"  Cn shape: {Cn.shape}")

    # ── Footprint max projection ──────────────────────────────────────────────
    A_max = A.max(axis=1).reshape(d1, d2, order='F')

    # ── Pick brightest 5 components and their COM ────────────────────────────
    component_masses = np.asarray(A.sum(axis=0)).ravel()
    top5 = np.argsort(component_masses)[::-1][:5]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: Cn alone
    axes[0].imshow(Cn, cmap='gray', origin='upper')
    axes[0].set_title(f'Cn  ({d1}×{d2})', fontsize=9)
    axes[0].axis('off')

    # Middle: A_max alone
    nz = A_max[A_max > 0]
    vmax = float(np.percentile(nz, 99)) if nz.size else 1.0
    axes[1].imshow(A_max, cmap='hot', vmin=0, vmax=vmax, origin='upper')
    axes[1].set_title('Max footprint projection', fontsize=9)
    axes[1].axis('off')

    # Right: Cn + contours of ALL components + markers for top5
    axes[2].imshow(Cn, cmap='gray', origin='upper')
    A_vol = A.reshape(d1, d2, K, order='F')
    for k in range(K):
        comp = A_vol[:, :, k]
        if comp.max() < 1e-10:
            continue
        axes[2].contour(comp, levels=[comp.max() * 0.3],
                        colors=['#00e5ff'], linewidths=0.4, alpha=0.7)

    # Mark top-5 brightest component COMs
    import scipy.ndimage
    for rank, k in enumerate(top5):
        fp = A_vol[:, :, k]
        cy, cx = scipy.ndimage.center_of_mass(fp)  # (row, col)
        axes[2].plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)
        axes[2].text(cx+3, cy+3, f'#{rank+1}', color='red', fontsize=7)

    axes[2].set_title('All contours + top-5 COM (red +)', fontsize=9)
    axes[2].axis('off')

    # ── Quadrant stats ────────────────────────────────────────────────────────
    h, w = d1 // 2, d2 // 2
    quad_names = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    quad_slices = [
        (slice(None, h), slice(None, w)),
        (slice(None, h), slice(w, None)),
        (slice(h, None), slice(None, w)),
        (slice(h, None), slice(w, None)),
    ]
    print("\nComponent COM distribution by quadrant:")
    quad_counts = {n: 0 for n in quad_names}
    for k in range(K):
        fp = A_vol[:, :, k]
        if fp.max() < 1e-10:
            continue
        cy, cx = scipy.ndimage.center_of_mass(fp)
        if cy < h and cx < w:   quad_counts['top-left']     += 1
        elif cy < h:             quad_counts['top-right']    += 1
        elif cx < w:             quad_counts['bottom-left']  += 1
        else:                    quad_counts['bottom-right'] += 1
    for n, c in quad_counts.items():
        cn_q = Cn[quad_slices[quad_names.index(n)]]
        print(f"  {n:15s}: {c:3d} components  Cn_mean={cn_q.mean():.3f}")

    fig.suptitle(hdf5_path.name, fontsize=9, y=1.01)
    plt.tight_layout()

    out = hdf5_path.with_name(hdf5_path.stem + '_alignment_check.png')
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
