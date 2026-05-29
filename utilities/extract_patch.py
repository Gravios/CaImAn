#!/usr/bin/env python3
"""
extract_patch.py — pull a small corner patch over a frame range from an
MSR (or TIFF) stack, save as a multi-page TIFF for fast inspection /
upload, plus a sanity-check PNG.

Usage
-----
    python extract_patch.py /path/to/session.msr
    python extract_patch.py /path/to/session.msr --y0 0 --x0 0 --size 64 \\
                            --start 1 --n 100 --skip-first

Notes
-----
- Uses caiman.utils.stack_io.StackReader so the same dispatch logic works
  for .msr / .tif / .tiff inputs.
- Default skips frame 0 because it has been observed to be all-zeros on
  the .msr exports from this Leica acquisition (a setup pulse / shutter-
  closed frame that would distort temporal statistics).
- Outputs (in the same directory as the input):
    <stem>_patch_<y0>x<x0>_<size>x<size>_<start>+<n>.tif
    <stem>_patch_<y0>x<x0>_<size>x<size>_<start>+<n>_inspect.png
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from caiman.utils.stack_io import StackReader


def extract(src: Path,
            y0: int, x0: int, size: int,
            start: int, n: int) -> np.ndarray:
    """Return an (n, size, size) array of the upper-left patch over the
    requested frame range."""
    with StackReader(src) as r:
        H, W = r.h, r.w
        T = r.n_frames
        if y0 + size > H or x0 + size > W:
            raise ValueError(
                f"patch ({y0}+{size}, {x0}+{size}) exceeds frame ({H}, {W})")
        if start + n > T:
            print(f"  warning: requested {start}+{n} > {T} frames; "
                  f"clipping to {T - start}")
            n = T - start
        out = np.empty((n, size, size), dtype=r.dtype)
        for i in range(n):
            fr = r.read_frame(start + i)
            if fr.ndim > 2:
                fr = fr.reshape(fr.shape[-2], fr.shape[-1])
            out[i] = fr[y0:y0 + size, x0:x0 + size]
    return out


def make_inspect_png(patch: np.ndarray, png_path: Path,
                     y0: int, x0: int, start: int) -> None:
    """Quick visual sanity check: 4 sample frames + temporal mean + std +
    a few pixel time series + 2D FFT of temporal mean."""
    T, H, W = patch.shape
    pf = patch.astype(np.float32)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # Row 1: four sample frames
    sample_idx = np.linspace(0, T - 1, 4).astype(int)
    vmin, vmax = np.percentile(pf, [2, 98])
    for i, fi in enumerate(sample_idx):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(pf[fi], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"frame {start + fi}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Row 2 left: temporal mean, std, max
    ax = fig.add_subplot(gs[1, 0])
    M = pf.mean(axis=0)
    ax.imshow(M, cmap="gray")
    ax.set_title(f"temporal mean\nrange {M.min():.0f}-{M.max():.0f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    S = pf.std(axis=0)
    ax.imshow(S, cmap="magma")
    ax.set_title(f"temporal std\nrange {S.min():.1f}-{S.max():.1f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    Mx = pf.max(axis=0)
    ax.imshow(Mx, cmap="gray")
    ax.set_title("temporal max", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2 right: 2D FFT of temporal mean
    ax = fig.add_subplot(gs[1, 3])
    F = np.fft.fftshift(np.fft.fft2(M - M.mean()))
    P = np.log10(np.abs(F)**2 + 1)
    ax.imshow(P, cmap="viridis")
    ax.set_title("log|FFT(temporal mean)|²", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Row 3: pixel time series — a small grid of 4×4 pixel locations
    ax = fig.add_subplot(gs[2, :])
    grid = np.linspace(H // 8, H - H // 8, 4).astype(int)
    for i, py in enumerate(grid):
        for j, px in enumerate(grid):
            trace = pf[:, py, px] + (i * 4 + j) * 30  # offset for visibility
            ax.plot(trace, lw=0.6, alpha=0.8)
    ax.set_xlabel("frame index (within patch)")
    ax.set_ylabel(f"intensity (DN, offset)")
    ax.set_title(f"pixel time series at 4×4 grid of locations within the patch ({T} frames)",
                 fontsize=10)
    ax.set_xlim(0, T - 1)

    fig.suptitle(
        f"patch at ({y0},{x0}) {H}×{W}, frames {start}–{start + T - 1}  "
        f"(T={T})", fontsize=12)
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract a corner patch over a frame range from a "
                    "stack file (.msr/.tif/.tiff).")
    ap.add_argument("src", type=Path, help="input stack file")
    ap.add_argument("--y0", type=int, default=0,
                    help="patch top-row index (default 0)")
    ap.add_argument("--x0", type=int, default=0,
                    help="patch left-column index (default 0)")
    ap.add_argument("--size", type=int, default=64,
                    help="patch size in pixels (square; default 64)")
    ap.add_argument("--start", type=int, default=0,
                    help="first frame to extract (default 0)")
    ap.add_argument("--n", type=int, default=100,
                    help="number of frames to extract (default 100)")
    ap.add_argument("--skip-first", action="store_true",
                    help="bump start to 1 if start=0 (avoid black setup frame)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: same as src)")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    if not src.exists():
        print(f"error: {src} does not exist", file=sys.stderr)
        return 2

    if args.skip_first and args.start == 0:
        print("  --skip-first: bumping start from 0 → 1")
        args.start = 1

    out_dir = (args.out_dir or src.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem + (f"_patch_{args.y0}x{args.x0}_"
                       f"{args.size}x{args.size}_{args.start}+{args.n}")
    tif_path = out_dir / f"{stem}.tif"
    png_path = out_dir / f"{stem}_inspect.png"

    print(f"  source: {src}")
    print(f"  patch : y0={args.y0} x0={args.x0} size={args.size}")
    print(f"  frames: {args.start} .. {args.start + args.n - 1}  (n={args.n})")
    patch = extract(src, args.y0, args.x0, args.size, args.start, args.n)
    T, H, W = patch.shape
    print(f"  → patch shape {patch.shape}, dtype {patch.dtype}, "
          f"mean {patch.mean():.1f}, std {patch.std():.1f}, "
          f"min {patch.min()}, max {patch.max()}")

    # Write multi-page TIFF (small enough to skip BigTIFF)
    with tifffile.TiffWriter(str(tif_path)) as w:
        for fr in patch:
            w.write(fr, contiguous=True)
    print(f"  wrote {tif_path}  ({tif_path.stat().st_size / 1024:.0f} KB)")

    make_inspect_png(patch, png_path, args.y0, args.x0, args.start)
    print(f"  wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
