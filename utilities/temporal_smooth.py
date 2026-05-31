#!/usr/bin/env python3
"""
temporal_smooth.py — apply a 1D Gaussian along the temporal axis to a
TIFF stack. Quick diagnostic for whether per-frame structured noise is
the dominant Cn-stripe driver: smoothing across ~5 frames damps white-
temporal noise by √5 ≈ 2.2× while leaving GCaMP6 transients (τ ≈ 30
frames at 30 Hz) essentially unchanged.

Usage
-----
    python temporal_smooth.py path/to/session_Ncorrected.tif
    python temporal_smooth.py path/to/file.tif --sigma 2 --chunk 1000

Output: alongside the input, ``<stem>_Tsmooth.tif`` (float32 BigTIFF).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter1d


def temporal_smooth(src: Path, sigma: float, chunk: int) -> Path:
    src = src.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    out = src.with_name(src.stem + "_Tsmooth.tif")

    with tifffile.TiffFile(src) as tf:
        n = len(tf.pages)
        H, W = tf.pages[0].shape
        dtype = tf.pages[0].dtype
        print(f"  input:  {src.name}")
        print(f"          {n} frames, {H}×{W}, dtype={dtype}")
        print(f"  output: {out.name}")
        print(f"          sigma={sigma} frames  (≈ {int(6*sigma+1)}-frame window)")

        # Gaussian truncates at ~3σ; we need that many extra frames as
        # context at each chunk boundary to avoid edge artifacts.
        pad = max(int(np.ceil(3 * sigma)), 1)
        print(f"  chunk={chunk}  pad={pad}")

        with tifffile.TiffWriter(out, bigtiff=True) as w:
            # Stream in [start - pad, end + pad) chunks; write [start, end)
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                a = max(start - pad, 0)
                b = min(end + pad, n)
                block = np.stack([tf.pages[i].asarray() for i in range(a, b)],
                                  axis=0).astype(np.float32)
                smoothed = gaussian_filter1d(block, sigma=sigma, axis=0,
                                              mode="nearest")
                # write the center [start, end) part of the smoothed result
                inner = smoothed[start - a: start - a + (end - start)]
                for fr in inner:
                    w.write(fr.astype(np.float32), contiguous=True)
                pct = 100 * end / n
                print(f"    [{end:>6d}/{n}]  {pct:5.1f}%", end="\r")
        print()
    sz_mb = out.stat().st_size / 1024 / 1024
    print(f"  wrote {out}  ({sz_mb:.0f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("src", type=Path, help="input TIFF stack")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Gaussian σ in frames (default 2 → ~5-frame window)")
    ap.add_argument("--chunk", type=int, default=1000,
                    help="frames per chunk for streaming (default 1000)")
    args = ap.parse_args()
    try:
        temporal_smooth(args.src, args.sigma, args.chunk)
    except (FileNotFoundError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
