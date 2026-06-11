#!/usr/bin/env python3
"""
Chunked temporal characterization of frame-varying spectral contamination.

The companion script ``diagnose_stationary_vs_perframe.py`` measures
how stationary vs frame-varying the contamination is across an entire
session (one pooled magnitude over all T frames). This script splits
the session into N temporal chunks and characterizes each chunk
separately, revealing whether:

  - Contamination amplitude changes over the session
    (warmup, drift, episodic bursts).
  - Different spectral peaks dominate at different times
    (would mean a single global notch can't catch everything).
  - The peak landscape is stable
    (the per-frame notch with global peak detection is sufficient).

For each chunk it computes the pooled magnitude spectrum on GPU (or
CPU fallback) using ``utilities.noise.noise_correction``'s detection
machinery, then characterizes peak structure with the annular-floor
detector.

Usage:
    python diagnose_chunked_perframe.py \\
        --input /data/source/.../session.tif \\
        --output ./diag_chunked \\
        --n-chunks 20 \\
        --use-gpu

For a 110880-frame session, n-chunks=20 gives ~5500 frames per chunk —
enough for stable per-chunk magnitude estimates while having sufficient
time resolution to see drift.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pool_magnitude_chunk(stack_chunk: np.ndarray, use_gpu: bool = False,
                          batch_size: int = 512) -> np.ndarray:
    """Pool |FFT(frame_i)| over a chunk of frames. Returns the pooled
    magnitude spectrum (fftshifted, DC at centre)."""
    if use_gpu:
        try:
            import cupy as xp
        except ImportError:
            xp = np
    else:
        xp = np

    T, H, W = stack_chunk.shape
    A_pooled = xp.zeros((H, W), dtype=xp.float32)
    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        batch = stack_chunk[s:e].astype(np.float32, copy=False)
        batch_xp = xp.asarray(batch) if xp is not np else batch
        batch_xp = batch_xp - batch_xp.mean(axis=(1, 2), keepdims=True)
        F = xp.fft.fft2(batch_xp)
        A_pooled += xp.abs(F).sum(axis=0)
    A_pooled = xp.fft.fftshift(A_pooled / T)
    return np.asarray(A_pooled) if xp is np else xp.asnumpy(A_pooled)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=Path, required=True,
                    help="path to .tif/.tiff/.msr stack")
    p.add_argument("--output", "-o", type=Path,
                    default=Path("./diag_chunked"))
    p.add_argument("--n-chunks", "-n", type=int, default=20,
                    help="number of temporal chunks (default 20)")
    p.add_argument("--max-frames", type=int, default=None,
                    help="cap total frames analysed; default uses all")
    p.add_argument("--use-gpu", action="store_true",
                    help="use CuPy for FFTs if available")
    p.add_argument("--probe", action="append", default=None,
                    metavar="dy,dx",
                    help="extra (dy, dx) coords to track per chunk; "
                         "may be repeated. Defaults cover known lattice "
                         "harmonics for the strohA scope.")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, "/data/dev/CaImAn")
    # Import the StackReader directly rather than through
    # utilities.support.io, because the support package's __init__
    # transitively imports torch (only needed for training/inference,
    # not for stack loading). This script only needs the I/O layer.
    from caiman.utils.stack_io import StackReader
    from utilities.noise.noise_correction import detect_fpn_peaks

    def read_stack(path: Path) -> np.ndarray:
        """Load an entire .tif/.tiff/.msr stack to a (T, H, W) float32
        host array via the same backend as utilities.support.io."""
        with StackReader(path) as r:
            T, H, W = int(r.n_frames), int(r.h), int(r.w)
            out = np.empty((T, H, W), dtype=np.float32)
            for i in range(T):
                out[i] = r.read_frame(i).astype(np.float32, copy=False)
        return out

    print(f"Loading {args.input}...")
    stack = read_stack(args.input)
    if args.max_frames is not None:
        stack = stack[:args.max_frames]
    T, H, W = stack.shape
    cy, cx = H // 2, W // 2
    print(f"  shape={stack.shape}, total frames T={T}")

    # Default probe coordinates: the known strohA lattice harmonics.
    # Negative-dy and conjugate pairs aren't tracked separately because
    # they're constrained to mirror.
    default_probes = [
        (52, 12, "lattice fund. (1×52, 12)"),
        (72, 12, "lattice harm. (72, 12)"),
        (112, 24, "lattice harm. (112, 24)"),
        (144, 24, "lattice harm. (144, 24)"),
        (184, 12, "lattice harm. (184, 12)"),
        (204, 12, "lattice harm. (204, 12)"),
        (23, 0,   "line pedestal Y (23, 0)"),
        (0, 25,   "line pedestal X (0, 25)"),
        (73, 41,  "baseline (cell)"),
    ]
    probes = list(default_probes)
    if args.probe:
        for spec in args.probe:
            try:
                dy, dx = (int(s) for s in spec.split(","))
                probes.append((dy, dx, f"user ({dy}, {dx})"))
            except ValueError:
                print(f"warning: cannot parse --probe {spec!r}, skipping")

    # Filter probes that would index out of bounds (default probes are
    # calibrated for 512×512 FOVs; smaller frames need them dropped)
    in_bounds = []
    for dy, dx, label in probes:
        y, x = cy + dy, cx + dx
        if 0 <= y < H and 0 <= x < W:
            in_bounds.append((dy, dx, label))
        else:
            print(f"  skip probe {label!r} (out of bounds for {H}×{W} FOV)")
    probes = in_bounds
    if not probes:
        print("warning: no probes in bounds; defaulting to a single probe at "
              "the lattice fundamental (1, 1) — coordinates were calibrated "
              "for 512×512")
        probes = [(1, 1, "fallback (1, 1)")]

    # Chunk boundaries
    chunk_size = T // args.n_chunks
    if chunk_size < 50:
        print(f"warning: chunk_size={chunk_size} frames is small; "
               f"consider reducing --n-chunks")
    chunk_bounds = [(i * chunk_size,
                      (i + 1) * chunk_size if i < args.n_chunks - 1 else T)
                     for i in range(args.n_chunks)]
    print(f"  {args.n_chunks} chunks, ~{chunk_size} frames each")

    # Per-chunk pooled magnitude + peak detection
    print(f"\nProcessing chunks (GPU={args.use_gpu}):")
    # Store per-chunk magnitudes at probe coordinates, plus the full
    # per-chunk pooled magnitudes for the heatmap viz
    probe_traces = np.zeros((len(probes), args.n_chunks), dtype=np.float32)
    n_peaks_per_chunk = np.zeros(args.n_chunks, dtype=np.int32)
    chunk_pooled = np.zeros((args.n_chunks, H, W), dtype=np.float32)
    detected_peaks_per_chunk = []

    for k, (s, e) in enumerate(chunk_bounds):
        t0 = time.perf_counter()
        chunk = stack[s:e]
        A = pool_magnitude_chunk(chunk, use_gpu=args.use_gpu)
        chunk_pooled[k] = A

        # Probe magnitudes at fixed bins
        for i, (dy, dx, _) in enumerate(probes):
            probe_traces[i, k] = A[cy + dy, cx + dx]

        # Detect peaks
        notch_mask, _ = detect_fpn_peaks(A, magnitude_in=True,
                                           prominence_db=15.0,
                                           max_peaks=64,
                                           annular_floor=True)
        from scipy import ndimage
        labeled, n_blobs = ndimage.label(notch_mask)
        n_peaks_per_chunk[k] = n_blobs
        peaks_here = []
        for blob_id in range(1, n_blobs + 1):
            ys, xs = np.where(labeled == blob_id)
            yi = ys[np.argmax(A[ys, xs])]
            xi = xs[np.argmax(A[ys, xs])]
            peaks_here.append((yi - cy, xi - cx, float(A[yi, xi])))
        detected_peaks_per_chunk.append(peaks_here)

        dt = time.perf_counter() - t0
        print(f"  chunk {k+1:2d}/{args.n_chunks}: frames [{s:6d}..{e:6d}), "
              f"{n_blobs} peaks, {dt:.1f}s")

    # Save raw outputs
    np.save(args.output / "chunk_pooled.npy", chunk_pooled)
    np.save(args.output / "probe_traces.npy", probe_traces)
    with open(args.output / "probes.txt", "w") as f:
        for dy, dx, label in probes:
            f.write(f"{dy:+5d}  {dx:+5d}  {label}\n")
    with open(args.output / "peaks_per_chunk.txt", "w") as f:
        for k, peaks in enumerate(detected_peaks_per_chunk):
            f.write(f"# chunk {k}\n")
            for dy, dx, mag in sorted(peaks, key=lambda t: -t[2]):
                f.write(f"  {dy:+5d}  {dx:+5d}  {mag:10.1f}\n")

    # =================== Visualisation =====================
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # (a) Time series of probe magnitudes
    ax = axes[0, 0]
    t_axis = np.arange(args.n_chunks)
    for i, (dy, dx, label) in enumerate(probes):
        ax.plot(t_axis, probe_traces[i], marker="o", markersize=3,
                 label=label)
    ax.set_xlabel("chunk index")
    ax.set_ylabel("pooled |FFT| at probe bin")
    ax.set_title("Probe-bin magnitude vs time")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_yscale("log")

    # (b) Heatmap of probe magnitudes
    ax = axes[0, 1]
    im = ax.imshow(np.log10(probe_traces + 1), aspect="auto",
                    cmap="viridis", origin="upper")
    ax.set_yticks(range(len(probes)))
    ax.set_yticklabels([p[2] for p in probes], fontsize=7)
    ax.set_xlabel("chunk index")
    ax.set_title("log₁₀(probe magnitude) heatmap")
    plt.colorbar(im, ax=ax, fraction=0.04)

    # (c) Number of detected peaks per chunk
    ax = axes[1, 0]
    ax.plot(t_axis, n_peaks_per_chunk, marker="o", color="C3")
    ax.set_xlabel("chunk index")
    ax.set_ylabel("# peaks detected")
    ax.set_title("Peak count per chunk\n(stable→single global notch OK; "
                  "variable→adaptive per-chunk notch helps)")
    ax.set_ylim(0, max(n_peaks_per_chunk.max() + 5, 20))

    # (d) Total contamination energy per chunk (sum at lattice probes)
    ax = axes[1, 1]
    lattice_idx = [i for i, p in enumerate(probes) if "lattice" in p[2]]
    if lattice_idx:
        lattice_total = probe_traces[lattice_idx].sum(axis=0)
        ax.plot(t_axis, lattice_total, marker="o", color="C2")
        ax.set_xlabel("chunk index")
        ax.set_ylabel("Σ |FFT| at lattice probes")
        ax.set_title("Total lattice contamination energy per chunk")
    else:
        ax.text(0.5, 0.5, "no lattice probes",
                 ha="center", va="center", transform=ax.transAxes)

    # (e) Pooled magnitude of first chunk (top-left FFT visualization)
    ax = axes[2, 0]
    A_first = chunk_pooled[0]
    A_log = np.log10(A_first + 1)
    vmin, vmax = np.percentile(A_log, [50, 99.7])
    ax.imshow(A_log, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"Pooled |FFT| — chunk 0 "
                  f"(frames {chunk_bounds[0][0]}-{chunk_bounds[0][1]})")

    # (f) Pooled magnitude of last chunk
    ax = axes[2, 1]
    A_last = chunk_pooled[-1]
    A_log = np.log10(A_last + 1)
    ax.imshow(A_log, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"Pooled |FFT| — chunk {args.n_chunks-1} "
                  f"(frames {chunk_bounds[-1][0]}-{chunk_bounds[-1][1]})")

    fig.suptitle(f"Chunked contamination characterization\n"
                  f"input: {args.input.name}  T={T}  n_chunks={args.n_chunks}",
                  fontsize=11)
    fig.tight_layout()
    out_path = args.output / "chunked_characterization.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nWrote {out_path}")

    # Summary statistics
    print("\n=== Summary ===")
    print(f"Peak count: min={n_peaks_per_chunk.min()}, "
          f"max={n_peaks_per_chunk.max()}, "
          f"mean={n_peaks_per_chunk.mean():.1f}")
    if lattice_idx:
        lt = probe_traces[lattice_idx].sum(axis=0)
        print(f"Lattice total energy: min={lt.min():.0f}, "
              f"max={lt.max():.0f}, "
              f"variation={(lt.max()-lt.min())/lt.mean()*100:.0f}%")
    print(f"\nProbe traces saved to {args.output / 'probe_traces.npy'}")
    print(f"Per-chunk peaks saved to {args.output / 'peaks_per_chunk.txt'}")
    print(f"Per-chunk pooled magnitudes saved to "
           f"{args.output / 'chunk_pooled.npy'} "
           f"(shape {chunk_pooled.shape}, ~{chunk_pooled.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
