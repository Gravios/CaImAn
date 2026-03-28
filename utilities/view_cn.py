#!/usr/bin/env python3
"""
plot_cn.py — load and display a CaImAn _Cn.npy correlation image.

Usage:
    python plot_cn.py                          # auto-find in cwd
    python plot_cn.py path/to/session_Cn.npy  # explicit path
    python plot_cn.py --dir /data/source/...  # search a directory
"""

import sys
import argparse
import numpy as np
from pathlib import Path


def find_cn(search_dir: Path):
    hits = sorted(search_dir.glob("*_Cn.npy"))
    if not hits:
        hits = sorted(search_dir.rglob("*_Cn.npy"))
    return hits


def main():
    parser = argparse.ArgumentParser(description="Plot CaImAn _Cn.npy correlation image")
    parser.add_argument("cn_path", nargs="?", help="Path to *_Cn.npy file")
    parser.add_argument("--dir", default=".", help="Directory to search if no path given")
    parser.add_argument("--pmin", type=float, default=1.0,
                        help="Low percentile clip for display (default 1)")
    parser.add_argument("--pmax", type=float, default=99.5,
                        help="High percentile clip for display (default 99.5)")
    parser.add_argument("--cmap", default="inferno",
                        help="Matplotlib colormap (default: inferno)")
    parser.add_argument("--save", action="store_true",
                        help="Save PNG next to the .npy file instead of showing")
    args = parser.parse_args()

    # ── Resolve path ─────────────────────────────────────────────────────────
    if args.cn_path:
        cn_file = Path(args.cn_path)
        if not cn_file.exists():
            sys.exit(f"ERROR: file not found: {cn_file}")
    else:
        hits = find_cn(Path(args.dir))
        if not hits:
            sys.exit(f"ERROR: no *_Cn.npy files found under {args.dir}")
        if len(hits) > 1:
            print("Multiple Cn files found — using most recent:")
            for h in hits:
                print(f"  {h}")
            cn_file = hits[-1]
        else:
            cn_file = hits[0]

    print(f"Loading: {cn_file}")
    Cn = np.load(str(cn_file))
    print(f"  shape  : {Cn.shape}")
    print(f"  dtype  : {Cn.dtype}")
    print(f"  min    : {Cn.min():.4f}")
    print(f"  max    : {Cn.max():.4f}")
    print(f"  median : {np.median(Cn):.4f}")
    print(f"  nonzero: {(Cn != 0).sum()} / {Cn.size} pixels  "
          f"({100*(Cn!=0).mean():.1f}%)")

    # Quadrant breakdown — helps distinguish biology from processing gaps
    d1, d2 = Cn.shape
    h, w = d1 // 2, d2 // 2
    quads = {
        "top-left    ": Cn[:h, :w],
        "top-right   ": Cn[:h, w:],
        "bottom-left ": Cn[h:, :w],
        "bottom-right": Cn[h:, w:],
    }
    print("\nQuadrant signal (mean absolute Cn):")
    for name, q in quads.items():
        print(f"  {name}: mean={q.mean():.4f}  max={q.max():.4f}  "
              f"nonzero={100*(q!=0).mean():.1f}%")

    # ── Plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    vmin = float(np.percentile(Cn, args.pmin))
    vmax = float(np.percentile(Cn, args.pmax))

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(cn_file.name, fontsize=11, y=1.01)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 2, 1])

    # Full FOV
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(Cn, cmap=args.cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax1.set_title(f"Correlation image  ({d1}×{d2})")
    ax1.set_xlabel("col (px)")
    ax1.set_ylabel("row (px)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Draw quadrant lines
    ax1.axhline(h, color="white", lw=0.5, alpha=0.4, ls="--")
    ax1.axvline(w, color="white", lw=0.5, alpha=0.4, ls="--")

    # Histogram
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(Cn.ravel(), bins=200, color="steelblue", alpha=0.8, density=True)
    ax2.axvline(vmin, color="red",  ls="--", lw=1, label=f"p{args.pmin:.0f}={vmin:.3f}")
    ax2.axvline(vmax, color="orange", ls="--", lw=1, label=f"p{args.pmax:.0f}={vmax:.3f}")
    ax2.set_xlabel("Correlation value")
    ax2.set_ylabel("Density")
    ax2.set_title("Pixel distribution")
    ax2.legend(fontsize=8)

    # Quadrant bar chart
    ax3 = fig.add_subplot(gs[2])
    q_labels = ["TL", "TR", "BL", "BR"]
    q_means  = [q.mean() for q in quads.values()]
    q_colors = ["steelblue", "salmon", "salmon", "salmon"]  # highlight non-TL
    ax3.barh(q_labels[::-1], q_means[::-1], color=q_colors[::-1])
    ax3.set_xlabel("Mean Cn")
    ax3.set_title("Quadrant\nsignal")
    ax3.axvline(0, color="k", lw=0.5)

    plt.tight_layout()

    if args.save:
        out_png = cn_file.with_suffix("_inspect.png")
        plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out_png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
