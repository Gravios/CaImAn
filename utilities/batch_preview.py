#!/usr/bin/env python3
"""
batch-preview
=============
One random frame from every stacked session (and channel) under a source /
user directory, written flat into ``<root>/preview/`` for checking the image
quality by eye.

    cd /data/source/stroh/strohA-ia
    batch-preview                         # -> strohA-ia/preview/*.png
    batch-preview --root strohA-ia --mean 5 --seed 1

Output (flat, one file per session and channel):

    preview/<session>-C00-t004211.png     frame 4211 of channel C00
    preview/<session>-C00-t004211-m5.png  mean of frames 4211–4215 (--mean 5)

Only the stacked BigTIFFs are sampled:

    <session>/<session>-C<NN>-fc<N>/<session>-C<NN>-fc<N>.tif

read as plain TIFF pages, at any depth below --root (subject / date folders
or not).  OME-TIFFs left in the session folders (raw frames, OME stacks)
are ignored; a folder holding only those is counted as "not stacked" in
the summary.  Unfinished *.partial.tif stacks are ignored.  Contrast: each image is
scaled from its 0.5th to its 99.5th percentile to 8 bit (--low / --high), so
dim and bright sessions both show detail; the scaling is the same for every
image's own range, not across images.

A session that already has a preview for a channel is skipped (whatever its
frame); --overwrite draws a new one.  Nothing outside preview/ is written.
"""
from __future__ import annotations

import argparse
import random
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

PREVIEW_DIR = "preview"
_CH_DIR_RE = re.compile(r"^(?P<session>.+)-C(?P<ch>\d{2})-fc(?P<n>\d+)$")
_FRAME_RE = re.compile(r"_C(?P<ch>\d{2})_t(?P<t>\d+)(?:\.ome)?\.tiff?$", re.I)
_STACK_RE = re.compile(r"_C(?P<ch>\d{2})\.ome\.tiff?$", re.I)


# ── discovery ────────────────────────────────────────────────────────────────

def _stacks(d: Path) -> dict:
    """{channel: stacked BigTIFF} of session dir *d*:
    <d>/<d.name>-C<NN>-fc<N>/<d.name>-C<NN>-fc<N>.tif."""
    found: dict = {}
    try:
        kids = sorted(d.iterdir())
    except OSError:
        return found
    for c in kids:
        m = _CH_DIR_RE.match(c.name)
        if m and m.group("session") == d.name and c.is_dir():
            f = c / f"{c.name}.tif"
            if f.is_file():
                found.setdefault(m.group("ch"), f)
    return found


def _has_ome(d: Path) -> bool:
    """Raw frames or an OME stack directly in *d* (data not stacked yet)."""
    try:
        return any(_FRAME_RE.search(f.name) or _STACK_RE.search(f.name)
                   for f in d.iterdir())
    except OSError:
        return False


def find_sessions(root: Path) -> tuple:
    """([(session dir, {channel: stack})], [folders with OME data but no
    stack]), sorted by path.  Doesn't descend into a session's channel
    folders, or into <root>/preview."""
    out, unstacked = [], []

    def walk(d: Path) -> None:
        st = _stacks(d)
        if st:
            out.append((d, st))
        elif _has_ome(d):
            unstacked.append(d)
        try:
            kids = sorted(c for c in d.iterdir() if c.is_dir() and not c.is_symlink())
        except OSError:
            return
        for c in kids:
            if c.name == PREVIEW_DIR and d == root:
                continue
            m = _CH_DIR_RE.match(c.name)
            if m and m.group("session") == d.name:      # a channel folder of d
                continue
            walk(c)
    walk(root)
    return out, unstacked


# ── reading ──────────────────────────────────────────────────────────────────

def _n_pages(tf) -> int:
    s = tf.series[0]
    return s.shape[0] if len(s.shape) == 3 else len(tf.pages)


def pick_frame(path: Path, rng: random.Random, mean: int = 1):
    """(index, image) of a random frame of a stack; *mean* > 1 averages that
    many consecutive frames starting there."""
    import tifffile
    with tifffile.TiffFile(str(path), is_ome=False) as tf:
        n = _n_pages(tf)
        i = rng.randrange(max(1, n - mean + 1))
        imgs = [np.squeeze(tf.pages[k].asarray()) for k in range(i, min(n, i + mean))]
    if any(a.ndim != 2 for a in imgs):
        raise ValueError(f"frame shape {imgs[0].shape} is not 2-D")
    img = imgs[0] if len(imgs) == 1 else np.mean(np.stack(imgs).astype(np.float64), 0)
    return i, img


def to_8bit(img: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)


def write_png(path: Path, img8: np.ndarray) -> None:
    """8-bit greyscale PNG, standard library only."""
    h, w = img8.shape
    raw = b"".join(b"\x00" + img8[r].tobytes() for r in range(h))

    def chunk(t: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(png)
    tmp.replace(path)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="batch-preview", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="source / user directory, e.g. strohA-ia (default: .)")
    ap.add_argument("--out", type=Path, help="output folder (default: <root>/preview)")
    ap.add_argument("--channel", action="append", metavar="NN",
                    help="only these channels (repeatable), e.g. --channel 00")
    ap.add_argument("--mean", type=int, default=1, metavar="K",
                    help="average K consecutive frames (less noise; default 1)")
    ap.add_argument("--seed", type=int, help="random seed (same frames on a rerun)")
    ap.add_argument("--low", type=float, default=0.5, help="lower percentile (default 0.5)")
    ap.add_argument("--high", type=float, default=99.5, help="upper percentile (default 99.5)")
    ap.add_argument("--overwrite", action="store_true",
                    help="draw a new frame where a preview exists")
    ap.add_argument("--dry-run", action="store_true", help="list what would be written")
    a = ap.parse_args(argv)
    if a.mean < 1:
        ap.error("--mean must be >= 1")
    if not 0 <= a.low < a.high <= 100:
        ap.error("need 0 <= --low < --high <= 100")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    root = a.root.resolve()
    if not root.is_dir():
        print(f"batch-preview: {root} is not a directory")
        return 2
    out = (a.out or root / PREVIEW_DIR).resolve()
    rng = random.Random(a.seed)
    want = {c.zfill(2) for c in a.channel} if a.channel else None
    sessions, unstacked = find_sessions(root)
    if not sessions:
        print(f"batch-preview: no stacked session under {root}"
              + (f" ({len(unstacked)} with OME-TIFFs only: run stack-sessions)"
                 if unstacked else ""))
        return 1
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    n_new = n_skip = n_err = 0
    for d, src in sessions:
        for ch, path in sorted(src.items()):
            if want is not None and ch not in want:
                continue
            stem = f"{d.name}-C{ch}"
            old = sorted(out.glob(f"{stem}-t*.png")) if out.is_dir() else []
            if old and not a.overwrite:
                n_skip += 1
                continue
            rel = d.relative_to(root)
            if a.dry_run:
                print(f"  would preview {rel} C{ch}")
                n_new += 1
                continue
            try:
                idx, img = pick_frame(path, rng, a.mean)
            except Exception as exc:                     # noqa: BLE001
                print(f"  ERROR {rel} C{ch}: {exc}")
                n_err += 1
                continue
            name = f"{stem}-t{idx:06d}" + (f"-m{a.mean}" if a.mean > 1 else "") + ".png"
            write_png(out / name, to_8bit(img, a.low, a.high))
            for f in old:                               # --overwrite: one per channel
                if f.name != name:
                    f.unlink()
            print(f"  {name}   ({img.shape[1]}x{img.shape[0]})")
            n_new += 1
    verb = "would write" if a.dry_run else "wrote"
    print(f"\n{len(sessions)} stacked sessions: {verb} {n_new}, kept {n_skip} existing"
          + (f", {n_err} errors" if n_err else "") + f"  -> {out}")
    if unstacked:
        print(f"{len(unstacked)} folders with OME-TIFFs but no stack, not sampled "
              f"(e.g. {unstacked[0].relative_to(root)})")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
