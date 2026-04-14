#!/usr/bin/env python
"""
stack_to_bigtiff.py  —  Collapse a directory of TIFF frames into a BigTIFF stack
==================================================================================

Collects all TIFF files in a source directory, sorts them, and writes a single
BigTIFF output using memory-efficient frame-by-frame streaming. Suitable for
large 2-photon or widefield datasets that exceed the 4 GB classic TIFF limit.

Multi-channel support
---------------------
If the input directory contains files with an embedded channel token of the form
``_CNN_tNNNN`` (e.g. ``_C00_t0000.ome.tif``, ``_C01_t0000.ome.tif``), the
script auto-detects each unique channel ID using a regular expression and writes
one BigTIFF stack per channel:

    <output_stem>_C00.tif
    <output_stem>_C01.tif
    ...
    <output_stem>.yaml          ← YAML sidecar (channel_count, per-channel metadata)

Single-channel folders (no ``_CNN_`` token found) behave exactly as before.

Supports:
  - Grayscale and RGB frames
  - uint8, uint16, float32 source data
  - Optional frame range selection (--start / --end)
  - Optional spatial downsampling (--downsample)
  - Preallocated writing via tifffile.memmap (--preallocate, uncompressed only)
  - Streaming compressed writing via TiffWriter (default)
  - Dry-run mode to preview what would be written

Write modes
-----------
Streaming (default):
    Frames are written one at a time through TiffWriter. Compatible with all
    compression codecs. Peak RAM ≈ 2× one frame. File grows incrementally,
    so a crash leaves a partial but valid TIFF.

Preallocated (--preallocate):
    Uses tifffile.memmap to reserve the exact final file size on disk before
    any pixel data is written. Eliminates filesystem fragmentation, fails fast
    if space is insufficient, and produces contiguous layout ideal for CaImAn
    memmap reads. Requires --compression none (compressed sizes are unknowable
    ahead of time). Peak RAM ≈ 2× one frame; the memmap itself is not held in
    RAM. Progress is flushed to disk periodically via msync.

Dependencies:
    pip install tifffile numpy pyyaml   # pyyaml optional; plain-text fallback used

Usage:
    python stack_to_bigtiff.py --input /data/frames/ --output /data/stack.tif
    python stack_to_bigtiff.py --input /data/frames/ --output /data/stack.tif \\
        --pattern "frame_*.tif" --start 100 --end 500 --downsample 2
    python stack_to_bigtiff.py --input /data/frames/ --output /data/stack.tif \\
        --preallocate --compression none
    # Multi-channel (auto-detected; writes stack_C00.tif, stack_C01.tif, stack.yaml)
    python stack_to_bigtiff.py --input /data/frames/ --output /data/stack.tif
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    print("Error: tifffile not installed.  Run:  pip install tifffile")
    sys.exit(1)

try:
    import yaml as _yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Matches the channel token embedded by most acquisition software, e.g.:
#   sample_C00_t0042.ome.tif  →  group(1) = "00"
#   FOV3_C01_t1200.tif        →  group(1) = "01"
_CHANNEL_RE = re.compile(r'_C(\d+)_t\d+', re.IGNORECASE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collapse a directory of TIFF frames into a single BigTIFF stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input", "-i", required=True,
        help="Source directory containing TIFF frames"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Destination BigTIFF file path (.tif / .tiff)"
    )
    parser.add_argument(
        "--pattern", default="*.tif*",
        help="Glob pattern used to match frames inside --input"
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="First frame index to include (0-based, inclusive)"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Last frame index to include (0-based, exclusive)"
    )
    parser.add_argument(
        "--downsample", type=int, default=1, metavar="FACTOR",
        help="Spatial downsampling factor applied to each frame (must be ≥ 1)"
    )
    parser.add_argument(
        "--compression", default="zlib", choices=["none", "zlib", "lzw", "zstd"],
        help="Per-tile compression codec"
    )
    parser.add_argument(
        "--compression-level", type=int, default=1, metavar="LEVEL",
        help="Compression level (codec-dependent; lower = faster)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without creating any file"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output file if it already exists"
    )
    parser.add_argument(
        "--keep-sources", action="store_true",
        help=(
            "Do not delete the source frame TIFFs after a successful write. "
            "By default the source frames are removed once the output file "
            "has been verified on disk."
        )
    )
    parser.add_argument(
        "--preallocate", action="store_true",
        help=(
            "Preallocate the full output file before writing via tifffile.memmap. "
            "Eliminates fragmentation and fails fast on insufficient space. "
            "Requires --compression none."
        )
    )
    parser.add_argument(
        "--flush-every", type=int, default=200, metavar="N",
        help=(
            "Preallocated mode only: call msync every N frames to flush dirty "
            "pages to disk and bound OS page-cache growth (default: 200)"
        )
    )
    parser.add_argument(
        "--in-memory", action="store_true",
        help=(
            "Load all frames into a single numpy array then write in one call. "
            "Fastest write and best compression ratio, but requires enough RAM "
            "to hold the full stack. Auto-selected when estimated size is below "
            "--memory-threshold."
        )
    )
    parser.add_argument(
        "--memory-threshold", type=float, default=15.0, metavar="GB",
        help=(
            "Auto-select in-memory mode when estimated uncompressed stack size "
            "is below this threshold in GB (default: 15.0). Ignored if "
            "--in-memory or --preallocate is set explicitly."
        )
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_frames(input_dir: Path, pattern: str, start, end) -> list[Path]:
    """Sort and slice the list of TIFF paths in input_dir."""
    frames = sorted(input_dir.glob(pattern))
    if not frames:
        logger.error(f"No files matching '{pattern}' found in {input_dir}")
        sys.exit(1)

    frames = frames[start:end]  # None slices are no-ops
    logger.info(f"Found {len(frames)} frame(s) after slice [{start}:{end}]")
    return frames


# ---------------------------------------------------------------------------
# Multi-channel helpers
# ---------------------------------------------------------------------------

def detect_channel_ids(input_dir: Path, pattern: str) -> list[str] | None:
    """
    Scan *all* files in ``input_dir`` matching ``pattern`` for an embedded
    channel token of the form ``_CNN_tNNNN`` (e.g. ``_C00_t0042``).

    Returns a sorted list of zero-padded channel ID strings (e.g.
    ``['00', '01', '02']``) when at least one such token is found, or
    ``None`` if the directory appears to be single-channel.
    """
    channel_ids: set[str] = set()
    for fpath in input_dir.glob(pattern):
        m = _CHANNEL_RE.search(fpath.name)
        if m:
            channel_ids.add(m.group(1).zfill(2))
    return sorted(channel_ids) if channel_ids else None


def collect_channel_frames(input_dir: Path, pattern: str, channel_id: str,
                           start, end) -> list[Path]:
    """
    Return a sorted, sliced list of frame paths belonging to *one* channel.

    Filters by an exact channel token ``_C{channel_id}_t\\d+`` so that
    channel "01" never accidentally matches "001".
    """
    ch_re = re.compile(rf'_C{re.escape(channel_id)}_t\d+', re.IGNORECASE)
    frames = sorted(
        fpath for fpath in input_dir.glob(pattern)
        if ch_re.search(fpath.name)
    )
    if not frames:
        logger.error(
            f"No frames found for channel C{channel_id} "
            f"in {input_dir} (pattern '{pattern}')"
        )
        sys.exit(1)
    frames = frames[start:end]
    logger.info(
        f"  Channel C{channel_id}: {len(frames)} frame(s) after slice [{start}:{end}]"
    )
    return frames


def channel_output_path(base_output: Path, channel_id: str) -> Path:
    """
    Derive the per-channel output path from the base ``--output`` argument.

    Example::

        base_output = Path('/data/stack/stack.tif')
        channel_id  = '01'
        →  Path('/data/stack/stack_C01.tif')
    """
    return base_output.parent / f"{base_output.stem}_C{channel_id}{base_output.suffix}"


def write_yaml_sidecar(
    yaml_path: Path,
    channel_ids: list[str],
    output_paths: dict[str, Path],
    n_frames_map: dict[str, int],
    frame_shape: tuple,
    dtype: np.dtype,
) -> None:
    """
    Write a YAML sidecar next to the per-channel stacks.

    Schema::

        channel_count: 2
        channels:
          C00:
            output: /data/stack/stack_C00.tif
            n_frames: 27720
          C01:
            output: /data/stack/stack_C01.tif
            n_frames: 27720
        frame_shape: [512, 512]
        dtype: uint16
    """
    data: dict = {
        'channel_count': len(channel_ids),
        'channels': {
            f'C{cid}': {
                'output': str(output_paths[cid]),
                'n_frames': n_frames_map[cid],
            }
            for cid in channel_ids
        },
        'frame_shape': list(frame_shape),
        'dtype': str(dtype),
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    if _HAVE_YAML:
        with open(yaml_path, 'w') as fh:
            _yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    else:
        # Minimal hand-rolled YAML — no extra dependencies required.
        lines = [
            f"channel_count: {data['channel_count']}",
            "channels:",
        ]
        for ch_key, ch_val in data['channels'].items():
            lines.append(f"  {ch_key}:")
            lines.append(f"    output: {ch_val['output']}")
            lines.append(f"    n_frames: {ch_val['n_frames']}")
        lines.append(f"frame_shape: {data['frame_shape']}")
        lines.append(f"dtype: {data['dtype']}")
        with open(yaml_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')

    logger.info(f"YAML sidecar : {yaml_path}")


def probe_frame(path: Path, downsample: int):
    """
    Read the first frame to determine shape and dtype.

    OME-TIFF files written by acquisition software often carry singleton
    dimensions for unused axes (e.g. Z, channel, time), giving shapes like
    (1, 1, 512, 512) for a plain grayscale frame. squeeze() collapses all
    size-1 axes before the shape check so these are handled transparently.

    is_ome=False bypasses OME-XML linking — without this, tifffile.imread on
    the master OME-TIFF follows companion file references and loads the entire
    series (e.g. 5000 frames) instead of the single plane in this file.

    Returns
    -------
    shape : tuple
        (H, W) or (H, W, C) after squeezing and downsampling
    dtype : np.dtype
    """
    with tifffile.TiffFile(str(path), is_ome=False) as tf:
        frame = tf.pages[0].asarray().squeeze()
    if frame.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2-D (grayscale) or 3-D (RGB) frame after squeezing, "
            f"got shape {frame.shape} from {path}. "
            f"File may contain multiple Z-planes or channels per frame."
        )
    if downsample > 1:
        frame = frame[::downsample, ::downsample]
    return frame.shape, frame.dtype


def load_frame(path: Path, downsample: int, target_dtype: np.dtype) -> np.ndarray:
    """Read a single frame, bypass OME-XML linking (is_ome=False), squeeze
    singleton OME axes, optionally downsample, and cast to target dtype."""
    with tifffile.TiffFile(str(path), is_ome=False) as tf:
        frame = tf.pages[0].asarray().squeeze()
    if downsample > 1:
        frame = frame[::downsample, ::downsample]
    if frame.dtype != target_dtype:
        frame = frame.astype(target_dtype)
    return frame


def estimate_output_bytes(n_frames: int, frame_shape: tuple, dtype: np.dtype) -> int:
    """Return estimated uncompressed output size in bytes."""
    return n_frames * int(np.prod(frame_shape)) * dtype.itemsize


def estimate_output_size(n_frames: int, frame_shape: tuple, dtype: np.dtype) -> str:
    """Return a human-readable estimate of the uncompressed output size."""
    nbytes = n_frames * int(np.prod(frame_shape)) * dtype.itemsize
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def preallocated_stack_shape(n_frames: int, frame_shape: tuple) -> tuple:
    """Return the full (T, H, W) or (T, H, W, C) stack shape."""
    return (n_frames,) + frame_shape


# ---------------------------------------------------------------------------
# Core writers
# ---------------------------------------------------------------------------

def write_stack_inmemory(frames: list[Path], output_path: Path, downsample: int,
                         compression: str, compression_level: int):
    """
    Load all frames into a single pre-allocated numpy array, then write the
    complete stack in one tifffile.imwrite call.

    Advantages over streaming:
      - Single write syscall — fastest possible throughput on NVMe/NAS
      - Compressor sees the full stack, enabling better inter-frame compression
        ratios with codecs like zstd
      - No IFD fragmentation

    Requirement: enough free RAM to hold the entire stack (n_frames × frame_shape).
    Auto-selected when estimated size < --memory-threshold (default 15 GB).
    """
    frame_shape, dtype = probe_frame(frames[0], downsample)
    n_frames = len(frames)
    stack_shape = (n_frames,) + frame_shape

    logger.info(f"Frame shape : {frame_shape}  dtype : {dtype}")
    logger.info(f"Stack shape : {stack_shape}")
    logger.info(f"Est. size   : {estimate_output_size(n_frames, frame_shape, dtype)} (uncompressed)")
    logger.info(f"Compression : {compression}  level {compression_level}")
    logger.info(f"Write mode  : in-memory")
    logger.info(f"Allocating array in RAM...")

    try:
        stack = np.empty(stack_shape, dtype=dtype)
    except MemoryError:
        logger.error(
            "MemoryError: not enough RAM for in-memory mode. "
            "Re-run without --in-memory to use streaming instead, "
            "or lower --memory-threshold."
        )
        sys.exit(1)

    for i, fpath in enumerate(frames):
        if i % 100 == 0 or i == n_frames - 1:
            logger.info(f"  [{i + 1:>{len(str(n_frames))}}/{n_frames}]  {fpath.name}")
        stack[i] = load_frame(fpath, downsample, dtype)

    logger.info(f"Writing to  : {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compression_arg = None if compression == "none" else compression
    tifffile.imwrite(
        str(output_path),
        stack,
        bigtiff=True,
        compression=compression_arg,
        compressionargs={"level": compression_level} if compression_arg else None,
        metadata=None,
    )

    output_size_mb = output_path.stat().st_size / 1024 ** 2
    logger.info(f"Done. Output file: {output_size_mb:.1f} MB")


def write_stack_preallocated(frames: list[Path], output_path: Path,
                             downsample: int, flush_every: int):
    """
    Preallocate the complete BigTIFF file via tifffile.memmap, then fill
    frames in-place.

    tifffile.memmap writes the TIFF header and a single large IFD immediately,
    reserving the exact final file size on disk before any pixel data arrives.
    This gives you:
      - A fast upfront failure if the filesystem lacks space
      - Contiguous, unfragmented layout (ideal for CaImAn's memmap reader)
      - No IFD chain overhead per frame

    The memmap is not held in RAM — the OS maps it into the virtual address
    space and pages in/out as needed. Peak RAM stays at roughly 2× one frame.

    flush_every controls how often os.msync is called to push dirty pages from
    the OS page cache to disk, which prevents unbounded cache growth during
    very long writes.

    Compression is not supported in this mode because compressed frame sizes
    are not known until after encoding, making preallocation impossible.
    """
    frame_shape, dtype = probe_frame(frames[0], downsample)
    n_frames = len(frames)
    stack_shape = preallocated_stack_shape(n_frames, frame_shape)

    uncompressed_size = estimate_output_size(n_frames, frame_shape, dtype)
    logger.info(f"Frame shape   : {frame_shape}  dtype : {dtype}")
    logger.info(f"Stack shape   : {stack_shape}")
    logger.info(f"Output size   : {uncompressed_size} (exact, uncompressed)")
    logger.info(f"Preallocating : {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # tifffile.memmap creates the file, writes the BigTIFF header + a single
    # IFD for the full stack, and returns a writable numpy memmap of shape
    # stack_shape backed by the pixel region of the file.
    try:
        stack_mm = tifffile.memmap(
            str(output_path),
            shape=stack_shape,
            dtype=dtype,
            bigtiff=True,
        )
    except OSError as exc:
        logger.error(f"Preallocation failed: {exc}")
        logger.error("Check available disk space and permissions.")
        if output_path.exists():
            output_path.unlink()
        sys.exit(1)

    logger.info("Preallocation complete — writing frames...")

    # msync requires the underlying file descriptor, accessible via the mmap
    # buffer object that backs the memmap array.
    mmap_buffer = stack_mm._mmap if hasattr(stack_mm, '_mmap') else None

    try:
        for i, fpath in enumerate(frames):
            if i % 100 == 0 or i == n_frames - 1:
                logger.info(
                    f"  [{i + 1:>{len(str(n_frames))}}/{n_frames}]  {fpath.name}"
                )

            stack_mm[i] = load_frame(fpath, downsample, dtype)

            # Periodically flush dirty pages to disk to bound page-cache growth
            if mmap_buffer is not None and (i + 1) % flush_every == 0:
                mmap_buffer.flush()

    finally:
        # Final flush and release — ensures all dirty pages hit disk even on
        # an early exit (e.g. KeyboardInterrupt after partial write)
        if mmap_buffer is not None:
            mmap_buffer.flush()
        del stack_mm

    output_size_mb = output_path.stat().st_size / 1024 ** 2
    logger.info(f"Done. Output file: {output_size_mb:.1f} MB")

def write_stack_streaming(frames: list[Path], output_path: Path, downsample: int,
                compression: str, compression_level: int):
    """
    Stream frames one-by-one into a BigTIFF file using tifffile.TiffWriter.

    Writing frame-by-frame keeps peak RAM at roughly 2× a single frame,
    regardless of total stack size.
    """
    frame_shape, dtype = probe_frame(frames[0], downsample)
    n_frames = len(frames)

    logger.info(f"Frame shape : {frame_shape}  dtype : {dtype}")
    logger.info(f"Stack size  : {n_frames} frames")
    logger.info(f"Est. output : {estimate_output_size(n_frames, frame_shape, dtype)} (uncompressed)")
    logger.info(f"Compression : {compression}  level {compression_level}")
    logger.info(f"Writing to  : {output_path}")

    compression_arg = None if compression == "none" else compression

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
        for i, fpath in enumerate(frames):
            if i % 100 == 0 or i == n_frames - 1:
                logger.info(f"  [{i + 1:>{len(str(n_frames))}}/{n_frames}]  {fpath.name}")

            frame = load_frame(fpath, downsample, dtype)

            tif.write(
                frame,
                contiguous=True,          # forces a single IFD strip — faster seeks
                compression=compression_arg,
                compressionargs={"level": compression_level} if compression_arg else None,
                metadata=None,            # suppress per-frame XML to keep file lean
            )

    output_size_mb = output_path.stat().st_size / 1024 ** 2
    logger.info(f"Done. Output file: {output_size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _resolve_output(raw_output: Path) -> Path:
    """
    Place the output TIF inside a directory named after its own stem::

        --output /data/stack.tif  →  /data/stack/stack.tif
    """
    return raw_output.parent / raw_output.stem / raw_output.name


def _write_single_channel(
    args,
    frames: list[Path],
    output_path: Path,
    label: str = "",
) -> None:
    """Run the appropriate write strategy for one channel (or a single-channel stack)."""
    frame_shape, dtype = probe_frame(frames[0], args.downsample)
    estimated_bytes = estimate_output_bytes(len(frames), frame_shape, dtype)
    threshold_bytes = args.memory_threshold * 1024 ** 3

    use_inmemory    = args.in_memory or (
        not args.preallocate and estimated_bytes < threshold_bytes
    )
    use_preallocate = args.preallocate

    if use_preallocate:
        mode_label = "preallocated (tifffile.memmap)"
    elif use_inmemory:
        mode_label = (
            f"in-memory (est. {estimated_bytes/1024**3:.1f} GB "
            f"< {args.memory_threshold} GB threshold)"
        )
    else:
        mode_label = (
            f"streaming (est. {estimated_bytes/1024**3:.1f} GB "
            f"≥ {args.memory_threshold} GB threshold)"
        )

    if label:
        logger.info(f"[{label}] Write mode : {mode_label}")
    else:
        logger.info(f"Write mode  : {mode_label}")

    if use_preallocate:
        if args.compression != "none":
            logger.error(
                "--preallocate requires --compression none. "
                "Compressed frame sizes are not known ahead of time, "
                "so the file cannot be preallocated."
            )
            sys.exit(1)
        write_stack_preallocated(
            frames=frames,
            output_path=output_path,
            downsample=args.downsample,
            flush_every=args.flush_every,
        )
    elif use_inmemory:
        write_stack_inmemory(
            frames=frames,
            output_path=output_path,
            downsample=args.downsample,
            compression=args.compression,
            compression_level=args.compression_level,
        )
    else:
        write_stack_streaming(
            frames=frames,
            output_path=output_path,
            downsample=args.downsample,
            compression=args.compression,
            compression_level=args.compression_level,
        )


def main():
    args = parse_args()

    if args.downsample < 1:
        logger.error("--downsample must be ≥ 1")
        sys.exit(1)

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error(f"--input is not a directory: {input_dir}")
        sys.exit(1)

    output_path = _resolve_output(Path(args.output))

    # ── Channel detection ────────────────────────────────────────────────────
    channel_ids = detect_channel_ids(input_dir, args.pattern)

    if channel_ids is not None:
        # ── Multi-channel mode ───────────────────────────────────────────────
        logger.info(
            f"Multi-channel mode: {len(channel_ids)} channel(s) detected: "
            + ", ".join(f"C{c}" for c in channel_ids)
        )

        # Collect frames per channel
        ch_frames: dict[str, list[Path]] = {
            cid: collect_channel_frames(input_dir, args.pattern, cid,
                                        args.start, args.end)
            for cid in channel_ids
        }

        # Derive per-channel output paths
        ch_outputs: dict[str, Path] = {
            cid: channel_output_path(output_path, cid)
            for cid in channel_ids
        }

        # Probe shape/dtype once (assumed identical across channels)
        frame_shape, dtype = probe_frame(ch_frames[channel_ids[0]][0], args.downsample)

        if args.dry_run:
            print("\n--- DRY RUN (multi-channel) ---")
            print(f"  Input dir     : {input_dir}")
            print(f"  Pattern       : {args.pattern}")
            print(f"  Channel count : {len(channel_ids)}")
            for cid in channel_ids:
                n = len(ch_frames[cid])
                print(f"  C{cid}  →  {n} frame(s)  →  {ch_outputs[cid]}")
            print(f"  Frame shape   : {frame_shape}  dtype: {dtype}")
            print(f"  Downsample    : {args.downsample}x")
            print(f"  Compression   : {args.compression}  level {args.compression_level}")
            yaml_path = output_path.parent / f"{output_path.stem}.yaml"
            print(f"  YAML sidecar  : {yaml_path}")
            if not args.keep_sources:
                total = sum(len(v) for v in ch_frames.values())
                print(f"  Delete src    : {total} source frame(s) will be deleted after write")
            print("--- nothing written ---\n")
            return

        # Check overwrites
        for cid, op in ch_outputs.items():
            if op.exists() and not args.overwrite:
                logger.error(
                    f"Output file already exists: {op}\n"
                    "  Pass --overwrite to replace it."
                )
                sys.exit(1)

        n_frames_map: dict[str, int] = {}

        for cid in channel_ids:
            frames = ch_frames[cid]
            op     = ch_outputs[cid]
            logger.info(f"─── Writing channel C{cid} ({len(frames)} frames) → {op}")
            _write_single_channel(args, frames, op, label=f"C{cid}")
            n_frames_map[cid] = len(frames)

        # ── YAML sidecar ─────────────────────────────────────────────────────
        yaml_path = output_path.parent / f"{output_path.stem}.yaml"
        write_yaml_sidecar(
            yaml_path=yaml_path,
            channel_ids=channel_ids,
            output_paths=ch_outputs,
            n_frames_map=n_frames_map,
            frame_shape=frame_shape,
            dtype=dtype,
        )

        # ── Delete sources ────────────────────────────────────────────────────
        if not args.keep_sources:
            all_frames = [f for frames in ch_frames.values() for f in frames]
            # Verify all outputs are intact before deleting anything
            for cid, op in ch_outputs.items():
                if not op.exists() or op.stat().st_size == 0:
                    logger.error(
                        f"Output for channel C{cid} missing or empty ({op}) — "
                        "source frames NOT deleted to avoid data loss."
                    )
                    sys.exit(1)
            _delete_sources(all_frames)
        else:
            logger.info("Source frames preserved (--keep-sources).")

    else:
        # ── Single-channel mode (original behaviour) ─────────────────────────
        logger.info("Single-channel mode")

        if output_path.exists() and not args.overwrite:
            logger.error(
                f"Output file already exists: {output_path}\n"
                "  Pass --overwrite to replace it."
            )
            sys.exit(1)

        frames = collect_frames(input_dir, args.pattern, args.start, args.end)

        frame_shape, dtype = probe_frame(frames[0], args.downsample)
        estimated_bytes    = estimate_output_bytes(len(frames), frame_shape, dtype)

        if args.dry_run:
            use_preallocate = args.preallocate
            use_inmemory    = args.in_memory or (
                not use_preallocate
                and estimated_bytes < args.memory_threshold * 1024 ** 3
            )
            mode = (
                "preallocated (tifffile.memmap)" if use_preallocate else
                "in-memory (numpy array)"        if use_inmemory    else
                "streaming (TiffWriter)"
            )
            print("\n--- DRY RUN ---")
            print(f"  Input dir   : {input_dir}")
            print(f"  Pattern     : {args.pattern}")
            print(f"  Frames      : {len(frames)}")
            print(f"  Frame shape : {frame_shape}  dtype: {dtype}")
            print(f"  Stack shape : {preallocated_stack_shape(len(frames), frame_shape)}")
            print(f"  Downsample  : {args.downsample}x")
            print(f"  Write mode  : {mode}")
            print(f"  Compression : {args.compression}  level {args.compression_level}")
            print(f"  Output dir  : {output_path.parent}  (will be created)")
            print(f"  Output file : {output_path}")
            print(f"  Est. size   : {estimate_output_size(len(frames), frame_shape, dtype)} (uncompressed)")
            if use_preallocate:
                print(f"  Flush every : {args.flush_every} frames")
            if not args.keep_sources:
                print(f"  Delete src  : {len(frames)} source frame(s) will be deleted after write")
            else:
                print(f"  Delete src  : --keep-sources set, source frames preserved")
            print("--- nothing written ---\n")
            return

        _write_single_channel(args, frames, output_path)

        # ── Delete source frames ──────────────────────────────────────────────
        if not args.keep_sources:
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(
                    "Output file missing or empty after write — "
                    "source frames NOT deleted to avoid data loss."
                )
                sys.exit(1)
            _delete_sources(frames)
        else:
            logger.info("Source frames preserved (--keep-sources).")


def _delete_sources(frames: list) -> None:
    """Remove the source frame TIFFs after a successful stack write."""
    logger.info(f"Deleting {len(frames)} source frame(s)...")
    failed = []
    for fpath in frames:
        try:
            Path(fpath).unlink()
        except OSError as exc:
            logger.warning(f"  Could not delete {fpath}: {exc}")
            failed.append(fpath)
    deleted = len(frames) - len(failed)
    logger.info(f"  Deleted {deleted}/{len(frames)} source frame(s).")
    if failed:
        logger.warning(f"  {len(failed)} file(s) could not be deleted (see above).")


if __name__ == "__main__":
    main()
