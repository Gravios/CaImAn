#!/usr/bin/env python3
"""
utilities/stack_sessions.py
============================
Stack OME-TIFF frame series for all session directories under a parent
directory.  Replaces ``stack_sessions.sh``.

Imports :mod:`caiman.utils.ome_meta` and :mod:`caiman.utils.stack_to_bigtiff`
directly — no ``--script-dir`` needed, no OS ARG_MAX constraint.

Usage
-----
Run from the parent date directory (``--parent`` defaults to CWD)::

    cd /data/source/strohA/.../strohA-ia-000000-20150709
    python ~/software/CaImAn/utilities/stack_sessions.py --prefix strohA-ia

Explicit parent::

    python ~/software/CaImAn/utilities/stack_sessions.py \\
        --prefix strohA-ia \\
        --parent /data/source/strohA/.../strohA-ia-000000-20150709

Options
-------
--prefix          Session directory prefix, e.g. ``strohA-ia`` (required)
--parent          Parent directory containing session subdirs (default: CWD)
--delete-sources  Delete source frame TIFFs after each successful stack write
--dry-run         Preview without writing any files
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# caiman.utils imports — ome_meta and stack_to_bigtiff now live there
# ---------------------------------------------------------------------------
try:
    from caiman.utils.ome_meta import extract_pixels, format_rate_str, update_yaml
    from caiman.utils.stack_to_bigtiff import stack_frames
except ImportError as _exc:
    sys.exit(
        f"ImportError: {_exc}\n"
        "Ensure the CaImAn package is installed or on PYTHONPATH:\n"
        "  cd ~/software/CaImAn && pip install -e . --no-deps"
    )

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Matches the channel token in OME-TIFF filenames: _C00_t0000.ome.tif
_CHANNEL_RE = re.compile(r'_C(\d+)_t', re.IGNORECASE)
# Matches the time index for padding: _tNNNN(N…)(.ome).tif
_TINDEX_RE  = re.compile(r'(_t)(\d+)((?:\.ome)?\.tif)$', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiff_magic(path: Path) -> bool:
    """Return True if *path* starts with a TIFF magic number (II or MM)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) in (b"II", b"MM")
    except OSError:
        return False


def _find_master(session_dir: Path) -> Path | None:
    """Return the first frame TIF that carries valid OME-XML metadata.

    Tries ``*_t0*.ome.tif`` (earliest time-point, handles both unpadded
    _t0000 and padded _t000000) then any ``*.ome.tif``.  Validates each
    candidate with a magic-byte check and an OME-XML probe before accepting.
    Resilient to partial deletions that may have removed the original t0000.
    """
    candidates = sorted(session_dir.glob("*_t0*.ome.tif")) \
               + sorted(session_dir.glob("*.ome.tif"))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if not _tiff_magic(path):
            continue
        try:
            extract_pixels(path)   # raises / exits if not OME
            return path
        except SystemExit:
            continue
    return None


def _discover_channels(session_dir: Path) -> list[str]:
    """Return sorted unique channel IDs from ``*_C{N}_t0*.ome.tif`` files."""
    ids: set[str] = set()
    for f in session_dir.glob("*_C*_t0*.ome.tif"):
        m = _CHANNEL_RE.search(f.name)
        if m:
            ids.add(m.group(1).zfill(2))
    return sorted(ids)


def _pad_time_indices(session_dir: Path, channel: str, width: int) -> int:
    """Zero-pad the ``_tNNNN`` index in all channel *channel* frame filenames
    to *width* digits.  Returns the number of files renamed."""
    renamed = 0
    for f in session_dir.glob(f"*_C{channel}_t*.tif"):
        m = _TINDEX_RE.search(f.name)
        if m and len(m.group(2)) < width:
            new_name = (
                f.name[: m.start(1)]
                + m.group(1)
                + m.group(2).zfill(width)
                + m.group(3)
            )
            f.rename(f.parent / new_name)
            renamed += 1
    return renamed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--prefix", required=True, metavar="PREFIX",
        help="Session directory prefix, e.g. strohA-ia",
    )
    p.add_argument(
        "--parent", default=None, metavar="DIR",
        help="Parent directory containing session subdirectories "
             "(default: current working directory)",
    )
    p.add_argument(
        "--delete-sources", action="store_true",
        help="Delete source frame TIFFs after each successful stack write. "
             "Off by default.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be done without writing any files.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    parent_dir = Path(args.parent).resolve() if args.parent else Path.cwd()
    if not parent_dir.is_dir():
        logger.error(f"--parent is not a directory: {parent_dir}")
        return 1

    session_dirs = sorted(
        d for d in parent_dir.glob(f"{args.prefix}-*/") if d.is_dir()
    )
    if not session_dirs:
        logger.error(
            f"No session directories found matching: {parent_dir}/{args.prefix}-*/"
        )
        return 1

    print(f"Parent dir  : {parent_dir}")
    print(f"Prefix      : {args.prefix}")
    print(f"Sessions    : {len(session_dirs)}")
    print(f"Dry run     : {args.dry_run}")
    print(f"Del sources : {args.delete_sources}")
    print()

    for session_dir in session_dirs:
        session_name = session_dir.name
        print(f"==> {session_name}")

        # ── Find master OME-TIFF ─────────────────────────────────────────────
        master = _find_master(session_dir)
        if master is None:
            print("    No OME-TIFF files found, skipping")
            continue

        # ── Read OME metadata ────────────────────────────────────────────────
        try:
            px = extract_pixels(master)
        except SystemExit:
            print(f"    ome_meta failed on {master.name}, skipping")
            continue

        frame_count  = px["size_t"] or 0
        sample_rate  = px["sample_rate_hz"]
        rate_str     = format_rate_str(sample_rate) if sample_rate else "?p??"
        fc_str       = f"{frame_count:06d}"

        print(f"    Master      : {master.name}")
        print(f"    Frame count : {fc_str}")
        print(f"    Sample rate : {rate_str} Hz")

        # ── Discover channels ────────────────────────────────────────────────
        channel_ids = _discover_channels(session_dir)
        if not channel_ids:
            print("    No channel frame files found, skipping")
            continue

        yaml_updated = False

        for ch in channel_ids:
            output = session_dir / f"{session_name}-C{ch}-fc{fc_str}.tif"

            if output.exists():
                print(f"    C{ch}: already exists, skipping")
                continue

            frames = sorted(session_dir.glob(f"*_C{ch}_t*.tif"))
            print(f"    C{ch}: {len(frames)} frames -> {output.name}")

            if args.dry_run:
                continue

            # ── Pad time indices (pure Python, no rename, no ARG_MAX) ────────
            sample_t = _TINDEX_RE.search(frames[0].name)
            cur_width = len(sample_t.group(2)) if sample_t else 4
            needed    = max(6, len(str(max(frame_count - 1, 0))))
            if cur_width < needed:
                print(f"    C{ch}: padding time indices → {needed} digits...")
                n = _pad_time_indices(session_dir, ch, needed)
                print(f"    C{ch}: padded {n} filename(s)")
                frames = sorted(session_dir.glob(f"*_C{ch}_t*.tif"))
                # Update master path if it was renamed
                if ch == channel_ids[0]:
                    new_master = session_dir / re.sub(
                        r'_t(\d+)((?:\.ome)?\.tif)$',
                        lambda m: f"_t{m.group(1).zfill(needed)}{m.group(2)}",
                        master.name,
                    )
                    if new_master.exists():
                        master = new_master

            # ── Stack ────────────────────────────────────────────────────────
            stack_frames(
                input_dir   = session_dir,
                output_path = output,
                pattern     = f"*_C{ch}_t*.tif",
                preallocate = True,
                compression = "none",
            )

            # ── Verify + optional source deletion ───────────────────────────
            if not output.exists() or output.stat().st_size == 0:
                print(f"    C{ch}: ERROR — output missing or empty, sources preserved")
                continue

            if args.delete_sources:
                print(f"    C{ch}: deleting {len(frames)} source frame(s)...")
                for f in frames:
                    try:
                        f.unlink()
                    except OSError as exc:
                        logger.warning(f"      Could not delete {f.name}: {exc}")
                print(f"    C{ch}: deleted")

            # ── Update Trial YAML (first channel only) ───────────────────────
            if not yaml_updated:
                yaml_path = session_dir / f"{session_name}.yaml"
                if yaml_path.exists():
                    print(f"    yaml: updating {yaml_path.name} with OME metadata")
                    try:
                        update_yaml(yaml_path, px)
                    except Exception as exc:
                        print(f"    yaml: WARNING update failed: {exc}")
                else:
                    print(f"    yaml: {yaml_path.name} not found, skipping OME update")
                yaml_updated = True

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
