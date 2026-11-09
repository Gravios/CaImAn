#!/usr/bin/env python3
"""
utilities/stack_sessions.py
============================
Stack OME-TIFF frame series into per-channel BigTIFF stacks for every
session directory under a parent directory.

For each session this script:
  1. Finds the master OME-TIFF and reads frame count / sample rate from
     its OME-XML header.
  2. Discovers channels from *_CNN_t0*.ome.tif sentinel files.
  3. Creates or updates the session Trial YAML, populating every field
     it can infer from the directory name, frame filename, and OME header.
  4. Validates each channel's frame set: time indices unique and
     contiguous from 0, count equal to the OME SizeT.  A gap or duplicate
     would shift every later frame in time, so the channel is refused.
  5. Zero-pads time indices to 10 digits (for tools that sort by text;
     stacking itself orders frames numerically).
  6. Stacks each channel into a single preallocated BigTIFF.  The stack is
     written as ``*.partial.tif`` and renamed only once complete, so an
     interrupted run is redone, not skipped, on the next invocation.

A failure in one channel or session is reported and the batch continues;
the exit status is 1 if anything failed.  --dry-run writes nothing.

Single session / trial
----------------------
  stack-sessions --session strohA-ia-000000-20150709-TL001_134044-25x-spont

Quick start
-----------
  cd /data/source/strohA/.../strohA-ia-000000-20150709
  stack-sessions --prefix strohA-ia --dry-run          # preview
  stack-sessions --prefix strohA-ia                    # live run (sources kept)
  stack-sessions --prefix strohA-ia --delete-sources   # delete frames after stack

Flags
-----
  --prefix PREFIX       Session directory prefix to match, e.g. strohA-ia.
                        Required unless --session is given.
  --session NAME        Process only this session directory (repeatable).
                        Combined with --prefix, both must match.
  --parent DIR          Parent directory containing session subdirs.
                        Default: current working directory.
  --delete-sources      Delete source frame TIFFs after the session's
                        channels are stacked. Off by default. The OME master
                        frame is kept while any channel is still unstacked.
  --allow-incomplete    Accept fewer frames than SizeT when indices are
                        contiguous from 0 (aborted acquisition). Gaps are
                        always refused.
  --dry-run             Preview what would be done — no files written.
  -h / --help           Show this help message.

Output layout (per session)
---------------------------
  <session>/
    <session>.yaml                                          Trial YAML
    <session>-C00-fc<NNNNNN>/<session>-C00-fc<NNNNNN>.tif   channel 00
    <session>-C01-fc<NNNNNN>/<session>-C01-fc<NNNNNN>.tif   channel 01 (if present)

This is the layout new-session and batch-sessions discover.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# caiman.utils imports
# ---------------------------------------------------------------------------
try:
    from caiman.utils.ome_meta import extract_pixels, format_rate_str, update_yaml
    from caiman.utils.stack_to_bigtiff import frame_sort_key, stack_frames
except ImportError as _exc:
    sys.exit(
        f"ImportError: {_exc}\n"
        "Ensure the CaImAn package is installed or on PYTHONPATH:\n"
        "  cd ~/software/CaImAn && pip install -e . --no-deps"
    )

# session-name → YAML fields (moved to the dependency-free utilities.session_name)
from utilities.session_name import parse_session_name as _parse_session_name  # noqa: E402

try:
    import yaml as _yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------


# Frame filename tokens, in any order:
#   tl1-FA1-LP11-90um-spont25x_C00_t0000.ome.tif
#   tl1-25xspont-FA1-LP4-87um_C00_t0000.ome.tif
#   FA{n}   → frame averaging factor
#   LP{n}   → laser power (integer percent; LP11_5 → 11.5)
#   {n}um   → imaging depth in µm
#   {n}x    → magnification (fallback if not in dirname)
# Token boundaries are any non-alphanumeric character.  \b is not used
# because "_" is a word character: "87um_C00" never matched \b(\d+)um\b.
# Magnification may touch letters on either side (25xspont, spont25x).
_A = r'(?<![A-Za-z0-9])'
_Z = r'(?![A-Za-z0-9])'
_FA_RE    = re.compile(_A + r'FA(\d+)' + _Z,              re.IGNORECASE)
_LP_RE    = re.compile(_A + r'LP(\d+(?:_\d+)?)' + _Z,     re.IGNORECASE)
_DEPTH_RE = re.compile(_A + r'(\d+)um' + _Z,              re.IGNORECASE)
_MAG_RE   = re.compile(r'(?<![0-9.])(\d+)x(?![0-9])',      re.IGNORECASE)


# Minimum zero-padded width of the _t<index> token (10 digits ≈ 10^10 frames,
# ~10 years at 30 Hz); widened automatically if a series ever needs more.
PAD_WIDTH = 10

# Time-index regex for padding
_TINDEX_RE = re.compile(r'(_t)(\d+)((?:\.ome)?\.tif)$', re.IGNORECASE)
_CHANNEL_RE = re.compile(r'_C(\d+)_t', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_frame_name(name: str) -> dict:
    """Extract acquisition settings embedded in a frame filename.

    Returns a flat dict with keys: ``fa``, ``laser_power``, ``depth_um``,
    ``magnification`` (fallback).
    """
    out: dict = {}
    m = _FA_RE.search(name)
    if m:
        try:
            out["fa"] = int(m.group(1))
        except ValueError:
            pass

    m = _LP_RE.search(name)
    if m:
        try:
            out["laser_power"] = float(m.group(1).replace("_", "."))
        except ValueError:
            pass

    m = _DEPTH_RE.search(name)
    if m:
        try:
            out["depth_um"] = -abs(int(m.group(1)))  # depth is negative (below surface)
        except ValueError:
            pass

    m = _MAG_RE.search(name)
    if m:
        try:
            out.setdefault("magnification", int(m.group(1)))
        except ValueError:
            pass

    return out


# ---------------------------------------------------------------------------
# YAML construction
# ---------------------------------------------------------------------------

def _find_template() -> Path | None:
    """Locate template_acquisition.yaml relative to this script."""
    candidates = [
        Path(__file__).resolve().parent / "pipelines" / "template_acquisition.yaml",
        Path(__file__).resolve().parent / "template_acquisition.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_yaml(
    session_name: str,
    master_frame: Path,
    px: dict,
    n_channels: int,
) -> dict:
    """Build an acquisition YAML dict from template + parsed metadata.

    The template is always loaded verbatim first so every field is present
    with a null value.  Parsed values are then filled in on top in priority
    order: dirname < frame filename < OME header.

    Priority (highest last, i.e. higher overwrites lower):
      template skeleton -> dirname parse -> frame filename parse -> OME header
    """
    # ── 1. Load template verbatim (all fields present, all null) ─────────
    template_path = _find_template()
    if template_path and _HAVE_YAML:
        doc = _yaml.safe_load(template_path.read_text()) or {}
    else:
        # Minimal fallback matching the template schema
        doc = {
            "experiment": {"datetime": None, "id": None, "class": None, "condition": None},
            "subject": {
                "id": None, "class": None, "species": None, "sex": None,
                "age": None, "genotype": None,
                "weight": {"value": None, "units": None},
                "heart_rate": {"value": None, "units": None},
                "body_temperature": {"value": None, "units": None},
            },
            "regions": {
                "type": None,
                "vis_ctx": {
                    "indicator": None,
                    "coordinates": {
                        "ap": {"value": None, "units": "mm"},
                        "ml": {"value": None, "units": "mm"},
                    },
                },
            },
            "acquisition_system": {"settings": {
                "sample_rate":  {"value": None, "units": "Hz"},
                "magnification": {"value": None, "units": "factor"},
                "gain":         {"value": None, "units": None},
                "laserPower":   {"value": None, "units": "percent"},
                "fa":           {"value": None, "units": None},
                "depth":        {"value": None, "units": "um"},
                "frame_size":   {"x": None, "y": None, "units": "pixels"},
                "pixel_size":   {"x": None, "y": None, "units": "um"},
                "n_channels": None, "n_planes": None,
                "n_frames": None, "pixel_dtype": None,
            }},
        }

    s = doc.setdefault("acquisition_system", {}).setdefault("settings", {})

    # ── 2. Dirname ────────────────────────────────────────────────────────
    dn = _parse_session_name(session_name)
    if dn.get("experiment_datetime"):
        doc["experiment"]["datetime"] = dn["experiment_datetime"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    if dn.get("experiment_id") is not None:
        doc["experiment"]["id"] = dn["experiment_id"]
    if dn.get("condition"):
        doc["experiment"]["condition"] = dn["condition"]
    if dn.get("subject_id") is not None:
        doc["subject"]["id"] = dn["subject_id"]
    if dn.get("magnification"):
        s["magnification"]["value"] = dn["magnification"]

    # ── 3. Frame filename ─────────────────────────────────────────────────
    fn = _parse_frame_name(master_frame.name)
    if fn.get("fa") is not None:
        s["fa"]["value"] = fn["fa"]
    if fn.get("laser_power") is not None:
        s["laserPower"]["value"] = fn["laser_power"]
    if fn.get("depth_um") is not None:
        s["depth"]["value"] = fn["depth_um"]
    if fn.get("magnification") and not dn.get("magnification"):
        s["magnification"]["value"] = fn["magnification"]

    # ── 4. OME header (highest priority) ─────────────────────────────────
    if px.get("sample_rate_hz"):
        s["sample_rate"]["value"] = round(px["sample_rate_hz"], 4)
    if px.get("size_x") and px.get("size_y"):
        s["frame_size"]["x"] = px["size_x"]
        s["frame_size"]["y"] = px["size_y"]
    if px.get("physical_size_x") and px.get("physical_size_y"):
        s["pixel_size"]["x"] = round(px["physical_size_x"], 6)
        s["pixel_size"]["y"] = round(px["physical_size_y"], 6)
    if px.get("pixel_type"):
        s["pixel_dtype"] = px["pixel_type"]
    if px.get("size_t"):
        s["n_frames"] = px["size_t"]
    if px.get("size_z"):
        s["n_planes"] = px["size_z"]

    s["n_channels"] = n_channels

    return doc


def _write_yaml(path: Path, doc: dict) -> None:
    if _HAVE_YAML:
        with open(path, "w") as fh:
            _yaml.dump(doc, fh, default_flow_style=False,
                       sort_keys=False, allow_unicode=True)
    else:
        # Minimal hand-rolled fallback (avoids hard pyyaml dependency)
        def _render(d, indent=0):
            lines = []
            pad = "  " * indent
            for k, v in d.items():
                if isinstance(v, dict):
                    lines.append(f"{pad}{k}:")
                    lines.extend(_render(v, indent + 1))
                elif v is None:
                    lines.append(f"{pad}{k}: null")
                elif isinstance(v, bool):
                    lines.append(f"{pad}{k}: {'true' if v else 'false'}")
                elif isinstance(v, str):
                    lines.append(f"{pad}{k}: {v}")
                else:
                    lines.append(f"{pad}{k}: {v}")
            return lines
        path.write_text("\n".join(_render(doc)) + "\n")


# ---------------------------------------------------------------------------
# Frame / session helpers
# ---------------------------------------------------------------------------

def _tiff_magic(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(2) in (b"II", b"MM")
    except OSError:
        return False


def _find_master(session_dir: Path) -> Path | None:
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
            extract_pixels(path)
            return path
        except SystemExit:
            continue
    return None


def _discover_channels(session_dir: Path) -> list[str]:
    ids: set[str] = set()
    for f in session_dir.glob("*_C*_t0*.ome.tif"):
        m = _CHANNEL_RE.search(f.name)
        if m:
            ids.add(m.group(1).zfill(2))
    return sorted(ids)


def _channel_frames(session_dir: Path, channel: str) -> list[Path]:
    """Raw frames of one channel in numeric time order (width-independent)."""
    return sorted(session_dir.glob(f"*_C{channel}_t*.tif"), key=frame_sort_key)


def _check_frames(frames: list[Path], frame_count: int,
                  allow_incomplete: bool) -> tuple[int | None, str]:
    """Validate a channel's frame set before stacking.

    The stack's time axis is the frame order, so a missing or duplicated
    index would silently shift every later frame.  Requires indices to be
    unique and contiguous from 0.  Fewer frames than the OME SizeT (an
    aborted acquisition) is accepted only with *allow_incomplete*; more is
    always an error (the header does not describe this series).

    Returns ``(n_frames, "")`` on success or ``(None, reason)``.
    """
    idx: dict[int, list[str]] = {}
    for f in frames:
        m = _TINDEX_RE.search(f.name)
        if m is None:
            return None, f"no _t<index> in {f.name}"
        idx.setdefault(int(m.group(2)), []).append(f.name)
    dups = {i: names for i, names in idx.items() if len(names) > 1}
    if dups:
        i, names = next(iter(sorted(dups.items())))
        return None, (f"{len(dups)} duplicated time index(es), e.g. t={i}: "
                      + ", ".join(names))
    n = len(idx)
    missing = sorted(set(range(max(idx) + 1)) - set(idx)) if idx else []
    if missing:
        shown = ", ".join(map(str, missing[:10])) + (" ..." if len(missing) > 10 else "")
        return None, f"{len(missing)} missing frame index(es): {shown}"
    if frame_count and n > frame_count:
        return None, f"{n} frames on disk but OME SizeT is {frame_count}"
    if frame_count and n < frame_count and not allow_incomplete:
        return None, (f"{n} of {frame_count} frames (indices 0-{n - 1}); "
                      "pass --allow-incomplete to stack the recorded frames")
    return n, ""


def _pad_time_indices(session_dir: Path, channel: str, width: int) -> int:
    """Zero-pad ``_t<N>`` to *width* digits so text order equals time order
    for downstream tools.  Never overwrites: an existing target name means
    the same index exists twice, which :func:`_check_frames` rejects first.
    """
    plan = []
    for f in session_dir.glob(f"*_C{channel}_t*.tif"):
        m = _TINDEX_RE.search(f.name)
        if m and len(m.group(2)) < width:
            new = f.parent / (f.name[: m.start(1)] + m.group(1)
                              + m.group(2).zfill(width) + m.group(3))
            if new.exists():
                raise FileExistsError(f"refusing to overwrite {new.name} with {f.name}")
            plan.append((f, new))
    for f, new in plan:
        f.rename(new)
    return len(plan)


def _stacked_output(session_dir: Path, session_name: str, channel: str) -> Path | None:
    """Existing finished stack for *channel*, whatever frame count its name carries."""
    for d in sorted(session_dir.glob(f"{session_name}-C{channel}-fc*")):
        out = d / f"{d.name}.tif"
        if d.is_dir() and out.exists():
            return out
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Output per session:\n  <session>-C00-fc<NNNNNN>/<session>-C00-fc<NNNNNN>.tif\n                             BigTIFF stack for channel 00\n  <session>.yaml             Trial YAML (created/updated)\n\nExamples:\n  stack-sessions --prefix strohA-ia\n  stack-sessions --prefix strohA-ia --delete-sources\n  stack-sessions --prefix strohA-ia --dry-run',
    )
    p.add_argument(
        "--prefix", default=None, metavar="PREFIX",
        help="Session directory prefix, e.g. strohA-ia "
             "(required unless --session is given)",
    )
    p.add_argument(
        "--session", action="append", default=None, metavar="NAME",
        help="Process only this session directory name (repeatable)",
    )
    p.add_argument(
        "--parent", default=None, metavar="DIR",
        help="Parent directory containing session subdirectories "
             "(default: current working directory)",
    )
    p.add_argument(
        "--delete-sources", action="store_true",
        help="Delete source frame TIFFs after each successful stack write.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be done without writing any files.",
    )
    p.add_argument(
        "--allow-incomplete", action="store_true",
        help="Stack a channel that has fewer frames than the OME SizeT, as long as "
             "its indices are contiguous from 0 (e.g. an aborted acquisition). "
             "The stem then carries the recorded count. Gaps are always refused.",
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

    if not args.prefix and not args.session:
        logger.error("one of --prefix or --session is required")
        return 2

    if args.session:
        missing = [n for n in args.session if not (parent_dir / n).is_dir()]
        if missing:
            logger.error(f"--session not found under {parent_dir}: {', '.join(missing)}")
            return 1
        session_dirs = sorted(
            parent_dir / n for n in dict.fromkeys(args.session)
            if not args.prefix or n.startswith(f"{args.prefix}-")
        )
    else:
        session_dirs = sorted(
            d for d in parent_dir.glob(f"{args.prefix}-*/") if d.is_dir()
        )
    if not session_dirs:
        logger.error(
            f"No session directories found matching: {parent_dir}/{args.prefix}-*/"
        )
        return 1

    print(f"Parent dir  : {parent_dir}")
    print(f"Prefix      : {args.prefix or '-'}")
    print(f"Sessions    : {len(session_dirs)}")
    print(f"Dry run     : {args.dry_run}")
    print(f"Del sources : {args.delete_sources}")
    print()

    failed: list[str] = []

    for session_dir in session_dirs:
        session_name = session_dir.name
        print(f"==> {session_name}")

        master = _find_master(session_dir)
        if master is None:
            print("    No OME-TIFF files found, skipping")
            continue

        try:
            px = extract_pixels(master)
        except SystemExit:
            print(f"    ome_meta failed on {master.name}, skipping")
            failed.append(session_name)
            continue

        channel_ids = _discover_channels(session_dir)
        if not channel_ids:
            print("    No channel frame files found, skipping")
            continue

        frame_count = px["size_t"] or 0
        if not frame_count:
            # No SizeT in the OME header: fall back to the on-disk frame count
            # of the first channel rather than stamping the stem with fc000000.
            frame_count = len(list(session_dir.glob(f"*_C{channel_ids[0]}_t*.tif")))
            print(f"    WARNING: OME header has no SizeT; using {frame_count} "
                  f"frames found on disk for C{channel_ids[0]}")
        sample_rate = px["sample_rate_hz"]
        rate_str    = format_rate_str(sample_rate) if sample_rate else "?p??"
        fc_str      = f"{frame_count:06d}"

        print(f"    Master      : {master.name}")
        print(f"    Frame count : {fc_str}")
        print(f"    Sample rate : {rate_str} Hz")

        # ── Build and write Trial YAML (before stacking) ─────────────────────
        yaml_path = session_dir / f"{session_name}.yaml"
        if yaml_path.exists():
            if args.dry_run:
                print(f"    yaml: {yaml_path.name} exists - would update OME fields only")
            else:
                print(f"    yaml: {yaml_path.name} already exists - updating OME fields only")
                try:
                    update_yaml(yaml_path, px)
                except (Exception, SystemExit) as exc:
                    print(f"    yaml: WARNING update failed: {exc}")
        else:
            if not args.dry_run:
                try:
                    doc = build_yaml(session_name, master, px, len(channel_ids))
                    _write_yaml(yaml_path, doc)
                except Exception as exc:
                    print(f"    yaml: WARNING write failed: {exc}")
                    doc = {}
                s   = doc.get("acquisition_system", {}).get("settings", {})
                exp = doc.get("experiment", {})
                sub = doc.get("subject", {})
                populated = {k: v for k, v in {
                    "datetime":   exp.get("datetime"),
                    "condition":  exp.get("condition"),
                    "subject_id": sub.get("id"),
                    "magnif":     (s.get("magnification") or {}).get("value"),
                    "fa":         (s.get("fa") or {}).get("value"),
                    "laser_%":    (s.get("laserPower") or {}).get("value"),
                    "depth_um":   (s.get("depth") or {}).get("value"),
                    "fr_Hz":      (s.get("sample_rate") or {}).get("value"),
                    "frame_size": s.get("frame_size"),
                    "n_channels": s.get("n_channels"),
                    "n_frames":   s.get("n_frames"),
                }.items() if v is not None}
                if doc:
                    fields_str = "  ".join(f"{k}={v}" for k, v in populated.items())
                    print(f"    yaml: wrote {yaml_path.name}  [{fields_str}]")
            else:
                dn = _parse_session_name(session_name)
                fn = _parse_frame_name(master.name)
                print(f"    yaml: would write {yaml_path.name}")
                print(f"           dirname -> {dn}")
                print(f"           frame   -> {fn}")

        to_delete: list[Path] = []
        for ch in channel_ids:
            # Writes are atomic (*.partial.tif → rename), so an existing
            # output is a complete stack.
            existing = _stacked_output(session_dir, session_name, ch)
            if existing is not None:
                print(f"    C{ch}: already exists ({existing.parent.name}/), skipping")
                left = _channel_frames(session_dir, ch)
                if left and args.delete_sources:
                    print(f"    C{ch}: {len(left)} raw frame(s) remain beside the existing "
                          "stack (e.g. a kept OME master); not deleted by this run")
                continue

            frames = _channel_frames(session_dir, ch)
            if not frames:
                print(f"    C{ch}: ERROR - no *_C{ch}_t*.tif frames, skipping")
                failed.append(f"{session_name} C{ch}")
                continue

            n_frames, why = _check_frames(frames, frame_count, args.allow_incomplete)
            if n_frames is None:
                print(f"    C{ch}: ERROR - {why}; not stacked, sources preserved")
                failed.append(f"{session_name} C{ch}")
                continue
            if n_frames != frame_count:
                print(f"    C{ch}: WARNING - {n_frames} of {frame_count} frames "
                      "(--allow-incomplete); stem carries the recorded count")

            stem   = f"{session_name}-C{ch}-fc{n_frames:06d}"
            output = session_dir / stem / f"{stem}.tif"
            print(f"    C{ch}: {n_frames} frames -> {stem}/")

            # ── Pad time indices ─────────────────────────────────────────────
            # Stacking orders frames numerically, so padding is not needed for
            # correctness; it keeps text order right for other tools.  Every
            # file is checked (not just the first), so a run interrupted
            # mid-rename is completed.
            needed = max(PAD_WIDTH, len(str(max(n_frames - 1, 0))))
            n_short = sum(len(_TINDEX_RE.search(f.name).group(2)) < needed for f in frames)

            if args.dry_run:
                if n_short:
                    print(f"    C{ch}: would rename {n_short} frame file(s) in place "
                          f"to pad time indices -> {needed} digits")
                continue

            if n_short:
                print(f"    C{ch}: padding time indices -> {needed} digits...")
                try:
                    n = _pad_time_indices(session_dir, ch, needed)
                except FileExistsError as exc:
                    print(f"    C{ch}: ERROR - {exc}; not stacked")
                    failed.append(f"{session_name} C{ch}")
                    continue
                print(f"    C{ch}: padded {n} filename(s)")
                frames = _channel_frames(session_dir, ch)

            # ── Create channel subdir and stack ──────────────────────────────
            # stack_to_bigtiff reports bad input via sys.exit; catch it here so
            # one broken channel does not abort the remaining sessions.
            try:
                stack_frames(
                    input_dir   = session_dir,
                    output_path = output,
                    pattern     = f"*_C{ch}_t*.tif",
                    preallocate = True,
                    compression = "none",
                )
            except (Exception, SystemExit) as exc:
                print(f"    C{ch}: ERROR - stacking failed ({exc!r}), sources preserved")
                failed.append(f"{session_name} C{ch}")
                continue

            if not output.exists():
                print(f"    C{ch}: ERROR - output missing, sources preserved")
                failed.append(f"{session_name} C{ch}")
                continue

            if args.delete_sources:
                to_delete.extend(frames)

        # ── Delete sources (deferred to the end of the session) ──────────────
        # The OME master frame is the session's only record of SizeT and the
        # sample rate.  Keep it while any channel is still unstacked so a
        # rerun for that channel can still read the header.
        if to_delete:
            keep: set[Path] = set()
            pending = [ch for ch in channel_ids
                       if _stacked_output(session_dir, session_name, ch) is None]
            if pending:
                cur_master = _find_master(session_dir)
                if cur_master is not None:
                    keep.add(cur_master)
                    print(f"    keeping OME master {cur_master.name} until "
                          f"{', '.join('C' + c for c in pending)} is stacked")
            doomed = [f for f in to_delete if f not in keep]
            print(f"    deleting {len(doomed)} source frame(s)...")
            for f in doomed:
                try:
                    f.unlink()
                except OSError as exc:
                    logger.warning(f"      Could not delete {f.name}: {exc}")

        print()

    if failed:
        print(f"FAILED ({len(failed)}):")
        for item in failed:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

