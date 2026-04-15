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
  3. Zero-pads time indices to 6 digits so alphabetic sort equals
     temporal order (fixes Olympus FluoView mixed-width indices).
  4. Stacks each channel into a single preallocated BigTIFF.
  5. Creates or updates the session Trial YAML, populating every field
     it can infer from the directory name, frame filename, and OME header.

Quick start
-----------
  cd /data/source/strohA/.../strohA-ia-000000-20150709
  stack-sessions --prefix strohA-ia            # dry run
  stack-sessions --prefix strohA-ia            # live run (sources kept)
  stack-sessions --prefix strohA-ia --delete-sources   # delete frames after stack

Flags
-----
  --prefix PREFIX       Session directory prefix to match (required).
                        e.g. strohA-ia
  --parent DIR          Parent directory containing session subdirs.
                        Default: current working directory.
  --delete-sources      Delete source frame TIFFs after each successful
                        stack write. Off by default.
  --dry-run             Preview what would be done — no files written.
  -h / --help           Show this help message.

Output layout (per session)
---------------------------
  <session>-C00-fc<N>.tif          BigTIFF stack, channel 00
  <session>-C01-fc<N>.tif          BigTIFF stack, channel 01  (if present)
  <session>.yaml                   Trial YAML (created or updated)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# caiman.utils imports
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

# Session directory name pattern:
#   strohA-ia-000000-20150709-TL001_134044-25x-spont
#   {prefix}-{subject_id}-{date8}-TL{trial}_{time6}-{mag}x-{condition}
_SESSION_RE = re.compile(
    r'^(?P<prefix>.+?)'            # strohA-ia
    r'-(?P<subject_id>\d+)'        # 000000
    r'-(?P<date>\d{8})'            # 20150709
    r'-TL(?P<trial>\d+)'           # TL001
    r'_(?P<time>\d{6})'            # 134044
    r'-(?P<mag>\d+)x'              # 25x
    r'-(?P<condition>[^/]+)$',     # spont
    re.IGNORECASE,
)

# Frame filename tokens:  tl1-FA1-LP11-90um-spont25x_C00_t0000.ome.tif
#   FA{n}   → frame averaging factor
#   LP{n}   → laser power (integer percent; LP11_5 → 11.5)
#   {n}um   → imaging depth in µm
#   {n}x    → magnification (fallback if not in dirname)
_FA_RE    = re.compile(r'\bFA(\d+)\b',              re.IGNORECASE)
_LP_RE    = re.compile(r'\bLP(\d+(?:_\d+)?)\b',     re.IGNORECASE)
_DEPTH_RE = re.compile(r'\b(\d+)um\b',              re.IGNORECASE)
_MAG_RE   = re.compile(r'\b(\d+)x\b',               re.IGNORECASE)

# Condition keywords found anywhere in session name after the magnification
_CONDITION_KEYWORDS = {
    "spont":  "spontaneous",
    "vstim":  "visual_stimulus",
    "vs":     "visual_stimulus",
    "astim":  "auditory_stimulus",
    "loco":   "locomotion",
    "dark":   "dark",
    "default": "default",
}

# Time-index regex for padding
_TINDEX_RE = re.compile(r'(_t)(\d+)((?:\.ome)?\.tif)$', re.IGNORECASE)
_CHANNEL_RE = re.compile(r'_C(\d+)_t', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_session_name(name: str) -> dict:
    """Extract structured metadata from a session directory name.

    Returns a flat dict with keys matching template_acquisition.yaml fields:
    ``subject_id``, ``experiment_datetime``, ``experiment_id``,
    ``condition``, ``magnification``.
    """
    out: dict = {}
    m = _SESSION_RE.match(name)
    if not m:
        return out

    # subject id
    try:
        out["subject_id"] = int(m.group("subject_id"))
    except ValueError:
        pass

    # datetime
    try:
        dt = datetime.strptime(
            m.group("date") + m.group("time"), "%Y%m%d%H%M%S"
        )
        out["experiment_datetime"] = dt
    except ValueError:
        pass

    # trial number → experiment id
    try:
        out["experiment_id"] = int(m.group("trial"))
    except ValueError:
        pass

    # magnification
    try:
        out["magnification"] = int(m.group("mag"))
    except ValueError:
        pass

    # condition - normalise known keywords
    cond_raw = m.group("condition").lower()
    for kw, label in _CONDITION_KEYWORDS.items():
        if kw in cond_raw:
            out["condition"] = label
            break
    else:
        out["condition"] = m.group("condition")

    return out


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
# Frame/session helpers (unchanged from previous version)
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


def _pad_time_indices(session_dir: Path, channel: str, width: int) -> int:
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
        epilog='Output per session:\n  <session>-C00-fc<N>.tif    BigTIFF stack for channel 00\n  <session>.yaml             Trial YAML (created/updated)\n\nExamples:\n  stack-sessions --prefix strohA-ia\n  stack-sessions --prefix strohA-ia --delete-sources\n  stack-sessions --prefix strohA-ia --dry-run',
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
        help="Delete source frame TIFFs after each successful stack write.",
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

        master = _find_master(session_dir)
        if master is None:
            print("    No OME-TIFF files found, skipping")
            continue

        try:
            px = extract_pixels(master)
        except SystemExit:
            print(f"    ome_meta failed on {master.name}, skipping")
            continue

        frame_count = px["size_t"] or 0
        sample_rate = px["sample_rate_hz"]
        rate_str    = format_rate_str(sample_rate) if sample_rate else "?p??"
        fc_str      = f"{frame_count:06d}"

        print(f"    Master      : {master.name}")
        print(f"    Frame count : {fc_str}")
        print(f"    Sample rate : {rate_str} Hz")

        channel_ids = _discover_channels(session_dir)
        if not channel_ids:
            print("    No channel frame files found, skipping")
            continue

        # ── Build and write Trial YAML (before stacking) ─────────────────────
        yaml_path = session_dir / f"{session_name}.yaml"
        if yaml_path.exists():
            print(f"    yaml: {yaml_path.name} already exists - updating OME fields only")
            try:
                update_yaml(yaml_path, px)
            except Exception as exc:
                print(f"    yaml: WARNING update failed: {exc}")
        else:
            if not args.dry_run:
                doc = build_yaml(session_name, master, px, len(channel_ids))
                _write_yaml(yaml_path, doc)
                # Report what was populated
                s   = doc.get("acquisition_system", {}).get("settings", {})
                exp = doc.get("experiment", {})
                sub = doc.get("subject", {})
                populated = {
                    k: v for k, v in {
                        "datetime":    exp.get("datetime"),
                        "condition":   exp.get("condition"),
                        "subject_id":  sub.get("id"),
                        "magnif":      (s.get("magnification") or {}).get("value"),
                        "fa":          s.get("fa"),
                        "laser_%":     (s.get("laserPower") or {}).get("value"),
                        "depth_um":    (s.get("depth") or {}).get("value"),
                        "fr_Hz":       (s.get("sample_rate") or {}).get("value"),
                        "frame_size":  s.get("frame_size"),
                        "n_channels":  s.get("n_channels"),
                        "n_frames":    s.get("n_frames"),
                    }.items() if v is not None
                }
                fields_str = "  ".join(f"{k}={v}" for k, v in populated.items())
                print(f"    yaml: wrote {yaml_path.name}  [{fields_str}]")
            else:
                # Dry-run: show what would be parsed
                dn = _parse_session_name(session_name)
                fn = _parse_frame_name(master.name)
                print(f"    yaml: would write {yaml_path.name}")
                print(f"           dirname → {dn}")
                print(f"           frame   → {fn}")

        for ch in channel_ids:
            output = session_dir / f"{session_name}-C{ch}-fc{fc_str}.tif"

            if output.exists():
                print(f"    C{ch}: already exists, skipping")
                continue

            frames = sorted(session_dir.glob(f"*_C{ch}_t*.tif"))
            print(f"    C{ch}: {len(frames)} frames -> {output.name}")

            if args.dry_run:
                continue

            # ── Pad time indices ─────────────────────────────────────────────
            sample_t  = _TINDEX_RE.search(frames[0].name) if frames else None
            cur_width = len(sample_t.group(2)) if sample_t else 4
            needed    = max(6, len(str(max(frame_count - 1, 0))))
            if cur_width < needed:
                print(f"    C{ch}: padding time indices → {needed} digits...")
                n = _pad_time_indices(session_dir, ch, needed)
                print(f"    C{ch}: padded {n} filename(s)")
                frames = sorted(session_dir.glob(f"*_C{ch}_t*.tif"))
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

            if not output.exists() or output.stat().st_size == 0:
                print(f"    C{ch}: ERROR - output missing or empty, sources preserved")
                continue

            if args.delete_sources:
                print(f"    C{ch}: deleting {len(frames)} source frame(s)...")
                for f in frames:
                    try:
                        f.unlink()
                    except OSError as exc:
                        logger.warning(f"      Could not delete {f.name}: {exc}")
                print(f"    C{ch}: deleted")

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

