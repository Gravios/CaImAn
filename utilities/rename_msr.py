#!/usr/bin/env python3
"""
Rename Imspector .msr files using metadata.

Input:  <subjectId>.msr            e.g.  2966.msr
Output: <prefix>-<sid6>-<YYYYMMDD>_TL001.msr   e.g.  stroh-sa-002966-20251222_TL001.msr

Usage:
    python rename_msr.py <prefix> <file_or_dir> [...]              # rename in place
    python rename_msr.py --dry-run <prefix> <file_or_dir> [...]    # show planned moves
    python rename_msr.py --inspect <prefix> <file>                 # dump metadata keys
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# --- adjust this import to wherever your reader lives -----------------------
sys.path.insert(0, str(Path(__file__).parent))
from imspectorreader import ImspectorReader  # noqa: E402
# ---------------------------------------------------------------------------

SUBJECT_RX = re.compile(r"^(\d+)\.msr$", re.IGNORECASE)

# Candidate metadata keys for the acquisition timestamp. Adjust/extend as
# needed once you've inspected one file with `--inspect`.
DATE_KEYS = (
    "AcquisitionDate", "acquisition_date", "DateTime", "datetime",
    "date", "Date", "CreatedOn", "created", "StartTime",
)
DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    "%Y%m%dT%H%M%S",     "%Y%m%d_%H%M%S",
    "%Y-%m-%d",          "%Y%m%d",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y",
)


def load_metadata(path: Path) -> dict:
    """Return a flat dict of metadata for an MSR file."""
    r = ImspectorReader(str(path))
    # Common shapes — pick whichever your reader actually uses.
    for attr in ("metadata", "meta", "info", "header"):
        m = getattr(r, attr, None)
        if isinstance(m, dict):
            return m
    raise RuntimeError(f"Could not locate metadata dict on ImspectorReader for {path.name}")


def parse_date(meta: dict) -> str:
    """Return YYYYMMDD by walking the metadata for a known date field."""
    for k in DATE_KEYS:
        if k not in meta:
            continue
        raw = str(meta[k]).strip().split(".")[0]  # drop fractional seconds
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(raw, fmt).strftime("%Y%m%d")
            except ValueError:
                continue
    raise ValueError(f"No parseable date field found (tried {DATE_KEYS})")


def build_name(subject_id: str, yyyymmdd: str, prefix: str) -> str:
    return f"{prefix}-{int(subject_id):06d}-{yyyymmdd}_TL001.msr"


def collect(paths) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.msr")))
        elif p.suffix.lower() == ".msr":
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="Name prefix, e.g. 'stroh-sa'")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="Print metadata for the first file and exit")
    args = ap.parse_args()

    files = collect(args.paths)
    if not files:
        print("No .msr files found.", file=sys.stderr)
        return 1

    if args.inspect:
        meta = load_metadata(files[0])
        print(f"# {files[0].name}")
        for k, v in meta.items():
            print(f"{k!r}: {v!r}")
        return 0

    rc = 0
    for src in files:
        m = SUBJECT_RX.match(src.name)
        if not m:
            print(f"SKIP   {src.name}  (not <subjectId>.msr)")
            continue
        sid = m.group(1)
        try:
            date_str = parse_date(load_metadata(src))
        except Exception as e:
            print(f"FAIL   {src.name}  {e}")
            rc = 2
            continue
        dst = src.with_name(build_name(sid, date_str, args.prefix))
        if dst.exists():
            print(f"SKIP   {src.name} -> {dst.name}  (target exists)")
            continue
        tag = "DRY  " if args.dry_run else "MOVE "
        print(f"{tag} {src.name} -> {dst.name}")
        if not args.dry_run:
            src.rename(dst)
    return rc


if __name__ == "__main__":
    sys.exit(main())
