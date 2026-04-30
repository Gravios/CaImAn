#!/usr/bin/env python3
"""
Organize Imspector .msr files into the lab's canonical tradition tree.

Source layout (input):
    <src_root>/<cond_dir>/<subjectId>.msr     e.g.  ./c1/70.msr, ./d3/70.msr
    cond_dir starts with c|d|r followed by digits and is preserved verbatim
    (e.g. 'c1' stays 'c1', 'd3' stays 'd3', 'r1' stays 'r1').

Canonical layout (output, under --dest):
    <dest>/<lab>/<prefix>/<prefix>-<sid6>/<prefix>-<sid6>-<YYYYMMDD>/<TL_dir>/<TL_dir>.msr
    where TL_dir = <prefix>-<sid6>-<YYYYMMDD>-TL<NNN>_<HHMMSS>-<MAG>x-<cond>

TL allocation:
    Per (subject, date), files are sorted by acquisition HHMMSS and assigned
    TL001, TL002, ... in order. Same subject on different days -> both TL001.

Optional mirror (--mirror-dir):
    Recreates the source layout (<cond_dir>/<orig>.msr) as symlinks pointing
    into the canonical tree. Useful for keeping the c1/d1/r1 view intact.

Usage:
    # 1) discover metadata keys
    python organize_msr.py strohA-sa ./ --inspect

    # 2) dry run (prints full plan with TL allocation)
    python organize_msr.py strohA-sa ./ --dest /data/source/ms2p --dry-run

    # 3) commit + build symlink mirror
    python organize_msr.py strohA-sa ./ \
        --dest /data/source/ms2p \
        --mirror-dir /data/projects/seventh
"""
import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# --- adjust this import to wherever your reader lives -----------------------
sys.path.insert(0, str(Path(__file__).parent))
from imspectorreader import IMSpectorReader  # noqa: E402
# ---------------------------------------------------------------------------

SUBJECT_RX = re.compile(r"^(\d+)\.msr$", re.IGNORECASE)
COND_RX    = re.compile(r"^[cdr]\d+$", re.IGNORECASE)

# Candidate metadata keys — refine with --inspect once on a real file.
DATE_KEYS = ("Creation Date", "AcquisitionDate", "acquisition_date",
             "DateTime", "datetime", "date", "Date",
             "CreatedOn", "created", "StartTime")
TIME_KEYS = ("AcquisitionTime", "acquisition_time", "Time", "StartTime")
MAG_KEYS  = ("Magnification", "ObjectiveMagnification", "Zoom",
             "objective_mag", "objective_magnification")
DATE_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y%m%dT%H%M%S",     "%Y%m%d_%H%M%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",          "%Y%m%d",
                "%d.%m.%Y %H:%M:%S", "%d.%m.%Y")


@dataclass
class Entry:
    src: Path
    sid6: str
    cond: str           # 'c1', 'd3', 'r1'
    yyyymmdd: str
    hhmmss: str
    mag: str
    mirror_rel: Path
    tl: str = ""        # 'TL001' etc., assigned in pass 2


def load_metadata(path: Path) -> dict:
    r = IMSpectorReader(str(path))
    for attr in ("metadata", "meta", "info", "header"):
        m = getattr(r, attr, None)
        if isinstance(m, dict):
            return m
    raise RuntimeError(f"No metadata dict on IMSpectorReader for {path.name}")


def parse_dt(meta: dict) -> tuple[str, str]:
    """Return (YYYYMMDD, HHMMSS) from any parseable datetime field."""
    for k in DATE_KEYS + TIME_KEYS:
        if k not in meta:
            continue
        raw = str(meta[k]).strip().split(".")[0]  # drop fractional seconds
        for fmt in DATE_FORMATS:
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.strftime("%Y%m%d"), dt.strftime("%H%M%S")
            except ValueError:
                continue
    raise ValueError(f"No parseable datetime field (tried {DATE_KEYS + TIME_KEYS})")


def parse_mag(meta: dict) -> str:
    for k in MAG_KEYS:
        if k in meta:
            v = str(meta[k]).rstrip("xX").strip()
            try:
                return str(int(round(float(v))))
            except ValueError:
                continue
    raise ValueError(f"No magnification field (tried {MAG_KEYS})")


def plan(src_root: Path) -> tuple[list[Entry], list[tuple[Path, str]]]:
    """Pass 1: read metadata for every file and assign TL numbers.
    Returns (planned_entries, skipped_with_reason)."""
    planned: list[Entry] = []
    skipped: list[tuple[Path, str]] = []

    for src in sorted(src_root.rglob("*.msr")):
        sm = SUBJECT_RX.match(src.name)
        if not sm:
            skipped.append((src, "filename not <subjectId>.msr"))
            continue
        if not COND_RX.match(src.parent.name):
            skipped.append((src, f"parent {src.parent.name!r} not c|d|r + digits"))
            continue
        try:
            meta = load_metadata(src)
            yyyymmdd, hhmmss = parse_dt(meta)
            mag = parse_mag(meta)
        except Exception as e:
            skipped.append((src, f"metadata: {e}"))
            continue

        try:
            mirror_rel = src.relative_to(src_root)
        except ValueError:
            mirror_rel = Path(src.parent.name) / src.name

        planned.append(Entry(
            src=src,
            sid6=f"{int(sm.group(1)):06d}",
            cond=src.parent.name.lower(),
            yyyymmdd=yyyymmdd,
            hhmmss=hhmmss,
            mag=mag,
            mirror_rel=mirror_rel,
        ))

    # Allocate TL per (subject, date) by acquisition time
    groups: dict[tuple[str, str], list[Entry]] = defaultdict(list)
    for e in planned:
        groups[(e.sid6, e.yyyymmdd)].append(e)
    for entries in groups.values():
        entries.sort(key=lambda e: e.hhmmss)
        for i, e in enumerate(entries, start=1):
            e.tl = f"TL{i:03d}"

    return planned, skipped


def dest_for(e: Entry, dest_root: Path, prefix: str) -> Path:
    lab      = prefix.split("-")[0]
    subj_dir = f"{prefix}-{e.sid6}"
    date_dir = f"{prefix}-{e.sid6}-{e.yyyymmdd}"
    tl_dir   = f"{prefix}-{e.sid6}-{e.yyyymmdd}-{e.tl}_{e.hhmmss}-{e.mag}x-{e.cond}"
    return dest_root / lab / prefix / subj_dir / date_dir / tl_dir / f"{tl_dir}.msr"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="e.g. strohA-sa")
    ap.add_argument("src", type=Path, help="root containing c1/, d3/, r1/ ...")
    ap.add_argument("--dest", type=Path, default=Path("/data/source/ms2p"))
    ap.add_argument("--mirror-dir", type=Path, default=None,
                    help="If set, create symlinks here matching the source layout")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="Print metadata for the first .msr found, then exit")
    args = ap.parse_args()

    if args.inspect:
        first = next(iter(sorted(args.src.rglob("*.msr"))), None)
        if first is None:
            print("No .msr files found.", file=sys.stderr)
            return 1
        meta = load_metadata(first)
        print(f"# {first}")
        for k, v in meta.items():
            print(f"{k!r}: {v!r}")
        return 0

    entries, skipped = plan(args.src)
    if not entries and not skipped:
        print("No .msr files found.", file=sys.stderr)
        return 1

    for src, why in skipped:
        print(f"SKIP  {src}  ({why})")

    rc = 0
    moved = 0
    for e in entries:
        dst = dest_for(e, args.dest, args.prefix)
        if dst.exists():
            print(f"SKIP  {e.src}  (target exists: {dst})")
            continue
        tag = "DRY " if args.dry_run else "MOVE"
        print(f"{tag}  {e.src} -> {dst}")
        if args.mirror_dir:
            link = args.mirror_dir / e.mirror_rel
            print(f"      link {link} -> {dst}")
        if not args.dry_run:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                e.src.replace(dst)
                if args.mirror_dir:
                    link = args.mirror_dir / e.mirror_rel
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(dst)
                moved += 1
            except Exception as ex:
                print(f"FAIL  {e.src}  {ex}")
                rc = 2

    suffix = "pending" if args.dry_run else "moved"
    count  = len(entries) if args.dry_run else moved
    print(f"\n{count} {suffix}, {len(skipped)} skipped")
    return rc


if __name__ == "__main__":
    sys.exit(main())
