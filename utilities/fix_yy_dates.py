#!/usr/bin/env python3
"""
Fix YY -> YYYY in canonical MSR tree (one-shot cleanup).

Walks under <root>, reads each .msr's acquisition date, and rewrites the
4th hyphen-separated field (the date) in:
  - the .msr filename
  - its parent (TL_dir)
  - its grandparent (date_dir)
from YYMMDD to YYYYMMDD.

Bottom-up. Idempotent. Default is dry-run; pass --apply to actually rename.

Usage:
    python3 fix_yy_dates.py /data/source/ms2p/strohA/strohA-sa
    python3 fix_yy_dates.py /data/source/ms2p/strohA/strohA-sa --apply
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from imspectorreader import IMSpectorReader  # noqa: E402

DATE_KEYS = ("Creation Date", "AcquisitionDate", "acquisition_date",
             "DateTime", "datetime", "date", "Date",
             "CreatedOn", "created", "StartTime", "Time")
FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
           "%Y%m%dT%H%M%S",     "%Y%m%d_%H%M%S",
           "%Y-%m-%d %H:%M",
           "%Y-%m-%d",          "%Y%m%d",
           "%d.%m.%Y %H:%M:%S", "%d.%m.%Y",
           "%a %b %d %H:%M:%S %Y",   # ctime-like, common in scope headers
           "%Y-%m-%dT%H:%M:%S%z")
DATE_FIELD = re.compile(r"^\d{6}$|^\d{8}$")


def parse_yyyymmdd(p: Path) -> str:
    r = IMSpectorReader(str(p))
    # Try direct attribute first (some forks store .date), then metadata dict
    candidates: list[str] = []
    for attr in ("date", "acquisition_date"):
        v = getattr(r, attr, None)
        if v:
            candidates.append(str(v))
    md = getattr(r, "metadata", None)
    if isinstance(md, dict):
        for k in DATE_KEYS:
            if k in md:
                candidates.append(str(md[k]))
    for raw in candidates:
        raw = raw.strip().split(".")[0]
        for f in FORMATS:
            try:
                return datetime.strptime(raw, f).strftime("%Y%m%d")
            except ValueError:
                continue
    raise ValueError(f"no parseable date (tried {len(candidates)} fields)")


def fix_date_field(name: str, idx: int, yyyymmdd: str) -> str:
    """Replace the idx-th hyphen-separated field with yyyymmdd. Idempotent."""
    stem, dot, ext = name.partition(".")
    parts = stem.split("-")
    if len(parts) <= idx or not DATE_FIELD.fullmatch(parts[idx]):
        return name
    parts[idx] = yyyymmdd
    return "-".join(parts) + dot + ext


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="Actually rename (default: dry-run)")
    ap.add_argument("--inspect", action="store_true",
                    help="Dump metadata + date attrs from first .msr and exit")
    ap.add_argument("--date-field-index", type=int, default=3,
                    help="Hyphen-separated index of the date field (default: 3, "
                         "matching <lab>-<exp>-<subj>-<date>-...)")
    args = ap.parse_args()

    if args.inspect:
        first = next(iter(sorted(args.root.rglob("*.msr"))), None)
        if first is None:
            print("# no .msr files found", file=sys.stderr)
            return 1
        print(f"# {first}")
        r = IMSpectorReader(str(first))
        for attr in ("date", "acquisition_date", "filename"):
            if hasattr(r, attr):
                print(f"r.{attr} = {getattr(r, attr)!r}")
        md = getattr(r, "metadata", None)
        if isinstance(md, dict):
            print("# metadata dict:")
            for k, v in md.items():
                print(f"  {k!r}: {v!r}")
        return 0

    plan: dict[Path, Path] = {}
    failed: list[Path] = []

    for msr in sorted(args.root.rglob("*.msr")):
        try:
            yyyy = parse_yyyymmdd(msr)
        except Exception as e:
            print(f"# skip {msr}: {e}", file=sys.stderr)
            failed.append(msr)
            continue
        for old in (msr, msr.parent, msr.parent.parent):
            new_name = fix_date_field(old.name, args.date_field_index, yyyy)
            if new_name == old.name:
                continue
            target = old.with_name(new_name)
            existing = plan.get(old)
            if existing is not None and existing != target:
                print(f"# WARNING conflict on {old}: {existing.name} vs {new_name}",
                      file=sys.stderr)
            plan[old] = target

    if not plan:
        print("# nothing to rename")
        return 0 if not failed else 2

    # Bottom-up: deepest paths first
    for old in sorted(plan, key=lambda p: -len(p.parts)):
        new = plan[old]
        tag = "MV " if args.apply else "DRY"
        print(f"{tag} {old} -> {new}")
        if args.apply:
            old.rename(new)

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
