#!/usr/bin/env python3
"""
utilities/rename_stem.py
========================
Replace a named field in directory/file stems, applied recursively.

Stem schema
-----------
  <sourceId>-<userId>-<subjectId>-<date>-<trialId>_<time>-<magnification>-<experimentId>
  e.g.  strohA-ia-000000-20151016-TL001_143915-25x-spont

Target modes
------------
  File      — renames just that file.
  Directory — renames contents and the directory itself (if it matches the schema).
  Parent    — renames all matching subdirectories and their contents.
              The parent itself is not renamed.
  Default   — current working directory (no target argument needed).

Quick start
-----------
  cd /data/source/strohA/.../strohA-ia-000000-20151016
  rename-stem --field magnification --value 40x            # dry run
  rename-stem --field magnification --value 40x --apply
  rename-stem --field subjectId --value STO0069 --apply
  rename-stem path/to/single_file.tif --field userId --value ej --apply

Flags
-----
  target                File, directory, or parent to process.
                        Default: current working directory.
  --field FIELD         Stem field to replace. Required.
                        One of: sourceId, userId, subjectId, date,
                                trialId, time, magnification, experimentId
  --value VALUE         New value for the field. Required.
  --apply               Actually perform the rename.
                        Default: dry run (preview only).
  --prefix PREFIX       Only rename entries whose name starts with PREFIX.
  -h / --help           Show this help message.

Fields
------
  sourceId      Lab/source identifier, e.g. strohA
  userId        Experimenter ID, e.g. ia
  subjectId     Animal/subject ID, e.g. 000000 or STO0069
  date          Acquisition date, e.g. 20151016
  trialId       Trial label, e.g. TL001
  time          Acquisition time, e.g. 143915
  magnification Objective magnification, e.g. 25x or 40x
  experimentId  Experiment/paradigm label, e.g. spont or vstim
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Stem parser / rebuilder
# ---------------------------------------------------------------------------

# Full stem regex — all seven dash-separated tokens.
# trialId and time are joined by an underscore within the 5th token.
#
#   strohA  -  ia  -  000000  -  20151016  -  TL001_143915  -  25x  -  spont
#     1          2       3           4              5             6       7
_STEM_RE = re.compile(
    r'^(?P<sourceId>[^-]+)'
    r'-(?P<userId>[^-]+)'
    r'-(?P<subjectId>[^-]+)'
    r'-(?P<date>\d{8})'
    r'-(?P<trialId>[^_]+)_(?P<time>\d+)'
    r'-(?P<magnification>[^-]+)'
    r'-(?P<experimentId>.+)$'
)

FIELDS = [
    "sourceId", "userId", "subjectId", "date",
    "trialId", "time", "magnification", "experimentId",
]


def parse_stem(stem: str) -> dict | None:
    m = _STEM_RE.match(stem)
    return m.groupdict() if m else None


def build_stem(parts: dict) -> str:
    return (
        f"{parts['sourceId']}"
        f"-{parts['userId']}"
        f"-{parts['subjectId']}"
        f"-{parts['date']}"
        f"-{parts['trialId']}_{parts['time']}"
        f"-{parts['magnification']}"
        f"-{parts['experimentId']}"
    )


def replace_in_name(name: str, field: str, new_value: str) -> str | None:
    """Return the new name with *field* replaced, or None if stem not matched."""
    # Split off any suffix (e.g. _pipeline.json, _C00.tif, .yaml)
    # The stem is the leading part that matches the schema; everything after
    # the first non-schema character is the suffix.
    # Strategy: try successively shorter prefixes until one parses.
    # For plain directory names the whole name is the stem.
    for cut in range(len(name), 0, -1):
        candidate_stem = name[:cut]
        parts = parse_stem(candidate_stem)
        if parts is not None:
            suffix = name[cut:]
            parts[field] = new_value
            return build_stem(parts) + suffix
    return None


# ---------------------------------------------------------------------------
# Recursive rename
# ---------------------------------------------------------------------------

def collect_renames(
    target: Path,
    field: str,
    new_value: str,
    prefix: str | None,
    *,
    rename_root: bool = False,
) -> list[tuple[Path, Path]]:
    """
    Collect (old_path, new_path) pairs for *target*, which may be:

    - A **file** — rename just that file.
    - A **directory** — recurse into contents depth-first post-order, and
      rename the directory itself when *rename_root* is True.
    """
    renames: list[tuple[Path, Path]] = []
    if target.is_file():
        _maybe_rename(target, field, new_value, prefix, renames)
    else:
        _collect(target, target, field, new_value, prefix, renames,
                 rename_root=rename_root)
    return renames


def _collect(
    current: Path,
    root: Path,
    field: str,
    new_value: str,
    prefix: str | None,
    out: list,
    *,
    rename_root: bool,
) -> Path:
    """Recurse into *current*, return its (possibly updated) path."""
    if current.is_dir():
        for child in sorted(current.iterdir()):
            _collect(child, root, field, new_value, prefix, out,
                     rename_root=True)
        # Rename this directory after all children have been processed.
        # Skip if it is the root and rename_root is False.
        if rename_root or current != root:
            new_name = _maybe_rename(current, field, new_value, prefix, out)
            if new_name:
                return current.parent / new_name
    else:
        _maybe_rename(current, field, new_value, prefix, out)
    return current


def _maybe_rename(
    path: Path,
    field: str,
    new_value: str,
    prefix: str | None,
    out: list,
) -> str | None:
    name = path.name
    if prefix and not name.startswith(prefix):
        return None
    new_name = replace_in_name(name, field, new_value)
    if new_name and new_name != name:
        out.append((path, path.parent / new_name))
        return new_name
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Fields:\n  sourceId  userId  subjectId  date  trialId  time  magnification  experimentId\n\nExamples:\n  rename-stem --field magnification --value 40x\n  rename-stem --field magnification --value 40x --apply\n  rename-stem --field subjectId --value STO0069 --apply\n  rename-stem path/to/file.tif --field userId --value ej --apply',
    )
    p.add_argument(
        "target", nargs="?", default=None,
        help="File, session directory, or parent directory to process "
             "(default: current working directory)",
    )
    p.add_argument(
        "--field", required=True, choices=FIELDS, metavar="FIELD",
        help=f"Stem field to replace. One of: {', '.join(FIELDS)}",
    )
    p.add_argument("--value", required=True, help="New value for the field")
    p.add_argument(
        "--apply", action="store_true",
        help="Perform the rename (default: dry run)",
    )
    p.add_argument(
        "--prefix", default=None,
        help="Only rename entries whose name starts with this prefix",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    target = Path(args.target).resolve() if args.target else Path.cwd()
    if not target.exists():
        print(f"Error: target does not exist: {target}", file=sys.stderr)
        return 1

    # For a single directory that itself matches the schema, rename it too.
    # For a parent directory containing session dirs, don't rename the parent.
    rename_self = target.is_dir() and parse_stem(target.name) is not None

    renames: list[tuple[Path, Path]] = []
    if target.is_file():
        _maybe_rename(target, args.field, args.value, args.prefix, renames)
    else:
        _collect(target, target, args.field, args.value, args.prefix, renames,
                 rename_root=rename_self)

    if not renames:
        print("No matching stems found.")
        return 0

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}]  field={args.field}  value={args.value}  target={target}")
    print(f"  {len(renames)} rename(s):\n")

    display_root = target.parent
    for old, new in renames:
        try:
            rel_old = old.relative_to(display_root)
            rel_new = new.relative_to(display_root)
        except ValueError:
            rel_old, rel_new = old, new
        print(f"  {rel_old}")
        print(f"  → {rel_new}\n")

    if not args.apply:
        print("Dry run — pass --apply to execute.")
        return 0

    errors = 0
    for old, new in renames:
        if not old.exists() and new.exists():
            continue  # already renamed by a parent
        if not old.exists():
            print(f"  WARNING: {old.name} not found, skipping")
            errors += 1
            continue
        try:
            old.rename(new)
        except OSError as exc:
            print(f"  ERROR renaming {old.name}: {exc}")
            errors += 1

    done = len(renames) - errors
    print(f"Done: {done}/{len(renames)} renamed.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
