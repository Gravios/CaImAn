#!/usr/bin/env python3
"""
utilities/batch_subject.py
==========================
Run batch-sessions for every date subdirectory under a subject directory.
All unrecognised flags are forwarded verbatim to batch-sessions.

Quick start
-----------
  cd /data/source/ms2p/strohA/strohA-sa/strohA-sa-000070
  batch-subject --run-mc --estimate-params --run -y --prefix strohA-sa

Flags (batch-subject)
---------------------
  subject (positional)  Subject directory containing date subdirs.
                        Default: current working directory.
  --subject DIR         Same as positional; explicit form. Overrides positional.
  --stop-on-error       Abort on first failed date. Default: log and continue.

Forwarded to batch-sessions (examples)
---------------------------------------
  --prefix PREFIX       TL_dir prefix filter inside each date.
  --skip-done           Skip sessions with a _pipeline.json.
  -y / --force          Overwrite existing pipeline files.
  --run-mc / --estimate-params / --run  Pipeline orchestration.
  --template-json PATH  Custom template_pipeline.json.
  (see: batch-sessions --help for the full list)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ── import batch_sessions so we can call main() directly (no subprocess) ─────
try:
    from utilities.batch_sessions import main as _bs_main
except ImportError:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "batch_sessions",
        Path(__file__).resolve().parent / "batch_sessions.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _bs_main = _mod.main


# Date dir naming convention: <subject>-<YYYYMMDD>
_DATE_RX = re.compile(r"-(\d{8})$")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "All other flags are forwarded verbatim to batch-sessions.\n"
            "Run: batch-sessions --help   for the full list of forwarded flags."
        ),
    )
    p.add_argument(
        "subject_pos", nargs="?", default=None, metavar="subject",
        help="Subject directory (default: CWD). Same as --subject.",
    )
    p.add_argument(
        "--subject", default=None, metavar="DIR",
        help="Subject directory (default: CWD). Overrides positional.",
    )
    p.add_argument(
        "--stop-on-error", action="store_true",
        help="Abort on first failed date (default: continue).",
    )
    return p.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)

    subj_arg = args.subject if args.subject is not None else args.subject_pos
    subject  = Path(subj_arg).resolve() if subj_arg else Path.cwd()
    if not subject.is_dir():
        print(f"Error: subject is not a directory: {subject}", file=sys.stderr)
        return 1

    date_dirs = sorted(
        d for d in subject.iterdir()
        if d.is_dir() and _DATE_RX.search(d.name)
    )
    if not date_dirs:
        print(f"No date subdirectories (matching *-YYYYMMDD) under {subject}")
        return 0

    print("Batch subject")
    print(f"  Subject : {subject}")
    print(f"  Dates   : {len(date_dirs)}")
    if passthrough:
        print(f"  Flags   : {' '.join(passthrough)}")
    print()

    failed: list[str] = []
    for i, date_dir in enumerate(date_dirs, 1):
        print(f"[{i}/{len(date_dirs)}] {date_dir.name}")
        print("=" * 70)

        bs_argv = [str(date_dir)] + passthrough
        try:
            rc = _bs_main(bs_argv)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            rc = 1

        if rc != 0:
            failed.append(date_dir.name)
            print(f"  DATE FAILED (exit {rc})\n")
            if args.stop_on_error:
                print("--stop-on-error set, aborting subject.")
                break
        else:
            print()

    print("=" * 70)
    succeeded = len(date_dirs) - len(failed)
    print(f"Done: {succeeded}/{len(date_dirs)} dates succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
