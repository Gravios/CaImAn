#!/usr/bin/env python3
"""
utilities/batch_sessions.py
============================
Run new-session for every session subdirectory under a parent directory,
passing through all new-session flags unchanged.

Usage
-----
Run from the date directory (--parent defaults to CWD)::

    cd /data/source/strohA/strohA-ia/strohA-ia-000000/strohA-ia-000000-20151016
    batch-sessions --prefix strohA-ia --run-mc --estimate-params -y

Explicit parent with pipeline execution::

    batch-sessions \\
        --parent /data/source/strohA/.../strohA-ia-000000-20151016 \\
        --prefix strohA-ia \\
        --run-mc --estimate-params --run -y

Any flag not listed below is forwarded verbatim to new-session.

Batch-only options
------------------
--parent      Parent directory containing session subdirectories
              (default: current working directory)
--prefix      Only process subdirectories whose name starts with this prefix
              (default: all subdirectories)
--skip-done   Skip sessions that already have a _pipeline.json
--stop-on-error
              Abort the batch on the first session that fails
              (default: log the error and continue)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── import new_session so we can call main() directly (no subprocess) ─────────
try:
    from utilities.new_session import main as _new_session_main
except ImportError:
    # Fallback when running the script directly without package install
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "new_session",
        Path(__file__).resolve().parent / "new_session.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _new_session_main = _mod.main


# Flags that belong to batch_sessions and must NOT be forwarded to new_session
_BATCH_FLAGS = {"--parent", "--prefix", "--skip-done", "--stop-on-error"}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # allow_abbrev=False prevents --prefix matching --preallocate etc.
        allow_abbrev=False,
    )
    p.add_argument(
        "--parent", default=None, metavar="DIR",
        help="Parent directory (default: CWD)",
    )
    p.add_argument(
        "--prefix", default=None, metavar="PREFIX",
        help="Only process subdirs whose name starts with this prefix",
    )
    p.add_argument(
        "--skip-done", action="store_true",
        help="Skip sessions that already have a _pipeline.json",
    )
    p.add_argument(
        "--stop-on-error", action="store_true",
        help="Abort on first failure (default: continue)",
    )
    # parse_known_args passes everything else through to new_session
    return p.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)

    parent = Path(args.parent).resolve() if args.parent else Path.cwd()
    if not parent.is_dir():
        print(f"Error: --parent is not a directory: {parent}", file=sys.stderr)
        return 1

    session_dirs = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and (args.prefix is None or d.name.startswith(args.prefix))
    )

    if not session_dirs:
        print(f"No session directories found in {parent}"
              + (f" matching prefix '{args.prefix}'" if args.prefix else ""))
        return 0

    print(f"Batch session setup")
    print(f"  Parent  : {parent}")
    print(f"  Sessions: {len(session_dirs)}")
    if passthrough:
        print(f"  Flags   : {' '.join(passthrough)}")
    print()

    failed = []

    for i, session_dir in enumerate(session_dirs, 1):
        session_name = session_dir.name
        print(f"[{i}/{len(session_dirs)}] {session_name}")
        print("-" * 70)

        # --skip-done: check for any existing _pipeline.json
        if args.skip_done:
            existing = list(session_dir.glob("*_pipeline.json"))
            if existing:
                print(f"  Skipping — pipeline JSON already exists: {existing[0].name}\n")
                continue

        # Build argv for new_session:
        #   new_session takes an optional positional `dest` and infers session
        #   from the TIF in that directory.
        ns_argv = [str(session_dir)] + passthrough

        try:
            rc = _new_session_main(ns_argv)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            rc = 1

        if rc != 0:
            failed.append(session_name)
            print(f"  FAILED (exit {rc})\n")
            if args.stop_on_error:
                print("--stop-on-error set, aborting batch.")
                break
        else:
            print()

    # Summary
    print("=" * 70)
    succeeded = len(session_dirs) - len(failed)
    print(f"Done: {succeeded}/{len(session_dirs)} sessions succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  {name}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
