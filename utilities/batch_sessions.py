#!/usr/bin/env python3
"""
utilities/batch_sessions.py
============================
Run new-session for every session subdirectory under a parent directory.
All unrecognised flags are forwarded verbatim to new-session.

Quick start
-----------
  cd /data/source/strohA/.../strohA-ia-000000-20151016
  batch-sessions --prefix strohA-ia --run-mc --estimate-params -y
  batch-sessions --prefix strohA-ia --run -y
  batch-sessions --prefix strohA-ia --run-mc --estimate-params --run -y

Flags (batch-sessions)
----------------------
  --parent DIR          Parent directory containing session subdirs.
                        Default: current working directory.
  --prefix PREFIX       Only process subdirs whose name starts with PREFIX.
                        Default: all subdirectories.
  --skip-done           Skip sessions that already have a _pipeline.json.
  --stop-on-error       Abort on first failure. Default: log and continue.

Forwarded to new-session (examples)
-------------------------------------
  -y / --force          Overwrite existing pipeline files without prompting.
  --run-mc              Run GPU motion correction before param estimation.
  --estimate-params     Estimate CNMF parameters from the MC'd movie.
  --run                 Run the CaImAn pipeline after setup.
  --template-json PATH  Use a custom template_pipeline.json.
  --gSig PX             Gaussian half-width in pixels.
  --rf PX               Patch half-size in pixels.
  --fr HZ               Acquisition frame rate.
  --decay-time S        GCaMP decay time constant in seconds.
  --species mouse|rat   Animal species (constrains gSig search range).
  --dry-run             Preview only — no files written.
  (see: new-session --help for the full list)
"""

from __future__ import annotations

import argparse
import re
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
        epilog=(
            "Batch-only flags:\n"
            "  --parent DIR          Parent directory (default: CWD)\n"
            "  --prefix PREFIX       Only process subdirs starting with PREFIX\n"
            "  --skip-done           Skip sessions that already have a _pipeline.json\n"
            "  --stop-on-error       Abort on first failure (default: continue)\n\n"
            "All other flags are forwarded verbatim to new-session.\n"
            "Run: new-session --help   for the full list of forwarded flags."
        ),
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

    # Collect all session candidates across all matching TL dirs.
    # Layouts:
    #   <parent>/
    #     <TL_dir>/                     <- matches --prefix
    #       <TL_dir>-C00-fc<N>/         <- channel subdir (TIF or MSR post-branch)
    #         <TL_dir>-C00-fc<N>.tif    or .msr
    #       <TL_dir>.msr                <- organize_msr.py output, pre-MSR-branch
    ch_pattern = re.compile(r'.+-C\d{2}-fc\d+$')

    tl_dirs = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and (args.prefix is None or d.name.startswith(args.prefix))
    )

    # Build list of (session_stem, dest_dir) pairs.
    #
    # Two layouts are recognised per TL_dir:
    #   A. Channel subdir(s) already exist (TIF case, or MSR after first
    #      new_session run): each <TL_dir>-C<NN>-fc<NNNNNN>/ becomes a
    #      session whose dest is the channel subdir itself.
    #   B. organize_msr.py output, pre-MSR-branch: <TL_dir>/<TL_dir>.msr
    #      sits at the TL_dir level and no channel subdir exists yet. The
    #      TL_dir itself is passed as dest; new_session's MSR branch will
    #      build the channel subdir on first call.
    sessions: list[tuple[str, Path]] = []
    for tl_dir in tl_dirs:
        ch_dirs = sorted(
            d for d in tl_dir.iterdir()
            if d.is_dir() and ch_pattern.match(d.name)
        )
        if ch_dirs:
            for ch_dir in ch_dirs:
                sessions.append((ch_dir.name, ch_dir))
        else:
            msr_at_tl = tl_dir / f"{tl_dir.name}.msr"
            if msr_at_tl.exists():
                sessions.append((tl_dir.name, tl_dir))

    if not sessions:
        print(f"No channel subdirectories found under {parent}"
              + (f" matching prefix '{args.prefix}'" if args.prefix else ""))
        return 0

    print(f"Batch session setup")
    print(f"  Parent  : {parent}")
    print(f"  Sessions: {len(sessions)}")
    if passthrough:
        print(f"  Flags   : {' '.join(passthrough)}")
    print()

    failed = []

    for i, (session_name, session_dir) in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] {session_name}")
        print("-" * 70)

        # --skip-done: check for any existing _pipeline.json in the channel subdir
        if args.skip_done:
            existing = list(session_dir.glob("*_pipeline.json"))
            if existing:
                print(f"  Skipping — pipeline JSON already exists: {existing[0].name}\n")
                continue

        # Pass session stem + dest path as positional args to new_session
        ns_argv = [session_name, str(session_dir)] + passthrough

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
    succeeded = len(sessions) - len(failed)
    print(f"Done: {succeeded}/{len(sessions)} sessions succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  {name}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
