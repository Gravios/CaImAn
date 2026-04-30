#!/usr/bin/env python3
"""
utilities/batch_subject_list.py
================================
Run batch-subject for each subject in an explicit list. All unrecognised
flags are forwarded verbatim to batch-subject.

Subjects are specified via flags only — there is no positional form,
because mixed ordering with passthrough flags (e.g. ``--prefix VALUE``)
is ambiguous to argparse and would silently swallow flag values into
the subject list. Use:

  -s NAME / --subject NAME   Repeat for each subject. Combinable with -f.
  -f PATH  / --from-file PATH    One subject per line (# comments OK).
                                 Use '-' to read from stdin.
  --root DIR                 Resolve bare names against this directory.

Quick start
-----------
  # Repeated -s flag
  batch-subject-list \\
      -s strohA-sa-000070 -s strohA-sa-000076 -s strohA-sa-000082 \\
      --root /data/source/ms2p/strohA/strohA-sa \\
      --run-mc --estimate-params --run -y \\
      --prefix strohA-sa \\
      --template-json /data/source/ms2p/strohA/strohA-ia/strohA-ia-000000/caiman_params.json

  # From a file
  batch-subject-list -f ./subject_list.txt \\
      --root /data/source/ms2p/strohA/strohA-sa \\
      --run-mc --estimate-params --run -y \\
      --prefix strohA-sa --min-pnr 6 --min-corr 0.45

  # From stdin
  ls -d /data/source/ms2p/strohA/strohA-sa/strohA-sa-000??? \\
      | batch-subject-list -f - --run-mc --estimate-params --run -y --prefix strohA-sa

Flags (batch-subject-list)
--------------------------
  -s NAME, --subject NAME   Add NAME to the subject list. Repeatable.
                            Bare names resolve against --root (or CWD).
  -f PATH, --from-file PATH Read additional subjects from PATH (one per
                            line; '-' for stdin).
  --root DIR                Resolve bare subject names against this directory.
  --stop-on-error           Abort on first failed subject. Default: continue.

Forwarded to batch-subject (examples)
---------------------------------------
  --prefix PREFIX           TL_dir prefix filter (forwarded to batch-sessions).
  --skip-done -y --run-mc --estimate-params --run --template-json PATH
  (see: batch-subject --help for the full list)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── import batch_subject so we can call main() directly (no subprocess) ─────
try:
    from utilities.batch_subject import main as _bsubj_main
except ImportError:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "batch_subject",
        Path(__file__).resolve().parent / "batch_subject.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _bsubj_main = _mod.main


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "All other flags are forwarded verbatim to batch-subject.\n"
            "Run: batch-subject --help   for the full list of forwarded flags."
        ),
    )
    p.add_argument(
        "-s", "--subject", action="append", default=[], metavar="NAME",
        help="Subject directory path or bare name (resolved with --root). "
             "Repeat for multiple subjects.",
    )
    p.add_argument(
        "-f", "--from-file", default=None, metavar="PATH",
        help="Read additional subjects from PATH (one per line). "
             "Use '-' for stdin.",
    )
    p.add_argument(
        "--root", default=None, metavar="DIR",
        help="Resolve bare subject names against this directory.",
    )
    p.add_argument(
        "--stop-on-error", action="store_true",
        help="Abort on first failed subject (default: continue).",
    )
    return p.parse_known_args(argv)


def _read_subject_file(path: str) -> list[str]:
    """One entry per line; '#' comments and blank lines stripped."""
    if path == "-":
        lines = sys.stdin.read().splitlines()
    else:
        lines = Path(path).read_text().splitlines()
    out: list[str] = []
    for line in lines:
        s = line.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _resolve(entry: str, root: Path | None) -> Path:
    """Resolve a subject entry to an absolute Path."""
    p = Path(entry)
    # If it's a bare name (no path separators) and --root given, prepend.
    if root is not None and not p.is_absolute() and len(p.parts) == 1:
        p = root / entry
    return p.resolve()


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)

    # Collect entries from -s/--subject + --from-file
    entries: list[str] = list(args.subject)
    if args.from_file:
        try:
            entries.extend(_read_subject_file(args.from_file))
        except OSError as exc:
            print(f"Error reading --from-file {args.from_file!r}: {exc}",
                  file=sys.stderr)
            return 1

    if not entries:
        print("Error: no subjects given. Pass them with -s/--subject\n"
              "       (repeatable) or --from-file PATH.", file=sys.stderr)
        return 1

    root = Path(args.root).resolve() if args.root else None

    # Resolve and validate
    subjects: list[Path] = []
    bad: list[tuple[str, str]] = []
    for entry in entries:
        try:
            p = _resolve(entry, root)
        except OSError as exc:
            bad.append((entry, str(exc)))
            continue
        if not p.is_dir():
            bad.append((entry, f"not a directory: {p}"))
            continue
        subjects.append(p)

    print("Batch subject-list")
    if root:
        print(f"  Root     : {root}")
    print(f"  Subjects : {len(subjects)}")
    if bad:
        print(f"  Skipped  : {len(bad)} (see below)")
    if passthrough:
        print(f"  Flags    : {' '.join(passthrough)}")
    print()

    for entry, reason in bad:
        print(f"  SKIP {entry}  ({reason})")
    if bad:
        print()

    if not subjects:
        print("No valid subjects to process.")
        return 1 if bad else 0

    failed: list[str] = []
    for i, subject in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {subject.name}")
        print("#" * 70)

        bsubj_argv = [str(subject)] + passthrough
        try:
            rc = _bsubj_main(bsubj_argv)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            rc = 1

        if rc != 0:
            failed.append(subject.name)
            print(f"  SUBJECT FAILED (exit {rc})\n")
            if args.stop_on_error:
                print("--stop-on-error set, aborting subject-list.")
                break
        else:
            print()

    print("#" * 70)
    succeeded = len(subjects) - len(failed)
    print(f"Done: {succeeded}/{len(subjects)} subjects succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
