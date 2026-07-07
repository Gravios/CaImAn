#!/usr/bin/env python3
"""
utilities/sync_pipeline_templates.py
====================================
Propagate the current pipeline templates to existing per-session pipeline
files. Recursively finds every ``*_pipeline.py`` and ``*_pipeline.json``
under --root and replaces its contents with the matching template in
``utilities/pipelines/template_pipeline.{py,json}``, keeping each file's
session-specific name.

Both templates are session-agnostic — the .py derives the session name from
its own filename (resolve_pipeline_path) and the .json carries no session_id —
so a verbatim copy under a renamed file is correct.

Safety
------
This OVERWRITES existing session pipeline files. If a session's _pipeline.json
was tuned by --estimate-params (or hand-edited), those values are replaced by
the template defaults; use --py-only to leave the JSON side alone. Runs as a
DRY-RUN by default (prints planned changes); pass --apply to write. A .bak
copy of each replaced file is kept unless --no-backup. Files already identical
to the template are skipped, and the templates themselves are never touched.

Quick start
-----------
  # preview from the data root
  sync-pipeline-templates --root /data/source/ms2p/strohA

  # apply, keeping .bak backups
  sync-pipeline-templates --root /data/source/ms2p/strohA --apply

  # only the .py side, filtered to one subject prefix
  sync-pipeline-templates --root . --py-only --prefix strohA-sa --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_TEMPLATE_NAMES = {"template_pipeline.py", "template_pipeline.json"}


def _default_template_dir() -> Path:
    return Path(__file__).resolve().parent / "pipelines"


def _iter_targets(root: Path, want_py: bool, want_json: bool, prefix):
    pats = []
    if want_py:
        pats.append("*_pipeline.py")
    if want_json:
        pats.append("*_pipeline.json")
    seen = set()
    for pat in pats:
        for f in sorted(root.rglob(pat)):
            if not f.is_file() or f.name in _TEMPLATE_NAMES:
                continue                              # never touch the templates
            if prefix and not f.name.startswith(prefix):
                continue
            if f in seen:
                continue
            seen.add(f)
            yield f


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Replace *_pipeline.{py,json} with the current templates",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path.cwd(),
                   help="directory tree to scan (default: CWD)")
    p.add_argument("--prefix", default=None,
                   help="only files whose name starts with this prefix")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--py-only", action="store_true",
                   help="only *_pipeline.py")
    g.add_argument("--json-only", action="store_true",
                   help="only *_pipeline.json")
    p.add_argument("--template-dir", type=Path, default=None,
                   help="dir holding template_pipeline.{py,json} "
                        "(default: utilities/pipelines next to this script)")
    p.add_argument("--template-py", type=Path, default=None,
                   help="override path to template_pipeline.py")
    p.add_argument("--template-json", type=Path, default=None,
                   help="override path to template_pipeline.json")
    p.add_argument("--apply", action="store_true",
                   help="actually write (default: dry-run preview)")
    p.add_argument("--no-backup", action="store_true",
                   help="do not keep a .bak copy of replaced files")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    tdir     = a.template_dir or _default_template_dir()
    tpl_py   = a.template_py   or (tdir / "template_pipeline.py")
    tpl_json = a.template_json or (tdir / "template_pipeline.json")

    want_py   = not a.json_only
    want_json = not a.py_only

    templates = {}
    if want_py:
        if not tpl_py.is_file():
            print(f"error: template not found: {tpl_py}", file=sys.stderr)
            return 2
        templates[".py"] = tpl_py.read_text(encoding="utf-8")
    if want_json:
        if not tpl_json.is_file():
            print(f"error: template not found: {tpl_json}", file=sys.stderr)
            return 2
        templates[".json"] = tpl_json.read_text(encoding="utf-8")

    root = a.root.resolve()
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2

    changed = same = json_changed = 0
    for f in _iter_targets(root, want_py, want_json, a.prefix):
        tpl_text = templates[f.suffix]
        if f.read_text(encoding="utf-8") == tpl_text:
            same += 1
            continue
        changed += 1
        json_changed += (f.suffix == ".json")
        print(f"  {'replace' if a.apply else 'would replace'}: {f}")
        if a.apply:
            if not a.no_backup:
                shutil.copy2(f, f.with_name(f.name + ".bak"))
            f.write_text(tpl_text, encoding="utf-8")

    mode = "APPLIED" if a.apply else "DRY-RUN (use --apply to write)"
    print(f"\n{mode}: {changed} to change, {same} already up-to-date "
          f"(root={root})")
    if json_changed and not a.apply:
        print("note: replacing *_pipeline.json overwrites any per-session "
              "params from --estimate-params; use --py-only to keep the JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
