#!/usr/bin/env python3
"""
pipelines/new_session.py
========================
Prepare a new pipeline session from the template files.

Creates ``<dest>/<session>_pipeline.py`` and ``<dest>/<session>_pipeline.json``
by copying the templates and patching the JSON with the correct ``data_root``
and ``experiment`` paths derived from the destination.

Usage
-----
Minimal (infers data_root and experiment from the destination path)::

    python pipelines/new_session.py \\
        stroh-ej-20140714-TL1 \\
        /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/14072014/

Override individual JSON parameters::

    python pipelines/new_session.py \\
        stroh-ej-20140714-TL1 \\
        /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/14072014/ \\
        --data-root /data/src/ \\
        --fr 15 \\
        --decay-time 0.4 \\
        --gSig 7 \\
        --rf 28 \\
        --method-init greedy_roi

Dry run (print what would be done without writing anything)::

    python pipelines/new_session.py \\
        stroh-ej-20140714-TL1 \\
        /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/14072014/ \\
        --dry-run

Positional arguments
--------------------
session
    Session identifier, e.g. ``stroh-ej-20140714-TL1``.  This becomes the
    stem of both output files and is used to locate the expected ``.tif``
    input.
dest
    Absolute path to the session folder.  The folder is created if it does
    not exist.  ``data_root`` and ``experiment`` are inferred as the longest
    prefix that matches ``/data/src/`` (or the value of ``--data-root``) and
    the remainder respectively.

Output files
------------
``<dest>/<session>_pipeline.py``
    Ready-to-run pipeline script (unchanged copy of the template).

``<dest>/<session>_pipeline.json``
    JSON config with ``session.data_root`` and ``session.experiment`` patched.
    Any additional ``--key value`` overrides are applied on top.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


# ── Locate templates ──────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_TPL_PY     = _SCRIPT_DIR / "template_pipeline.py"
_TPL_JSON   = _SCRIPT_DIR / "template_pipeline.json"


def _find_templates() -> tuple[Path, Path]:
    """Return (template.py, template.json), raising if not found."""
    missing = [p for p in (_TPL_PY, _TPL_JSON) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Template files not found: {[str(p) for p in missing]}\n"
            f"Expected in: {_SCRIPT_DIR}"
        )
    return _TPL_PY, _TPL_JSON


# ── JSON patching ─────────────────────────────────────────────────────────────

def _infer_paths(dest: Path, data_root: str | None) -> tuple[str, str]:
    """Infer data_root and experiment from *dest*.

    Walks upward from *dest* to find the longest prefix that matches
    *data_root* (default ``/data/src/``).  The remainder is the experiment
    path.

    Returns ``(data_root, experiment)`` as strings with trailing slashes.
    """
    if data_root is None:
        # Heuristic: try common roots in order
        for candidate in ["/data/src", "/data", "/mnt/data/src"]:
            if str(dest).startswith(candidate):
                data_root = candidate
                break
        else:
            # Fall back to parent of dest
            data_root = str(dest.parent.parent)

    data_root = data_root.rstrip("/")
    dest_str  = str(dest).rstrip("/")

    if dest_str.startswith(data_root):
        experiment = dest_str[len(data_root):].lstrip("/") + "/"
    else:
        # dest is outside data_root — use absolute path as experiment
        experiment = dest_str.lstrip("/") + "/"

    return data_root + "/", experiment


def _strip_comments(raw: dict) -> dict:
    """Recursively remove ``_comment`` keys from a dict (in-place copy)."""
    if isinstance(raw, dict):
        return {k: _strip_comments(v) for k, v in raw.items()
                if not k.startswith("_comment")}
    if isinstance(raw, list):
        return [_strip_comments(v) for v in raw]
    return raw


def _patch_json(
    template_path: Path,
    session: str,
    dest: Path,
    data_root: str | None,
    overrides: dict,
    strip_comments: bool,
) -> dict:
    """Load the template JSON and apply all patches.

    Parameters
    ----------
    template_path
        Path to ``template_pipeline.json``.
    session
        Session identifier.
    dest
        Session destination folder.
    data_root
        Explicit data root override, or ``None`` to infer.
    overrides
        Dict of ``{section.key: value}`` overrides from CLI flags.
    strip_comments
        If ``True``, remove ``_comment`` keys from the output.

    Returns
    -------
    dict
        Patched JSON as a plain dict, ready for ``json.dump``.
    """
    raw = json.loads(template_path.read_text())

    dr, exp = _infer_paths(dest, data_root)
    raw["session"]["data_root"]  = dr
    raw["session"]["experiment"] = exp

    # Remove placeholder _comment from session (not useful after patching)
    raw["session"].pop("_comment", None)

    # Apply CLI overrides: section.key → value
    for dotkey, value in overrides.items():
        parts = dotkey.split(".", 1)
        if len(parts) == 2:
            section, key = parts
            if section in raw and isinstance(raw[section], dict):
                raw[section][key] = value
            else:
                raw[section] = {key: value}
        else:
            raw[dotkey] = value

    if strip_comments:
        raw = _strip_comments(raw)

    return raw


# ── Validation ────────────────────────────────────────────────────────────────

def _check_tif(dest: Path, session: str) -> str | None:
    """Return a warning string if the expected TIF is not found, else None."""
    tif = dest / f"{session}.tif"
    if not tif.exists():
        return (f"  ⚠  {tif.name} not found in {dest}\n"
                f"     Place the TIFF there before running the pipeline.")
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="new_session.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("session",
        help="Session identifier, e.g. stroh-ej-20140714-TL1")
    p.add_argument("dest",
        help="Absolute path to the session folder (created if missing)")

    # Session / data
    p.add_argument("--data-root", metavar="PATH",
        help="Override inferred data_root in the JSON (default: auto)")
    p.add_argument("--fr", type=float, metavar="HZ",
        help="Acquisition frame rate [Hz]")
    p.add_argument("--decay-time", type=float, metavar="S",
        help="GCaMP decay time constant [s]")

    # CNMF
    p.add_argument("--gSig", type=int, metavar="PX",
        help="Gaussian half-width [px]; sets gSig=[N,N] and gSiz=[4N+1,4N+1]")
    p.add_argument("--rf", type=int, metavar="PX",
        help="Patch half-size [px]")
    p.add_argument("--K", type=int, metavar="N",
        help="Max components per patch")
    p.add_argument("--min-corr", type=float, metavar="F",
        help="Minimum local correlation for seed pixel")
    p.add_argument("--min-pnr", type=float, metavar="F",
        help="Minimum peak-to-noise ratio for seed pixel")
    p.add_argument("--method-init", choices=["corr_pnr", "greedy_roi"],
        help="Initialisation method")
    p.add_argument("--n-processes", type=int, metavar="N",
        help="CNMF worker count (default: null = all CPUs)")

    # Behaviour
    p.add_argument("--dry-run", action="store_true",
        help="Print what would be done without writing any files")
    p.add_argument("-y", "--force", action="store_true",
        help="Overwrite existing pipeline files without prompting (useful in batch scripts)")
    p.add_argument("--no-comments", action="store_true",
        help="Strip _comment keys from the output JSON")

    p.add_argument("--estimate-params", action="store_true",
        help="Run parameter estimation from the TIF after creating the session "
             "files and update the JSON with the suggestions (requires caiman)")
    p.add_argument("--n-frames", type=int, default=500, metavar="N",
        help="Frames to subsample for parameter estimation (default 500)")

    return p


def main(argv: list[str] | None = None) -> int:
    args   = _build_parser().parse_args(argv)
    session = args.session
    dest    = Path(args.dest).resolve()

    # ── Locate templates ──────────────────────────────────────────────────────
    try:
        tpl_py, tpl_json = _find_templates()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ── Output paths ──────────────────────────────────────────────────────────
    out_py   = dest / f"{session}_pipeline.py"
    out_json = dest / f"{session}_pipeline.json"

    # ── Conflict check ────────────────────────────────────────────────────────
    existing = [p for p in (out_py, out_json) if p.exists()]
    if existing and not args.force and not args.dry_run:
        print("The following files already exist:")
        for p in existing:
            print(f"  {p}")
        ans = input("Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 0

    # ── Build CLI overrides dict ───────────────────────────────────────────────
    overrides: dict = {}
    if args.fr           is not None: overrides["data.fr"]            = args.fr
    if args.decay_time   is not None: overrides["data.decay_time"]    = args.decay_time
    if args.rf           is not None: overrides["cnmf.rf"]            = args.rf
    if args.K            is not None: overrides["cnmf.K"]             = args.K
    if args.min_corr     is not None: overrides["cnmf.min_corr"]      = args.min_corr
    if args.min_pnr      is not None: overrides["cnmf.min_pnr"]       = args.min_pnr
    if args.method_init  is not None: overrides["cnmf.method_init"]   = args.method_init
    if args.n_processes  is not None: overrides["cluster.n_processes"] = args.n_processes

    if args.gSig is not None:
        g = args.gSig
        overrides["cnmf.gSig"] = [g, g]
        overrides["cnmf.gSiz"] = [g * 4 + 1, g * 4 + 1]

    # ── Patch JSON ────────────────────────────────────────────────────────────
    patched = _patch_json(
        tpl_json, session, dest,
        data_root      = args.data_root,
        overrides      = overrides,
        strip_comments = args.no_comments,
    )
    json_txt = json.dumps(patched, indent=4)

    # ── Dry-run summary ───────────────────────────────────────────────────────
    dr  = patched["session"]["data_root"]
    exp = patched["session"]["experiment"]

    print()
    print("Session preparation")
    print("=" * 60)
    print(f"  Session    : {session}")
    print(f"  Dest       : {dest}")
    print(f"  data_root  : {dr}")
    print(f"  experiment : {exp}")
    if overrides:
        print("  Overrides  :")
        for k, v in overrides.items():
            print(f"    {k} = {v}")
    print()
    print("  Files to write:")
    print(f"    {out_py}")
    print(f"    {out_json}")

    tif_warn = _check_tif(dest, session)
    if tif_warn:
        print()
        print(tif_warn)

    if args.dry_run:
        print()
        print("Dry run — no files written.")
        return 0

    # ── Write files ───────────────────────────────────────────────────────────
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tpl_py, out_py)
    out_json.write_text(json_txt + "\n")

    print()
    print("Done.")

    # ── Optional parameter estimation ────────────────────────────────────
    if args.estimate_params:
        tif_path = dest / f"{session}.tif"
        if not tif_path.exists():
            print(f"\n  ⚠  Cannot estimate params: {tif_path.name} not found.")
            print(f"     Place the TIFF and re-run with --estimate-params.")
        else:
            print(f"\nEstimating parameters from {tif_path.name}...")
            try:
                import logging as _logging
                _est_logger = _logging.getLogger("caiman")
                _est_logger.setLevel(_logging.INFO)
                if not _est_logger.handlers:
                    _h = _logging.StreamHandler()
                    _h.setFormatter(_logging.Formatter("%(message)s"))
                    _est_logger.addHandler(_h)
                # Find the MC mmap if it exists, else use the raw TIF
                import glob as _glob
                _mc = sorted(_glob.glob(str(
                    dest / f"*{session}*rig*order_F*.mmap")))
                _caiman_temp = os.environ.get("CAIMAN_TEMP", "/data/caiman/temp")
                _mc_temp = sorted(_glob.glob(
                    os.path.join(_caiman_temp, f"*{session}*rig*order_F*.mmap")))
                _mc_path = (_mc or _mc_temp)
                if _mc_path:
                    from caiman.utils.params_estimator import estimate_params, apply_suggestions
                    _suggestions = estimate_params(
                        _mc_path[-1],
                        n_frames  = args.n_frames,
                        out_path  = out_json.parent / f"{session}_qc_00_param_estimate.png",
                        logger    = _est_logger,
                    )
                    apply_suggestions(out_json, _suggestions)
                    print(f"\n  JSON updated with estimated parameters.")
                else:
                    print(f"  No MC mmap found for {session} — run motion correction first,"
                          f" then re-run with --estimate-params.")
            except Exception as _exc:
                print(f"  Parameter estimation failed: {_exc}")

    print()
    print("Next steps:")
    print(f"  1. Review and tune:  {out_json.name}")
    print(f"  2. Place TIFF:       {dest}/{session}.tif")
    print(f"  3. Run:              python {out_py.name}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
