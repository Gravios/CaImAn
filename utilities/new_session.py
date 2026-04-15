#!/usr/bin/env python3
"""
utilities/new_session.py
========================
Prepare a new pipeline session from the template files.

Creates ``<dest>/<session>_pipeline.py`` and ``<dest>/<session>_pipeline.json``
by copying the templates and patching the JSON with the correct ``data_root``
and ``experiment`` paths derived from the destination.

Usage
-----
Minimal — run from the TL directory containing the .tif::

    cd /data/source/strohA/…/strohA-ia-000000-20140813-TL001_121103-25x-default
    python ~/software/CaImAn/utilities/new_session.py

Or run from an already-created session directory::

    cd /data/source/…/TL001_…/<session>/
    python ~/software/CaImAn/utilities/new_session.py

Explicit (session and dest still accepted for scripted / batch use)::

    python utilities/new_session.py \\
        stroh-ej-20140714-TL1 \\
        /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/14072014/

Override individual JSON parameters::

    python utilities/new_session.py \\
        stroh-ej-20140714-TL1 \\
        /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/14072014/ \\
        --data-root /data/src/ \\
        --fr 15 \\
        --decay-time 0.4 \\
        --gSig 7 \\
        --rf 28 \\
        --method-init greedy_roi

Dry run (print what would be done without writing anything)::

    python utilities/new_session.py \\
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
import re
import shutil
import sys
from pathlib import Path

# Matches per-channel TIF names produced by stack_to_bigtiff multi-channel mode,
# e.g.  "record-0001_C00.tif"  →  group(1)="record-0001"  group(2)="00"
_MULTICHAN_TIF_RE = re.compile(r'^(.+)_C(\d{2})\.tiff?$', re.IGNORECASE)



# ── YAML reader ───────────────────────────────────────────────────────────────

def _read_yaml(path) -> dict:
    """Load an acquisition YAML and return a flat dict of useful defaults.

    Handles both {value, units} nodes and plain scalars for all fields.
    Reads pixel_dtype (new) with fallback to pixel_type (old).
    Reads regions.vis_ctx.indicator and coordinates.

    Returns an empty dict if the file cannot be parsed.
    """
    try:
        import yaml as _yaml
        doc = _yaml.safe_load(path.read_text())
    except Exception:
        return {}
    if not isinstance(doc, dict):
        return {}

    out: dict = {}
    acq = doc.get("acquisition_system", {}).get("settings", {})

    def _val(node):
        if isinstance(node, dict):
            return node.get("value")
        return node

    fr = _val(acq.get("sample_rate"))
    if isinstance(fr, (int, float)):
        out["fr"] = float(fr)

    mag = _val(acq.get("magnification"))
    if isinstance(mag, (int, float)):
        out["magnification"] = f"{int(mag)}x"

    if isinstance(acq.get("n_channels"), int):
        out["n_channels"] = acq["n_channels"]
    if isinstance(acq.get("n_planes"), int):
        out["n_planes"] = acq["n_planes"]
    if isinstance(acq.get("n_frames"), int):
        out["n_frames_acq"] = acq["n_frames"]

    fs = acq.get("frame_size")
    if isinstance(fs, dict) and "x" in fs and "y" in fs:
        out["frame_size"] = {"x": fs["x"], "y": fs["y"]}

    ps = acq.get("pixel_size")
    if isinstance(ps, dict):
        out["pixel_size"] = dict(ps)

    # pixel_dtype (new schema) with fallback to pixel_type (old)
    if isinstance(acq.get("pixel_dtype"), str):
        out["pixel_dtype"] = acq["pixel_dtype"]
    elif isinstance(acq.get("pixel_type"), str):
        out["pixel_dtype"] = acq["pixel_type"]

    depth = _val(acq.get("depth"))
    if isinstance(depth, (int, float)):
        out["depth_um"] = float(depth)

    fa = _val(acq.get("fa"))
    if fa is not None:
        out["fa"] = fa

    lp = _val(acq.get("laserPower"))
    if isinstance(lp, (int, float)):
        out["laser_power"] = float(lp)

    gain = _val(acq.get("gain"))
    if isinstance(gain, (int, float)):
        out["gain"] = float(gain)

    vis_ctx = (doc.get("regions") or {}).get("vis_ctx") or {}
    if vis_ctx.get("indicator"):
        out["indicator"] = vis_ctx["indicator"]

    species_raw = (doc.get("subject", {}) or {}).get("species", "") or ""
    if species_raw:
        out["species"] = "rat" if "rat" in species_raw.lower() else "mouse"

    rec = doc.get("caiman_recommended") or {}
    if isinstance(rec.get("gSig"), int):
        out["gSig"] = rec["gSig"]
    if isinstance(rec.get("rf"), int):
        out["rf"] = rec["rf"]
    if isinstance(rec.get("decay_time"), (int, float)):
        out["decay_time"] = float(rec["decay_time"])

    return out


def _find_yaml(session: str, dest) -> "Path | None":
    """Locate the acquisition YAML for a session.

    Layout::

        <date>/<TL_dir>/              ← dest.parent
          <TL_dir>.yaml              ← YAML (stem = TL dir name)
          <TL_dir>-C00-fc*.tif       ← source TIF
          <TL_dir>-C00-fc*/          ← dest (session dir)

    Returns ``dest.parent / (dest.parent.name + ".yaml")`` if it exists.
    """
    candidate = dest.parent / (dest.parent.name + ".yaml")
    if candidate.exists() and ".bak" not in candidate.suffixes:
        return candidate
    return None


# ── Locate templates ──────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINES  = _SCRIPT_DIR / "pipelines"
_TPL_PY     = _PIPELINES / "template_pipeline.py"
_TPL_JSON   = _PIPELINES / "template_pipeline.json"


def _find_templates() -> tuple[Path, Path]:
    """Return (template.py, template.json), raising if not found."""
    missing = [p for p in (_TPL_PY, _TPL_JSON) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Template files not found: {[str(p) for p in missing]}\n"
            f"Expected in: {_PIPELINES}"
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

def _check_tif(dest: Path, session: str, channel_id: str | None = None) -> str | None:
    """Return a warning string if the expected TIF is not found, else None.

    For single-channel sessions expects ``<session>.tif``.
    For multi-channel sessions (``channel_id`` provided) expects
    ``<session>_C{channel_id}.tif`` (e.g. ``<session>_C00.tif``).
    """
    if channel_id is not None:
        tif = dest / f"{session}_C{channel_id}.tif"
    else:
        tif = dest / f"{session}.tif"
    if not tif.exists():
        return (f"  ⚠  {tif.name} not found in {dest}\n"
                f"     Place the TIFF in {dest} before running the pipeline.")
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="new_session.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("session", nargs="?", default=None,
        help="Session identifier (TIF stem). "
             "Omit to infer from the current working directory.")
    p.add_argument("dest", nargs="?", default=None,
        help="Absolute path to the session folder (created if missing). "
             "Omit to infer from the current working directory.")

    # Session / data
    p.add_argument("--data-root", metavar="PATH",
        help="Override inferred data_root in the JSON (default: auto)")
    p.add_argument("--yaml", metavar="PATH",
        help="Acquisition YAML to read defaults from. "
             "Auto-detected from dest parent if omitted.")
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

    p.add_argument("--run-mc", action="store_true",
        help="Run GPU motion correction before parameter estimation if no MC mmap "
             "exists yet. Uses conservative defaults for 512×512 stacks. "
             "Implies --estimate-params.")

    p.add_argument("--estimate-params", action="store_true",
        help="Run parameter estimation from the TIF after creating the session "
             "files and update the JSON with the suggestions (requires caiman)")
    p.add_argument("--n-frames", type=int, default=500, metavar="N",
        help="Frames to subsample for parameter estimation (default 500)")
    p.add_argument("--species", choices=["mouse", "rat"], default="mouse",
        help="Animal species — constrains gSig search range (default: mouse)")
    p.add_argument("--magnification", choices=["20x", "40x"], default="20x",
        help="Objective magnification — combined with species to bound gSig (default: 20x)")

    p.add_argument("--run", action="store_true",
        help="Run the CaImAn pipeline script after session setup (and MC/estimation "
             "if those flags are also set). Equivalent to: python <session>_pipeline.py")

    return p



def _run_motion_correction(
    tif_path: Path,
    session:  str,
    caiman_temp: str,
    logger,
) -> tuple:
    """Run GPU rigid motion correction with conservative 512×512 defaults.

    Writes the F-order mmap to ``caiman_temp`` and returns
    ``(fname_mc, mc_object)`` so that the caller can inspect shifts
    for better MC parameter suggestions, then delete the mmap.
    Returns ``(None, None)`` on failure.

    Default parameters are tuned for 512×512 galvo 2P stacks:
    - ``max_shifts = [6, 6]`` — ±6 px per axis, catches typical brain motion
    - ``strides / overlaps`` — standard piecewise-rigid tile size (unused for
      rigid MC, included for forward-compatibility)
    - ``border_nan = "copy"`` — no black borders after shift
    - ``pw_rigid = False`` — rigid-only; faster and sufficient for most cases
    """
    try:
        from caiman.motion_correction import MotionCorrect as _MC
        import numpy as _np

        logger.info(f"MC: rigid GPU correction on {tif_path}")
        mc = _MC(
            [str(tif_path)],
            dview               = None,
            max_shifts          = [6, 6],
            strides             = [64, 64],
            overlaps            = [32, 32],
            max_deviation_rigid = 3,
            shifts_opencv       = True,
            nonneg_movie        = True,
            border_nan          = "copy",
            pw_rigid            = False,
            use_gpu             = True,
        )
        mc.motion_correct(save_movie=True)
        fname_mc = mc.mmap_file[0]

        shifts = _np.array(mc.shifts_rig)
        mag    = _np.hypot(shifts[:, 0], shifts[:, 1])
        logger.info(
            f"MC done: {fname_mc}  "
            f"(median shift {_np.median(mag):.2f} px, "
            f"max {mag.max():.2f} px)"
        )
        return fname_mc, mc

    except Exception as exc:
        logger.warning(f"MC failed: {exc}")
        return None, None


def _infer_from_cwd() -> tuple[str, Path]:
    """Infer (session, dest) from the current working directory.

    Expected layouts::

        Single-channel:
          <session>/
            <session>.tif
            <session>_pipeline.py

        Multi-channel (produced by stack_to_bigtiff multi-channel mode):
          <session>/
            <session>_C00.tif
            <session>_C01.tif   ← all share the same base stem
            <session>_C00_pipeline.json
            ...

    Returns ``(base_session, cwd)`` in both cases. ``base_session`` is the
    stem without any ``_CNN`` suffix; multi-channel dispatch is handled later
    in ``main()`` via ``n_channels`` read from the acquisition YAML.
    """
    cwd  = Path.cwd()
    tifs = sorted(p for p in cwd.glob("*.tif") if p.is_file())

    if not tifs:
        raise SystemExit(
            "new_session.py: no .tif found in the current directory "
            f"({cwd}).\n"
            "  cd into the directory that contains the .tif and re-run,\n"
            "  or supply session and dest arguments explicitly."
        )

    # ── Check for multi-channel set: stem_CNN.tif ─────────────────────────
    multichan: dict[str, list[str]] = {}
    for t in tifs:
        m = _MULTICHAN_TIF_RE.match(t.name)
        if m:
            multichan.setdefault(m.group(1), []).append(m.group(2))

    if multichan:
        if len(multichan) == 1:
            stem     = next(iter(multichan))
            chan_ids = sorted(multichan[stem])
            print(f"  Multi-channel TIFs detected: {len(chan_ids)} channel(s) "
                  f"({', '.join('C'+c for c in chan_ids)})")
            return stem, cwd
        else:
            names = "\n    ".join(sorted(multichan.keys()))
            raise SystemExit(
                f"new_session.py: multiple multi-channel TIF sets in {cwd}:\n"
                f"    {names}\n"
                "  Supply the session argument explicitly."
            )

    # ── Single TIF ────────────────────────────────────────────────────────
    if len(tifs) == 1:
        return tifs[0].stem, cwd

    names = "\n    ".join(t.name for t in tifs[:8])
    raise SystemExit(
        f"new_session.py: multiple .tif files in {cwd}:\n"
        f"    {names}\n"
        "  Supply the session and dest arguments explicitly."
    )



def main(argv: list[str] | None = None) -> int:
    args   = _build_parser().parse_args(argv)

    if args.session is None or args.dest is None:
        _session, _dest = _infer_from_cwd()
        session = args.session or _session
        dest    = Path(args.dest).resolve() if args.dest else _dest.resolve()
        print(f"  Inferred   : session={session}")
        print(f"             : dest={dest}")
    else:
        session = args.session
        dest    = Path(args.dest).resolve()

    # ── Load YAML defaults — create from template if not found ───────────────
    _yaml_path = Path(args.yaml).resolve() if getattr(args, "yaml", None) else _find_yaml(session, dest)

    if _yaml_path is None or not _yaml_path.exists():
        # No YAML found — create one from template_acquisition.yaml in the
        # session directory (dest) so the experimentalist can fill it in.
        _tpl_acq = _PIPELINES / "template_acquisition.yaml"
        if _tpl_acq.exists() and not args.dry_run:
            _yaml_path = dest / f"{session}.yaml"
            import shutil as _sh
            dest.mkdir(parents=True, exist_ok=True)
            _sh.copy2(_tpl_acq, _yaml_path)
            print(f"  YAML       : created {_yaml_path.name} from template")
        elif args.dry_run:
            print(f"  YAML       : would create {dest / (session + '.yaml')} from template")

    _yd: dict = _read_yaml(_yaml_path) if _yaml_path and _yaml_path.exists() else {}
    if _yd:
        print("  YAML       :", _yaml_path.name)
        for _k, _v in _yd.items(): print(f"    {_k} = {_v}")

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

    # ── Build overrides (YAML defaults < CLI flags) ───────────────────────────
    _fr      = args.fr         if args.fr         is not None else _yd.get("fr")
    _decay   = args.decay_time if args.decay_time is not None else _yd.get("decay_time")
    _rf      = args.rf         if args.rf         is not None else _yd.get("rf")
    _gSig    = args.gSig       if args.gSig       is not None else _yd.get("gSig")
    _species = args.species    if args.species    != "mouse"  else _yd.get("species", args.species)
    _magnif  = args.magnification if args.magnification != "20x" else _yd.get("magnification", args.magnification)

    # ── Channel / plane metadata from YAML ────────────────────────────────────
    _n_channels = _yd.get("n_channels", 1)
    _n_planes   = _yd.get("n_planes",   1)

    overrides: dict = {}
    if _fr      is not None: overrides["data.fr"]             = _fr
    if _decay   is not None: overrides["data.decay_time"]     = _decay
    if _rf      is not None: overrides["cnmf.rf"]             = _rf
    if args.K            is not None: overrides["cnmf.K"]             = args.K
    if args.min_corr     is not None: overrides["cnmf.min_corr"]      = args.min_corr
    if args.min_pnr      is not None: overrides["cnmf.min_pnr"]       = args.min_pnr
    if args.method_init  is not None: overrides["cnmf.method_init"]   = args.method_init
    if args.n_processes  is not None: overrides["cluster.n_processes"] = args.n_processes

    # Always write channel/plane counts into JSON so the pipeline script sees them.
    overrides["data.n_channels"] = _n_channels
    overrides["data.n_planes"]   = _n_planes

    if _gSig is not None:
        g = _gSig
        overrides["cnmf.gSig"] = [g, g]
        overrides["cnmf.gSiz"] = [g * 4 + 1, g * 4 + 1]
        # Auto-derive rf and stride from gSig when not explicitly supplied:
        #   rf = 5 × gSig  (guarantees ring fits: 0.9 × gSiz < rf for all gSig≥2)
        #   stride = rf // 2
        if _rf is None and args.rf is None:
            _rf_auto = 5 * g
            overrides.setdefault("cnmf.rf",     _rf_auto)
            overrides.setdefault("cnmf.stride", _rf_auto // 2)

    # ── Determine channel list ────────────────────────────────────────────────
    # Scan the dest directory for per-channel TIFs to enumerate channels when
    # the YAML reports n_channels > 1.  Fall back to a synthetic list when TIFs
    # are not yet present (e.g. during session setup before stacking).
    if _n_channels > 1:
        # Try to discover IDs from existing TIFs first.
        _ch_tifs = sorted(dest.glob(f"{session}_C*.tif")) if dest.exists() else []
        _discovered = []
        for _t in _ch_tifs:
            _m = _MULTICHAN_TIF_RE.match(_t.name)
            if _m and _m.group(1) == session:
                _discovered.append(_m.group(2))
        if _discovered:
            channel_ids = sorted(set(_discovered))
        else:
            # No TIFs yet — synthesise IDs from n_channels (C00, C01, …)
            channel_ids = [f"{i:02d}" for i in range(_n_channels)]
    else:
        channel_ids = []   # empty → single-channel path

    _multichannel = bool(channel_ids)

    # ── Output paths ──────────────────────────────────────────────────────────
    # Single-channel: one py + one json, same as before.
    # Multi-channel:  one shared py  + one json per channel (_C00, _C01, …).
    out_py = dest / f"{session}_pipeline.py"
    if _multichannel:
        out_jsons = {
            cid: dest / f"{session}_C{cid}_pipeline.json"
            for cid in channel_ids
        }
        all_outputs = [out_py] + list(out_jsons.values())
    else:
        out_json  = dest / f"{session}_pipeline.json"
        all_outputs = [out_py, out_json]

    # ── Conflict check ────────────────────────────────────────────────────────
    existing = [p for p in all_outputs if p.exists()]
    if existing and not args.force and not args.dry_run:
        print("The following files already exist:")
        for p in existing:
            print(f"  {p}")
        ans = input("Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 0

    # ── Patch JSON(s) ─────────────────────────────────────────────────────────
    # Build base patched dict (channel-independent overrides applied here).
    _base_patched = _patch_json(
        tpl_json, session, dest,
        data_root      = args.data_root,
        overrides      = overrides,
        strip_comments = args.no_comments,
    )
    dr  = _base_patched["session"]["data_root"]
    exp = _base_patched["session"]["experiment"]

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("Session preparation")
    print("=" * 60)
    print(f"  Session      : {session}")
    print(f"  Dest         : {dest}")
    print(f"  data_root    : {dr}")
    print(f"  experiment   : {exp}")
    if _multichannel:
        print(f"  Channels     : {len(channel_ids)}  ({', '.join('C'+c for c in channel_ids)})")
    print(f"  n_channels   : {_n_channels}")
    print(f"  n_planes     : {_n_planes}")
    if args.estimate_params or args.run_mc:
        print(f"  Species      : {_species}")
        print(f"  Magnif.      : {_magnif}")
    if overrides:
        print("  Overrides    :")
        for k, v in overrides.items():
            print(f"    {k} = {v}")
    print()
    print("  Files to write:")
    for p in all_outputs:
        print(f"    {p}")

    if _multichannel:
        for cid in channel_ids:
            tif_warn = _check_tif(dest, session, channel_id=cid)
            if tif_warn:
                print()
                print(tif_warn)
    else:
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

    if _multichannel:
        for cid in channel_ids:
            import copy as _copy
            _ch_patched = _copy.deepcopy(_base_patched)
            _ch_patched["data"]["channel_id"] = int(cid)
            _ch_json_path = out_jsons[cid]
            _ch_json_path.write_text(json.dumps(_ch_patched, indent=4) + "\n")
            print(f"  Wrote  {_ch_json_path.name}")
    else:
        out_json.write_text(json.dumps(_base_patched, indent=4) + "\n")

    print()
    print("Done.")

    # ── Optional motion correction + parameter estimation ────────────────
    # In multi-channel mode, run estimation on C00 only (primary channel).
    _do_estimate = args.estimate_params or args.run_mc
    if _do_estimate:
        # Apply env vars from the pipeline JSON before importing caiman so that
        # CAIMAN_TEMP / CAIMAN_DATA etc. take effect at caiman import time and
        # CaImAn never falls back to its compiled-in default (/data/proc/...).
        _env_section = _base_patched.get("env", {})
        for _ekey, _eval in _env_section.items():
            if not _ekey.startswith("_comment") and isinstance(_eval, str):
                os.environ.setdefault(_ekey, _eval)
        # CAIMAN_TEMP / CAIMAN_DATA are always forced (not setdefault) because
        # caiman reads them at import; a stale shell value could point anywhere.
        for _force_key in ("CAIMAN_TEMP", "CAIMAN_DATA", "CAIMAN_SHM"):
            if _force_key in _env_section:
                os.environ[_force_key] = _env_section[_force_key]

        if _multichannel:
            _primary_cid = channel_ids[0]
            tif_path  = dest / f"{session}_C{_primary_cid}.tif"
            out_json  = out_jsons[_primary_cid]
            print(f"\n  Multi-channel: running MC/estimation on primary channel C{_primary_cid}.")
        else:
            tif_path = dest / f"{session}.tif"

        if not tif_path.exists():
            print(f"\n  ⚠  Cannot proceed: {tif_path.name} not found.")
            print(f"     Place the TIFF and re-run.")
        else:
            try:
                import logging as _logging
                _est_logger = _logging.getLogger("caiman")
                _est_logger.setLevel(_logging.INFO)
                if not _est_logger.handlers:
                    _h = _logging.StreamHandler()
                    _h.setFormatter(_logging.Formatter("%(message)s"))
                    _est_logger.addHandler(_h)

                import glob as _glob
                _caiman_temp = os.environ.get("CAIMAN_TEMP", "/data/caiman/temp")
                _mc = sorted(_glob.glob(str(
                    dest / f"*{session}*rig*order_F*.mmap")))
                _mc_temp = sorted(_glob.glob(
                    os.path.join(_caiman_temp, f"*{session}*rig*order_F*.mmap")))
                _mc_path = (_mc or _mc_temp)

                # ── Run MC if requested and not already done ──────────────
                _mc_created = False
                _mc_obj     = None
                if args.run_mc and not _mc_path:
                    print(f"\nRunning motion correction on {tif_path.name}...")
                    _new_mc, _mc_obj = _run_motion_correction(
                        tif_path, session, _caiman_temp, _est_logger)
                    if _new_mc:
                        _mc_path   = [_new_mc]
                        _mc_created = True
                elif args.run_mc and _mc_path:
                    print(f"\n  MC mmap already exists — skipping motion correction.")
                    print(f"  ({_mc_path[-1]})")

                # ── MC parameter suggestions from shifts ──────────────────
                if _mc_created and _mc_obj is not None:
                    import numpy as _np_mc
                    _shifts = _np_mc.array(_mc_obj.shifts_rig)
                    _p99_r  = float(_np_mc.percentile(_np_mc.abs(_shifts[:, 0]), 99))
                    _p99_c  = float(_np_mc.percentile(_np_mc.abs(_shifts[:, 1]), 99))
                    _ms_r   = max(4, int(_np_mc.ceil(_p99_r / 2)) * 2)
                    _ms_c   = max(4, int(_np_mc.ceil(_p99_c / 2)) * 2)
                    _mc_overrides = {"max_shifts": [_ms_r, _ms_c]}
                    _est_logger.info(
                        f"MC shift analysis: p99 row={_p99_r:.2f} col={_p99_c:.2f} px "
                        f"→ max_shifts=[{_ms_r}, {_ms_c}]"
                    )
                    del _mc_obj
                    from caiman.utils.params_estimator import apply_suggestions as _apply
                    _apply(out_json, {f"motion_correction.{k}": v
                                      for k, v in _mc_overrides.items()})
                    print(f"  MC parameter update: max_shifts={[_ms_r, _ms_c]}")

                # ── Parameter estimation ──────────────────────────────────
                try:
                    if _mc_path and _mc_path[-1]:
                        print(f"\nEstimating parameters...")
                        from caiman.utils.params_estimator import estimate_params, apply_suggestions
                        _suggestions = estimate_params(
                            _mc_path[-1],
                            species       = _species,
                            magnification = _magnif,
                            n_frames      = args.n_frames,
                            out_path      = out_json.parent / f"{session}_qc_00_param_estimate.png",
                            logger        = _est_logger,
                        )
                        apply_suggestions(out_json, _suggestions)
                        print(f"\n  JSON updated with estimated parameters.")
                    elif not args.run_mc:
                        print(f"  No MC mmap found for {session}.")
                        print(f"  Re-run with --run-mc to run motion correction first.")
                finally:
                    if _mc_created and _mc_path and _mc_path[-1]:
                        try:
                            import os as _os_del
                            _os_del.unlink(_mc_path[-1])
                            _est_logger.info(f"Deleted temporary MC mmap: {_mc_path[-1]}")
                            print(f"  Deleted temporary MC mmap: {Path(_mc_path[-1]).name}")
                        except OSError as _del_exc:
                            _est_logger.warning(f"Could not delete MC mmap: {_del_exc}")

            except Exception as _exc:
                import traceback as _tb
                print(f"  Failed: {_exc}")
                _tb.print_exc()

    print()
    print("Next steps:")
    if _multichannel:
        for cid in channel_ids:
            _cj = out_jsons[cid]
            print(f"  Review  :  {_cj.name}")
        print(f"  Run     :  python {out_py.name}  (set channel_id in each JSON first)")
    else:
        print(f"  1. Review and tune:  {out_json.name}")
        print(f"  2. Place TIFF:       {dest}/{session}.tif")
        print(f"  3. Estimate params:  python utilities/new_session.py {session} {dest} --run-mc --estimate-params -y")
        print(f"  4. Run:              python {out_py.name}")
    print()

    # ── Optional pipeline run ─────────────────────────────────────────────────
    if getattr(args, "run", False):
        import subprocess as _sp
        print(f"Running pipeline: {out_py}")
        print("=" * 60)
        result = _sp.run([sys.executable, str(out_py)], cwd=str(dest))
        if result.returncode != 0:
            print(f"\n  Pipeline exited with code {result.returncode}")
            return result.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
