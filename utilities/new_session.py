#!/usr/bin/env python3
"""
utilities/new_session.py
========================
Prepare a new CaImAn pipeline session from the template files.

Creates <dest>/<session>_pipeline.py and <dest>/<session>_pipeline.json
by copying the templates and patching the JSON with the correct paths and
any parameter overrides.  Optionally runs GPU motion correction, estimates
CNMF parameters, and executes the pipeline.

Quick start
-----------
  # From the channel subdir (session and dest inferred automatically):
  cd /data/source/.../TL001_143915-25x-spont/TL001_143915-25x-spont-C00-fc011170
  new-session -y
  new-session --run-mc --estimate-params -y
  new-session --run-mc --estimate-params --run -y

  # Explicit:
  new-session <session-stem> <dest-path> --run-mc --estimate-params -y

Positional arguments
--------------------
  session               Session identifier (TIF stem). Inferred from CWD if omitted.
  dest                  Path to the channel subdir. Inferred from CWD if omitted.

Session / data flags
---------------------
  --yaml PATH           Acquisition YAML. Auto-detected from dest parent if omitted.
                        Created from template if not found.
  --fr HZ               Acquisition frame rate [Hz].
  --decay-time S        GCaMP decay time constant [s]. (GCaMP6f ~0.4, GCaMP6s ~1.0)

CNMF parameter flags
---------------------
  --gSig PX             Gaussian half-width [px]. Sets gSig=[N,N] and gSiz=[4N+1,4N+1].
                        Also auto-derives rf and stride when --rf is omitted.
  --rf PX               Patch half-size [px]. Ring constraint: ring_size_factor x gSiz < rf.
  --K N                 Max components per patch.
  --min-corr F          Minimum local correlation for seed pixel.
  --min-pnr F           Minimum peak-to-noise ratio for seed pixel.
  --method-init         Initialisation method: corr_pnr (default) or greedy_roi.
  --n-processes N       CNMF worker count. Default: all CPUs.

Processing flags
-----------------
  --run-mc              Run GPU rigid motion correction. Implies --estimate-params.
  --estimate-params     Estimate CNMF parameters from the MC'd movie.
  --n-frames N          Frames to subsample for param estimation. Default: 500.
  --species mouse|rat   Animal species. Default: mouse.
  --magnification 20x|40x  Objective magnification. Default: 20x.
  --run                 Run the pipeline script after setup.

Behaviour flags
---------------
  --dry-run             Preview only -- no files written.
  -y / --force          Overwrite existing files without prompting.
  --no-comments         Strip _comment keys from the output JSON.
  -h / --help           Show this help message.

Output files
------------
  <dest>/<session>_pipeline.py    Ready-to-run pipeline script.
  <dest>/<session>_pipeline.json  JSON config with all parameters.
  <dest.parent>/<TL_dir>.yaml     Acquisition YAML (created from template if absent).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Template locations ────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINES  = _SCRIPT_DIR / "pipelines"
_TPL_PY     = _PIPELINES / "template_pipeline.py"
_TPL_JSON   = _PIPELINES / "template_pipeline.json"
_TPL_YAML   = _PIPELINES / "template_acquisition.yaml"


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _read_yaml(path: Path) -> dict:
    """Load acquisition YAML -> flat dict of pipeline defaults.

    Handles both {value, units} nodes and plain scalars.
    Returns {} if the file cannot be parsed.
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
        return node.get("value") if isinstance(node, dict) else node

    fr = _val(acq.get("sample_rate"))
    if isinstance(fr, (int, float)):
        out["fr"] = float(fr)

    mag = _val(acq.get("magnification"))
    if isinstance(mag, (int, float)):
        out["magnification"] = f"{int(mag)}x"

    for key in ("n_channels", "n_planes"):
        if isinstance(acq.get(key), int):
            out[key] = acq[key]
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

    if isinstance(_val(acq.get("depth")), (int, float)):
        out["depth_um"] = float(_val(acq["depth"]))
    if _val(acq.get("fa")) is not None:
        out["fa"] = _val(acq["fa"])
    if isinstance(_val(acq.get("laserPower")), (int, float)):
        out["laser_power"] = float(_val(acq["laserPower"]))
    if isinstance(_val(acq.get("gain")), (int, float)):
        out["gain"] = float(_val(acq["gain"]))

    vis_ctx = (doc.get("regions") or {}).get("vis_ctx") or {}
    if vis_ctx.get("indicator"):
        out["indicator"] = vis_ctx["indicator"]

    species_raw = (doc.get("subject") or {}).get("species") or ""
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


def _find_yaml(session: str, dest: Path) -> Path | None:
    """Locate the acquisition YAML.

    Layout::
        <TL_dir>/              <- dest.parent
          <TL_dir>.yaml        <- canonical location
          <TL_dir>-C00-fc<N>/ <- dest
    """
    for candidate in [
        dest.parent / (dest.parent.name + ".yaml"),
        dest / (session + ".yaml"),
    ]:
        if candidate.exists() and ".bak" not in candidate.suffixes:
            return candidate
    return None


# ── Template helpers ──────────────────────────────────────────────────────────

def _find_templates() -> tuple[Path, Path]:
    missing = [p for p in (_TPL_PY, _TPL_JSON) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Template files not found: {[str(p) for p in missing]}\n"
            f"Expected in: {_PIPELINES}"
        )
    return _TPL_PY, _TPL_JSON


# ── JSON patching ─────────────────────────────────────────────────────────────

def _strip_comments(raw: dict) -> dict:
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
    overrides: dict,
    strip_comments: bool,
) -> dict:
    raw = json.loads(template_path.read_text())
    raw.pop("session", None)   # session section removed — outdir derived from script location

    for dotkey, value in overrides.items():
        parts = dotkey.split(".", 1)
        if len(parts) == 2:
            section, key = parts
            raw.setdefault(section, {})[key] = value
        else:
            raw[dotkey] = value

    return _strip_comments(raw) if strip_comments else raw


# ── Validation ────────────────────────────────────────────────────────────────

def _check_tif(dest: Path, session: str) -> str | None:
    tif = dest / f"{session}.tif"
    if not tif.exists():
        return (f"  ⚠  {tif.name} not found in {dest}\n"
                f"     Place the TIFF before running the pipeline.")
    return None


# ── Inference ────────────────────────────────────────────────────────────────

def _infer_from_cwd() -> tuple[str, Path]:
    """Infer (session, dest) from CWD (the channel subdir)."""
    cwd  = Path.cwd()
    tifs = sorted(p for p in cwd.glob("*.tif") if p.is_file())
    if not tifs:
        raise SystemExit(
            f"new_session.py: no .tif found in {cwd}.\n"
            "  cd into the channel subdir, or supply session and dest explicitly."
        )
    if len(tifs) == 1:
        return tifs[0].stem, cwd
    names = "\n    ".join(t.name for t in tifs[:8])
    raise SystemExit(
        f"new_session.py: multiple .tif files in {cwd}:\n    {names}\n"
        "  Supply session and dest arguments explicitly."
    )


# ── MC / param estimation ─────────────────────────────────────────────────────

def _apply_pipeline_env(env_section: dict) -> None:
    """Apply env vars from the pipeline JSON before caiman import."""
    for key, val in env_section.items():
        if not key.startswith("_comment") and isinstance(val, str):
            os.environ.setdefault(key, val)
    # Always force CaImAn path vars so stale shell values don't misdirect
    for key in ("CAIMAN_TEMP", "CAIMAN_DATA", "CAIMAN_SHM"):
        if key in env_section:
            os.environ[key] = env_section[key]


def _run_motion_correction(tif_path: Path, caiman_temp: str, log) -> tuple:
    """GPU rigid MC. Returns (mmap_path, mc_object) or (None, None) on failure."""
    try:
        from caiman.motion_correction import MotionCorrect as _MC
        import numpy as _np
        log.info(f"MC: rigid GPU correction on {tif_path}")
        mc = _MC(
            [str(tif_path)],
            dview=None, max_shifts=[6, 6], strides=[64, 64], overlaps=[32, 32],
            max_deviation_rigid=3, shifts_opencv=True, nonneg_movie=True,
            border_nan="copy", pw_rigid=False, use_gpu=True,
        )
        mc.motion_correct(save_movie=True)
        fname = mc.mmap_file[0]
        shifts = _np.array(mc.shifts_rig)
        mag    = _np.hypot(shifts[:, 0], shifts[:, 1])
        log.info(f"MC done: {fname}  (median {_np.median(mag):.2f} px, max {mag.max():.2f} px)")
        return fname, mc
    except Exception as exc:
        log.warning(f"MC failed: {exc}")
        return None, None


def _run_mc_and_estimate(args, session: str, dest: Path, out_json: Path) -> bool:
    """Run motion correction and/or param estimation; updates out_json in place.
    Returns True if estimation ran (or was skipped intentionally), False if TIF missing.
    """
    import glob as _glob
    import logging as _logging

    log = _logging.getLogger("caiman")
    log.setLevel(_logging.INFO)
    if not log.handlers:
        _h = _logging.StreamHandler()
        _h.setFormatter(_logging.Formatter("%(message)s"))
        log.addHandler(_h)

    tif_path    = dest / f"{session}.tif"
    caiman_temp = os.environ.get("CAIMAN_TEMP", "/data/caiman/temp")

    if not tif_path.exists():
        print(f"\n  ⚠  Cannot proceed: {tif_path.name} not found.")
        return False

    mc_path = (
        sorted(_glob.glob(str(dest / f"*{session}*rig*order_F*.mmap")))
        or sorted(_glob.glob(os.path.join(caiman_temp, f"*{session}*rig*order_F*.mmap")))
    )

    mc_created = False
    mc_obj     = None

    if args.run_mc and not mc_path:
        print(f"\nRunning motion correction on {tif_path.name}...")
        new_mc, mc_obj = _run_motion_correction(tif_path, caiman_temp, log)
        if new_mc:
            mc_path    = [new_mc]
            mc_created = True
    elif args.run_mc and mc_path:
        print(f"\n  MC mmap already exists -- skipping.\n  ({mc_path[-1]})")

    if mc_created and mc_obj is not None:
        import numpy as _np
        shifts = _np.array(mc_obj.shifts_rig)
        p99_r  = float(_np.percentile(_np.abs(shifts[:, 0]), 99))
        p99_c  = float(_np.percentile(_np.abs(shifts[:, 1]), 99))
        ms_r   = max(4, int(_np.ceil(p99_r / 2)) * 2)
        ms_c   = max(4, int(_np.ceil(p99_c / 2)) * 2)
        log.info(f"MC shift p99: row={p99_r:.2f} col={p99_c:.2f} -> max_shifts=[{ms_r},{ms_c}]")
        del mc_obj
        from caiman.utils.params_estimator import apply_suggestions
        apply_suggestions(out_json, {"motion_correction.max_shifts": [ms_r, ms_c]})
        print(f"  MC parameter update: max_shifts=[{ms_r}, {ms_c}]")

    try:
        if mc_path and mc_path[-1]:
            print("\nEstimating parameters...")
            from caiman.utils.params_estimator import estimate_params, apply_suggestions
            suggestions = estimate_params(
                mc_path[-1],
                species       = args.species,
                magnification = args.magnification,
                n_frames      = args.n_frames,
                out_path      = dest / f"{session}_qc_00_param_estimate.png",
                logger        = log,
            )
            apply_suggestions(out_json, suggestions)
            print("\n  JSON updated with estimated parameters.")
        elif not args.run_mc:
            print("  No MC mmap found. Re-run with --run-mc first.")
    finally:
        if mc_created and mc_path and mc_path[-1]:
            try:
                os.unlink(mc_path[-1])
                print(f"  Deleted temporary MC mmap: {Path(mc_path[-1]).name}")
            except OSError as exc:
                log.warning(f"Could not delete MC mmap: {exc}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="new_session.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "CNMF ring constraint: ring_size_factor x gSiz must be < rf\n"
            "  Safe rule: rf = 5 x gSig,  stride = rf // 2,  gSiz = 4 x gSig + 1\n\n"
            "Examples:\n"
            "  new-session -y\n"
            "  new-session --run-mc --estimate-params -y\n"
            "  new-session --run-mc --estimate-params --run -y\n"
            "  new-session --gSig 6 --rf 32 --K 15 -y"
        ),
    )
    p.add_argument("session", nargs="?", default=None,
                   help="Session stem. Inferred from CWD if omitted.")
    p.add_argument("dest", nargs="?", default=None,
                   help="Channel subdir path. Inferred from CWD if omitted.")
    p.add_argument("--template-json",   metavar="PATH",
                   help="Path to a custom template_pipeline.json. "
                        "Defaults to utilities/pipelines/template_pipeline.json.")
    p.add_argument("--yaml",            metavar="PATH")
    p.add_argument("--fr",              type=float, metavar="HZ")
    p.add_argument("--decay-time",      type=float, metavar="S")
    p.add_argument("--gSig",            type=int,   metavar="PX")
    p.add_argument("--rf",              type=int,   metavar="PX")
    p.add_argument("--K",               type=int,   metavar="N")
    p.add_argument("--min-corr",        type=float, metavar="F")
    p.add_argument("--min-pnr",         type=float, metavar="F")
    p.add_argument("--method-init",     choices=["corr_pnr", "greedy_roi"])
    p.add_argument("--n-processes",     type=int,   metavar="N")
    p.add_argument("--run-mc",          action="store_true")
    p.add_argument("--estimate-params", action="store_true")
    p.add_argument("--n-frames",        type=int, default=500, metavar="N")
    p.add_argument("--species",         choices=["mouse", "rat"], default="mouse")
    p.add_argument("--magnification",   choices=["20x", "40x"],   default="20x")
    p.add_argument("--run",             action="store_true")
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("-y", "--force",     action="store_true")
    p.add_argument("--no-comments",     action="store_true")
    return p


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # ── Resolve session + dest ─────────────────────────────────────────────
    if args.session is None or args.dest is None:
        _session, _dest = _infer_from_cwd()
        session = args.session or _session
        dest    = Path(args.dest).resolve() if args.dest else _dest
        print(f"  Inferred   : session={session}")
        print(f"             : dest={dest}")
    else:
        session = args.session
        dest    = Path(args.dest).resolve()

    # ── YAML: find or create from template ────────────────────────────────
    yaml_path = Path(args.yaml).resolve() if args.yaml else _find_yaml(session, dest)
    if yaml_path is None or not yaml_path.exists():
        if _TPL_YAML.exists() and not args.dry_run:
            yaml_path = dest.parent / f"{dest.parent.name}.yaml"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_TPL_YAML, yaml_path)
            print(f"  YAML       : created {yaml_path.name} from template")
        elif args.dry_run:
            print(f"  YAML       : would create {dest.parent / (dest.parent.name + '.yaml')}")

    yd: dict = _read_yaml(yaml_path) if yaml_path and yaml_path.exists() else {}
    if yd:
        print(f"  YAML       : {yaml_path.name}")
        for k, v in yd.items():
            print(f"    {k} = {v}")

    # ── Locate templates ───────────────────────────────────────────────────
    tpl_json_override = Path(args.template_json).resolve() if args.template_json else None
    if tpl_json_override and not tpl_json_override.exists():
        print(f"ERROR: --template-json not found: {tpl_json_override}", file=sys.stderr)
        return 1
    try:
        tpl_py, tpl_json = _find_templates()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if tpl_json_override:
        tpl_json = tpl_json_override
        print(f"  Template JSON : {tpl_json}")

    # ── Build overrides (YAML defaults < CLI flags) ────────────────────────
    fr      = args.fr         if args.fr         is not None else yd.get("fr")
    decay   = args.decay_time if args.decay_time is not None else yd.get("decay_time")
    rf      = args.rf         if args.rf         is not None else yd.get("rf")
    gSig    = args.gSig       if args.gSig       is not None else yd.get("gSig")
    species = args.species    if args.species    != "mouse"  else yd.get("species", args.species)
    magnif  = args.magnification if args.magnification != "20x" else yd.get("magnification", args.magnification)

    overrides: dict = {}
    if fr    is not None: overrides["data.fr"]           = fr
    if decay is not None: overrides["data.decay_time"]   = decay
    if rf    is not None: overrides["cnmf.rf"]           = rf
    if args.K            is not None: overrides["cnmf.K"]           = args.K
    if args.min_corr     is not None: overrides["cnmf.min_corr"]    = args.min_corr
    if args.min_pnr      is not None: overrides["cnmf.min_pnr"]     = args.min_pnr
    if args.method_init  is not None: overrides["cnmf.method_init"] = args.method_init
    if args.n_processes  is not None: overrides["cluster.n_processes"] = args.n_processes
    overrides["data.n_channels"] = yd.get("n_channels", 1)
    overrides["data.n_planes"]   = yd.get("n_planes",   1)

    # Extract channel_id from session stem (e.g. "...-C01-fc015000" -> 1)
    _ch_match = re.search(r'-C(\d{2})-fc\d+', session)
    overrides["data.channel_id"] = int(_ch_match.group(1)) if _ch_match else 0

    if gSig is not None:
        overrides["cnmf.gSig"] = [gSig, gSig]
        overrides["cnmf.gSiz"] = [gSig * 4 + 1, gSig * 4 + 1]
        if rf is None and args.rf is None:
            rf_auto = 5 * gSig
            overrides.setdefault("cnmf.rf",     rf_auto)
            overrides.setdefault("cnmf.stride", rf_auto // 2)

    # ── Output paths ───────────────────────────────────────────────────────
    out_py   = dest / f"{session}_pipeline.py"
    out_json = dest / f"{session}_pipeline.json"

    # ── Conflict check ─────────────────────────────────────────────────────
    existing = [p for p in (out_py, out_json) if p.exists()]
    if existing and not args.force and not args.dry_run:
        print("The following files already exist:")
        for p in existing:
            print(f"  {p}")
        if input("Overwrite? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return 0

    # ── Patch JSON ─────────────────────────────────────────────────────────
    patched = _patch_json(tpl_json, session, dest,
                          overrides=overrides,
                          strip_comments=args.no_comments)

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("Session preparation")
    print("=" * 60)
    print(f"  Session    : {session}")
    print(f"  Dest       : {dest}")
    if args.estimate_params or args.run_mc:
        print(f"  Species    : {species}    Magnif: {magnif}")
    if overrides:
        print("  Overrides  :")
        for k, v in overrides.items():
            print(f"    {k} = {v}")
    print()
    print("  Files to write:")
    print(f"    {out_py}")
    print(f"    {out_json}")

    warn = _check_tif(dest, session)
    if warn:
        print(f"\n{warn}")

    if args.dry_run:
        print("\nDry run -- no files written.")
        return 0

    # ── Write ──────────────────────────────────────────────────────────────
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tpl_py, out_py)
    out_json.write_text(json.dumps(patched, indent=4) + "\n")
    print("\nDone.")

    # ── MC / param estimation ──────────────────────────────────────────────
    tif_ok = (dest / f"{session}.tif").exists()
    if args.estimate_params or args.run_mc:
        _apply_pipeline_env(patched.get("env", {}))
        try:
            tif_ok = _run_mc_and_estimate(args, session, dest, out_json)
        except Exception as exc:
            import traceback
            print(f"  Failed: {exc}")
            traceback.print_exc()
            tif_ok = False

    # ── Next steps ─────────────────────────────────────────────────────────
    print()
    print("Next steps:")
    print(f"  1. Review : {out_json.name}")
    print(f"  2. TIF    : {dest / (session + '.tif')}")
    print(f"  3. Params : new-session {session} {dest} --run-mc --estimate-params -y")
    print(f"  4. Run    : python {out_py.name}")
    print()

    # ── Optional pipeline run ──────────────────────────────────────────────
    if args.run:
        if not tif_ok:
            print("  Skipping --run: TIF not found.")
        else:
            import subprocess
            print(f"Running pipeline: {out_py}")
            print("=" * 60)
            result = subprocess.run([sys.executable, str(out_py)], cwd=str(dest))
            if result.returncode != 0:
                print(f"\n  Pipeline exited with code {result.returncode}")
                return result.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
