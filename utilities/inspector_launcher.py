#!/usr/bin/env python3
"""
utilities/inspector_launcher.py — installed launcher for the component inspector.

Registered as the ``caiman-inspector`` console script. The GUI itself lives in
``bin/caiman_inspector.py``; this thin launcher loads and runs it so there is a
stable installed entry point without moving the GUI into the package (same
pattern as ``caiman-batch-gui``).

    caiman-inspector /path/to/<session>_results.hdf5
    caiman-inspector /path/to/<session>.tif        # stack + its results
    caiman-inspector                               # synthetic demo
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_GUI = Path(__file__).resolve().parent.parent / "bin" / "caiman_inspector.py"


def _load_gui():
    if not _GUI.is_file():
        raise SystemExit(f"inspector not found: {_GUI}")
    spec = importlib.util.spec_from_file_location("caiman_inspector", _GUI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["caiman_inspector"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    return _load_gui().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
