#!/usr/bin/env python3
"""
utilities/batch_gui_launcher.py — installed launcher for the batch GUI.

Registered as the ``caiman-batch-gui`` console script. The GUI itself lives in
``bin/caiman_batch_gui.py`` (next to caiman_inspector.py); this thin launcher
loads and runs it so there is a stable installed entry point without moving
the GUI into the package.

    caiman-batch-gui /data/source/ms2p/strohA/strohA-sa
    caiman-batch-gui            # no arg -> folder picker
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_GUI = Path(__file__).resolve().parent.parent / "bin" / "caiman_batch_gui.py"


def _load_gui():
    if not _GUI.is_file():
        raise SystemExit(f"batch GUI not found: {_GUI}")
    spec = importlib.util.spec_from_file_location("caiman_batch_gui", _GUI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["caiman_batch_gui"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    return _load_gui().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
