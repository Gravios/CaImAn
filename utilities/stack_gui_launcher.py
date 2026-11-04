#!/usr/bin/env python3
"""
utilities/stack_gui_launcher.py — installed launcher for the stack-sessions GUI.

Registered as the ``stack-sessions-gui`` console script. The GUI itself lives
in ``bin/stack_sessions_gui.py`` (next to the other GUIs); this thin launcher
loads and runs it so there is a stable installed entry point.

    stack-sessions-gui /data/source/strohA/.../strohA-ia-000000-20150709
    stack-sessions-gui --prefix strohA-ia      # current directory
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_GUI = Path(__file__).resolve().parent.parent / "bin" / "stack_sessions_gui.py"


def _load_gui():
    if not _GUI.is_file():
        raise SystemExit(f"stack-sessions GUI not found: {_GUI}")
    spec = importlib.util.spec_from_file_location("stack_sessions_gui", _GUI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stack_sessions_gui"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    return _load_gui().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
