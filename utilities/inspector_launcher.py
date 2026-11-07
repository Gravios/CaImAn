#!/usr/bin/env python3
"""
utilities/inspector_launcher.py — installed launcher for the component inspector.

Registered as the ``caiman-inspector`` console script.  The GUI is the
``utilities.inspector`` package (``bin/caiman_inspector.py`` is a shim).

    caiman-inspector /path/to/<session>_results.hdf5
    caiman-inspector /path/to/<session>.tif        # stack + its results
    caiman-inspector                               # session in the current folder
    caiman-inspector --demo                        # synthetic demo
"""
from __future__ import annotations


def main(argv=None):
    from utilities.inspector import main as _main
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
