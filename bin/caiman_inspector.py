#!/usr/bin/env python
"""CaImAn Component Inspector — compatibility shim.

The inspector lives in the ``utilities.inspector`` package (see its
docstring for usage and layout).  This file keeps ``python
bin/caiman_inspector.py …`` working and re-exports every name, so code that
loads it by path keeps working.  Patch names in their defining module
(e.g. ``utilities.inspector.theme._system_scheme``), not here.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities.inspector import *          # noqa: E402,F401,F403
from utilities.inspector import main       # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
