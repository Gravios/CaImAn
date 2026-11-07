"""
CaImAn Component Inspector  (Qt6 / PyQt6)
==========================================
Interactive GUI for reviewing, merging, and deleting CNMF components.

Requires
--------
    PyQt6, pyqtgraph, numpy, scipy
    caiman, h5py — loading / saving a results .hdf5
    tifffile     — movie playback and image backgrounds (optional)

Usage
-----
    caiman-inspector                         # session in the current folder:
                                             #   <dir>_results.hdf5 + <dir>.tif
    caiman-inspector /path/to/<session>/     # same, for another folder
    caiman-inspector /path/to/results.hdf5
    caiman-inspector /path/to/<session>.tif  # stack + its results
    caiman-inspector --demo                  # synthetic demo
    (python bin/caiman_inspector.py … or python -m utilities.inspector …
     work the same)

Package layout (utilities/inspector/)
-------------------------------------
    constants  trace keys, history depth, overlay colours / opacity
    theme      UI / plot colour tables, colour maps, palette + font
    prefs      preferences file and the editor's field tables
    deps       lazy caiman.utils.event_detection / roi_filter loaders
    store      ComponentStore: row-aligned model, edits, undo, caches, save
    loading    results / stack / session loading, backgrounds, demo data
    movie      raw-movie access for playback
    review     event-table export, blinded review files
    overlay    RGBA footprint overlays
    views      cell viewer, movie bar, trace / event views, matrices
    table      sortable component table
    dialogs    Preferences, Run ROI filter, Help
    window     InspectorWindow
    cli        caiman-inspector entry point

Controls
--------
    Rows         — every component of the file is loaded and saved.  Eval
                   shows CaImAn's verdict (✓ idx_components / census active,
                   ✗ rejected / census inactive); View ▸ Hide rejected /
                   inactive ROIs hides them (they are still saved).
    Table        — click to select; Ctrl/Shift for multi-select;
                   click a column header to sort; right-click it (or
                   View ▸ Columns) to show / hide columns.  ROI-filter columns
                   (QC, Active, Fail, Area, Solid, Ecc, Edge, Rate,
                   Pk ΔF/F, Skew, Act %) appear when the results file has a
                   roi_filter group; Incl then starts from the QC verdict.
                   Analysis ▸ Run ROI filter… (Ctrl+Shift+F) adds or
                   recomputes them, e.g. for sessions processed before the
                   pipeline step existed; thresholds live in Edit ▸
                   Preferences… (Ctrl+,), ~/.config/caiman/inspector_prefs.json
    P / H        — Preferences (theme, plot background, font size, ROI overlay
                   opacity, ROI-filter
                   thresholds) / Help with every shortcut and a usage guide
                   (also Ctrl+, and F1).
    Layout       — window and pane sizes, hidden columns, sort, view toggles,
                   background / trace choice and the last folder are
                   remembered in the same file (View ▸ Reset layout).
    Movie bar    — under the cell viewer: play / scrub the raw stack
                   (<session>.tif / .msr next to the results is found;
                   View > Open movie… / Ctrl+M for another).  With the
                   image focused: Space play/pause, ←/→ ±1, Shift+←/→ ±10,
                   Home/End, Shift+wheel steps frames.  Drag the yellow
                   cursor in the trace view to seek.
    Matrices     — Ctrl+click an off-diagonal cell of the correlation or
                   distance matrix to select that pair (orange / magenta)
    ⊕ Merge      — merge ≥ 2 selected (footprint = pixel max, trace = mean)
    ✕ Delete     — permanently remove selected
    ↩ Undo       — Ctrl+Z  — undo the last merge, delete or ROI-filter run
                             (up to 5 steps)
    ↪ Redo       — Ctrl+Y  — redo last undone operation
    File > Save  — write every component (A / C, Eval split, events,
                   include flags, roi_filter fields) to <name>_curated.hdf5

Event detection & curation (ported from SLabAnalysisPackage)
-------------------------------------------------------------
    Analysis > Detect events  — Ctrl+D — SLab peak detection on ΔF/F
                                (caiman.utils.event_detection); include
                                flags start at the automatic approval
    ✓ Include / exclude       — I — non-destructive approval; excluded
                                footprints are red ("Hide excluded" hides them)
    Event panel               — filtered ΔF/F, threshold, peaks, binary trace
                                of the last-selected component
    Analysis > Export         — Ctrl+E — <session>_PeakStats / _ROIStats
    Analysis > Batch detect   — stroh_results into every file of a folder
    Analysis > Blinded review — Ctrl+R — shuffled, name-blind rating;
                                Ctrl+Return saves SLabRandomization/<RATER>
    Events and include flags are saved as an SLab-compatible stroh_results
    group, so SLabAnalysisApp.py opens curated files.
"""

import os

# pyqtgraph: PyQt6 binding and image orientation, before any view is built
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")
import pyqtgraph as _pg  # noqa: E402

_pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)

from . import (  # noqa: E402
    constants, theme, deps, prefs, store, loading, movie, review,
    overlay, views, table, dialogs, window, cli,
)
from .constants import *  # noqa: E402,F401,F403
from .theme import *  # noqa: E402,F401,F403
from .deps import *  # noqa: E402,F401,F403
from .prefs import *  # noqa: E402,F401,F403
from .store import *  # noqa: E402,F401,F403
from .loading import *  # noqa: E402,F401,F403
from .movie import *  # noqa: E402,F401,F403
from .review import *  # noqa: E402,F401,F403
from .overlay import *  # noqa: E402,F401,F403
from .views import *  # noqa: E402,F401,F403
from .table import *  # noqa: E402,F401,F403
from .dialogs import *  # noqa: E402,F401,F403
from .window import *  # noqa: E402,F401,F403
from .cli import *  # noqa: E402,F401,F403

_MODULES = (constants, theme, deps, prefs, store, loading, movie, review,
            overlay, views, table, dialogs, window, cli)

# Every name of every module, private ones included (the bin/ shim and the
# tests reach them).  Monkeypatch a name in its defining module, e.g.
# utilities.inspector.theme._system_scheme — rebinding it here does nothing.
__all__ = sorted({n for m in _MODULES for n in m.__all__}
                 | {m.__name__.rsplit(".", 1)[-1] for m in _MODULES})
