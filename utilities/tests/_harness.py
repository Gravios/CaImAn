"""Shared header for the suites in this folder.

Each suite is a plain script (``python utilities/tests/suite_x.py``) that
prints ``ok …`` lines and ``ALL OK``; ``test_suites.py`` runs every suite in
its own process under pytest.  The harness lets them run without the
compiled CaImAn stack:

* :func:`stub_caiman` — a bare ``caiman`` package whose ``caiman.utils`` is
  this checkout's ``caiman/utils`` folder (its ``__init__`` and the rest of
  ``caiman`` are never imported); the repo goes on ``sys.path``.
* :func:`load_inspector` — the above, offscreen Qt and a throw-away
  preferences file (unless the suite set one), then ``utilities.inspector``.
* :func:`light_root` — the same stand-in as a folder, for child processes.
* :func:`require` — exit with :data:`SKIP` when a dependency is missing.
"""
import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

SKIP = 77                                   # exit status: suite skipped
REPO = Path(os.environ.get("CAIMAN_REPO") or Path(__file__).resolve().parents[2])


def ok(msg: str) -> None:
    print("ok ", msg, flush=True)


def require(*modules: str) -> None:
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as exc:            # ImportError, or a broken install
            print(f"SKIP: {m} not importable ({exc})")
            sys.exit(SKIP)


def stub_caiman() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if getattr(sys.modules.get("caiman"), "_suite_stub", False):
        return
    cm = types.ModuleType("caiman"); cm.__path__ = []; cm._suite_stub = True
    cu = types.ModuleType("caiman.utils"); cu.__path__ = [str(REPO / "caiman" / "utils")]
    cm.utils = cu
    sys.modules["caiman"] = cm
    sys.modules["caiman.utils"] = cu


def load_inspector():
    require("numpy", "scipy", "PyQt6.QtWidgets", "pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("CAIMAN_INSPECTOR_PREFS",
                          str(Path(tempfile.mkdtemp()) / "inspector_prefs.json"))
    stub_caiman()
    import utilities.inspector as insp
    return insp


def light_root() -> Path:
    """A temp folder with ``caiman/`` (empty inits, caiman/utils/*.py linked
    from this checkout) and ``utilities`` linked — put it on PYTHONPATH of
    a child process instead of the repo."""
    root = Path(tempfile.mkdtemp(prefix="caiman_light_"))
    utils = root / "caiman" / "utils"
    utils.mkdir(parents=True)
    (root / "caiman" / "__init__.py").write_text("")
    (utils / "__init__.py").write_text("")
    for f in (REPO / "caiman" / "utils").glob("*.py"):
        if f.name != "__init__.py":
            _link(f, utils / f.name)
    _link(REPO / "utilities", root / "utilities")
    return root


def _link(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:                         # no symlinks (Windows): copy
        import shutil
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
