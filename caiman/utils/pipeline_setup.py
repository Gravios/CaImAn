"""
caiman/utils/pipeline_setup.py
===============================
One-time pipeline infrastructure: path resolution, CNN model bootstrapping,
logger construction, and stale shared-memory cleanup.

``resolve_pipeline_path`` is the very first call in every pipeline script —
it must be importable before any caiman import (it has no caiman dependencies).

Usage
-----
    # ── Must come before any caiman import ─────────────────────────────────
    # The import itself is safe because pipeline_setup only uses stdlib.
    from caiman.utils.pipeline_setup import resolve_pipeline_path

    _SCRIPT_PATH, _CONFIG_PATH, session = resolve_pipeline_path()

    # ── After caiman import ─────────────────────────────────────────────────
    from caiman.utils.pipeline_setup import ensure_model_files, setup_logging, clean_stale_shm

    _cnn_available = ensure_model_files(os.path.join(CAIMAN_DATA, "model"))
    logger         = setup_logging(outdir / f"{session}.log")
    clean_stale_shm(CAIMAN_SHM, CAIMAN_TEMP, logger)
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Union

from caiman.utils.timing import log_call


# ── Script path resolution ───────────────────────────────────────────────────

def resolve_pipeline_path(
    suffix: str = "_pipeline",
    argv_index: int = 1,
) -> tuple[Path, Path, str]:
    """Resolve the pipeline script path, config path, and session name.

    Must be called before any caiman import — it has zero caiman dependencies
    and uses only the Python standard library.

    Resolution order for the script path:

    1. ``__file__`` from the calling frame — set by CPython for normal
       execution and IPython ``%run``.
    2. ``inspect.getfile`` walking up the call stack — works in Emacs
       ``python-el`` eval and any ``exec()`` context where ``__file__``
       is not injected.
    3. ``sys.argv[0]`` if it names an existing ``.py`` file.
    4. ``Path.cwd() / "pipeline.py"`` as a last resort.

    The config path is resolved as:

    - ``sys.argv[argv_index]`` if that argument is present (explicit override).
    - Otherwise ``<script_dir>/<script_stem>.json``.

    The session name is ``script_stem.removesuffix(suffix)``, so
    ``stroh-ej-20140708-TL2_pipeline.py`` → ``stroh-ej-20140708-TL2``.

    Parameters
    ----------
    suffix
        Suffix to strip from the script stem to form the session name
        (default ``"_pipeline"``).
    argv_index
        ``sys.argv`` index to check for an explicit config path override
        (default ``1``).

    Returns
    -------
    tuple[Path, Path, str]
        ``(script_path, config_path, session)``

        - *script_path* — absolute path of the pipeline ``.py`` file.
        - *config_path* — absolute path of the JSON config.
        - *session*     — session identifier string.

    Examples
    --------
    Place this at the very top of every pipeline script, before any
    ``import caiman``::

        from caiman.utils.pipeline_setup import resolve_pipeline_path

        _SCRIPT_PATH, _CONFIG_PATH, session = resolve_pipeline_path()
        _SCRIPT_DIR = _SCRIPT_PATH.parent
    """
    script_path = _find_script_path()
    script_dir  = script_path.parent

    if len(sys.argv) > argv_index:
        config_path = Path(sys.argv[argv_index]).resolve()
    else:
        config_path = script_dir / (script_path.stem + ".json")

    session = script_path.stem.removesuffix(suffix)
    return script_path, config_path, session


def _find_script_path() -> Path:
    """Return the absolute path of the calling pipeline script.

    Walks the call stack to find the outermost ``.py`` file that is not
    this module itself.  Internal helper for :func:`resolve_pipeline_path`.
    """
    _this_file = Path(__file__).resolve()

    # 1. __file__ from direct execution or IPython %run
    # Walk the call stack — __file__ is available on the frame globals.
    frame = inspect.currentframe()
    while frame is not None:
        fname = frame.f_globals.get("__file__")
        if fname:
            p = Path(fname).resolve()
            if p.suffix == ".py" and p.exists() and p != _this_file:
                return p
        frame = frame.f_back

    # 2. sys.argv[0] — set when run as `python script.py`
    if sys.argv and sys.argv[0]:
        p = Path(sys.argv[0]).resolve()
        if p.suffix == ".py" and p.exists():
            return p

    # 3. Last resort
    return Path.cwd() / "pipeline.py"


# Module-level logger used by @log_call decorators on the public functions.
# These decorators fire before the pipeline's per-session logger is configured,
# so they write to the root 'caiman' logger at its current level.
_module_logger = logging.getLogger('caiman')

# ── CNN model bootstrap ───────────────────────────────────────────────────────

@log_call(_module_logger, level=logging.INFO, show_result=True)
def ensure_model_files(model_dir: Union[str, Path]) -> bool:
    """Copy CNN classifier weights into *model_dir* if they are missing.

    CaImAn looks for ``cnn_model.pkl`` and ``cnn_model_online.pkl`` under
    ``CAIMAN_DATA/model/``.  On a fresh conda environment the files live in
    ``sys.prefix/share/caiman/model/`` — this function copies them once so
    the pipeline can run without a prior ``caimanmanager install``.

    Parameters
    ----------
    model_dir
        Destination directory, typically ``os.path.join(CAIMAN_DATA, "model")``.

    Returns
    -------
    bool
        ``True`` if both files are present (or were copied successfully),
        ``False`` if either file could not be located — the caller should pass
        this as ``cnn_available=False`` to :func:`~caiman.utils.params_io.build_cnmf_opts`.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    needed = ["cnn_model.pkl", "cnn_model_online.pkl"]

    import importlib.resources
    try:
        pkg_root = str(importlib.resources.files("caiman").joinpath(".."))
    except Exception:
        pkg_root = ""

    candidates = [
        os.path.join(sys.prefix, "share", "caiman", "model"),
        os.path.join(pkg_root, "model"),
    ]

    all_present = True
    for fname in needed:
        dst = model_dir / fname
        if dst.exists():
            continue
        copied = False
        for src_dir in candidates:
            src = Path(src_dir) / fname
            if src.exists():
                shutil.copy2(src, dst)
                print(f"[setup] Copied {fname} from {src_dir}")
                copied = True
                break
        if not copied:
            print(f"[setup] WARNING: {fname} not found — CNN classifier will be disabled")
            all_present = False

    return all_present


# ── Logger construction ───────────────────────────────────────────────────────

def setup_logging(
    logfile: Union[str, Path],
    *,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the ``"caiman"`` logger.

    Each call truncates *logfile* so every pipeline run starts at line 1.
    Existing handlers are removed first to prevent duplicate output when
    re-running in the same interpreter session (Emacs, Jupyter, IPython).

    Parameters
    ----------
    logfile
        Absolute path to the ``.log`` file.  Parent directory must exist.
    file_level
        Logging level for the file handler (default: ``DEBUG`` — verbose,
        includes internal CaImAn debug messages).
    console_level
        Logging level for the stderr handler (default: ``INFO``).

    Returns
    -------
    logging.Logger
        The configured ``"caiman"`` logger.
    """
    logfile = Path(logfile)

    logger = logging.getLogger("caiman")
    logger.setLevel(logging.DEBUG)  # handlers filter independently

    # Remove any handlers from a previous run in the same session
    for h in logger.handlers[:]:
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)

    # Truncate log file — fresh start every run
    logfile.write_text("")

    file_fmt = logging.Formatter(
        "%(asctime)s %(relativeCreated)12d "
        "[%(filename)s:%(funcName)20s():%(lineno)s] "
        "[%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(logfile, mode="a")
    fh.setLevel(file_level)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    return logger


# ── Stale shared-memory cleanup ───────────────────────────────────────────────

@log_call(_module_logger, level=logging.DEBUG, show_result=False)
def clean_stale_shm(
    shm_dir: Union[str, Path],
    temp_dir: Union[str, Path],
    logger: logging.Logger,
) -> None:
    """Remove stale CaImAn worker mmaps and SHM files from crashed runs.

    Safe to call at any point before starting a new CNMF cluster.  PID-tagged
    ``caiman_{pid}_*`` files are only deleted when the owning process no longer
    exists; all other transient SHM files (``psm_*``, ``sem.loky-*``,
    ``__KMP_REGISTERED_LIB_*``, ``_caiman_tile_*``, ``_caiman_filt_*``) are
    always removed.

    Parameters
    ----------
    shm_dir
        Shared-memory directory, typically ``/dev/shm``.
    temp_dir
        CaImAn temp directory, typically ``CAIMAN_TEMP``.
    logger
        Logger to write removal messages to.
    """
    import glob

    shm_dir  = str(shm_dir)
    temp_dir = str(temp_dir)

    caiman_mmaps = (
        glob.glob(os.path.join(shm_dir,  "caiman_*.mmap"))
        + glob.glob(os.path.join(shm_dir, "_caiman_tile_*.mmap"))
        + glob.glob(os.path.join(shm_dir, "_caiman_filt_*.mmap"))
        + glob.glob(os.path.join(temp_dir, "caiman_*.mmap"))
    )
    transient = (
        glob.glob(os.path.join(shm_dir, "psm_*"))
        + glob.glob(os.path.join(shm_dir, "sem.loky-*"))
        + glob.glob(os.path.join(shm_dir, "__KMP_REGISTERED_LIB_*"))
    )

    for path in caiman_mmaps:
        m = re.search(r"caiman_(\d+)_", os.path.basename(path))
        if m:
            pid = int(m.group(1))
            try:
                os.kill(pid, 0)
                continue           # process alive — leave it alone
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
        try:
            os.unlink(path)
            logger.info(f"Cleared stale worker mmap: {path}")
        except OSError:
            pass

    for path in transient:
        try:
            os.unlink(path)
            logger.info(f"Cleared stale SHM file: {path}")
        except OSError:
            pass
