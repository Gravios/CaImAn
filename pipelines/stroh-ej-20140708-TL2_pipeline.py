"""
CaImAn 2-photon pipeline: motion correction → CNMF → dF/F
===========================================================

Naming convention
-----------------
The session name is derived from this script's own filename by stripping the
``_pipeline`` suffix.  Rename the script and JSON together to process a new
session — all output files (log, QC images, mmaps, results, report) follow
automatically.

    stroh-ej-20140708-TL2_pipeline.py   ← rename this
    stroh-ej-20140708-TL2_pipeline.json ← and this
    stroh-ej-20140708-TL2.tif           ← input (unchanged)

See docs/ for full parameter reference and troubleshooting guide.
"""

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Path resolution and env application MUST happen before any caiman import so
# that CAIMAN_DATA / CAIMAN_TEMP are set before CaImAn touches the filesystem.
import os
import sys
import json as _j
import inspect as _i
from pathlib import Path


def _resolve_script_path() -> Path:
    """Return this script's absolute path under all invocation styles."""
    try:
        p = Path(__file__).resolve()
        if p.suffix == ".py" and p.exists():
            return p
    except NameError:
        pass
    try:
        frame = _i.currentframe()
        while frame is not None:
            p = Path(_i.getfile(frame)).resolve()
            if p.suffix == ".py":
                return p
            frame = frame.f_back
    except (TypeError, OSError):
        pass
    return Path.cwd() / "pipeline.py"


_SCRIPT_PATH = _resolve_script_path()
_SCRIPT_DIR  = _SCRIPT_PATH.parent
_CONFIG_PATH = (
    Path(sys.argv[1]).resolve() if len(sys.argv) > 1
    else _SCRIPT_DIR / (_SCRIPT_PATH.stem + ".json")
)

# Apply env section before any caiman import
try:
    _env = _j.load(open(_CONFIG_PATH)).get("env", {})
except Exception as _e:
    import warnings
    warnings.warn(f"{_SCRIPT_PATH.name}: env load failed: {_e}")
    _env = {}

_FORCE = {"CAIMAN_DATA", "CAIMAN_TEMP", "CAIMAN_SHM", "CAIMAN_TILE_SLOTS"}
for _k, _v in _env.items():
    if _k in _FORCE:
        os.environ[_k] = str(_v)
    else:
        os.environ.setdefault(_k, str(_v))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

CAIMAN_DATA = os.environ.get("CAIMAN_DATA", "/data/caiman")
CAIMAN_TEMP = os.environ.get("CAIMAN_TEMP", "/data/caiman/temp")
CAIMAN_SHM  = os.environ.get("CAIMAN_SHM",  "/dev/shm")
for _d in [CAIMAN_DATA, os.path.join(CAIMAN_DATA, "model"), CAIMAN_TEMP]:
    os.makedirs(_d, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
import gc
import glob
import warnings

import numpy as np
import cv2
try:
    cv2.setNumThreads(0)
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in remainder",
    category=RuntimeWarning,
    module=r"scipy\.sparse\._dia",
)

import caiman as cm
import caiman.mmapping
import caiman.summary_images as csi
from caiman.motion_correction import MotionCorrect

from caiman.utils.tiff_io        import ensure_multipage_tiff, fc_convert_parallel
from caiman.utils.params_io      import load_pipeline_params
from caiman.utils.param_summary  import log_params
from caiman.utils.pipeline_setup import ensure_model_files, setup_logging, clean_stale_shm
from caiman.utils.timing         import PipelineTimer, write_report
from caiman.utils.memory         import malloc_trim, cupy_flush
from caiman.utils.cnmf_runner    import CNMFRunner
from caiman.utils.qc             import QCRunner

import dill as _dill
import multiprocessing.reduction as _mpr
_mpr.ForkingPickler.dumps = _dill.dumps

# ── Session identity ──────────────────────────────────────────────────────────
_P      = load_pipeline_params(_CONFIG_PATH)
session = _SCRIPT_PATH.stem.removesuffix("_pipeline")
datsrc  = Path(_P.session.data_root)
expsrc  = Path(_P.session.experiment)
outdir  = datsrc / expsrc

# ── Infrastructure ────────────────────────────────────────────────────────────
logger         = setup_logging(outdir / f"{session}.log")
_cnn_available = ensure_model_files(os.path.join(CAIMAN_DATA, "model"))
log_params(_P, logger, session=session)

ADD_BASELINE = _P.data.add_baseline
border_nan   = _P.motion_correction.border_nan

timer = PipelineTimer(logger)
qc    = QCRunner(_P, session, outdir)

# ── 1. TIFF check ─────────────────────────────────────────────────────────────
fnames = str(outdir / f"{session}.tif")

@timer.step("TIFF format check")
def check_tiff():
    global fnames
    fnames = ensure_multipage_tiff(fnames)

@timer.step("QC: raw sample")
def qc_raw():
    qc.raw_sample(fnames)

check_tiff()
qc_raw()

# ── 2. Motion correction ──────────────────────────────────────────────────────
_mc_existing = sorted(glob.glob(
    os.path.join(CAIMAN_TEMP, f"*{session}*rig*order_F*.mmap")))

if _mc_existing:
    fname_mc, shifts_rig = _mc_existing[-1], [(0, 0)]
    logger.info(f"Reusing MC mmap: {fname_mc}")
else:
    mc = MotionCorrect(fnames, dview=None, use_gpu=True, nonneg_movie=True,
                       **{k: v for k, v in _P.motion_correction.items()
                          if not k.startswith("_")})

    @timer.step("Motion correction")
    def run_mc():
        mc.motion_correct(save_movie=True)

    @timer.step("QC: motion correction")
    def qc_mc():
        qc.motion_correction(mc)

    run_mc()
    fname_mc, shifts_rig = mc.mmap_file[0], mc.shifts_rig
    logger.info(f"MC done: {fname_mc}")
    qc_mc()
    del mc

bord_px = 0 if border_nan == "copy" else int(np.ceil(np.max(np.abs(shifts_rig))))

# ── 3. F→C mmap conversion ────────────────────────────────────────────────────
_cnmf_existing = sorted(glob.glob(
    os.path.join(CAIMAN_TEMP, f"*{session}_cnmf*order_C*.mmap")))

if _cnmf_existing and os.path.getmtime(_cnmf_existing[-1]) >= os.path.getmtime(fname_mc):
    fname_cnmf = _cnmf_existing[-1]
    _, dims, T = cm.mmapping.load_memmap(fname_mc)
    logger.info(f"Reusing C-order mmap: {fname_cnmf}")
else:
    Yr_F, dims, T = cm.mmapping.load_memmap(fname_mc)
    n_px = int(np.prod(dims))
    fname_cnmf = cm.paths.fn_relocated(
        cm.paths.memmap_frames_filename(session + "_cnmf", dims, T, "C"),
        force_temp=True,
    )
    Yr_C = np.memmap(fname_cnmf, mode="w+", dtype=np.float32,
                     shape=cm.mmapping.prepare_shape((n_px, T)), order="C")

    @timer.step("F→C mmap conversion")
    def convert():
        fc_convert_parallel(Yr_F, Yr_C, n_px, T, ADD_BASELINE, logger)

    convert()
    del Yr_C, Yr_F

# ── 4. Correlation image ──────────────────────────────────────────────────────
Yr, dims, T = cm.mmapping.load_memmap(fname_cnmf)
images = np.reshape(Yr.T, [T] + list(dims), order="F")
images.filename = Yr.filename
logger.info(f"Data: {images.shape}  dtype={images.dtype}")

cupy_flush(logger, label="before summary-image step")

@timer.step("Correlation image (Cn)")
def compute_cn():
    global Cn
    Cn = csi.local_correlations_fft(images[::5], swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    np.save(str(outdir / f"{session}_Cn.npy"), Cn)

@timer.step("QC: correlation image")
def qc_cn():
    qc.correlation_image(Cn)

compute_cn()
qc_cn()

# Release Cn-step RSS so filt_full writes don't compete for page cache
del images, Yr
gc.collect()
malloc_trim(logger)
Yr, dims, T = cm.mmapping.load_memmap(fname_cnmf)
images = np.reshape(Yr.T, [T] + list(dims), order="F")
images.filename = Yr.filename

# ── 5. CNMF ───────────────────────────────────────────────────────────────────
clean_stale_shm(CAIMAN_SHM, CAIMAN_TEMP, logger)
logger.info("Starting CNMF cluster")
_, dview, n_processes = cm.cluster.setup_cluster(
    backend="multiprocessing",
    n_processes=getattr(_P, "cluster", None) and _P.cluster.n_processes or None,
    single_thread=False,
)

runner = CNMFRunner(
    _P, session, outdir,
    fname_mc      = fname_mc,
    fname_cnmf    = fname_cnmf,
    dims          = dims,
    bord_px       = bord_px,
    dview         = dview,
    n_processes   = n_processes,
    cnn_available = _cnn_available,
)

try:
    cnm2 = runner.run_all(images, Yr=Yr, qc=qc, Cn=Cn, timer=timer)
    del images, Yr, Cn

finally:
    logger.info("Stopping cluster")
    cm.stop_server(dview=dview)

# ── 6. Report ─────────────────────────────────────────────────────────────────
write_report(timer, session, outdir, logger)
