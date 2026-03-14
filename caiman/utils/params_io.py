"""
caiman/utils/params_io.py
=========================
Elegant JSON parameter loading for CaImAn pipelines.

``ParamBag`` gives dot-access to nested dicts and auto-coerces JSON
lists to tuples so CaImAn never receives a plain list where it expects a
tuple (e.g. ``gSig``, ``max_shifts``).

``load_pipeline_params`` loads a JSON file and wraps it in a
``ParamBag``.

``build_cnmf_opts`` constructs a fully-configured ``CNMFParams`` object
from the bag, accepting the small set of runtime values that are only
known after motion correction (dims, bord_px, fname, …).

Usage
-----
    from caiman.utils.params_io import load_pipeline_params, build_cnmf_opts

    P    = load_pipeline_params("pipeline_p2.json")

    # Dot-access, any depth
    print(P.session.session_id)      # "stroh-sa-2966-…"
    print(P.cnmf.gSig)               # (6, 6)   ← tuple, not list
    print(P.motion_correction.pw_rigid)  # False

    # Build CNMFParams when the runtime values are ready
    opts = build_cnmf_opts(
        P,
        fname_cnmf    = fname_cnmf,
        dims          = dims,
        bord_px       = bord_px,
        n_processes   = n_processes,
        cnn_available = True,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger("caiman")


# ── ParamBag ──────────────────────────────────────────────────────────────────

class ParamBag:
    """Dot-access wrapper around a (possibly nested) dict.

    Lists are recursively converted to tuples so that every leaf value
    that came from a JSON array is already tuple-typed — matching what
    CaImAn's internal parameter validators expect.

    Attributes mirror the JSON keys; nested objects become nested
    ``ParamBag`` instances.

    The underlying dict is still accessible as ``bag._data`` and the bag
    can be iterated like a dict via ``bag.items()``.
    """

    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", {})
        for k, v in data.items():
            self._data[k] = self._wrap(v)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _wrap(value: Any) -> Any:
        """Recursively convert dicts → ParamBag, lists → tuples."""
        if isinstance(value, dict):
            return ParamBag(value)
        if isinstance(value, list):
            return tuple(ParamBag._wrap(v) for v in value)
        return value

    # ── attribute-style read / write ──────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, "_data")[name]
        except KeyError:
            raise AttributeError(
                f"ParamBag has no parameter '{name}'. "
                f"Available: {list(self._data)}"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = self._wrap(value)

    # ── dict-like iteration ───────────────────────────────────────────────────

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    # ── unwrap back to a plain dict (for serialisation) ───────────────────────

    def to_dict(self) -> dict:
        """Recursively convert back to a plain dict (lists, not tuples)."""
        out = {}
        for k, v in self._data.items():
            if isinstance(v, ParamBag):
                out[k] = v.to_dict()
            elif isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
        return out

    def __repr__(self) -> str:
        keys = list(self._data)
        return f"ParamBag({keys})"


# ── load_pipeline_params ──────────────────────────────────────────────────────

def load_pipeline_params(path: Union[str, Path]) -> ParamBag:
    """Load a pipeline JSON parameter file and return a ``ParamBag``.

    Parameters
    ----------
    path
        Path to the JSON file (absolute or relative to cwd).

    Returns
    -------
    ParamBag
        Nested bag; lists are already tuples, nested dicts are nested bags.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")
    with path.open() as fh:
        raw = json.load(fh)
    # Strip comment keys (conventionally "_comment")
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    bag = ParamBag(raw)
    logger.info(f"Parameters loaded from {path}")
    return bag


# ── build_cnmf_opts ───────────────────────────────────────────────────────────

def build_cnmf_opts(
    P: ParamBag,
    *,
    fname_cnmf: str,
    dims: tuple,
    bord_px: int,
    n_processes: int,
    cnn_available: bool = True,
) -> "caiman.source_extraction.cnmf.params.CNMFParams":  # noqa: F821
    """Build a ``CNMFParams`` object from a ``ParamBag`` and runtime values.

    All JSON-sourced parameters come from *P*; the small set of values
    that are only known after motion correction (file paths, spatial
    dimensions, border pixels, cluster size) are passed as keyword
    arguments.

    Parameters
    ----------
    P
        ParamBag produced by :func:`load_pipeline_params`.
    fname_cnmf
        Path to the C-order mmap file produced by F→C conversion.
    dims
        Spatial dimensions ``(d1, d2)`` from ``load_memmap``.
    bord_px
        Border pixels to zero out (0 when ``border_nan="copy"``).
    n_processes
        Number of worker processes from ``cm.cluster.setup_cluster``.
    cnn_available
        Whether the CNN classifier model files are present on disk.
        Gates ``use_cnn`` to avoid a hard crash if models are missing.

    Returns
    -------
    CNMFParams
        Fully configured params object, ready to pass to ``CNMF()``.
    """
    from caiman.source_extraction.cnmf.params import CNMFParams

    d  = P.data
    mc = P.motion_correction
    c  = P.cnmf
    q  = P.quality

    opts = CNMFParams()

    opts.set("data", {
        "fnames"     : [fname_cnmf],
        "fr"         : d.fr,
        "decay_time" : d.decay_time,
        "dims"       : dims,
    })

    # cluster JSON keys forwarded to the patch group.
    # ram_budget_frac   : fraction of vm.available allocated to patch workers (default 0.75).
    # worker_overhead_frac: multiplier on analytical per-worker RAM estimate (default 1.6).
    #   Lower to e.g. 1.1 if workers consistently use less RAM than estimated;
    #   raise if you see OOM kills.
    _cl = getattr(P, "cluster", None)
    _ram_frac      = float(getattr(_cl, "ram_budget_frac",        0.75)) if _cl else 0.75
    _overhead_frac = float(getattr(_cl, "worker_overhead_frac",   1.6))  if _cl else 1.6
    _blas_threads  = int(getattr(_cl,   "blas_threads_per_worker", 1))   if _cl else 1

    opts.set("patch", {
        "rf"                  : c.rf,
        "stride"              : c.stride,
        "n_processes"         : n_processes,
        "only_init"           : True,
        "p_patch"             : 0,
        "nb_patch"            : c.gnb,
        "border_pix"          : bord_px,
        "ram_budget_frac"      : _ram_frac,
        "worker_overhead_frac" : _overhead_frac,
        "blas_threads_per_worker": _blas_threads,
    })

    opts.set("init", {
        "K"          : c.K,
        "gSig"       : c.gSig,
        "gSiz"       : c.gSiz,
        "method_init": c.method_init,
        "ssub"       : c.ssub,
        "tsub"       : c.tsub,
        "nb"         : c.gnb,
        **({"min_corr": c.min_corr} if hasattr(c, "min_corr") else {}),
        **({"min_pnr":  c.min_pnr}  if hasattr(c, "min_pnr")  else {}),
    })

    opts.set("preprocess", {
        "p"          : 0,       # p=0 during initial fit; set to c.p at refit
    })

    opts.set("merging", {
        "merge_thr"  : c.merge_thr,
    })

    opts.set("spatial", {
        "nb"         : c.gnb,
    })

    opts.set("temporal", {
        "nb"                 : c.gnb,
        "method_deconvolution": c.method_deconv,
        "p"                  : 0,
    })

    opts.set("quality", {
        "min_SNR"    : q.min_SNR,
        "rval_thr"   : q.rval_thr,
        "use_cnn"    : q.use_cnn and cnn_available,
        "min_cnn_thr": q.min_cnn_thr,
        "cnn_lowest" : q.cnn_lowest,
    })

    logger.info(
        f"CNMFParams built: method_init={c.method_init}  K={c.K}  "
        f"gSig={c.gSig}  rf={c.rf}  p={c.p}  gnb={c.gnb}  "
        f"decay_time={d.decay_time}  n_proc={n_processes}"
    )
    return opts
