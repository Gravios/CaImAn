"""
param_summary.py — log a formatted parameter table from a pipeline ParamBag.

Usage in pipeline_p2.py:
    from param_summary import log_params
    log_params(_P, logger)
"""
import logging


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(str(x) for x in v) + ']'
    return str(v)


def log_params(P, logger: logging.Logger | None = None) -> None:
    """Print a formatted parameter table from a ParamBag to *logger*.

    Parameters
    ----------
    P:
        ParamBag produced by :func:`caiman.utils.params_io.load_pipeline_params`.
    logger:
        Logger to write to.  If None, uses ``logging.getLogger("caiman")``.
    """
    if logger is None:
        logger = logging.getLogger("caiman")

    def _get(section, key, default="—"):
        try:
            return getattr(getattr(P, section), key)
        except AttributeError:
            return default

    rows = [
        # (label, value, unit)
        ("SESSION",           "",                                              ""),
        ("  data_root",       _get("session", "data_root"),                   ""),
        ("  experiment",      _get("session", "experiment"),                  ""),
        ("  session_id",      _get("session", "session_id"),                  ""),
        ("DATA",              "",                                              ""),
        ("  fr",              _get("data", "fr"),                             "Hz"),
        ("  decay_time",      _get("data", "decay_time"),                     "s"),
        ("  add_baseline",    _get("data", "add_baseline"),                   ""),
        ("MOTION CORRECTION", "",                                              ""),
        ("  pw_rigid",        _get("motion_correction", "pw_rigid"),          ""),
        ("  max_shifts",      _fmt(_get("motion_correction", "max_shifts")),  "px"),
        ("  strides",         _fmt(_get("motion_correction", "strides")),     "px"),
        ("  overlaps",        _fmt(_get("motion_correction", "overlaps")),    "px"),
        ("  border_nan",      _get("motion_correction", "border_nan"),        ""),
        ("CNMF",              "",                                              ""),
        ("  method_init",     _get("cnmf", "method_init"),                    ""),
        ("  K",               _get("cnmf", "K"),                              "components/patch"),
        ("  gSig",            _fmt(_get("cnmf", "gSig")),                     "px"),
        ("  gSiz",            _fmt(_get("cnmf", "gSiz")),                     "px"),
        ("  rf",              _get("cnmf", "rf"),                             "px half-size"),
        ("  stride",          _get("cnmf", "stride"),                         "px"),
        ("  gnb",             _get("cnmf", "gnb"),                            ""),
        ("  p",               _get("cnmf", "p"),                              "AR order"),
        ("  merge_thr",       _get("cnmf", "merge_thr"),                      ""),
        ("  ssub / tsub",     f"{_get('cnmf','ssub')} / {_get('cnmf','tsub')}", ""),
        ("  method_deconv",   _get("cnmf", "method_deconv"),                  ""),
        ("QUALITY",           "",                                              ""),
        ("  min_SNR",         _get("quality", "min_SNR"),                     ""),
        ("  rval_thr",        _get("quality", "rval_thr"),                    ""),
        ("  min_cnn_thr",     _get("quality", "min_cnn_thr"),                 ""),
        ("  cnn_lowest",      _get("quality", "cnn_lowest"),                  ""),
        ("CLUSTER",           "",                                              ""),
        ("  n_processes",     _get("cluster", "n_processes", "auto"),         ""),
        ("  ram_budget_frac",         _get("cluster", "ram_budget_frac",        0.75), ""),
        ("  blas_threads/worker",      _get("cluster", "blas_threads_per_worker", 1),    ""),
    ]

    val_width = max(len(str(v)) for _, v, _ in rows if v != "")

    logger.info("=" * 60)
    logger.info("PIPELINE PARAMETERS")
    logger.info("=" * 60)
    for label, value, unit in rows:
        if value == "":
            logger.info(f"  {label}")
        else:
            unit_str = f"  [{unit}]" if unit else ""
            logger.info(f"    {label:<22} {_fmt(value):<{val_width}}{unit_str}")
    logger.info("=" * 60)
