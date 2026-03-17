"""
caiman.utils
============
Utility modules for the CaImAn calcium imaging analysis package.

Submodules
----------
image_preprocessing_keras   Keras image augmentation utilities
labelling                   Movie labelling helpers
memory                      malloc_trim / madvise_dontneed / cupy_flush
nn_models                   Neural network model components
param_summary               Formatted parameter table logger
params_io                   JSON parameter loading with dot-access (ParamBag)
pipeline_setup              resolve_pipeline_path, CNN model bootstrap, logging setup, stale SHM cleanup
sbx_utils                   Scanbox file I/O
stats                       Robust statistics (mode, compressive NMF, …)
tiff_io                     Fast TIFF I/O for large NVMe-backed stacks
cnmf_runner                 CNMFRunner: config-driven CNMF orchestrator
qc                          QC figure generation (raw, MC, Cn, footprints, evaluation, traces)
timing                      PipelineTimer context manager, step decorator, log_call, write_report
utils                       Miscellaneous helpers (download, SI metadata, …)
visualization               Component and patch visualisation (bokeh/holoviews)
"""

from caiman.utils.params_io import (
    ParamBag,
    load_pipeline_params,
    build_cnmf_opts,
)
from caiman.utils.tiff_io import (
    ensure_multipage_tiff,
    fc_convert_parallel,
    madvise_sequential,
)
from caiman.utils.param_summary import log_params
from caiman.utils.pipeline_setup import (
    resolve_pipeline_path,
    ensure_model_files,
    setup_logging,
    clean_stale_shm,
)
from caiman.utils.timing import (
    PipelineTimer,
    write_report,
    fmt_elapsed,
    log_call,
)
from caiman.utils.cnmf_runner import CNMFRunner as CnmfRunner
from caiman.utils.qc import (
    QCRunner,
    qc_raw_sample,
    qc_motion_correction,
    qc_correlation_image,
    qc_pnr_image,
    qc_cnmf_fit,
    qc_cnmf_refit,
    qc_component_evaluation,
    qc_traces,
    save_all_post_cnmf,
)
from caiman.utils.memory import (
    malloc_trim,
    madvise_dontneed,
    cupy_flush,
)

__all__ = [
    # params_io
    "ParamBag",
    "load_pipeline_params",
    "build_cnmf_opts",
    # tiff_io
    "ensure_multipage_tiff",
    "fc_convert_parallel",
    "madvise_sequential",
    # param_summary
    "log_params",
    # pipeline_setup
    "resolve_pipeline_path",
    "ensure_model_files",
    "setup_logging",
    "clean_stale_shm",
    # timing
    "PipelineTimer",
    "write_report",
    "fmt_elapsed",
    "log_call",
    # cnmf_runner
    "CnmfRunner",
    # qc
    "QCRunner",
    "qc_raw_sample",
    "qc_motion_correction",
    "qc_correlation_image",
    "qc_pnr_image",
    "qc_cnmf_fit",
    "qc_cnmf_refit",
    "qc_component_evaluation",
    "qc_traces",
    "save_all_post_cnmf",
    # memory
    "malloc_trim",
    "madvise_dontneed",
    "cupy_flush",
]
