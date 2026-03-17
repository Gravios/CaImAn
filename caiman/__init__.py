#!/usr/bin/env python

import importlib.metadata

from caiman.base.movies import movie, load, load_movie_chain, _load_behavior, play_movie
from caiman.base.timeseries import concatenate
from caiman.cluster import start_server, stop_server
from caiman.keras_model_arch import keras_cnn_model_from_pickle
from caiman.mmapping import load_memmap, save_memmap, save_memmap_each, save_memmap_join
from caiman.summary_images import local_correlations
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

__version__ = importlib.metadata.version('caiman')
