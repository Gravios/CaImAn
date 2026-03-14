"""
caiman.utils
============
Utility modules for the CaImAn calcium imaging analysis package.

Submodules
----------
image_preprocessing_keras   Keras image augmentation utilities
labelling                   Movie labelling helpers
nn_models                   Neural network model components
sbx_utils                   Scanbox file I/O
stats                       Robust statistics (mode, compressive NMF, …)
params_io                   JSON parameter loading with dot-access (ParamBag)
param_summary               Formatted parameter table logger
tiff_io                     Fast TIFF I/O for large NVMe-backed stacks
utils                       Miscellaneous helpers (download, SI metadata, …)
visualization               Component and patch visualisation (bokeh/holoviews)
"""

# tiff_io — commonly needed outside pipelines
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

__all__ = [
    # params_io
    "ParamBag",
    "load_pipeline_params",
    "build_cnmf_opts",
    # tiff_io
    "ensure_multipage_tiff",
    "fc_convert_parallel",
    "madvise_sequential",
]
