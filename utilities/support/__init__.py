"""
SUPPORT — self-supervised denoising for fluorescence imaging.

Vendored from NICALab/SUPPORT (https://github.com/NICALab/SUPPORT,
MIT-licensed) with a custom CLI layer and architecture-sidecar JSON
mechanism. See ``utilities/support/README.md`` for the import history
and what was changed.

Public API for use by pipeline code (``template_pipeline.py`` stage 1d):

    from utilities.support import denoise_stack, load_model, ModelConfig

    cfg = ModelConfig.load_for_checkpoint(checkpoint_path)
    model = load_model(checkpoint_path, cfg, use_cuda=True)
    denoise_stack(input_tiff, output_tiff, model,
                  patch_size=[61, 64, 64],
                  patch_interval=[1, 32, 32],
                  batch_size=8)

CLI usage from the repo root::

    python -m utilities.support train --noisy-data ... --exp-name ...
    python -m utilities.support test  --input ... --checkpoint ...
    python -m utilities.support info  --checkpoint ...
    python -m utilities.support list  --results-dir ...

Pretrained checkpoints (vendored from upstream's ``src/GUI/trained_models/``)
live in ``utilities/support/trained_models/``. To smoke-test the install
without training first::

    python -m utilities.support test \\
        --checkpoint utilities/support/trained_models/bs3.pth \\
        --bs-size 3 3 \\
        --input <session>_Ncorrected.tif
"""

from .cli.paths import (
    ModelConfig,
    add_architecture_arguments,
    checkpoint_path,
    default_output_path,
    experiment_dir,
    find_checkpoints,
    latest_checkpoint,
    resolve_checkpoint,
    resolve_input_paths,
)
from .cli.test import denoise_array, denoise_mmap, denoise_stack, load_model
from .network import SUPPORT as SUPPORTNet

__all__ = [
    "ModelConfig",
    "SUPPORTNet",
    "add_architecture_arguments",
    "checkpoint_path",
    "default_output_path",
    "denoise_array",
    "denoise_mmap",
    "denoise_stack",
    "experiment_dir",
    "find_checkpoints",
    "latest_checkpoint",
    "load_model",
    "resolve_checkpoint",
    "resolve_input_paths",
]
