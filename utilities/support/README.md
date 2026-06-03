# utilities/support — SUPPORT denoising (vendored)

Self-supervised denoising for fluorescence imaging. Vendored from
[NICALab/SUPPORT](https://github.com/NICALab/SUPPORT) with a custom
CLI layer.

## Provenance

- Upstream: <https://github.com/NICALab/SUPPORT> at HEAD ``9fba0f4``
  (audited 2026-05-31).
- Restructured for CLI use and Python 3.11+ / NumPy 2.x compatibility
  (work tracked in conversation: SUPPORT-restructured v1.3.0).
- Vendored into CaImAn fork as ``utilities/support/`` for pipeline
  integration.

## Layout

```
utilities/support/
├── __init__.py             # public API exports
├── __main__.py             # `python -m utilities.support`
├── README.md               # this file
├── network.py              # SUPPORT 3D-UNet architecture
├── conv_layers.py          # partial-conv blind-spot layers
├── dataset.py              # train + stitched-inference Dataset classes
├── cli/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py             # argparse dispatch (train, test, ...)
│   ├── paths.py            # ModelConfig sidecar + path resolution
│   ├── train.py            # `support train` subcommand
│   ├── test.py             # `support test` subcommand
│   └── test_batch.py       # `support test-batch` subcommand
└── trained_models/         # shipped pretrained .pth (5.4 MB each)
    ├── bs1.pth             # 1x1 blind-spot — old voltage-imaging
    ├── bs3.pth             # 3x3 blind-spot — recommended GCaMP default
    ├── L1_generalization.pth
    └── zebrafish_voltage.pth
```

## Modifications from upstream

1. **CLI layer** (``cli/``) added: argparse-driven ``train`` / ``test`` /
   ``test-batch`` / ``info`` / ``list`` subcommands. Upstream had only
   ``src/train.py`` (with argparse) and ``src/test.py`` /
   ``test_directory.py`` (hardcoded paths in ``__main__`` blocks).

2. **Architecture sidecar JSON**: every checkpoint saved via
   ``support train`` is accompanied by a ``model_N.json`` capturing the
   architecture (``mid_channels``, ``bs_size``, etc.) and provenance
   (``training_data``, ``exp_name``, ``epoch``). Inference reads this
   automatically — no more manually matching ``--unet-channels`` between
   train and test invocations, which was a real upstream footgun
   (train default = ``[64,128,256,512,1024]``, test hardcoded =
   ``[16,32,64,128,256]``).

3. **Bug fixes** caught during the restructure audit:
   - ``test.py`` ``denoised[pad:-pad]`` silently produces an empty array
     when ``pad==0`` (2D mode) — fixed with explicit guard.
   - ``test.py`` ``args.patch_size or cfg.patch_size`` fallback never
     fires because argparse default ``[61,64,64]`` is truthy — argparse
     default changed to ``None``.
   - ``--start-epoch`` resume did not load the sidecar — now does, with
     a loud warning on architecture drift.
   - ``_expand_edges(mirror)`` silently failed for very short stacks —
     now raises ``ValueError`` with a helpful message.
   - Stack with ``T < model.in_channels`` produced garbage output —
     now raises ``ValueError`` upfront.

4. **Architectural cleanup**: ``ModelConfig`` is the single source of
   truth for architecture. ``ModelConfig.from_namespace(args)`` and
   ``.to_model_kwargs()`` keep argparse and the model constructor in
   sync. The duplicated ``_build_model_from_config`` /
   ``_build_model_from_args`` functions are gone.

5. **Python 3.11+ / NumPy 2.x**: dropped ``from __future__ import
   annotations``, adopted ``typing.Self``, ``@dataclass(slots=True)``,
   ``match/case``. Verified NumPy 2.x compatibility — no removed-API
   usage anywhere in this subpackage.

6. **Vendored layout**: import paths rewritten from upstream's
   ``from src.utils.dataset import ...`` / ``from model.SUPPORT import
   ...`` to use relative imports within this subpackage. The
   ``get_coordinate()`` helper from upstream's ``src/utils/util.py`` is
   inlined into ``dataset.py`` (it was the only function used; the
   rest of ``util.py`` was legacy CLI argparse that's superseded here).

7. **Dropped from upstream**: ``src/train.py``, ``src/test.py``,
   ``src/test_directory.py`` (replaced by the new CLI), ``src/GUI/``
   (PyQt GUIs — broken on headless servers anyway), ``src/utils/util.py``
   (only ``get_coordinate`` was needed, inlined), ``env.yml``,
   ``pyproject.toml`` (no longer a standalone package), ``colab/``,
   ``docs/``.

## Public API

```python
from utilities.support import (
    ModelConfig,        # architecture + provenance dataclass
    denoise_stack,      # single-TIFF inference function
    load_model,         # build SUPPORT net + load weights
    resolve_checkpoint, # find .pth by --exp-name or explicit path
)
```

See ``utilities/pipelines/template_pipeline.py`` stage 1d for the
canonical pipeline integration.

## CLI

```bash
# From the CaImAn repo root:
python -m utilities.support train  --noisy-data ... --exp-name ...
python -m utilities.support test   --input ... --checkpoint ...
python -m utilities.support info   --checkpoint ...
python -m utilities.support list   --results-dir ...
```

## License

NICALab/SUPPORT is MIT-licensed. This vendored copy retains the upstream
copyright; see the upstream LICENSE file at
<https://github.com/NICALab/SUPPORT/blob/main/LICENSE>.
