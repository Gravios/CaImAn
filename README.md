# Gravios/CaImAn

A GPU-accelerated fork of [CaImAn](https://github.com/flatironinstitute/CaImAn) with memory-safe large-dataset processing and a structured pipeline framework for batch 2-photon calcium imaging analysis.

## Quick start

```bash
# Install
conda create -n caiman python=3.11 && conda activate caiman
pip install git+https://github.com/Gravios/CaImAn.git
pip install cupy-cuda12x dill psutil tifffile scikit-image

# Create a session (runs MC + estimates parameters automatically)
python pipelines/new_session.py \
    stroh-ej-20140708-TL2 \
    /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/08072014/ \
    -y --run-mc --estimate-params

# Run
cd /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/08072014/
python stroh-ej-20140708-TL2_pipeline.py
```

Full documentation: `docs/` or [ReadTheDocs](https://caiman.readthedocs.io).

---

## What this fork adds

### Pipeline framework (`pipelines/`)

| File | Purpose |
|---|---|
| `template_pipeline.py` | Ready-to-run pipeline script (copy + rename per session) |
| `template_pipeline.json` | Fully-commented parameter template |
| `new_session.py` | CLI: create session files, run MC, estimate parameters |

### `caiman/utils/` — new modules

| Module | Provides |
|---|---|
| `pipeline_setup.py` | `resolve_pipeline_path`, `setup_logging`, `ensure_model_files`, `clean_stale_shm` |
| `timing.py` | `PipelineTimer`, `@timer.step`, `@log_call`, `write_report` |
| `memory.py` | `malloc_trim`, `madvise_dontneed`, `cupy_flush` |
| `cnmf_runner.py` | `CNMFRunner` — config-driven fit → refit → evaluate → dF/F |
| `qc.py` | `QCRunner` + 8 standalone QC figure functions |
| `params_estimator.py` | `estimate_params`, `apply_suggestions` |
| `params_io.py` | `ParamBag`, `load_pipeline_params`, `build_cnmf_opts` |
| `tiff_io.py` | `ensure_multipage_tiff`, `fc_convert_parallel` |

### Core fixes (GPU + memory)

| File | Changes |
|---|---|
| `summary_images.py` | Chunked GPU `correlation_pnr`; float32 fix in `local_correlations_fft` |
| `motion_correction.py` | GPU path via `gpu_motion_correction.py`; `use_gpu` parameter |
| `source_extraction/cnmf/initialization.py` | Shape guards on all precomp arrays; mmap-based memory management |
| `source_extraction/cnmf/map_reduce.py` | SHM tile dispatcher; precomp cache transfer; exception barrier |
| `source_extraction/cnmf/pre_processing.py` | Batched CuPy rfft for noise estimation |
| `source_extraction/cnmf/spatial.py` | GPU gram-matrix spatial update |

See [docs/source/gpu_acceleration.rst](docs/source/gpu_acceleration.rst) for full details.

---

## Install from tarball (existing environment)

```bash
tar -xzf pipeline_refactor.tar.gz -C ~/software/CaImAn/
```

---

## Citation

> Giovannucci et al. (2019). CaImAn: An open source tool for scalable Calcium Imaging data Analysis. *eLife* 8:e38173. https://doi.org/10.7554/eLife.38173
