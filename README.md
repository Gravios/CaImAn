# CaImAn drop-in memory & GPU fix — v2

Install by unpacking over your existing CaImAn installation:

    tar -xf caiman_dropin_mem_fix_v2.tar.gz --strip-components=1 \
        -C /path/to/caiman/install/

## Files changed

| File | Changes |
|---|---|
| `caiman/__init__.py` | Exposes `ensure_multipage_tiff`, `fc_convert_parallel`, `madvise_sequential` at top level |
| `caiman/utils/__init__.py` | New — exposes tiff_io public API from `caiman.utils` |
| `caiman/utils/tiff_io.py` | New — fast TIFF I/O utilities (madvise, parallel F→C, multipage conversion) |
| `caiman/gpu_motion_correction.py` | Parallel GPU MC: double-buffering, async mmap writes, PW-rigid batch registration |
| `caiman/motion_correction.py` | Routes GPU path through parallel implementation with serial fallback |
| `caiman/gpu_spatial.py` | **NEW** — GPU-accelerated spatial update via precomputed gram matrices |
| `caiman/summary_images.py` | Fix float64 OOM in `local_correlations_fft` (sum2_gpu) |
| `caiman/source_extraction/cnmf/cnmf.py` | Fix refit rf override; fix memmap detection after reshape |
| `caiman/source_extraction/cnmf/initialization.py` | greedyROI_corr mmap rewrite; init_neurons_corr_pnr copy elimination |
| `caiman/source_extraction/cnmf/map_reduce.py` | SHM RAM guard; F_tot offset fix (f_bgr_count); NMF n_components guard |
| `caiman/source_extraction/cnmf/pre_processing.py` | GPU batched rfft path in get_noise_fft (1p/CNMFe) |
| `caiman/source_extraction/cnmf/spatial.py` | GPU gram path for update_spatial_components; GPU Y@f.T residual |

## Key fixes

### Memory
- **SHM headroom guard** (`map_reduce.py`): uses `total − used − movie − worker_overhead`
  instead of `available`, preventing the 27 GB SHM segment from spilling to swap
- **F_tot offset** (`map_reduce.py`): `f_bgr_count` counter replaces `patch_id × nb_patch`,
  fixing silent background misalignment when any patch returns fewer components than `nb_patch`
- **NMF guard** (`map_reduce.py`): clamps `n_components` to `min(gnb, F_tot.shape[0])`
  so the NMF never crashes when edge patches return fewer background rows than `gnb`

### GPU acceleration
- **Spatial update** (`spatial.py` + `gpu_spatial.py`): precomputes `YC = Y @ Cf_scaled.T`
  on GPU in ~2 tiles (~1s), then solves each pixel's NNLS using a K×K gram system
  instead of the K×T system — ~2700× fewer FLOPs per pixel. Falls back to CPU silently.
- **get_noise_fft** (`pre_processing.py`): batched CuPy rfft replaces 262k serial cv2.dft
  calls on the full-FOV path (1p/CNMFe, fires when n_pixels > 4096 and T > 3072)
- **Y @ f.T residual** (`spatial.py`): background residual also computed on GPU

### Bug fixes
- `refit()` no longer overrides `rf=None`, preserving patch structure across refits
- Memmap detection uses `hasattr(images, 'filename')` instead of `isinstance(np.memmap)`
  so patched images after reshape still use the zero-copy Yr path

## Requirements
- CuPy matching your CUDA version (GPU paths fall back to CPU if unavailable):
  `conda install -n caiman cupy`
- psutil (already a CaImAn dependency)
- MKL_NUM_THREADS=1 / OMP_NUM_THREADS=1 recommended to prevent BLAS oversubscription
  across multiprocessing workers
