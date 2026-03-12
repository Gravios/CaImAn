# CaImAn GPU acceleration — drop-in patch

Drop these files into your fork of CaImAn, preserving the directory structure
shown below.  No other files need to be changed.

## File layout

```
caiman/
├── cpu_topology.py               ← NEW      cache-aware worker affinity
├── gpu_motion_correction.py      ← NEW      batched cuFFT motion correction
├── shared_memory_utils.py        ← NEW      POSIX shared-memory movie buffer
├── motion_correction.py          ← MODIFIED use_gpu param + SHM dispatch
├── summary_images.py             ← MODIFIED correlation_pnr gains GPU path
└── source_extraction/
    └── cnmf/
        └── map_reduce.py         ← MODIFIED ShmHandle-aware patch loading
```

## Installation

```bash
# from the root of your CaImAn fork
tar -xzf caiman_gpu_drop_in.tar.gz --strip-components=1

# GPU dependency (match your CUDA version)
pip install cupy-cuda12x     # CUDA 12.x
# pip install cupy-cuda11x   # CUDA 11.x
```

## What each file does

| File | Role |
|---|---|
| `gpu_motion_correction.py` | `motion_correction_piecewise_gpu()` — batched phase-correlation via cuFFT. Replaces the worker-pool entirely for the MC stage. Auto-batches frames to fit VRAM. |
| `shared_memory_utils.py` | `SharedMovieBuffer` — loads the movie into POSIX shared memory once; workers attach zero-copy via `ShmHandle`. Includes `SharedMemory.close()` monkey-patch to suppress harmless `BufferError` at GC. |
| `cpu_topology.py` | `cache_aware_chunk_order()` — reorders temporal chunks so workers sharing L3 process adjacent windows. `apply_affinity()` pins workers to physical cores. |
| `motion_correction.py` | Adds `use_gpu` to `MotionCorrect` and the batch functions. GPU branch injected before multiprocessing dispatch; CPU fallback gains SHM + cache-aware ordering. |
| `summary_images.py` | `correlation_pnr()` gains a `use_gpu` parameter (default `None` = auto-detect). GPU path replaces 262,144 serial `cv2.dft` calls with a single batched `cp.fft.rfft`. Falls back to original CPU code when CuPy is absent. All existing call sites work unchanged. |
| `map_reduce.py` | `cnmf_patches` accepts an `ShmHandle` in place of a file path for zero-copy patch reads during CNMF. |

## API changes

### `correlation_pnr` (summary_images.py)

```python
# Existing call — unchanged, auto-selects GPU if available
cn, pnr = caiman.summary_images.correlation_pnr(images, gSig=6, swap_dim=False)

# Explicit control
cn, pnr = caiman.summary_images.correlation_pnr(
    images, gSig=6, swap_dim=False,
    use_gpu=True,           # True / False / None (auto)
    noise_range=[0.25, 0.5],
    noise_method='mean',
)
```

### `MotionCorrect` (motion_correction.py)

```python
mc = MotionCorrect(fname, use_gpu=None, ...)   # None = auto-detect
mc = MotionCorrect(fname, use_gpu=True, ...)   # force GPU
mc = MotionCorrect(fname, use_gpu=False, ...)  # force CPU
```

## Expected performance (RTX 5070 Ti, 10k × 512×512)

| Stage | CPU | GPU |
|---|---|---|
| Motion correction | ~90 s | ~2–3 s |
| `correlation_pnr` (5× subsampled) | ~30 s | ~2 s |
