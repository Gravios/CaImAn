Performance Guide
========================
Do not start with this guide.
* First, please get Caiman working with your data, using the minimal set of changes to defaults needed to verify that it is working
* Second, become familiar with the Caiman software; without that it will be hard for you to know what to change and why

This guide will change according to feedback and common questions, and is designed to address both CPU optimisation and memory optimisation. As of this writing, the guide is minimal; more substantial advice will be filled in over time.

Online versus offline
---------------------
The first choice with Caiman is whether to use it in online or offline mode. This is not purely a performance concern - the features available between the two modes are not identical, with the online mode having some newer features and having generally better performance. The difference is covered better elsewhere (including the Caiman paper), but in brief:
* data passes through the offline mode in a series of steps that run to completion before the next
* data passes through the online mode streaming through those steps, where a chunk of data may be entering an early stage (like motion correction) while other chunks may already be in a later stage (like signal extraction)

Clustering options
------------------
Caiman supports running in linear mode (which is very slow and useful mainly for debugging), ipyparallel, and multiprocessing. You can also set the number of processes Caiman will use (when not in linear mode). Clustering options are set in the call to `cm.cluster.setup_cluster()` and are one knob you will want to look at (although note that you may find bugs with unusual settings, or find that some settings require an unreasonable amount of RAM with your data).

Patch size
----------
Patch size is the other side of clustering options, and specify the granularity of breaking down the data for processing. This setting must be set in ways that fit your clustering options; spinning up a number of worker processes will not be helpful if there are not sufficient patches to hand out to them for work. Patches are the unit of potential work in most stages in the Caiman pipeline (online or offline).

Alternate builds of dependencies and unsupported environment variables
----------------------------------------------------------------------
If you cannot get the performance (or memory usage) you need with Caiman, you may be able to explore alternate versions of Caiman dependencies in the conda environment, as well as some of the undocumented (such as GPU-based function) features in the Caiman codebase. This is only for the adventurous, as we will not be able to support such features very readily, and our environment might not match yours. If you do this, please familiarise yourself with the codebase first.

If you have suitable GPU hardware, look into enabling GPU support in tensorflow, by reading the README-GPU.md doc.

For some of these alternative dependencies, they may be installable through conda. Some platforms, for example, have alternate builds of math libraries that either will use CPU instructions specific to certain CPUs, or some OS parallelisation settings. Look in particular at `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`, but know that adjustments to these can cause the code to hang during processing. 

GPU acceleration
----------------

This fork adds GPU paths for the most expensive pipeline stages.  All GPU
paths fall back to CPU silently when CuPy is not available.

**Motion correction** (`use_gpu=True`) uses a batched cuFFT phase-correlation
implementation.  Typical speedup: 30–45× over the CPU multiprocessing path
for a 512×512 × 5000 frame movie.

**Full-FOV filter precompute** replaces per-patch CPU filtering.  The GPU
filters the entire FOV once in chunks of `precompute_chunk_frames` frames
and writes a float16 mmap to `/dev/shm`.  Workers copy their slice from RAM
instead of running T filter calls per patch.

Tune `precompute_chunk_frames` in the JSON `gpu` section:

```text
peak VRAM per chunk ≈ 2 × chunk_frames × d1 × d2 × 4 bytes
500 frames × 512 × 512 × 4 × 2 = 1.05 GB
```

**correlation_pnr** uses the same chunked GPU path for summary image
computation.  Pass `use_gpu=True` to force GPU or `use_gpu=None` (default)
for automatic dispatch.

**Spatial update** uses a GPU gram-matrix path that is ~10–20× faster than
the CPU path for K=50, T=6000.

See [gpu_acceleration](gpu_acceleration) for a full table and details.

Memory management
-----------------

Three explicit memory management calls are integrated at critical points in
the pipeline:

- `malloc_trim()`: returns glibc heap free-pages to the OS immediately after
  large array deletions.  Without this Python holds freed pages for minutes,
  artificially inflating RSS and causing the worker budget check to spawn
  fewer processes than available RAM allows.

- `madvise_dontneed(Yr)`: evicts the C-order mmap from the kernel page cache
  before CNMF workers start.  Workers read the F-order mmap from NVMe; the
  C-order copy does not need to compete for page-cache slots during the fit.

- `cupy_flush()`: fully reclaims VRAM after MC and Cn computation.  Sequence:
  `gc.collect()` → FFT plan cache clear → pool `free_all_blocks()` → device
  sync.  The pool flush alone is insufficient — Python gc must run first to
  drop references, and the FFT plan cache maintains its own CUDA allocations
  separately.

Storage: NVMe vs FUSE
---------------------

`CAIMAN_TEMP` must be on a local NVMe drive formatted ext4.  NTFS-3g FUSE
causes D-state stalls (uninterruptible I/O waits) during tile loading because
FUSE silently ignores `posix_fadvise` hints.  The pipeline uses
`madvise(MADV_DONTNEED)` instead, which is handled by the kernel VM
subsystem regardless of filesystem.

| Filesystem | Tile read time | MC mmap write speed |
|---|---|---|
| NTFS-3g (FUSE) | ~40 s/tile | ~94 MB/s |
| ext4 NVMe | ~1 s/tile | ~3 GB/s |
