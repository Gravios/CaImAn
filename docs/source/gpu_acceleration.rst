GPU Acceleration
================

GPU status table
----------------

.. list-table::
   :header-rows: 1

   * - Step
     - Function
     - Status
   * - Motion correction
     - ``MotionCorrect(use_gpu=True)``
     - ✅ Full GPU (parallel cuFFT)
   * - Filter precompute
     - ``precompute_corr_pnr_filtered_fov``
     - ✅ Full GPU (chunked float16)
   * - Correlation image
     - ``correlation_pnr``
     - ✅ Full GPU (chunked)
   * - Noise estimation (1P)
     - ``get_noise_fft``
     - ✅ GPU batched rfft
   * - Spatial update
     - ``update_spatial_components``
     - ✅ GPU gram-matrix path
   * - Y@f.T residual
     - inside ``update_spatial_components``
     - ✅ GPU
   * - Temporal update
     - ``update_temporal_components``
     - ❌ CPU
   * - HALS shape/activity
     - ``HALS4shape``, ``HALS4activity``
     - ❌ CPU
   * - Ring background
     - ``compute_W``
     - ❌ CPU
   * - Deconvolution
     - ``constrained_oasisAR2``
     - ❌ CPU
   * - Merging
     - ``merge_components``
     - ❌ CPU
   * - Component evaluation
     - ``evaluate_components``
     - ❌ CPU (CNN inference uses TF/Keras GPU if configured)
   * - Multisession registration
     - ``register_multisession``
     - ❌ CPU

Precompute pipeline
-------------------

The most impactful GPU operation is the full-FOV filtered movie precompute.
Without it, each of the ~200 patches runs T=5000 calls to ``cv2.filter2D``
(~20 s per patch, ~65 min total).  With the GPU precompute:

1. Filter full FOV on GPU in chunks of ``chunk_frames`` frames (configured via
   ``gpu.precompute_chunk_frames`` in the JSON)
2. Write float16 mmap to ``/dev/shm`` (~14.5 GB, RAM-backed)
3. Each worker copies its ``(d1p × d2p × T)`` slice from SHM (~76 ms)
4. Compute per-patch ``sn``, ``data_max``, Cn, PNR from precomp arrays

Total precompute: ~2 s.  Per-patch slice copy: ~76 ms vs ~20 s (CPU).

VRAM management
---------------

Motion correction and Cn computation leave ~12 GB stranded in CuPy's pool.
The pipeline flushes before CNMF:

.. code:: python

    from caiman.utils.memory import cupy_flush
    cupy_flush(logger, label="before CNMF fit")

The flush sequence is: ``gc.collect()`` → FFT plan cache clear →
``pool.free_all_blocks()`` → pinned pool → ``Device().synchronize()``.
The pool flush alone is insufficient — gc must run first to drop Python
references, and the FFT plan cache is a separate CUDA allocation.

Chunk size tuning
-----------------

.. code:: text

    peak VRAM per chunk ≈ 2 × chunk_frames × d1 × d2 × 4 bytes

    512×512, 500 frames:  500 × 512 × 512 × 4 × 2 = 1.05 GB
    516×512, 500 frames:  500 × 516 × 512 × 4 × 2 = 1.06 GB

Default 500 frames is safe for 16 GB GPU after MC leaves ~12 GB allocated.
Reduce if you see GPU OOM during precompute.

Spatial update GPU path
------------------------

``update_spatial_components`` uses a gram-matrix path when CuPy is available:

**Original:** K×T NNLS per pixel — 2700× more FLOPs than necessary.

**GPU path:** Precompute ``G = C @ C.T`` (K×K) and ``YC = Y @ C.T``
(pixels×K), then solve each pixel's NNLS on the K×K system.

Speedup: ~10–20× for K=50, T=6000.  Falls back to CPU silently.

Fallback behaviour
------------------

All GPU paths fall back to CPU silently when:

- CuPy is not installed
- No CUDA-capable device is found
- ``use_gpu=False`` is passed explicitly

No code changes are required to run without a GPU.
