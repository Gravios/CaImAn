Memory Management
=================

The pipeline applies explicit memory management at three points to keep RSS
controlled on large datasets and ensure the worker budget check allocates the
correct number of processes.

``malloc_trim``
---------------

After deleting large arrays, call ``malloc_trim`` to return glibc heap
free-pages to the OS immediately:

.. code:: python

    del images, Yr
    import gc; gc.collect()
    from caiman.utils.memory import malloc_trim
    malloc_trim(logger)

Without this, Python's allocator holds freed pages in an internal free-list
for minutes.  The worker RAM budget check reads ``/proc/meminfo`` — if freed
pages have not been returned to the OS, the check underestimates available
RAM and spawns fewer workers than it should.

``madvise_dontneed``
--------------------

Before starting CNMF workers, evict the C-order mmap from the kernel page
cache:

.. code:: python

    from caiman.utils.memory import madvise_dontneed
    madvise_dontneed(Yr, logger)

Workers read the F-order MC mmap from NVMe; the C-order copy does not need to
compete for page-cache slots during the fit.

Uses ``madvise(MADV_DONTNEED)`` which is handled by the kernel VM subsystem
regardless of filesystem.  More reliable than ``posix_fadvise`` on FUSE
(NTFS-3g silently ignores fadvise).

``cupy_flush``
--------------

Before the CNMF fit, fully reclaim VRAM:

.. code:: python

    from caiman.utils.memory import cupy_flush
    cupy_flush(logger, label="before CNMF fit")

Flush sequence:

1. ``gc.collect()`` — drop Python references so arrays enter the free-list
2. ``cp.fft.config.get_plan_cache().clear()`` — cuFFT plan cache (separate from pool)
3. ``cp.get_default_memory_pool().free_all_blocks()``
4. ``cp.get_default_pinned_memory_pool().free_all_blocks()``
5. ``cp.cuda.Device().synchronize()``

The pool flush alone is insufficient.  Motion correction and correlation-image
computation leave cuFFT plans in the plan cache that maintain their own CUDA
allocations outside the MemoryPool.

SHM tile buffer layout
-----------------------

During CNMF the GPU precompute writes ``CAIMAN_TILE_SLOTS`` (default 3) tile
slots to ``/dev/shm``.  The dispatcher fills slots in a rolling fashion while
workers drain them:

.. code:: text

    /dev/shm/_caiman_tile_<session>_slot0.mmap  (d1_tile × d2_tile × T × 2 bytes)
    /dev/shm/_caiman_tile_<session>_slot1.mmap
    /dev/shm/_caiman_tile_<session>_slot2.mmap

Each slot holds one tile's filtered movie as float16.  Workers read their
patch slice from the slot and release it; the dispatcher writes the next tile
into the freed slot.

Peak SHM usage per slot:

.. code:: text

    d1_tile × d2_tile × T × 2 bytes
    For a 180×180 tile, T=5000: 180 × 180 × 5000 × 2 = 324 MB

With 3 slots: ~1 GB of SHM consumed during CNMF.

``/dev/shm`` is RAM-backed tmpfs — writes do not touch disk.  The pipeline
cleans up all SHM files at cluster start via ``clean_stale_shm``.

Worker RAM budget
-----------------

The worker count is capped by available RAM.  The budget formula:

.. code:: text

    budget = vm.total × ram_budget_frac - vm.used - movie_ram
    n_workers = min(requested, floor(budget / (per_worker_estimate × overhead_frac)))

Tune ``cluster.ram_budget_frac`` (default 0.85) and
``cluster.worker_overhead_frac`` (default 1.1) if workers OOM or if fewer
processes are allocated than expected.

F-order vs C-order mmaps
--------------------------

The pipeline maintains two mmap representations of the movie:

.. list-table::
   :header-rows: 1

   * - File
     - Order
     - Location
     - Used by
   * - ``*rig*order_F*.mmap``
     - F (Fortran)
     - ``CAIMAN_TEMP``
     - Motion correction output; GPU precompute reads frames as contiguous blocks
   * - ``*cnmf*order_C*.mmap``
     - C
     - ``CAIMAN_TEMP``
     - CNMF patch workers; pixel access across full T is contiguous

F-order is optimal for frame-sequential reads (precompute filter, GPU MC).
C-order is optimal for pixel-sequential reads (CNMF component extraction).
