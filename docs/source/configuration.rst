Configuration Reference
=======================

All parameters live in ``<session>_pipeline.json``.  The session name is
derived from the script filename — it is never stored in the JSON.

``env`` section
---------------

Applied to ``os.environ`` **before** any CaImAn import.  ``CAIMAN_*``
variables are set unconditionally; thread counts use ``setdefault`` so a
shell value takes precedence.

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``CAIMAN_DATA``
     - ``/data/caiman``
     - Root for CNN model files and persistent data
   * - ``CAIMAN_TEMP``
     - ``/data/caiman/temp``
     - Temporary mmaps (MC output, C-order mmap). **Must be NVMe ext4.**
   * - ``CAIMAN_SHM``
     - ``/dev/shm``
     - Shared-memory tile buffers. ``/dev/shm`` is RAM-backed tmpfs.
   * - ``CAIMAN_TILE_SLOTS``
     - ``3``
     - Number of concurrent SHM tile slots during CNMF
   * - ``MKL_NUM_THREADS``
     - ``1``
     - Prevent BLAS oversubscription across workers
   * - ``OMP_NUM_THREADS``
     - ``1``
     - Same
   * - ``MPLBACKEND``
     - ``Agg``
     - Headless matplotlib (no display required)

``session`` section
-------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Description
   * - ``data_root``
     - Absolute path prefix, e.g. ``/data/src/``
   * - ``experiment``
     - Relative path from data_root to session folder, e.g.
       ``stroh-ej/RawDataSel_AD_Project/G1_B6J/08072014/``

The session name (script stem minus ``_pipeline``) is not stored here.

``data`` section
----------------

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``fr``
     - ``30``
     - Acquisition frame rate [Hz]
   * - ``decay_time``
     - ``1.0``
     - GCaMP decay time constant [s]. GCaMP6f ≈ 0.4, GCaMP6s ≈ 1.0
   * - ``add_baseline``
     - ``100.0``
     - Added to all pixels during F→C conversion to keep values > 0

``motion_correction`` section
-------------------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``max_shifts``
     - ``[6, 6]``
     - Maximum [row, col] shift in pixels. Estimated automatically by ``--run-mc``
   * - ``strides``
     - ``[64, 64]``
     - Patch stride for piecewise-rigid MC (unused when ``pw_rigid=false``)
   * - ``overlaps``
     - ``[32, 32]``
     - Patch overlap for piecewise-rigid MC
   * - ``pw_rigid``
     - ``false``
     - Piecewise-rigid (true) or rigid (false). Rigid is faster and sufficient
       for most 2P data
   * - ``border_nan``
     - ``"copy"``
     - ``"copy"`` replicates border pixels; ``"min"`` fills with movie minimum
   * - ``max_deviation_rigid``
     - ``3``
     - Max deviation from rigid shift allowed per patch (piecewise-rigid only)

``cnmf`` section
-----------------

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``p``
     - ``1``
     - AR model order for deconvolution. 0 = off, 1 = fast rise, 2 = visible rise
   * - ``gnb``
     - ``2``
     - Background components. 2 = standard 2P; -1 = ring model (CNMF-E)
   * - ``merge_thr``
     - ``0.7``
     - Correlation threshold for merging duplicate components
   * - ``rf``
     - ``36``
     - Patch half-size [px]. Must satisfy: ``ring_size_factor × gSiz < rf``
   * - ``stride``
     - ``18``
     - Patch overlap [px]. Typically ``rf // 2``
   * - ``K``
     - ``15``
     - Max components per patch
   * - ``gSig``
     - ``[9, 9]``
     - Gaussian half-width of neuron PSF [px]. **Crucial — must match your data**
   * - ``gSiz``
     - ``[37, 37]``
     - Spatial support [px]. Rule of thumb: ``4 × gSig + 1``
   * - ``ring_size_factor``
     - ``0.9``
     - Ring neuropil outer radius as multiple of gSiz. Constraint: ``rsf × gSiz < rf``
   * - ``min_corr``
     - ``0.6``
     - Minimum local correlation for seed pixels (corr_pnr init)
   * - ``min_pnr``
     - ``7``
     - Minimum peak-to-noise ratio for seed pixels
   * - ``method_init``
     - ``"corr_pnr"``
     - Initialisation method: ``corr_pnr`` or ``greedy_roi``
   * - ``method_deconv``
     - ``"oasis"``
     - Deconvolution method: ``oasis`` or ``cvxpy``
   * - ``dff_quantile_min``
     - ``8``
     - Baseline percentile for dF/F (``detrend_df_f quantileMin``)
   * - ``dff_frames_window``
     - ``500``
     - Sliding window length for dF/F baseline

Ring constraint
~~~~~~~~~~~~~~~

When ``method_init = "corr_pnr"``, the ring neuropil model requires::

    ring_size_factor × gSiz[0] < rf

With defaults (ring_size_factor=0.9, gSiz=37, rf=36):
``0.9 × 37 = 33.3 < 36 ✓``

Violating this causes ``compute_W`` to return a degenerate weight matrix,
collapsing hundreds of raw components to very few after assembly.

``quality`` section
--------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``min_SNR``
     - ``1.5``
     - Minimum peak SNR for acceptance
   * - ``rval_thr``
     - ``0.6``
     - Minimum spatial correlation with raw data
   * - ``use_cnn``
     - ``true``
     - CNN shape classifier. Auto-disabled if model files are missing
   * - ``min_cnn_thr``
     - ``0.6``
     - High CNN threshold (component must pass this OR min_SNR OR rval_thr)
   * - ``cnn_lowest``
     - ``0.1``
     - Low CNN threshold (component must pass ALL low thresholds)

``cluster`` section
--------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``n_processes``
     - ``null``
     - Worker count. ``null`` = all CPUs. Reduce if workers OOM
   * - ``ram_budget_frac``
     - ``0.85``
     - Fraction of available VM to allocate to patch workers
   * - ``worker_overhead_frac``
     - ``1.1``
     - Multiplier on estimated per-worker RAM. Raise if workers OOM
   * - ``blas_threads_per_worker``
     - ``1``
     - BLAS threads per worker. Always keep at 1

``gpu`` section
----------------

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``precompute_chunk_frames``
     - ``500``
     - Frames per GPU filter chunk. Peak VRAM ≈ ``2 × frames × d1 × d2 × 4`` bytes

Dot-notation overrides
----------------------

``apply_suggestions`` and ``new_session.py`` accept dotted ``section.key``
notation to update any JSON key programmatically:

.. code:: python

    from caiman.utils.params_estimator import apply_suggestions

    apply_suggestions("session_pipeline.json", {
        "cnmf.min_corr":                0.5,
        "cnmf.min_pnr":                 5.0,
        "motion_correction.max_shifts": [4, 4],
        "gpu.precompute_chunk_frames":  300,
    })
