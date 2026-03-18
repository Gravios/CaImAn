GPU & Memory Modernization Targets
====================================

Functions not yet GPU-accelerated that would benefit most.  Listed by
priority based on fraction of total wall time in typical runs.

Priority 1 — High impact, achievable
--------------------------------------

compute_W — ring neuropil background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``caiman/source_extraction/cnmf/initialization.py:2352``

``compute_W`` solves the ring-background weight matrix via NNLS.  The inner
product ``Y_ds @ residual.T`` is a dense ``(pixels × pixels_in_ring)`` matmul —
the dominant cost (~30–60 s CPU for 512×512, T=5000).

**Modernization path:** Compute ``Y_ds @ residual.T`` on GPU in spatial tiles;
solve constrained NNLS per-pixel using GPU-batched LSQR.  **Expected
speedup:** 5–10×.

HALS4activity / HALS4shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``initialization.py:1181``, ``initialization.py:1190``,
``online_cnmf.py:1633``, ``online_cnmf.py:1652``

Hierarchical ALS updates ``C`` and ``A`` iteratively.  The inner loop is
serial over K components, each performing a dense vector op on the residual.

**Modernization path:** Vectorize the K loop with CuPy batched matmul; keep
residual ``R`` as a GPU array updated in-place each iteration.  **Expected
speedup:** 3–8×.

update_temporal_components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``caiman/source_extraction/cnmf/temporal.py:64``

The bottleneck is ``A.T @ Y`` (K × T dense matmul), currently serialized
component-by-component.  Dominates refit wall time (~60% for large K).

**Modernization path:** Compute ``A.T @ Y`` on GPU via CuSPARSE SpMM;
deconvolution (OASIS) remains CPU per-component.  **Expected speedup:** 2–4×.

Priority 2 — Medium impact
---------------------------

local_correlations_fft (standalone path)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``caiman/summary_images.py:89``

The GPU path already exists in ``correlation_pnr``.  The standalone
``local_correlations_fft`` called per-patch inside ``init_neurons_corr_pnr``
still uses CPU.  Add a GPU dispatch path matching the existing
``_correlation_pnr_gpu`` structure.

merge_components
~~~~~~~~~~~~~~~~~

**Location:** ``caiman/source_extraction/cnmf/merging.py:19``

``A.T @ A`` pairwise correlation currently uses SciPy sparse matmul.  GPU via
CuSPARSE → CuPy sparse would remove the bottleneck for large K.

register_multisession
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``caiman/base/rois.py:553``

Pairwise spatial overlap matrix ``A_s1.T @ A_s2`` scales O(K²).  GPU matmul
would accelerate the dominant cost; ``linear_sum_assignment`` remains CPU.

Priority 3 — Lower impact or high complexity
---------------------------------------------

constrained_oasisAR2 (deconvolution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``caiman/source_extraction/cnmf/deconvolution.py:632``

Sequential AR filter — hard to parallelize across T.  Batch K components on
GPU (K-parallel, one time step at a time).  **Expected speedup:** 2–3×.

threshold_components
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``caiman/source_extraction/cnmf/spatial.py:443``

Per-component morphological operations.  Use ``cupyx.scipy.ndimage`` for
dilation and connected-component labelling; batch K components on GPU.

Memory management candidates
-----------------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Location
     - Issue
     - Fix
   * - ``update_temporal_components``
     - ``temporal.py``
     - Full-movie residual copy ``Y - A@C - b@f``
     - Compute in T-axis chunks
   * - ``merge_components``
     - ``merging.py``
     - ``C / norm`` full K×T copy
     - In-place normalisation
   * - ``get_noise_fft`` (CPU path)
     - ``pre_processing.py``
     - FFT of full (pixels, T) array
     - Apply chunking from GPU path to CPU path
   * - ``creatememmap``
     - ``spatial.py:1075``
     - Loads full Y before creating mmap
     - Superseded by tile-based dispatch; candidate for removal

SyntaxWarning fix
------------------

``caiman/source_extraction/volpy/spikepursuit.py`` contains invalid escape
sequences (``\d`` in non-raw strings) that produce ``SyntaxWarning`` on
Python 3.12+.  Fix: prepend ``r`` to affected string literals.
