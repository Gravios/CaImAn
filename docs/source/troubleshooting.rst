Troubleshooting
===============

This page covers errors encountered during development of the Gravios fork.

ValueError: operands could not be broadcast — shapes (73,19) (73,73)
----------------------------------------------------------------------

.. code:: text

    _frame[_frame < _thresh] = 0
    ValueError: operands could not be broadcast together with shapes (73,19) (73,73)

**Cause:** Tile-boundary shape mismatch.  ``precomp['sn']`` is sliced to
``(73, 73)`` from the full-FOV npz using the patch FOV coordinates, but the
tile delivered to this worker was clipped at the movie edge to 19 columns.
``_thresh = thresh_init * noise_pixel`` then fails the broadcast.

**Fix:** Shape guard at line ~1716 of ``initialization.py``:

.. code:: python

    _sn_candidate = precomp.get('sn')
    if _sn_candidate is not None and _sn_candidate.shape == (d1, d2):
        noise_pixel = _sn_candidate
        data_max    = precomp['data_max']
        pnr         = np.divide(data_max, noise_pixel + 1e-10)
    else:
        # fall through to local noise estimation
        data_filtered -= data_filtered.mean(axis=0)
        data_max    = np.max(data_filtered, axis=0)
        noise_pixel = get_noise_fft(data_filtered.T, noise_method='mean')[0].T
        pnr         = np.divide(data_max, noise_pixel + 1e-10)

**Verify:**

.. code:: bash

    grep -n "_sn_candidate" ~/software/CaImAn/caiman/source_extraction/cnmf/initialization.py

Should show a match at ~line 1716.

IndexError: boolean index did not match — shapes (49,13) vs (49,49)
--------------------------------------------------------------------

.. code:: text

    IndexError: boolean index did not match indexed array along dimension 1;
    dimension is 13 but corresponding boolean dimension is 49

**Cause:** Same tile-boundary issue for ``precomp['cn']``.  ``cn`` is sliced
to ``(d1, d2)`` from the npz, but the tile is ``(d1, d2_tile)`` where
``d2_tile < d2`` at the movie edge.

**Fix:** Shape guard for ``cn``/``pnr`` at the precomp branch in
``init_neurons_corr_pnr``:

.. code:: python

    _precomp_cn = precomp.get('cn') if precomp is not None else None
    if _precomp_cn is not None and _precomp_cn.shape == (d1, d2):
        cn  = _precomp_cn.copy()
        pnr = precomp['pnr'].copy()
    else:
        # fall through to local computation

Pipeline executes multiple times
---------------------------------

**Symptom:** Log shows ``STARTED TIFF format check`` 9–16 times at the same
timestamp.

**Cause:** Pipeline body is outside ``if __name__ == "__main__":``.  With
multiprocessing ``spawn``, workers re-import the script as ``__main__`` and
re-execute all top-level statements.

**Fix:**

.. code:: python

    # Bootstrap (runs in every process)
    from caiman.utils.pipeline_setup import resolve_pipeline_path
    _SCRIPT_PATH, _CONFIG_PATH, session = resolve_pipeline_path()
    # ... env vars ...

    if __name__ == "__main__":   # ← add this guard
        import caiman as cm
        ...

FileNotFoundError: Parameter file not found
--------------------------------------------

.. code:: text

    FileNotFoundError: Parameter file not found:
        .../stroh-ej-20140221-G25FAD_TL002_pipeline.json

**Cause:** JSON not yet created, or was created with the template placeholder
``<lab>/<experiment>/<date>/`` still in the ``experiment`` field.

**Fix:** Run ``new_session.py`` to (re-)create the JSON:

.. code:: bash

    python pipelines/new_session.py \
        stroh-ej-20140221-G25FAD_TL002 \
        /data/src/stroh-ej/RawDataSel_AD_Project/G2_5FAD/21022014/ \
        -y

FileNotFoundError when creating log file
-----------------------------------------

.. code:: text

    FileNotFoundError: [Errno 2] No such file or directory:
        '/data/src/<lab>/<experiment>/<date>/stroh-ej-....log'

**Cause:** The JSON ``experiment`` field still contains the template
placeholder ``<lab>/<experiment>/<date>/`` — the output directory does not
exist.

**Fix:** Re-run ``new_session.py`` with the correct destination path.  The
``setup_logging`` function now also creates parent directories automatically
(``logfile.parent.mkdir(parents=True, exist_ok=True)``).

Very few components found (e.g. 8 instead of ~50)
--------------------------------------------------

**Cause:** Ring constraint violated.  When ``ring_size_factor × gSiz ≥ rf``,
``compute_W`` returns a degenerate weight matrix.  NNLS then zeros all spatial
footprints during assembly, collapsing hundreds of raw patch components to
very few.

**Diagnosis:** Check the constraint in the JSON:

.. code:: python

    ring_size_factor * gSiz[0] < rf

**Fix (quick):** Add ``ring_size_factor: 0.9`` to the ``cnmf`` section.
**Fix (correct):** Increase ``rf`` so the ring fits inside the patch.

GPU OOM during precompute
--------------------------

**Symptom:** CuPy ``OutOfMemoryError`` during ``precompute_corr_pnr_filtered_fov``.

**Fix:** Reduce ``gpu.precompute_chunk_frames`` in the JSON.  At 300 frames
and 512×512: ``300 × 512 × 512 × 4 × 2 = 629 MB`` peak per chunk.

Non-picklable exception from worker
-------------------------------------

.. code:: text

    _flapack.error: (internal LAPACK error)
    ...
    _pickle.PicklingError: Can't pickle <class '_flapack.error'>

**Cause:** Scipy's compiled LAPACK extension raises a non-picklable exception
that cannot cross the multiprocessing boundary.

**Fix:** The exception barrier in ``map_reduce.py`` catches non-picklable
exceptions and re-raises them as ``RuntimeError`` before the process boundary.
If you see this error, update ``map_reduce.py`` from the latest tarball.

NTFS-3g D-state stalls during tile loading
-------------------------------------------

**Symptom:** Workers hang in D-state (uninterruptible I/O wait) for 30–60 s
per tile.  ``cat /proc/<pid>/status`` shows ``State: D``.

**Cause:** ``CAIMAN_TEMP`` is on an NTFS-3g FUSE filesystem.  FUSE ignores
``posix_fadvise`` hints, causing kernel page-cache thrashing.

**Fix:** Move ``CAIMAN_TEMP`` to an ext4 NVMe drive.  Tile read time drops
from ~40 s to ~1 s.  The pipeline uses ``madvise(MADV_DONTNEED)`` which is
handled by the kernel VM regardless of filesystem.
