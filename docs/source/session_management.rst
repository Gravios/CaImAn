Session Management
==================

``new_session.py`` — CLI reference
-------------------------------------

.. code:: bash

    python pipelines/new_session.py <session> <dest> [options]

Positional arguments
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Argument
     - Description
   * - ``session``
     - Session identifier, e.g. ``stroh-ej-20140708-TL2``
   * - ``dest``
     - Absolute path to the session folder (created if missing)

Options
~~~~~~~

.. list-table::
   :header-rows: 1

   * - Flag
     - Description
   * - ``--data-root PATH``
     - Override inferred ``data_root`` in the JSON
   * - ``--fr HZ``
     - Acquisition frame rate
   * - ``--decay-time S``
     - GCaMP decay time constant
   * - ``--gSig N``
     - Soma half-width [px]; sets ``gSig=[N,N]`` and ``gSiz=[4N+1,4N+1]``
   * - ``--rf N``
     - Patch half-size [px]
   * - ``--K N``
     - Max components per patch
   * - ``--min-corr F``
     - Cn threshold for seed pixels
   * - ``--min-pnr F``
     - PNR threshold for seed pixels
   * - ``--method-init``
     - ``corr_pnr`` or ``greedy_roi``
   * - ``--n-processes N``
     - CNMF worker count
   * - ``--run-mc``
     - Run GPU MC if no mmap exists; implies ``--estimate-params``
   * - ``--estimate-params``
     - Estimate gSig/thresholds and write into JSON
   * - ``--n-frames N``
     - Frames to subsample for estimation (default 500)
   * - ``-y`` / ``--force``
     - Overwrite existing files without prompting (batch-safe)
   * - ``--no-comments``
     - Strip ``_comment`` keys from output JSON
   * - ``--dry-run``
     - Print what would be done without writing anything

``--run-mc`` behaviour
~~~~~~~~~~~~~~~~~~~~~~~

#. Search for existing MC mmap in ``dest/`` and ``CAIMAN_TEMP``
#. If found: skip MC, use existing mmap for estimation
#. If not found: run rigid GPU MC with defaults (``max_shifts=[6,6]``)
#. Analyse shift distribution → update ``motion_correction.max_shifts``
   in JSON (p99 per-axis, rounded up, minimum 4)
#. Run ``estimate_params()`` on the mmap
#. **Delete** the temporary mmap (pre-existing mmaps are never deleted)

Examples
~~~~~~~~

.. code:: bash

    # Minimal — infer paths
    python pipelines/new_session.py \
        stroh-ej-20140708-TL2 \
        /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/08072014/

    # Full one-shot setup
    python pipelines/new_session.py \
        stroh-ej-20140708-TL2 \
        /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/08072014/ \
        -y --run-mc --estimate-params --fr 30 --decay-time 1.0

    # Override parameters manually
    python pipelines/new_session.py \
        stroh-ej-20140708-TL2 \
        /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/08072014/ \
        -y --gSig 7 --min-corr 0.4 --min-pnr 5 --method-init greedy_roi

    # Dry run
    python pipelines/new_session.py ... --dry-run

Parameter estimation
---------------------

``estimate_params`` subsamples the MC mmap, computes Cn and PNR images, and
returns a dict of suggested ``cnmf`` section values:

.. code:: python

    from caiman.utils.params_estimator import estimate_params, apply_suggestions

    suggestions = estimate_params(
        fname_mc,
        n_frames = 500,
        out_path = outdir / f"{session}_qc_00_param_estimate.png",
        logger   = logger,
    )
    # → {"gSig": [8,8], "gSiz": [33,33], "rf": 36, "stride": 18,
    #    "min_corr": 0.52, "min_pnr": 6.1}

    apply_suggestions("session_pipeline.json", suggestions)

**gSig estimation:** LoG blob detection on the normalised PNR image via
``skimage.feature.blob_log``.  Median blob radius → gSig.  Falls back to
gSig=5 if fewer than 5 blobs are detected.

**Threshold estimation:** histogram inflection point (steepest negative
gradient in the smoothed density), clipped to [25th, 85th] percentile.

**Interpreting low-signal results:** If ``gSig`` falls back to 5 and
``min_corr``/``min_pnr`` are very permissive (< 0.1 / < 4), the dataset
has weak signal-to-background separation.  Consider switching to
``method_init = "greedy_roi"`` or increasing ``n_frames`` for estimation.

Batch processing
----------------

.. code:: bash

    GROUP=/data/src/stroh-ej/RawDataSel_AD_Project/G2_5FAD

    # Prepare all sessions
    for dir in "$GROUP"/*/; do
        python pipelines/new_session.py "$(basename $dir)_TL002" "$dir" \
            -y --run-mc --estimate-params --n-frames 500
    done

    # Run all sessions, skip completed ones
    for dir in "$GROUP"/*/; do
        session="$(basename $dir)_TL002"
        script="$dir/${session}_pipeline.py"
        results="$dir/${session}_results.hdf5"
        [ -f "$results" ] && echo "Skipping $session" && continue
        python "$script"
    done
