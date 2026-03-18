Pipeline Framework
==================

The Gravios fork adds a structured pipeline framework that eliminates
boilerplate across sessions.  Every session is a renamed copy of the same
template script; all parameters live in a companion JSON file.

Session lifecycle
-----------------

.. code:: bash

    # 1. Create session files, run MC, estimate parameters
    python pipelines/new_session.py \
        stroh-ej-20140708-TL2 \
        /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/08072014/ \
        -y --run-mc --estimate-params

    # 2. Review the JSON, then run
    python stroh-ej-20140708-TL2_pipeline.py

Step 1 produces:

- ``stroh-ej-20140708-TL2_pipeline.py`` — copy of ``template_pipeline.py``
- ``stroh-ej-20140708-TL2_pipeline.json`` — patched copy of ``template_pipeline.json``
- ``stroh-ej-20140708-TL2_qc_00_param_estimate.png`` — Cn/PNR inspection figure

Step 2 runs all 17 pipeline steps and produces:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File
     - Contents
   * - ``<session>.log``
     - Full debug log with timestamps
   * - ``<session>_Cn.npy``
     - Local correlation image (d1 × d2)
   * - ``<session>_results.hdf5``
     - CNMF estimates (A, C, S, F_dff, …)
   * - ``<session>_report.txt``
     - Timing and resource table
   * - ``<session>_qc_01_raw_sample.png``
     - Raw TIFF frame grid
   * - ``<session>_qc_02_motion_correction.png``
     - Shift traces + mean frames
   * - ``<session>_qc_03_correlation_image.png``
     - Cn image
   * - ``<session>_qc_04_fit_footprints.png``
     - Initial fit component footprints
   * - ``<session>_qc_05_refit_footprints.png``
     - Refit footprints
   * - ``<session>_qc_06_evaluation.png``
     - Accepted/rejected + histogram
   * - ``<session>_qc_07_traces.png``
     - Stacked normalised dF/F traces

Naming convention
-----------------

The session name is the script filename stem with the ``_pipeline`` (or
``.pipeline``) suffix removed.  Both separators are recognised
(case-insensitive):

.. list-table::
   :header-rows: 1

   * - Script filename
     - Session name
   * - ``stroh-ej-20140708-TL2_pipeline.py``
     - ``stroh-ej-20140708-TL2``
   * - ``stroh-ej-20140708-TL2.pipeline.py``
     - ``stroh-ej-20140708-TL2``

``resolve_pipeline_path()`` handles path resolution under all invocation
styles (``python script.py``, IPython ``%run``, Emacs ``python-el``).

Multiprocessing guard
---------------------

The pipeline body must be inside ``if __name__ == "__main__":``.  Under the
``spawn`` multiprocessing start method (default on macOS and Windows, common
on Linux), each worker re-imports the script as ``__main__``.  Any top-level
code outside the guard re-executes in every worker.

The bootstrap block (path resolution + env-var application) runs in every
process.  It uses only the standard library and is safe to run in workers:

.. code:: python

    # Bootstrap — runs in EVERY process
    from caiman.utils.pipeline_setup import resolve_pipeline_path

    _SCRIPT_PATH, _CONFIG_PATH, session = resolve_pipeline_path()

    # Apply env vars from JSON before any caiman import
    import json, os
    _env = json.load(open(_CONFIG_PATH)).get("env", {})
    for k, v in _env.items():
        os.environ.setdefault(k, str(v))

    # Pipeline body — parent process ONLY
    if __name__ == "__main__":
        import caiman as cm
        ...

Resume behaviour
----------------

Expensive steps check for existing output before running:

.. list-table::
   :header-rows: 1

   * - Step
     - Skipped when
   * - Motion correction
     - ``*{session}*rig*order_F*.mmap`` exists in ``CAIMAN_TEMP``
   * - F→C conversion
     - C-order mmap exists and is newer than the MC mmap
   * - CNMF fit
     - Never skipped automatically (delete HDF5 to force re-run)

Batch processing
----------------

.. code:: bash

    for dir in /data/src/stroh-ej/RawDataSel_AD_Project/G2_5FAD/*/; do
        session=$(basename "$dir")_TL002
        python pipelines/new_session.py "$session" "$dir" \
            -y --run-mc --estimate-params --n-frames 500
    done

    for dir in /data/src/stroh-ej/RawDataSel_AD_Project/G2_5FAD/*/; do
        session=$(basename "$dir")_TL002
        results="$dir/${session}_results.hdf5"
        [ -f "$results" ] && continue   # skip completed sessions
        python "$dir/${session}_pipeline.py"
    done

Loading results
---------------

.. code:: python

    import caiman as cm

    cnm = cm.source_extraction.cnmf.cnmf.load_CNMF(
        "stroh-ej-20140708-TL2_results.hdf5")

    A   = cnm.estimates.A       # spatial footprints  (pixels × K)  CSC sparse
    C   = cnm.estimates.C       # denoised traces      (K × T)
    S   = cnm.estimates.S       # spike trains         (K × T)
    dff = cnm.estimates.F_dff   # dF/F                 (K × T)
