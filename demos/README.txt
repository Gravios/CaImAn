CaImAn Demo Scripts
===================

demos/general/
--------------

Updated for the Gravios fork.  All demos use CNMFRunner, QCRunner, and
PipelineTimer from caiman.utils instead of manual fit/refit/evaluate calls.

demo_pipeline.py
    Two-photon batch CNMF pipeline.  Downloads the Sue_2x_3000 demo dataset
    automatically.  Replaces the upstream demo_pipeline.py with GPU MC,
    memory-safe F→C conversion, CNMFRunner orchestration, and headless QC
    figures via QCRunner.

demo_pipeline_cnmfE.py
    One-photon / CNMF-E microendoscope pipeline.  Downloads data_endoscope.tif
    automatically.  Uses corr_pnr initialisation with ring background model
    (gnb=-1).  Includes ring size constraint check at startup.

demo_OnACID.py
    OnACID online analysis.  Processes frames streaming through the algorithm.
    Uses PipelineTimer and QCRunner for post-hoc figures on the final estimates.

demo_param_estimation.py
    Standalone parameter estimation tool.  Takes an existing MC mmap (or runs
    MC from a TIF with --run-mc) and estimates gSig, min_corr, min_pnr, rf
    from the data.  Can write suggestions directly into a pipeline JSON.

demo_pipeline_NWB.py
    Two-photon pipeline with NWB output (upstream, unchanged).

demo_pipeline_voltage_imaging.py
    Voltage imaging pipeline using VolPy (upstream, unchanged).

demo_behavior.py
    Behavioral video analysis (upstream, unchanged).

demos/notebooks/
----------------

Jupyter notebooks for interactive exploration.  Launch with:

    jupyter lab --ZMQChannelsWebsocketConnection.iopub_data_rate_limit=1.0e10

See docs/source/Getting_Started.rst for details.
