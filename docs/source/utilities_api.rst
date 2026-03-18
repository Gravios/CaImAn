Utilities API
=============

All public utilities are importable from ``caiman.utils`` or ``caiman`` directly.

pipeline_setup
--------------

.. automodule:: caiman.utils.pipeline_setup
   :members: resolve_pipeline_path, setup_logging, ensure_model_files, clean_stale_shm
   :undoc-members:
   :show-inheritance:

timing
------

.. automodule:: caiman.utils.timing
   :members: PipelineTimer, write_report, log_call, fmt_elapsed
   :undoc-members:
   :show-inheritance:

memory
------

.. automodule:: caiman.utils.memory
   :members: malloc_trim, madvise_dontneed, cupy_flush
   :undoc-members:
   :show-inheritance:

cnmf_runner
-----------

.. automodule:: caiman.utils.cnmf_runner
   :members: CNMFRunner
   :undoc-members:
   :show-inheritance:

qc
--

.. automodule:: caiman.utils.qc
   :members: QCRunner, qc_raw_sample, qc_motion_correction, qc_correlation_image,
             qc_pnr_image, qc_cnmf_fit, qc_cnmf_refit, qc_component_evaluation,
             qc_traces, save_all_post_cnmf
   :undoc-members:
   :show-inheritance:

params_estimator
----------------

.. automodule:: caiman.utils.params_estimator
   :members: estimate_params, apply_suggestions
   :undoc-members:
   :show-inheritance:

params_io
---------

.. automodule:: caiman.utils.params_io
   :members: ParamBag, load_pipeline_params, build_cnmf_opts
   :undoc-members:
   :show-inheritance:

tiff_io
-------

.. automodule:: caiman.utils.tiff_io
   :members: ensure_multipage_tiff, fc_convert_parallel, madvise_sequential
   :undoc-members:
   :show-inheritance:

param_summary
-------------

.. automodule:: caiman.utils.param_summary
   :members: log_params
   :undoc-members:
   :show-inheritance:
