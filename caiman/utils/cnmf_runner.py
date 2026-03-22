"""
caiman/utils/cnmf_runner.py
===========================
Config-driven CNMF orchestrator.

``CNMFRunner`` is constructed once from the pipeline ``ParamBag``, session
name, output directory, and runtime values known after motion correction
(movie path, spatial dims, border pixels, cluster handle).  Each method
maps directly to one CNMF pipeline stage, pulling all parameter values from
the stored config so call sites contain only the data objects that change.

Typical usage
-------------
    from caiman.utils.cnmf_runner import CNMFRunner

    runner = CNMFRunner(
        _P, session, outdir,
        fname_mc   = fname_mc,
        fname_cnmf = fname_cnmf,
        dims       = dims,
        bord_px    = bord_px,
        dview      = dview,
        n_processes= n_processes,
        cnn_available = _cnn_available,
    )

    cnm  = runner.fit(images, forder_movie_path=fname_mc)
    cnm2 = runner.refit(cnm, images)
    runner.evaluate(cnm2, images)
    runner.select(cnm2)
    runner.detrend_dff(cnm2)
    runner.save(cnm2)

All CNMF parameter values (gSig, rf, K, merge_thr, …) come from the JSON
config via the ``ParamBag``.  Memory management calls (madvise, malloc_trim,
cupy_flush) are integrated at the appropriate points so the pipeline body
does not need to know about them.

The ``CNMFRunner`` does not own the cluster lifecycle — the pipeline is
responsible for ``cm.cluster.setup_cluster`` and ``cm.stop_server`` so the
``finally`` block remains in the pipeline where it belongs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("caiman")


class CNMFRunner:
    """Config-driven CNMF orchestrator.

    Parameters
    ----------
    P
        ``ParamBag`` from :func:`~caiman.utils.params_io.load_pipeline_params`.
    session
        Session identifier (script stem minus ``_pipeline``).
    outdir
        Output directory for result files.
    fname_mc
        Path to the F-order motion-corrected mmap.  Passed to CNMF as
        ``_forder_movie_path`` to enable fast frame reads during precompute.
    fname_cnmf
        Path to the C-order CNMF mmap.
    dims
        Spatial dimensions ``(d1, d2)`` from ``load_memmap``.
    bord_px
        Border pixels to exclude (0 when ``border_nan="copy"``).
    dview
        ipyparallel or multiprocessing dview from ``cm.cluster.setup_cluster``.
    n_processes
        Number of worker processes.
    cnn_available
        Whether CNN classifier model files are present.  If ``False``,
        ``use_cnn`` is forced off regardless of the JSON value.

    Examples
    --------
    >>> runner = CNMFRunner(_P, session, outdir,
    ...     fname_mc=fname_mc, fname_cnmf=fname_cnmf,
    ...     dims=dims, bord_px=bord_px,
    ...     dview=dview, n_processes=n_processes,
    ...     cnn_available=_cnn_available)
    >>>
    >>> cnm  = runner.fit(images)
    >>> cnm2 = runner.refit(cnm, images)
    >>> runner.evaluate(cnm2, images)
    >>> runner.select(cnm2)
    >>> runner.detrend_dff(cnm2)
    >>> runner.save(cnm2)
    """

    def __init__(
        self,
        P,
        session: str,
        outdir: Union[str, Path],
        *,
        fname_mc:     str,
        fname_cnmf:   str,
        dims:         tuple,
        bord_px:      int,
        dview,
        n_processes:  int,
        cnn_available: bool = True,
    ) -> None:
        self._P             = P
        self._session       = session
        self._outdir        = Path(outdir)
        self._fname_mc      = fname_mc
        self._fname_cnmf    = fname_cnmf
        self._dims          = dims
        self._bord_px       = bord_px
        self._dview         = dview
        self._n_processes   = n_processes
        self._cnn_available = cnn_available

        # Cache frequently-used leaves
        c = P.cnmf
        self._p      = int(c.p)
        self._rf     = int(c.rf)
        self._stride = int(c.stride)
        self._dff_quantile = int(getattr(c, "dff_quantile_min", 8))
        self._dff_window   = int(getattr(c, "dff_frames_window", 500))

        # Build the CNMFParams object once — reused across fit / refit
        from caiman.utils.params_io import build_cnmf_opts
        self._opts = build_cnmf_opts(
            P,
            fname_cnmf    = fname_cnmf,
            dims          = dims,
            bord_px       = bord_px,
            n_processes   = n_processes,
            cnn_available = cnn_available,
        )

    # ── Memory management helpers ─────────────────────────────────────────────

    def _pre_fit_memory(self, Yr: np.ndarray) -> None:
        """Evict page-cache and flush GPU pool before fitting.

        Called automatically by :meth:`fit`.  Exposed as a public-ish method
        so it can be called manually if the pipeline reloads ``images`` between
        construction and the fit call.

        Parameters
        ----------
        Yr
            The raw mmap array backing *images* — used for madvise eviction.
        """
        from caiman.utils.memory import madvise_dontneed, malloc_trim, cupy_flush
        madvise_dontneed(Yr, logger)
        malloc_trim(logger)
        cupy_flush(logger, label="before CNMF fit")

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def fit(
        self,
        images: np.ndarray,
        Yr: Optional[np.ndarray] = None,
    ):
        """Construct a ``CNMF`` object and run the initial fit.

        Applies memory eviction before fitting, sets ``_forder_movie_path``
        on the CNMF object so the precompute step reads frames as fast
        sequential blocks rather than scattered reads.

        Parameters
        ----------
        images
            Zero-copy ``(T, d1, d2)`` mmap view of the C-order movie.
        Yr
            Underlying 2-D mmap ``(pixels, T)`` — used for madvise page
            eviction.  If ``None`` the eviction step is skipped.

        Returns
        -------
        caiman.source_extraction.cnmf.cnmf.CNMF
            Fitted CNMF object.
        """
        from caiman.source_extraction import cnmf as _cnmf_mod

        cnm = _cnmf_mod.CNMF(self._n_processes,
                              params=self._opts, dview=self._dview)
        cnm._forder_movie_path = self._fname_mc

        if Yr is not None:
            self._pre_fit_memory(Yr)

        logger.info("CNMF fit: starting")
        cnm.fit(images)
        logger.info(f"CNMF fit: done  —  {cnm.estimates.A.shape[1]} raw components")
        return cnm

    def refit(self, cnm, images: np.ndarray):
        """Run the full-AR refit on the accepted components from ``fit()``.

        Switches ``p`` from 0 (initialisation) to the configured AR order,
        disables ``only_init`` so the full spatial/temporal update runs, and
        calls ``cnm.refit()``.

        Parameters
        ----------
        cnm
            ``CNMF`` object returned by :meth:`fit`.
        images
            Same ``(T, d1, d2)`` mmap view used for the initial fit.

        Returns
        -------
        caiman.source_extraction.cnmf.cnmf.CNMF
            New ``CNMF`` object produced by ``refit()``.
        """
        self._opts.set("preprocess", {"p": self._p})
        self._opts.set("patch", {
            "only_init": False,
            "rf":        self._rf,
            "stride":    self._stride,
        })
        logger.info("CNMF refit: starting")
        cnm2 = cnm.refit(images, dview=self._dview)
        logger.info(f"CNMF refit: done  —  {cnm2.estimates.A.shape[1]} components")
        return cnm2

    def evaluate(self, cnm2, images: np.ndarray) -> tuple[int, int]:
        """Run component quality evaluation.

        Scores each component against SNR, spatial correlation, and CNN
        criteria configured in the JSON ``quality`` section.  Retries once
        if ``idx_components`` is ``None`` after the first call (CaImAn bug
        guard).

        Parameters
        ----------
        cnm2
            ``CNMF`` object returned by :meth:`refit`.
        images
            Movie array — same as used for fit / refit.

        Returns
        -------
        tuple[int, int]
            ``(n_accepted, n_rejected)`` component counts.
        """
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=None)
        if cnm2.estimates.idx_components is None:
            logger.warning("idx_components is None — re-running evaluate_components")
            cnm2.estimates.evaluate_components(images, cnm2.params, dview=None)

        idx_acc = cnm2.estimates.idx_components
        idx_rej = cnm2.estimates.idx_components_bad
        if idx_acc is None:
            logger.error("evaluate_components: idx_components still None after retry — "
                         "treating all components as rejected")
            idx_acc, idx_rej = [], list(range(cnm2.estimates.A.shape[1]))
        if idx_rej is None:
            idx_rej = []
        n_acc = len(idx_acc)
        n_rej = len(idx_rej)
        logger.info(f"Components: {n_acc} accepted / {n_rej} rejected")
        return n_acc, n_rej

    def select(self, cnm2) -> None:
        """Discard rejected components in-place.

        Calls ``select_components(use_object=True)`` which prunes ``A``,
        ``C``, ``S``, etc. to the accepted subset only.

        Parameters
        ----------
        cnm2
            Evaluated ``CNMF`` object (after :meth:`evaluate`).
        """
        cnm2.estimates.select_components(use_object=True)
        logger.info(f"Selected {cnm2.estimates.A.shape[1]} components")

    def detrend_dff(self, cnm2) -> None:
        """Compute dF/F traces in-place.

        Uses ``quantileMin`` and ``frames_window`` from the JSON ``cnmf``
        section (keys ``dff_quantile_min`` and ``dff_frames_window``,
        defaulting to 8 and 500 respectively).

        Parameters
        ----------
        cnm2
            Selected ``CNMF`` object (after :meth:`select`).
        """
        logger.info(
            f"dF/F: quantileMin={self._dff_quantile}  "
            f"frames_window={self._dff_window}"
        )
        cnm2.estimates.detrend_df_f(
            quantileMin   = self._dff_quantile,
            frames_window = self._dff_window,
        )

    def save(self, cnm2) -> str:
        """Save the CNMF estimates to ``<outdir>/<session>_results.hdf5``.

        Parameters
        ----------
        cnm2
            Final ``CNMF`` object.

        Returns
        -------
        str
            Absolute path of the saved HDF5 file.
        """
        path = str(self._outdir / f"{self._session}_results.hdf5")
        cnm2.save(path)
        logger.info(f"Results saved: {path}")
        return path

    def load(self):
        """Load a previously saved result from ``<outdir>/<session>_results.hdf5``.

        Returns
        -------
        caiman.source_extraction.cnmf.cnmf.CNMF
            The loaded CNMF object, or ``None`` if the file does not exist.
        """
        import caiman as cm
        path = self._outdir / f"{self._session}_results.hdf5"
        if not path.exists():
            logger.warning(f"Results file not found: {path}")
            return None
        logger.info(f"Loading results from {path}")
        return cm.source_extraction.cnmf.cnmf.load_CNMF(str(path))

    # ── Convenience: run all stages in sequence ───────────────────────────────

    def run_all(
        self,
        images: np.ndarray,
        Yr: Optional[np.ndarray] = None,
        qc=None,
        Cn: Optional[np.ndarray] = None,
        timer=None,
    ):
        """Run fit → refit → evaluate → select → dF/F → save in one call.

        Optionally integrates with a :class:`~caiman.utils.qc.QCRunner` and a
        :class:`~caiman.utils.timing.PipelineTimer` if supplied.

        Parameters
        ----------
        images
            Movie array ``(T, d1, d2)``.
        Yr
            Underlying mmap for page eviction (optional).
        qc
            :class:`~caiman.utils.qc.QCRunner` instance.  If supplied,
            QC figures are saved at each stage.
        Cn
            Correlation image passed to QC footprint plots (optional).
        timer
            :class:`~caiman.utils.timing.PipelineTimer` instance.  If
            supplied each stage is wrapped in a timed step.

        Returns
        -------
        caiman.source_extraction.cnmf.cnmf.CNMF
            Final selected CNMF object with dF/F computed.
        """
        def _step(label, fn):
            if timer is not None:
                import contextlib
                with timer(label):
                    return fn()
            return fn()

        cnm = _step("CNMF fit",    lambda: self.fit(images, Yr=Yr))
        if qc is not None:
            _step("QC: initial fit", lambda: qc.cnmf_fit(cnm, Cn))

        cnm2 = _step("CNMF refit", lambda: self.refit(cnm, images))
        del cnm

        n_acc, n_rej = _step("Component evaluation",
                             lambda: self.evaluate(cnm2, images))

        if qc is not None:
            _step("QC: refit + evaluation", lambda: (
                qc.cnmf_refit(cnm2, Cn),
                qc.component_evaluation(cnm2, Cn),
            ))

        self.select(cnm2)
        _step("dF/F computation",  lambda: self.detrend_dff(cnm2))

        if qc is not None:
            _step("QC: traces", lambda: qc.traces(cnm2))

        _step("Save results", lambda: self.save(cnm2))
        return cnm2
