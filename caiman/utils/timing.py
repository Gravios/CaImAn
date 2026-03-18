"""
caiman/utils/timing.py
======================
Step-level timing and resource tracking for CaImAn pipelines.

Two usage patterns are supported:

Context manager (primary interface for inline pipeline steps)::

    with timer("Motion correction"):
        mc.motion_correct(save_movie=True)

Decorator (for steps expressed as named functions)::

    @timer.step("Motion correction")
    def run_motion_correction():
        mc.motion_correct(save_movie=True)

    run_motion_correction()   # logs START/DONE, records the event

Both produce identical log output and event records.

``@log_call`` is a lightweight decorator for utility functions.  Logs the
function name, key arguments, and elapsed time at DEBUG level without
creating a full timer event::

    @log_call(logger)
    def ensure_model_files(model_dir):
        ...
    # DEBUG  → ensure_model_files(model_dir=/data/caiman/model)
    # DEBUG  ← ensure_model_files  0.02s  → True

See :func:`write_report` to produce a timing summary after the pipeline.
"""

from __future__ import annotations

import contextlib
import datetime
import functools
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Callable, Union


# ── Resource helpers ──────────────────────────────────────────────────────────

def _rss_gb() -> float:
    """Resident Set Size of the current process in GiB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return 0.0


def _shm_gb() -> float:
    """Used space under ``/dev/shm`` in GiB (0 if unavailable)."""
    try:
        import psutil
        return psutil.disk_usage("/dev/shm").used / 2**30
    except Exception:
        return 0.0


def _vram_mb() -> float:
    """CuPy pool used VRAM in MiB (0 if CuPy unavailable or no GPU)."""
    try:
        import cupy as cp
        return cp.get_default_memory_pool().used_bytes() / 2**20
    except Exception:
        return 0.0


def fmt_elapsed(s: float) -> str:
    """Format elapsed seconds as a compact human string.

    Examples: ``"0.83s"``, ``"2m 15.4s"``, ``"1h 03m 42.1s"``
    """
    m, s = divmod(s, 60)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h}h {m:02d}m {s:04.1f}s"
    elif m:
        return f"{m}m {s:04.1f}s"
    else:
        return f"{s:.2f}s"


# ── PipelineTimer ─────────────────────────────────────────────────────────────

class PipelineTimer:
    """Context-manager and decorator factory for timing pipeline steps.

    Parameters
    ----------
    logger
        Logger to write START / DONE messages to.

    Examples
    --------
    Context-manager usage::

    >>> timer = PipelineTimer(logger)
    >>> with timer("Motion correction"):
    ...     mc.motion_correct(save_movie=True)

    Decorator usage::

    >>> @timer.step("Motion correction")
    ... def run_motion_correction():
    ...     mc.motion_correct(save_movie=True)
    ...
    >>> run_motion_correction()   # logs START/DONE and records event

    Both styles produce identical log lines and event records.

    Accumulated records are in ``timer.events`` — a list of dicts with keys
    ``label``, ``elapsed``, ``human``, ``rss_start_gb``, ``rss_end_gb``,
    ``rss_delta_gb``, ``shm_end_gb``, ``vram_end_mb``.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.events: list[dict] = []
        self.t0: float | None = None

    def _record(self, label: str):
        """Internal context manager shared by __call__ and step."""
        @contextlib.contextmanager
        def _ctx():
            if self.t0 is None:
                self.t0 = time.perf_counter()

            rss0  = _rss_gb()
            shm0  = _shm_gb()
            vram0 = _vram_mb()
            t     = time.perf_counter()

            self.logger.info(
                f"STARTED  {label}  "
                f"[RSS {rss0:.1f} GB  SHM {shm0:.1f} GB  VRAM {vram0:.0f} MB]"
            )
            try:
                yield
            finally:
                elapsed = time.perf_counter() - t
                rss1    = _rss_gb()
                shm1    = _shm_gb()
                vram1   = _vram_mb()
                human   = fmt_elapsed(elapsed)
                self.logger.info(
                    f"DONE     {label}  —  {human}  ({elapsed:.1f} s)  "
                    f"[RSS {rss1:.1f} GB (Δ{rss1-rss0:+.1f})  "
                    f"SHM {shm1:.1f} GB  VRAM {vram1:.0f} MB]"
                )
                self.events.append({
                    "label":         label,
                    "elapsed":       round(elapsed, 3),
                    "human":         human,
                    "rss_start_gb":  round(rss0, 2),
                    "rss_end_gb":    round(rss1, 2),
                    "rss_delta_gb":  round(rss1 - rss0, 2),
                    "shm_end_gb":    round(shm1, 2),
                    "vram_end_mb":   round(vram1, 1),
                })
        return _ctx()

    @contextlib.contextmanager
    def __call__(self, label: str):
        """Context manager: ``with timer("label"): ...``"""
        with self._record(label):
            yield

    def step(self, label: str) -> Callable:
        """Decorator factory: ``@timer.step("label")``

        Wraps a function so that calling it is equivalent to::

            with timer("label"):
                fn(*args, **kwargs)

        The decorated function's name and docstring are preserved via
        ``functools.wraps``.

        Parameters
        ----------
        label
            Human-readable step label — identical to what you would pass to
            the context-manager form.

        Examples
        --------
        >>> @timer.step("F\\u2192C mmap conversion")
        ... def convert():
        ...     fc_convert_parallel(Yr_F, Yr_C, n_px, T, ADD_BASELINE, logger)
        ...
        >>> convert()
        """
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self._record(label):
                    return fn(*args, **kwargs)
            return wrapper
        return decorator

    def total_elapsed(self) -> float:
        """Seconds since the first timed step (0 if no steps have run yet)."""
        if self.t0 is None:
            return 0.0
        return time.perf_counter() - self.t0


# ── log_call ──────────────────────────────────────────────────────────────────

def log_call(
    logger: logging.Logger,
    *,
    level: int = logging.DEBUG,
    show_args: bool = True,
    show_result: bool = True,
) -> Callable:
    """Decorator that logs a function's entry, result, and elapsed time.

    Intended for utility functions (e.g. ``ensure_model_files``,
    ``cupy_flush``) where you want call-level visibility in the debug log
    without creating a full timer event.

    Log format
    ----------
    On entry::

        → ensure_model_files(model_dir=/data/caiman/model)

    On return::

        ← ensure_model_files  0.02s  → True

    On exception::

        ✗ ensure_model_files  0.01s  raised ValueError: ...

    Parameters
    ----------
    logger
        Logger to write to.
    level
        Logging level for entry/exit lines (default: ``DEBUG``).
    show_args
        Log the bound arguments on entry.  Arguments whose ``repr`` exceeds
        120 characters are truncated to keep log lines readable.
    show_result
        Log the return value on exit.  Truncated at 80 characters.

    Examples
    --------
    >>> @log_call(logger)
    ... def ensure_model_files(model_dir):
    ...     ...

    >>> @log_call(logger, level=logging.INFO, show_result=False)
    ... def clean_stale_shm(shm_dir, temp_dir, logger):
    ...     ...
    """
    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if show_args:
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    parts = []
                    for k, v in bound.arguments.items():
                        if k == "logger":      # never log logger objects
                            continue
                        r = repr(v)
                        if len(r) > 120:
                            r = r[:117] + "…"
                        parts.append(f"{k}={r}")
                    arg_str = f"({', '.join(parts)})"
                except Exception:
                    arg_str = "(...)"
            else:
                arg_str = "()"

            logger.log(level, f"→ {fn.__name__}{arg_str}")
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                if show_result:
                    r = repr(result)
                    if len(r) > 80:
                        r = r[:77] + "…"
                    logger.log(level,
                        f"← {fn.__name__}  {fmt_elapsed(elapsed)}  → {r}")
                else:
                    logger.log(level, f"← {fn.__name__}  {fmt_elapsed(elapsed)}")
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.log(level,
                    f"✗ {fn.__name__}  {fmt_elapsed(elapsed)}  "
                    f"raised {type(exc).__name__}: {exc}")
                raise

        return wrapper
    return decorator


# ── write_report ──────────────────────────────────────────────────────────────

def write_report(
    timer: PipelineTimer,
    session: str,
    outdir: Union[str, Path],
    logger: logging.Logger,
) -> None:
    """Write a human-readable timing table and a JSON summary.

    Output files
    ------------
    ``<outdir>/<session>_report.txt``
        Fixed-width table: step name, wall time, % of total, RSS end,
        ΔRSS, SHM, VRAM.  Slowest three steps highlighted.
    ``<outdir>/<session>_report.json``
        Machine-readable dict with the same data plus peak resource values.

    Parameters
    ----------
    timer
        :class:`PipelineTimer` whose ``events`` list to summarise.
    session
        Session identifier used in file names and the report header.
    outdir
        Output directory — same folder as the pipeline results.
    logger
        Logger to announce the written paths.
    """
    outdir = Path(outdir)
    report_path = outdir / f"{session}_report.txt"
    json_path   = outdir / f"{session}_report.json"

    events        = timer.events
    total_elapsed = timer.total_elapsed()
    now           = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 72)
    lines.append("  CaImAn Pipeline Report")
    lines.append(f"  Session   : {session}")
    lines.append(f"  Generated : {now}")
    lines.append(f"  Wall time : {fmt_elapsed(total_elapsed)}  ({total_elapsed:.1f} s)")
    lines.append("=" * 72)
    lines.append("")

    if events:
        col_w  = max(len(e["label"]) for e in events) + 2
        header = (
            f"  {'Step':<{col_w}}  {'Time':>10}  {'%':>5}  "
            f"{'RSS end':>9}  {'ΔRSS':>7}  {'SHM':>6}  {'VRAM':>8}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for e in events:
            pct = 100.0 * e["elapsed"] / total_elapsed if total_elapsed > 0 else 0
            lines.append(
                f"  {e['label']:<{col_w}}  {e['human']:>10}  {pct:>4.1f}%  "
                f"{e['rss_end_gb']:>7.1f} GB  "
                f"{e['rss_delta_gb']:>+6.1f}  "
                f"{e['shm_end_gb']:>4.1f} GB  "
                f"{e['vram_end_mb']:>6.0f} MB"
            )

        lines.append("")
        lines.append("  " + "-" * (len(header) - 2))

        top = sorted(events, key=lambda e: e["elapsed"], reverse=True)[:3]
        lines.append("  Slowest steps:")
        for i, e in enumerate(top, 1):
            pct = 100.0 * e["elapsed"] / total_elapsed if total_elapsed > 0 else 0
            lines.append(f"    {i}. {e['label']}  {e['human']}  ({pct:.1f}%)")
        lines.append("")

        peak_rss  = max(e["rss_end_gb"]  for e in events)
        peak_shm  = max(e["shm_end_gb"]  for e in events)
        peak_vram = max(e["vram_end_mb"] for e in events)
    else:
        peak_rss = peak_shm = peak_vram = 0.0

    lines.append(f"  Peak RSS  : {peak_rss:.1f} GB")
    lines.append(f"  Peak SHM  : {peak_shm:.1f} GB")
    lines.append(f"  Peak VRAM : {peak_vram:.0f} MB")
    lines.append("")
    lines.append("=" * 72)

    report_txt = "\n".join(lines) + "\n"
    report_path.write_text(report_txt)

    summary = {
        "session":          session,
        "generated":        now,
        "total_elapsed_s":  round(total_elapsed, 3),
        "steps":            events,
        "peak_rss_gb":      round(peak_rss, 2),
        "peak_shm_gb":      round(peak_shm, 2),
        "peak_vram_mb":     round(peak_vram, 1),
    }
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + report_txt)
    logger.info(f"Report written to {report_path}")
    logger.info(f"JSON summary written to {json_path}")
