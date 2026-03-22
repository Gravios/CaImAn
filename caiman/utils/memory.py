"""
caiman/utils/memory.py
======================
Low-level memory management helpers for large-dataset CaImAn pipelines.

Three independent utilities, each a one-liner at the call site:

``malloc_trim(logger)``
    Releases Python/C heap free-pages back to the OS immediately after
    a large ``del`` + ``gc.collect()``.  Without this, RSS stays elevated
    for minutes, causing the worker budget check to under-count available
    RAM and spawn fewer workers than optimal.

``madvise_dontneed(arr, logger)``
    Evicts a memory-mapped array from the kernel page cache via
    ``madvise(MADV_DONTNEED)``.  More reliable than ``posix_fadvise`` on
    FUSE filesystems (ntfs-3g silently ignores fadvise; MADV_DONTNEED is
    handled by the kernel VM subsystem directly).  Falls back to
    ``posix_fadvise`` if madvise fails.

``cupy_flush(logger, label)``
    Forces a full VRAM reclaim: Python GC → cuFFT plan cache → CuPy memory
    pool → CUDA device sync.  ``free_all_blocks()`` alone only returns *free*
    pool blocks; GC must run first to drop references from MC / Cn arrays so
    those blocks enter the free-list.  Without the FFT plan cache clear, cuFFT
    retains its own CUDA allocations outside the MemoryPool, leaving 12+ GB
    stranded before the precompute GPU filter attempts its first chunk alloc.

Usage
-----
    from caiman.utils.memory import malloc_trim, madvise_dontneed, cupy_flush

    # After a large del + gc.collect():
    malloc_trim(logger)

    # Before spawning CNMF workers — evict Cn-step page-cache pages:
    madvise_dontneed(Yr, logger)

    # Before GPU precompute — reclaim MC + Cn VRAM:
    cupy_flush(logger, label="before CNMF fit")
"""

from __future__ import annotations

import gc
import logging
from typing import Union

import numpy as np

from caiman.utils.timing import log_call as _log_call

_logger = logging.getLogger('caiman')


# ── malloc_trim ───────────────────────────────────────────────────────────────

@_log_call(_logger, level=logging.DEBUG, show_result=False)
def malloc_trim(logger: logging.Logger | None = None) -> None:
    """Release glibc free-list pages back to the OS.

    Calls ``malloc_trim(0)`` via ctypes.  No-op on non-Linux systems.

    Parameters
    ----------
    logger
        If provided, a DEBUG message is logged after the call.
    """
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        if logger is not None:
            logger.debug("malloc_trim(0): heap free-pages returned to OS")
    except Exception as exc:
        if logger is not None:
            logger.debug(f"malloc_trim skipped: {exc}")


# ── madvise_dontneed ──────────────────────────────────────────────────────────

def madvise_dontneed(
    arr: np.ndarray,
    logger: logging.Logger | None = None,
) -> None:
    """Evict a memory-mapped array from the kernel page cache.

    Uses ``madvise(MADV_DONTNEED)`` on the mapped VA range, which instructs the
    kernel to reclaim the physical pages immediately.  Falls back to
    ``posix_fadvise(POSIX_FADV_DONTNEED)`` via the file descriptor if madvise
    fails (e.g. on anonymous mappings or non-Linux systems).

    Silently does nothing if neither syscall is available.

    Parameters
    ----------
    arr
        The numpy memmap (or array with a ``_mmap`` attribute) to evict.
        If *arr* is not memory-mapped the call is a no-op.
    logger
        If provided, INFO messages are logged on success, DEBUG on failure.
    """
    _MADV_DONTNEED  = 4
    _FADV_DONTNEED  = 4

    # Walk the base chain to find the actual mmap.mmap object
    target = arr
    while hasattr(target, "base") and target.base is not None:
        target = target.base

    mm = getattr(target, "_mmap", None)
    if mm is None:
        return

    try:
        import ctypes
        libc  = ctypes.CDLL("libc.so.6", use_errno=True)
        buf   = (ctypes.c_char * 1).from_buffer(mm)
        addr  = ctypes.c_void_p(ctypes.addressof(buf))
        size  = ctypes.c_size_t(len(mm))
        rc    = libc.madvise(addr, size, ctypes.c_int(_MADV_DONTNEED))
        if rc == 0:
            if logger is not None:
                logger.info(
                    f"madvise(MADV_DONTNEED): evicted {len(mm) / 2**30:.1f} GB "
                    f"from page cache"
                )
            return
        # madvise returned non-zero — fall through to posix_fadvise
        errno = ctypes.get_errno()
        if logger is not None:
            logger.debug(
                f"madvise returned {rc} errno={errno} — trying posix_fadvise"
            )
    except Exception as exc:
        if logger is not None:
            logger.debug(f"madvise unavailable: {exc}")

    # posix_fadvise fallback (Linux/macOS file-backed mmaps)
    try:
        import os
        os.posix_fadvise(mm.fileno(), 0, 0, _FADV_DONTNEED)
        if logger is not None:
            logger.info("posix_fadvise(DONTNEED): page cache hint sent")
        return
    except Exception as exc:
        if logger is not None:
            logger.debug(f"posix_fadvise skipped: {exc}")

    # Windows: DiscardVirtualMemory (Win8+) for anonymous/pagefile-backed
    # regions. Has no effect on file-backed mmaps (no Windows equivalent).
    # Silently skipped on older Windows or if ctypes unavailable.
    try:
        import sys, ctypes
        if sys.platform == "win32":
            _kernel32 = ctypes.windll.kernel32   # type: ignore[attr-defined]
            _DiscardVirtualMemory = getattr(_kernel32, "DiscardVirtualMemory", None)
            if _DiscardVirtualMemory is not None:
                buf  = (ctypes.c_char * 1).from_buffer(mm)
                addr = ctypes.c_void_p(ctypes.addressof(buf))
                size = ctypes.c_size_t(len(mm))
                rc   = _DiscardVirtualMemory(addr, size)
                if logger is not None and rc != 0:
                    logger.debug(
                        f"DiscardVirtualMemory: evicted {len(mm)/2**30:.1f} GB "
                        f"(rc={rc})")
    except Exception as exc:
        if logger is not None:
            logger.debug(f"DiscardVirtualMemory skipped: {exc}")


# ── cupy_flush ────────────────────────────────────────────────────────────────

@_log_call(_logger, level=logging.DEBUG, show_result=False)
def cupy_flush(
    logger: logging.Logger | None = None,
    label: str = "",
) -> None:
    """Fully reclaim VRAM before a GPU-intensive step.

    Notes
    -----
    **Flush sequence:**

    1. ``gc.collect()`` — drops Python references so arrays are added to
       CuPy's free-list rather than staying live.
    2. ``cp.fft.config.get_plan_cache().clear()`` — releases cuFFT plans,
       which maintain their own CUDA allocations outside the MemoryPool.
    3. ``cp.get_default_memory_pool().free_all_blocks()`` — returns the now-
       fully-free pool blocks to CUDA.
    4. ``cp.get_default_pinned_memory_pool().free_all_blocks()`` — same for
       pinned (page-locked) CPU memory used by async transfers.
    5. ``cp.cuda.Device().synchronize()`` — ensures all CUDA work is complete
       before the caller queries free VRAM.

    No-op if CuPy is not installed or no GPU is present.

    Parameters
    ----------
    logger
        If provided, an INFO message is logged with pool stats after flushing.
    label
        Optional description appended to the log message, e.g.
        ``"before CNMF fit"``.
    """
    try:
        import cupy as cp
    except ImportError:
        return

    try:
        gc.collect()

        try:
            cp.fft.config.get_plan_cache().clear()
        except Exception:
            pass

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()

        free_mb = cp.get_default_memory_pool().free_bytes()  // 2**20
        used_mb = cp.get_default_memory_pool().used_bytes()  // 2**20

        if logger is not None:
            suffix = f" {label}" if label else ""
            logger.info(
                f"CuPy pool flushed{suffix} "
                f"(pool free={free_mb} MB  used={used_mb} MB)"
            )
    except Exception as exc:
        if logger is not None:
            logger.debug(f"cupy_flush failed: {exc}")
