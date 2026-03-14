"""
caiman/utils/tiff_io.py
=======================
Fast TIFF I/O utilities for large calcium imaging datasets.

All functions are designed to exploit the sequential bandwidth of fast NVMe
drives (e.g. Samsung 9100 Pro, PCIe 5.0, ~14 GB/s read / ~13 GB/s write) by:

  * Using time-slab I/O — reads/writes contiguous byte ranges that match the
    on-disk layout of F-order mmaps, so the kernel's sequential readahead is
    always correct.
  * Calling madvise(MADV_SEQUENTIAL | MADV_WILLNEED) on every mmap to pre-fault
    pages before they are needed.
  * Parallelising the memory-bound F→C transpose across all physical cores via
    multiprocessing.Pool (bypasses the GIL; each worker reopens files by path).

Public API
----------
    ensure_multipage_tiff(src_path)             -> str
    fc_convert_parallel(Yr_F, Yr_C, n_pixels, T, baseline, logger) -> None

Both functions are safe to import in worker subprocesses (no top-level side
effects, no GPU imports).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import multiprocessing
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("caiman")

# ── madvise ──────────────────────────────────────────────────────────────────

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_MADV_SEQUENTIAL = 2
_MADV_WILLNEED   = 3


def madvise_sequential(arr: np.ndarray) -> None:
    """Hint the kernel that *arr* will be read sequentially.

    Applies MADV_SEQUENTIAL (expand readahead window) and MADV_WILLNEED
    (start async prefetch immediately).  Silently no-ops on non-Linux or
    if madvise is unavailable.
    """
    try:
        addr   = arr.ctypes.data
        length = arr.nbytes
        _libc.madvise(
            ctypes.c_void_p(addr), ctypes.c_size_t(length),
            ctypes.c_int(_MADV_SEQUENTIAL),
        )
        _libc.madvise(
            ctypes.c_void_p(addr), ctypes.c_size_t(length),
            ctypes.c_int(_MADV_WILLNEED),
        )
    except Exception:
        pass


# ── Multipage TIFF conversion ─────────────────────────────────────────────────

def ensure_multipage_tiff(src_path: str) -> str:
    """Ensure *src_path* is a contiguous multi-page BigTIFF.

    If the file already has more than one page it is returned unchanged.
    If it is a single-strip TIFF (all frames in one page), a new
    ``<stem>_mp.tif`` BigTIFF is written next to it and its path returned.

    The source is memory-mapped (no full-file RAM copy).  Frames are
    batched into ~256 MB chunks before being passed to tifffile so that
    each write is a large sequential I/O rather than thousands of small
    ones.  madvise(SEQUENTIAL) is applied to the source mmap.

    Re-conversion is skipped if the destination is newer than the source.
    """
    import tifffile

    dst_path = os.path.splitext(src_path)[0] + "_mp.tif"

    with tifffile.TiffFile(src_path) as tf:
        if len(tf.pages) != 1:
            return src_path

    if (os.path.exists(dst_path) and
            os.path.getmtime(dst_path) >= os.path.getmtime(src_path)):
        logger.info(f"Reusing multi-page TIFF: {dst_path}")
        return dst_path

    logger.info(f"Converting single-page TIFF → multi-page: {dst_path}")

    with tifffile.TiffFile(src_path) as tf:
        arr = tf.asarray(out="memmap")          # mmap — no RAM copy
        madvise_sequential(arr)

        T_src = arr.shape[0]
        H, W  = arr.shape[1], arr.shape[2]

        # Batch size: target ~256 MB per write burst
        target_bytes = 256 * 2**20
        batch = max(1, target_bytes // (H * W * arr.dtype.itemsize))

        with tifffile.TiffWriter(dst_path, bigtiff=True) as tw:
            for b0 in range(0, T_src, batch):
                b1    = min(b0 + batch, T_src)
                chunk = np.ascontiguousarray(arr[b0:b1])   # one large read
                for i in range(b1 - b0):
                    tw.write(chunk[i], contiguous=True)
                del chunk

    logger.info(f"Multi-page TIFF ready: {dst_path}")
    return dst_path


# ── Parallel F→C mmap conversion ─────────────────────────────────────────────

def _slab_worker(args: tuple) -> tuple[int, int]:
    """
    Subprocess worker: copy one pixel-slab from the F-order source mmap into
    the C-order destination mmap, fusing *baseline* addition in-place.

    Pixel slabs are the correct outer loop for F→C conversion:

      F-order (n_pixels, T): pixel is fast axis → Yr_F[p0:p1, :] requires
        T strided reads of (p1-p0) pixels each, but strides are 1 MB apart
        so the OS read-ahead still works reasonably.

      C-order (n_pixels, T): pixel is slow axis → Yr_C[p0:p1, :] is a
        single contiguous block of (p1-p0)*T*4 bytes → sequential write ✓

    Time slabs look appealing (sequential read) but produce 262,144 scattered
    1 KB writes at 110 KB stride per slab — catastrophic for any storage.
    """
    import numpy as np

    fname_in, fname_out, shape_in, p0, p1, n_pixels, T, baseline = args

    Yr_F_w = np.memmap(fname_in,  dtype=np.float32, mode='r',
                       shape=shape_in, order='F')
    Yr_C_w = np.memmap(fname_out, dtype=np.float32, mode='r+',
                       shape=(n_pixels, T), order='C')

    # Read: T strided reads of (p1-p0) pixels — acceptable for OS prefetcher.
    # np.array forces a contiguous copy so the write below is buffer→mmap.
    slab = np.array(Yr_F_w[p0:p1, :], dtype=np.float32)   # (p1-p0, T)

    # Write: single contiguous block at byte offset p0*T*4 → sequential ✓
    np.add(slab, baseline, out=Yr_C_w[p0:p1, :])

    # No flush here — let the kernel coalesce dirty pages from all
    # workers into a single sequential writeback after pool.join().
    del Yr_F_w, Yr_C_w, slab
    return p0, p1


def fc_convert_parallel(
    Yr_F: np.ndarray,
    Yr_C: np.ndarray,
    n_pixels: int,
    T: int,
    baseline: float,
    log: Optional[logging.Logger] = None,
) -> None:
    """Convert a (n_pixels, T) F-order mmap to a (n_pixels, T) C-order mmap.

    Replaces the single-core pixel-slab loop with a multiprocessing pool
    that transposes time-slabs in parallel, exploiting all physical cores
    and the sequential bandwidth of fast NVMe drives.

    Parameters
    ----------
    Yr_F      : np.memmap, shape (n_pixels, T), F-order — source
    Yr_C      : np.memmap, shape (n_pixels, T), C-order — destination
    n_pixels  : int — product of spatial dimensions
    T         : int — number of frames
    baseline  : float — scalar added to every element (fused with transpose)
    log       : logging.Logger — defaults to the caiman logger if None
    """
    import psutil

    if log is None:
        log = logger

    # madvise both arrays.
    # Yr_F: strided reads (pixel fast, time slow) — SEQUENTIAL hint still
    #        helps the kernel expand its readahead window for the file.
    # Yr_C: pixel-slab writes are contiguous → SEQUENTIAL is exact.
    madvise_sequential(Yr_F)
    madvise_sequential(Yr_C)

    n_cores = psutil.cpu_count(logical=False) or 4

    # Pixel-slab sizing: each worker writes one contiguous C-order block.
    # Cap per-worker RAM at 40 % of free memory divided across all workers.
    free_bytes   = psutil.virtual_memory().available
    px_per_worker = max(1, min(
        int(free_bytes * 0.40) // (n_cores * T * 4),
        n_pixels // n_cores,
    ))
    # Round up to a clean multiple so the last worker isn't tiny
    n_passes  = -(-n_pixels // (px_per_worker * n_cores))
    # Recompute with the rounded pass count so work is evenly distributed
    px_total_per_pass = -(-n_pixels // n_passes)
    px_per_worker     = -(-px_total_per_pass // n_cores)

    n_slabs  = -(-n_pixels // px_per_worker)
    slab_mb  = px_per_worker * T * 4 / 2**20

    log.info(
        f"F→C parallel: pixel_chunk={px_per_worker} px  "
        f"slab={slab_mb:.0f} MB  slabs={n_slabs}  workers={n_cores}"
    )

    fname_in  = Yr_F.filename
    fname_out = Yr_C.filename
    shape_in  = Yr_F.shape

    slabs = [
        (fname_in, fname_out, shape_in,
         p, min(p + px_per_worker, n_pixels),
         n_pixels, T, baseline)
        for p in range(0, n_pixels, px_per_worker)
    ]

    with multiprocessing.Pool(processes=min(n_cores, n_slabs)) as pool:
        for p0, p1 in pool.imap_unordered(_slab_worker, slabs, chunksize=1):
            log.debug(f"  F→C slab pix {p0}–{p1} done")

    # Single flush after all workers complete — coalesces dirty pages
    # into one sequential writeback stream at the 9100 Pro's TLC rate.
    Yr_C.flush()
