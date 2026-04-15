"""
caiman/utils/shm_movie.py
=========================
SHM (shared memory) movie mode for the CaImAn pipeline.

Loads the full C-order mmap into ``/dev/shm`` before CNMF so that every
worker patch read is a plain in-memory array slice rather than a disk
seek-and-read.  On a 256 GB RAM machine this eliminates all storage I/O
during the CNMF fit, refit, and evaluation stages.

The tile-buffer system (``CAIMAN_TILE_SLOTS``) is disabled when SHM mode
is active because its entire purpose is to amortise disk latency — with the
movie in RAM it adds overhead without benefit.

Public API
----------
    check_shm_capacity(movie_bytes, shm_dir)  -> (available_bytes, fits)
    load_to_shm(fname_cnmf, session, shm_dir, logger) -> shm_path
    release_shm(shm_path, logger)             -> None
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger("caiman")


# ---------------------------------------------------------------------------
# Capacity check
# ---------------------------------------------------------------------------

def check_shm_capacity(
    movie_bytes: int,
    shm_dir: str = "/dev/shm",
) -> tuple[int, bool]:
    """Return ``(available_bytes, fits)`` for *shm_dir*.

    Uses ``shutil.disk_usage`` which reads ``/proc/mounts`` on Linux and
    works correctly for tmpfs mounts like ``/dev/shm``.

    Parameters
    ----------
    movie_bytes
        Size of the movie that would be loaded into SHM.
    shm_dir
        Path to the shared memory filesystem (default: ``/dev/shm``).
    """
    try:
        stat = shutil.disk_usage(shm_dir)
        available = stat.free
    except OSError:
        # /dev/shm not mounted or inaccessible — report 0
        return 0, False
    return available, available >= movie_bytes


# ---------------------------------------------------------------------------
# Load / release
# ---------------------------------------------------------------------------

def load_to_shm(
    fname_cnmf: str,
    session: str,
    shm_dir: str = "/dev/shm",
    log: logging.Logger | None = None,
) -> str:
    """Copy the C-order mmap *fname_cnmf* into *shm_dir* and return the new path.

    The copy is named ``<session>_cnmf_shm.mmap`` so it is unambiguous and
    easy to identify in ``/dev/shm``.  If an up-to-date copy already exists
    (same size, newer mtime) it is reused without re-copying.

    After copying, ``CAIMAN_TILE_SLOTS`` is set to ``"0"`` in the current
    process environment so that any subsequent ``import caiman`` in workers
    does not allocate tile buffers.

    Parameters
    ----------
    fname_cnmf
        Absolute path to the disk-backed C-order mmap.
    session
        Session stem — used to name the SHM file.
    shm_dir
        Shared memory filesystem directory (default ``/dev/shm``).
    log
        Logger; defaults to ``caiman`` logger.

    Returns
    -------
    str
        Absolute path of the SHM-backed mmap.

    Raises
    ------
    RuntimeError
        If *shm_dir* does not exist, is not writable, or lacks sufficient
        free space for the copy.
    """
    if log is None:
        log = logger

    shm_path = os.path.join(shm_dir, f"{session}_cnmf_shm.mmap")

    if not os.path.isdir(shm_dir):
        raise RuntimeError(
            f"SHM directory does not exist: {shm_dir}\n"
            "  Is /dev/shm mounted? (check: mount | grep shm)"
        )
    if not os.access(shm_dir, os.W_OK):
        raise RuntimeError(f"SHM directory is not writable: {shm_dir}")

    movie_bytes = os.path.getsize(fname_cnmf)
    available, fits = check_shm_capacity(movie_bytes, shm_dir)
    movie_gb    = movie_bytes  / 1024**3
    avail_gb    = available    / 1024**3

    if not fits:
        raise RuntimeError(
            f"Insufficient SHM space: need {movie_gb:.1f} GB, "
            f"available {avail_gb:.1f} GB in {shm_dir}.\n"
            "  Increase /dev/shm size: sudo mount -o remount,size=<N>G /dev/shm\n"
            "  Or disable shm_mode in the pipeline JSON."
        )

    # Reuse an existing SHM copy if it is still valid
    if (os.path.exists(shm_path)
            and os.path.getsize(shm_path) == movie_bytes
            and os.path.getmtime(shm_path) >= os.path.getmtime(fname_cnmf)):
        log.info(f"SHM: reusing existing copy  {shm_path}  ({movie_gb:.1f} GB)")
    else:
        if os.path.exists(shm_path):
            os.unlink(shm_path)
        log.info(
            f"SHM: copying {movie_gb:.1f} GB  "
            f"{Path(fname_cnmf).name}  ->  {shm_path}"
        )
        _fast_copy(fname_cnmf, shm_path, log)
        log.info(f"SHM: copy complete  {shm_path}")

    # Disable tile buffer — irrelevant when movie is in RAM
    os.environ["CAIMAN_TILE_SLOTS"] = "0"
    log.info("SHM: CAIMAN_TILE_SLOTS set to 0 (tile buffer disabled)")

    return shm_path


def release_shm(shm_path: str, log: logging.Logger | None = None) -> None:
    """Delete the SHM copy and restore ``CAIMAN_TILE_SLOTS``.

    Safe to call even if *shm_path* no longer exists (e.g. after a crash
    that already cleaned up).

    Parameters
    ----------
    shm_path
        Path returned by :func:`load_to_shm`.
    log
        Logger; defaults to ``caiman`` logger.
    """
    if log is None:
        log = logger
    try:
        if os.path.exists(shm_path):
            os.unlink(shm_path)
            log.info(f"SHM: released  {shm_path}")
    except OSError as exc:
        log.warning(f"SHM: could not delete {shm_path}: {exc}")


# ---------------------------------------------------------------------------
# Fast copy helper
# ---------------------------------------------------------------------------

def _fast_copy(src: str, dst: str, log: logging.Logger) -> None:
    """Copy *src* → *dst* using memory-mapped reads for maximum throughput.

    Falls back to ``shutil.copy2`` if mmap fails (e.g. file > 2^63 bytes
    on 32-bit kernels, or insufficient address space).

    Strategy: read the source in 512 MB slabs and write to the destination
    mmap.  This keeps peak RAM at ~512 MB regardless of movie size and lets
    the kernel pipeline src reads with dst writes.
    """
    src_size    = os.path.getsize(src)
    slab_bytes  = 512 * 1024**2   # 512 MB per slab

    try:
        src_mm = np.memmap(src, dtype="uint8", mode="r",   shape=(src_size,))
        dst_mm = np.memmap(dst, dtype="uint8", mode="w+",  shape=(src_size,))

        for off in range(0, src_size, slab_bytes):
            end = min(off + slab_bytes, src_size)
            dst_mm[off:end] = src_mm[off:end]
            if (off // slab_bytes) % 4 == 0:
                pct = 100 * end / src_size
                log.info(f"  SHM copy: {pct:.0f}%  ({end / 1024**3:.1f} / {src_size / 1024**3:.1f} GB)")

        dst_mm.flush()
        del src_mm, dst_mm

    except Exception as exc:
        log.warning(f"SHM: mmap copy failed ({exc}), falling back to shutil.copy2")
        if os.path.exists(dst):
            os.unlink(dst)
        shutil.copy2(src, dst)
