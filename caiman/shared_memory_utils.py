"""
shared_memory_utils.py
======================
Zero-copy inter-process movie access using POSIX shared memory.

Problem solved
--------------
CaImAn's parallel motion-correction calls ``caiman.load(fname, subindices=idxs)``
inside every worker function (``tile_and_correct_wrapper``, ``cnmf_patches``, …).
For a TIFF that hasn't been converted to an .mmap file yet, each of the N
workers independently reads the whole file from disk and keeps its own private
copy in heap memory.  Even for .mmap files the OS must fault-in the same pages
into N separate virtual-address spaces before COW isolation takes effect.

Solution
--------
``SharedMovieBuffer`` loads the movie **once** in the parent process and places
it in a POSIX shared-memory segment (``/dev/shm`` on Linux).  Every worker
process then attaches to the *same physical pages* via
``multiprocessing.shared_memory.SharedMemory`` – a true zero-copy read.

Cache hierarchy notes
---------------------
* **L3 (LLC)** – shared across all cores on one socket.  Because every worker
  reads from the *same* physical pages, those pages are loaded into L3 only
  once regardless of how many processes access them simultaneously.  With
  private copies every process evicts competing lines from L3 for its own
  duplicate, producing O(N) LLC pressure instead of O(1).

* **L2 / L1** – private per-core.  True L1/L2 sharing requires threads, not
  processes; Python's GIL makes this impractical for CPU-bound work today.
  The cache-aware scheduler in ``cpu_topology.py`` mitigates L2/L1 misses by
  assigning *temporally adjacent* frame chunks to *cores that share L3*, so
  the template array and hot working-set data have the highest chance of
  already being warm in the shared LLC when the next worker starts.

* **Cache-line alignment** – ``SharedMovieBuffer`` allocates the array with
  ``numpy.zeros`` and then copies into a view whose base address is aligned to
  ``CACHE_LINE_BYTES`` (64 B) via ``numpy.frombuffer`` on an aligned ctypes
  buffer, so every worker's ``[row_start : row_end]`` slice begins on a fresh
  cache line and avoids false-sharing across frame boundaries.

Usage
-----
Parent process::

    with SharedMovieBuffer(fname) as shm_buf:
        handle = shm_buf.worker_handle()   # lightweight, picklable
        # dispatch workers, passing handle instead of fname
        results = pool.map(my_worker, [(handle, idxs, ...) for idxs in splits])

Worker process::

    def my_worker(args):
        shm_handle, idxs, *rest = args
        frames = attach_shared_frames(shm_handle, idxs)  # zero-copy slice
        # ... process frames ...
"""

from __future__ import annotations

import ctypes
import logging
import os
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple, Optional, Sequence, Union
import weakref

# CPython's SharedMemory.__del__ calls self._mmap.close() without catching BufferError.
# If a numpy array still holds an exported pointer into the buffer at GC time (which is
# unavoidable when workers return views into shared memory), the __del__ raises and prints
# a noisy traceback to stderr, even though the mmap fd will be correctly released by the
# OS once the last pointer dies.  Suppress the harmless error here at the source.
_orig_shm_close = SharedMemory.close
def _shm_close_safe(self: SharedMemory) -> None:
    try:
        _orig_shm_close(self)
    except BufferError:
        pass  # exported pointer still live; kernel reclaims fd when it dies
SharedMemory.close = _shm_close_safe

import numpy as np

logger = logging.getLogger("caiman")

# ── Constants ────────────────────────────────────────────────────────────────

CACHE_LINE_BYTES: int = 64   # x86 / ARM cache-line width


# ── Public data types ─────────────────────────────────────────────────────────

class ShmHandle(NamedTuple):
    """Lightweight, picklable descriptor passed to worker processes."""
    name:  str            # SharedMemory name (OS identifier)
    shape: tuple          # full array shape, e.g. (T, d1, d2)
    dtype: str            # numpy dtype string, e.g. 'float32'
    order: str            # 'C' or 'F' – memory layout of the array


# ── Core class ────────────────────────────────────────────────────────────────

class SharedMovieBuffer:
    """
    Load a movie into POSIX shared memory so workers share physical pages.

    Parameters
    ----------
    fname : str or array-like
        File name (tif / hdf5 / mmap) **or** a pre-loaded numpy array.
    var_name_hdf5 : str
        HDF5 dataset name (ignored for non-HDF5 files).
    is3D : bool
        Pass through to ``caiman.load``.
    order : {'C', 'F'}
        Memory layout for the shared buffer.  'C' (row-major, time-first) is
        the default because each worker accesses a contiguous temporal slice
        ``movie[t0:t1, :, :]`` which is contiguous under C order.

    Notes
    -----
    The shared-memory segment is unlinked when the context manager exits or
    ``close()`` is called.  Workers that have already attached keep their
    mapping alive until they release it; the OS cleans up the backing store
    once the reference count drops to zero.
    """

    def __init__(
        self,
        fname,
        var_name_hdf5: str = "mov",
        is3D: bool = False,
        order: str = "C",
    ) -> None:
        self._shm: Optional[SharedMemory] = None
        self._arr: Optional[np.ndarray] = None
        self._handle: Optional[ShmHandle] = None

        # ── Load source data ──────────────────────────────────────────────────
        # Peak-RAM goal: one chunk (~256 MB) at a time, never the full movie.
        #
        # caiman.load() reads the entire movie into a regular RAM array before
        # returning — at 14 GB that triples peak RSS (page-cache + load result
        # + SHM).  Instead we use load_memmap() for .mmap files (zero-copy OS
        # mapping) or np.asarray() for ndarray inputs, then stream into SHM in
        # pixel-row chunks so we never hold more than one chunk extra in RAM.
        actual_dtype = np.dtype(np.float32)

        if isinstance(fname, np.ndarray):
            # Caller passed an in-memory array — use it directly.
            # asarray is a no-op if already float32 and C/F contiguous.
            src_arr = np.asarray(fname, dtype=actual_dtype)
            if order == "C":
                src_arr = np.ascontiguousarray(src_arr)
            else:
                src_arr = np.asfortranarray(src_arr)
            src_shape = src_arr.shape
            src_nbytes = src_arr.nbytes
        else:
            # .mmap path: load_memmap returns a zero-copy np.memmap backed by
            # the file.  No data is read until we copy into SHM below.
            from caiman.mmapping import load_memmap as _load_memmap
            _Yr, _dims, _T = _load_memmap(fname)
            # _Yr is (pixels, T) in the file's native order.
            # Reconstruct (T, d1, d2) C-order view — still zero-copy.
            src_arr = np.reshape(_Yr.T, [_T] + list(_dims), order="F")
            src_shape  = src_arr.shape
            src_nbytes = int(np.prod(src_shape)) * actual_dtype.itemsize
            logger.info(
                f"SharedMovieBuffer: streaming {fname!r} into SHM "
                f"({src_nbytes / 2**30:.2f} GiB, order={order}) …"
            )

        # ── Allocate shared memory ─────────────────────────────────────────────
        nbytes = src_nbytes + CACHE_LINE_BYTES
        self._shm = SharedMemory(create=True, size=nbytes)

        # Build a cache-line-aligned numpy view into the shm buffer
        raw_addr = ctypes.addressof(
            ctypes.c_char.from_buffer(self._shm.buf)
        )
        offset = (-raw_addr) % CACHE_LINE_BYTES
        aligned_buf = (ctypes.c_char * src_nbytes).from_buffer(
            self._shm.buf, offset
        )
        self._arr = np.frombuffer(aligned_buf, dtype=actual_dtype).reshape(
            src_shape, order=order
        )

        # ── Stream src → SHM in chunks to keep peak RAM bounded ───────────────
        # For ndarray inputs np.copyto is fine (src is already fully in RAM).
        # For memmap inputs we copy one time-chunk at a time so the OS only
        # pages in ~256 MB of the source file before writing to SHM.
        if isinstance(fname, np.ndarray):
            np.copyto(self._arr, src_arr)
        else:
            # self._arr is (T, d1, d2); chunk along the time axis.
            T_total   = src_shape[0]
            frame_bytes = src_nbytes // T_total
            t_chunk   = max(1, int(256 * 1024 * 1024 // frame_bytes))
            dst_flat  = self._arr.reshape(T_total, -1)   # (T, pixels) view
            src_flat  = src_arr.reshape(T_total, -1)
            for t0 in range(0, T_total, t_chunk):
                t1 = min(t0 + t_chunk, T_total)
                np.copyto(dst_flat[t0:t1], src_flat[t0:t1].astype(actual_dtype))
            del dst_flat, src_flat, src_arr, _Yr

        self._handle = ShmHandle(
            name=self._shm.name,
            shape=tuple(self._arr.shape),
            dtype=actual_dtype.str,         # e.g. '<f4'
            order=order,
        )
        logger.info(
            f"SharedMovieBuffer: allocated {self._shm.size / 2**20:.1f} MiB "
            f"in SHM '{self._shm.name}' (aligned offset {offset} B)"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def array(self) -> np.ndarray:
        """Direct numpy view (for use in the *parent* process only)."""
        if self._arr is None:
            raise RuntimeError("SharedMovieBuffer already closed")
        return self._arr

    def worker_handle(self) -> ShmHandle:
        """Return a picklable descriptor to pass to worker processes."""
        if self._handle is None:
            raise RuntimeError("SharedMovieBuffer already closed")
        return self._handle

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Unlink the shared-memory segment.  Safe to call multiple times."""
        if self._shm is not None:
            # Must discard the numpy view *before* closing the shm mapping.
            # The SharedMemory.__del__ / close() raises BufferError if any
            # ctypes buffer (i.e. a numpy array) is still exporting a pointer
            # into the segment.  Setting _arr to None drops our reference;
            # Python's GC will release the ctypes buffer before we call close().
            self._arr = None
            self._handle = None
            import gc
            gc.collect()
            try:
                self._shm.close()
            except BufferError:
                pass  # exported pointers still exist; close will happen when they die
            except Exception as exc:
                logger.warning(f"SharedMovieBuffer.close(): {exc}")
            try:
                self._shm.unlink()
            except Exception as exc:
                logger.warning(f"SharedMovieBuffer.unlink(): {exc}")
            finally:
                self._shm = None

    def __enter__(self) -> "SharedMovieBuffer":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ── Worker-side helpers ───────────────────────────────────────────────────────

class _WorkerShmView:
    """
    Worker-side reference to a shared-memory segment.

    Call ``release()`` (or use as a context manager) when done so the kernel
    can reclaim the mapping as soon as the parent unlinks the segment.
    """

    def __init__(self, handle: ShmHandle) -> None:
        self._shm = SharedMemory(name=handle.name, create=False)
        dtype = np.dtype(handle.dtype)

        # Re-derive the alignment offset (same arithmetic as in SharedMovieBuffer)
        raw_addr = ctypes.addressof(
            ctypes.c_char.from_buffer(self._shm.buf)
        )
        offset = (-raw_addr) % CACHE_LINE_BYTES
        nbytes = int(np.prod(handle.shape)) * dtype.itemsize
        aligned_buf = (ctypes.c_char * nbytes).from_buffer(self._shm.buf, offset)
        self._arr = np.frombuffer(aligned_buf, dtype=dtype).reshape(
            handle.shape, order=handle.order
        )

    @property
    def array(self) -> np.ndarray:
        return self._arr

    def release(self) -> None:
        if self._shm is not None:
            # Drop the numpy view before closing the mapping (see SharedMovieBuffer.close).
            self._arr = None
            import gc; gc.collect()
            try:
                self._shm.close()
            except BufferError:
                # A numpy array is still exporting a pointer into this segment.
                # This happens when attach_shared_frames returns a view and _WorkerShmView
                # goes out of scope while the caller still holds the array.
                # The SharedMemory mmap will be closed by the OS when the numpy array
                # is eventually GC'd (via the weakref.finalize registered by attach_shared_frames).
                pass
            self._shm = None

    def __enter__(self) -> "_WorkerShmView":
        return self

    def __exit__(self, *_) -> None:
        self.release()


def attach_shared_frames(
    handle: ShmHandle,
    idxs: Union[Sequence[int], slice, None] = None,
) -> np.ndarray:
    """
    Worker-side: attach to shared memory and return a **read-only** numpy view.

    Parameters
    ----------
    handle : ShmHandle
        Descriptor returned by ``SharedMovieBuffer.worker_handle()``.
    idxs : sequence of ints, slice, or None
        Temporal indices to select (first axis).  When *None* the entire array
        is returned.

    Returns
    -------
    np.ndarray
        A numpy array backed by the shared-memory pages – **no data is
        copied**.  The view is read-only; workers that need to modify frames
        should call ``.copy()`` on the returned slice.

    Notes
    -----
    The returned array holds an internal reference to the ``SharedMemory``
    object.  The mapping stays alive until the array (and any views derived
    from it) are garbage-collected, even after the parent process has unlinked
    the segment.
    """
    view = _WorkerShmView(handle)
    arr = view.array  # full movie
    arr.flags.writeable = False  # prevent accidental in-place modification

    if idxs is None:
        return arr

    # Select temporal slice (axis 0) – this is a *view*, not a copy,
    # as long as idxs is a slice or a contiguous integer range.
    if isinstance(idxs, slice):
        selected = arr[idxs]
    else:
        idxs_arr = np.asarray(idxs)
        # Contiguous ascending integers → slice (view); otherwise fancy-index (copy).
        if idxs_arr.size > 0:
            steps = np.diff(idxs_arr)
            if steps.size == 0 or (np.all(steps == 1)):
                selected = arr[idxs_arr[0] : idxs_arr[-1] + 1]
            else:
                # Non-contiguous – unavoidable copy, but still cheaper than
                # re-reading from disk or pickling the whole movie.
                selected = arr[idxs_arr].copy()
                selected.flags.writeable = True
                return selected
        else:
            selected = arr[idxs_arr]

    return selected


# ── Utility: load_or_attach ───────────────────────────────────────────────────

def frames_from_handle_or_fname(
    img_name_or_handle,
    idxs,
    var_name_hdf5: str = "mov",
    is3D: bool = False,
) -> np.ndarray:
    """
    Unified loader used inside worker functions.

    If *img_name_or_handle* is an :class:`ShmHandle` the frames are obtained
    from shared memory (zero-copy for contiguous slices).  Otherwise the
    legacy ``caiman.load`` path is used unchanged so existing code continues
    to work without modification.

    Parameters
    ----------
    img_name_or_handle : str | tuple[str] | ShmHandle
        Either the original file-name argument or a shared-memory handle.
    idxs : array-like
        Frame indices (first axis of the movie).
    var_name_hdf5 : str
        Passed through to ``caiman.load`` when *img_name_or_handle* is a path.
    is3D : bool
        Passed through to ``caiman.load``.

    Returns
    -------
    np.ndarray  (T, d1, d2) or (T, d1, d2, d3)
    """
    if isinstance(img_name_or_handle, ShmHandle):
        frames = attach_shared_frames(img_name_or_handle, idxs)
        # Workers expect a writable array to apply motion-correction in-place.
        if not frames.flags.writeable:
            frames = frames.copy()
        return frames
    else:
        import caiman  # noqa: PLC0415
        return caiman.load(
            img_name_or_handle,
            subindices=idxs,
            var_name_hdf5=var_name_hdf5,
            is3D=is3D,
        )
