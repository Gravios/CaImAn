"""
caiman/utils/stack_io.py
========================
Format-agnostic helpers for sampling frames out of imaging stacks.

Most call sites in CaImAn need one of two operations on a raw movie:

    1. "How big is this stack?"          -> ``(H, W), T``
    2. "Give me N frames at indices X."  -> ndarray of shape (N, H, W)

Both are supplied by ``cm.load`` and ``get_file_size``, which dispatch on
file extension (TIF, MSR, HDF5, NWB, etc.).  This module is pure ergonomics
on top of those — short call sites, no new dispatch layer, no new format
registry.  When a new format is added to ``movies.load`` / ``get_file_size``,
the helpers below pick it up automatically.

Scope: stack files (TIF, MSR, HDF5, …).  Memory-mapped F-order arrays
written by ``MotionCorrect`` should still be read via
``caiman.mmapping.load_memmap`` — they're a different beast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def stack_size(path: PathLike) -> Tuple[Tuple[int, int], int]:
    """Return ``((H, W), T)`` for the stack at *path*.

    Thin wrapper around ``caiman.base.movies.get_file_size`` that coerces
    the path to ``str`` and unpacks for typical 2-D-time-series use.

    For volumetric (3-D-per-frame) stacks ``get_file_size`` returns a
    longer ``dims`` tuple; this helper passes whatever was returned through
    unchanged in the first slot.

    Parameters
    ----------
    path : str | Path
        Stack file. Extension determines the loader.

    Returns
    -------
    (dims, T) : tuple
        ``dims`` is the spatial shape (typically ``(H, W)``), ``T`` is the
        number of frames.
    """
    from caiman.base.movies import get_file_size
    dims, T = get_file_size(str(path))
    return dims, int(T)


def stack_sample(path: PathLike,
                 indices: Iterable[int],
                 *,
                 dtype=np.float32) -> np.ndarray:
    """Read frames at *indices* from the stack at *path*.

    Parameters
    ----------
    path : str | Path
        Stack file.
    indices : iterable of int
        Frame indices to read. Need not be sorted or contiguous, but
        loaders are most efficient on contiguous ranges.
    dtype : numpy dtype, default ``np.float32``
        Output dtype. The cast happens in the wrapper, not in the loader.

    Returns
    -------
    ndarray, shape ``(len(indices), H, W)``
        Stacked frames in the requested order.
    """
    import caiman as cm
    # CaImAn's `subindices` API: a *list* means per-axis indexers
    # ([time, H, W]); a non-list iterable means a flat sequence of frame
    # indices on the time axis. Convert to ndarray to take the latter
    # path in both the TIF and MSR branches of cm.load.
    idx = np.asarray(list(indices), dtype=int)
    arr = np.asarray(cm.load(str(path), subindices=idx))
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def stack_evenly_sampled(path: PathLike,
                         n: int,
                         *,
                         dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``n`` evenly-spaced frames spanning the stack.

    If ``n`` is greater than the stack length ``T``, only ``T`` frames are
    returned (one per index).

    Parameters
    ----------
    path : str | Path
    n : int
        Target number of frames.
    dtype : numpy dtype, default ``np.float32``

    Returns
    -------
    frames : ndarray of shape ``(min(n, T), H, W)``
    indices : ndarray of shape ``(min(n, T),)``
        The frame indices actually read (useful for axis labels).
    """
    _, T = stack_size(path)
    n = max(1, min(int(n), T))
    indices = np.linspace(0, T - 1, n, dtype=int)
    frames  = stack_sample(path, indices, dtype=dtype)
    return frames, indices


# ── Persistent-handle reader (for viewers / random access) ────────────────────
#
# The helpers above are batch-oriented (cm.load + get_file_size). That's the
# right shape for a one-shot N-frame sample, but it is the wrong shape for an
# interactive viewer that pages through frames in arbitrary order, dozens of
# times per second. A viewer wants a *persistent* file handle and a
# random-access read_frame(idx) call.
#
# StackReader is exactly that: a tiny extension-dispatching adapter with a
# uniform shape across formats. Backends are kept private — they are an
# implementation detail of the dispatch.

class StackReader:
    """Format-agnostic persistent reader for stack files.

    Dispatches to a backend based on file extension (.tif/.tiff/.msr).
    Holds an open handle for the lifetime of the object so per-frame
    random access stays cheap.

    Attributes
    ----------
    path : Path
    n_frames : int
    h, w : int            # frame dimensions
    dtype : np.dtype      # native dtype as returned by ``read_frame``

    Usage
    -----
        with StackReader(path) as r:
            frame = r.read_frame(123)
    """

    def __init__(self, path: PathLike):
        path = Path(path)
        ext  = path.suffix.lower()
        if ext in (".tif", ".tiff"):
            self._backend = _TiffStackBackend(path)
        elif ext == ".msr":
            self._backend = _MsrStackBackend(path)
        else:
            raise ValueError(
                f"Unsupported stack format: {ext!r} (path: {path})"
            )
        self.path     = path
        self.n_frames = self._backend.n_frames
        self.h        = self._backend.h
        self.w        = self._backend.w
        self.dtype    = self._backend.dtype

    def read_frame(self, idx: int) -> np.ndarray:
        return self._backend.read_frame(int(idx))

    def close(self) -> None:
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class _TiffStackBackend:
    def __init__(self, path: Path):
        import tifffile
        self._tf = tifffile.TiffFile(str(path))
        self.n_frames = len(self._tf.pages)
        if self.n_frames == 0:
            raise ValueError(f"TIFF has no pages: {path}")
        probe      = self._tf.pages[0].asarray()
        self.h     = int(probe.shape[0])
        self.w     = int(probe.shape[1]) if probe.ndim > 1 else 1
        self.dtype = probe.dtype

    def read_frame(self, idx: int) -> np.ndarray:
        return self._tf.pages[idx].asarray()

    def close(self) -> None:
        try:
            self._tf.close()
        except Exception:
            pass


class _MsrStackBackend:
    def __init__(self, path: Path):
        from caiman.utils.imspectorreader import IMSpectorReader
        self._reader  = IMSpectorReader(str(path))
        self.n_frames = int(self._reader.slices_count or 0)
        if self.n_frames == 0:
            raise ValueError(f"MSR has no slices: {path}")
        self.h     = int(self._reader.size_y)
        self.w     = int(self._reader.size_x)
        self.dtype = np.dtype(np.uint16)

    def read_frame(self, idx: int) -> np.ndarray:
        return self._reader.read_slice(idx)

    def close(self) -> None:
        # IMSpectorReader opens/closes the file per read_slice — nothing
        # to release. If a persistent-handle variant lands later, plug it
        # in here and read_frame stays unchanged.
        pass
