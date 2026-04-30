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
    idx = list(indices)
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
