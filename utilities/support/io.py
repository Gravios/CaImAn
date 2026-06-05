"""
Format-agnostic stack I/O for SUPPORT.

The training Dataset and the TIFF inference wrapper both need to load
an entire `(T, H, W)` stack into RAM as float32. This module funnels
both call sites through ``caiman.utils.stack_io.StackReader`` so the
same code path handles ``.tif``, ``.tiff``, and ``.msr`` inputs.

MSR support is provided by ``caiman.utils.imspectorreader.IMSpectorReader``
(part of the AG Stroh Lab fork — Leica Imspector's native format).
TIFF support is via ``tifffile``.

This module makes SUPPORT depend on ``caiman.utils.stack_io``. That is
fine because SUPPORT now lives inside the CaImAn fork as a subpackage;
the standalone-package use case is no longer supported (see
``utilities/support/README.md``).
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm


def read_stack_to_array(path: str | Path,
                         dtype: np.dtype = np.float32,
                         progress: bool = True,
                         desc: str | None = None) -> np.ndarray:
    """Read a ``.tif``/``.tiff``/``.msr`` stack into a contiguous
    ``(T, H, W)`` numpy array.

    Parameters
    ----------
    path
        File path. Extension determines the backend (TIFF or MSR);
        ``StackReader`` raises ``ValueError`` for anything else.
    dtype
        Output dtype. Defaults to ``np.float32`` because that is what
        SUPPORT consumes. Each frame is cast with ``astype(copy=False)``
        — for TIFF this is usually a copy (TIFF is typically uint16),
        for MSR always a copy.
    progress
        Show a tqdm progress bar over frames. Disable when called from
        a non-interactive context.
    desc
        Optional progress-bar label. Defaults to the file name.

    Returns
    -------
    numpy.ndarray, shape ``(T, H, W)``, dtype as requested.

    Raises
    ------
    FileNotFoundError
        Path does not exist.
    ValueError
        Format not supported, or stack has 0 frames.
    """
    from caiman.utils.stack_io import StackReader

    path = Path(path)
    with StackReader(path) as r:
        T, H, W = int(r.n_frames), int(r.h), int(r.w)
        out = np.empty((T, H, W), dtype=dtype)
        it = range(T)
        if progress:
            it = tqdm(it, desc=desc or path.name, unit="frame", leave=False)
        for i in it:
            out[i] = r.read_frame(i).astype(dtype, copy=False)
    return out
