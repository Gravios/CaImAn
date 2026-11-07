"""Raw-movie access for playback (.tif / .msr next to the results)."""

import os
from typing import TYPE_CHECKING

import numpy as np

from .loading import _os_dirname, _session_prefix

if TYPE_CHECKING:                      # annotations only
    from .store import ComponentStore


# ── Movie source for the cell-view background (no Qt) ────────────────────────

MOVIE_EXTS = (".tif", ".tiff", ".msr")


def find_session_movie(store: 'ComponentStore'):
    """The session's raw stack, if it sits next to the results file.

    Same rule as the background scan: ``<session>.tif`` / ``.tiff`` /
    ``.msr`` (session = results name without _results/_curated tails), or a
    file named after the folder.  A stack already projected as the
    "stack mean" background is used first.
    """
    src = store.bg_sources.get("stack mean")
    if src and os.path.isfile(src) and src.lower().endswith(MOVIE_EXTS):
        return src
    if not store.fpath:
        return None
    folder = _os_dirname(store.fpath)
    for name in dict.fromkeys((_session_prefix(store.fpath),
                               os.path.basename(folder))):
        for ext in MOVIE_EXTS:
            p = os.path.join(folder, name + ext)
            if os.path.isfile(p):
                return p
    return None


class MovieSource:
    """Random-access frames of a .tif / .msr stack, oriented to ``dims``.

    Wraps :class:`caiman.utils.stack_io.StackReader` (persistent handle)
    with a small frame cache, a running mean for denoised playback, and
    fixed display levels so playback does not flicker.
    """

    CACHE_FRAMES = 64
    MAX_AVG      = 32

    def __init__(self, path: str, dims):
        from caiman.utils.stack_io import StackReader
        from collections import OrderedDict
        self.path = path
        self._r   = StackReader(path)
        self.n_frames = int(self._r.n_frames)
        shape, dims = (int(self._r.h), int(self._r.w)), tuple(int(d) for d in dims)
        if shape == dims:
            self._transpose = False
        elif shape == dims[::-1]:
            self._transpose = True
        else:
            self._r.close()
            raise ValueError(f"movie frames are {shape[0]}×{shape[1]} but the "
                             f"results FOV is {dims[0]}×{dims[1]}")
        self._cache  = OrderedDict()
        self._levels = None

    def _read(self, i: int) -> np.ndarray:
        a = np.asarray(self._r.read_frame(i), dtype=np.float32)
        return np.ascontiguousarray(a.T if self._transpose else a)

    def frame(self, i: int) -> np.ndarray:
        i = int(min(max(i, 0), self.n_frames - 1))
        f = self._cache.get(i)
        if f is None:
            f = self._read(i)
            self._cache[i] = f
            if len(self._cache) > self.CACHE_FRAMES:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(i)
        return f

    def mean(self, i: int, n: int) -> np.ndarray:
        """Mean of frames i-n+1 … i (clipped at 0)."""
        n = int(min(max(n, 1), self.MAX_AVG))
        if n == 1:
            return self.frame(i)
        lo = max(0, int(i) - n + 1)
        acc = np.zeros_like(self.frame(lo))
        for j in range(lo, int(i) + 1):
            acc += self.frame(j)
        return acc / (int(i) + 1 - lo)

    def levels(self) -> tuple:
        """Display range from robust percentiles of ~24 frames across the movie."""
        if self._levels is None:
            idx = np.unique(np.linspace(0, self.n_frames - 1,
                                        min(24, self.n_frames)).astype(int))
            sample = np.stack([self._read(int(j)) for j in idx])
            lo, hi = np.percentile(sample, [0.5, 99.8])
            self._levels = (float(lo), float(hi) if hi > lo else float(lo) + 1.0)
        return self._levels

    def close(self) -> None:
        try:
            self._r.close()
        except Exception:
            pass
        self._cache.clear()


__all__ = ['MOVIE_EXTS', 'find_session_movie', 'MovieSource']
