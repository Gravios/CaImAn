"""RGBA footprint overlays (numpy only)."""

import numpy as np

from .constants import (
    CORR_A,
    CORR_B,
    CYAN,
    EXCLUDED,
    OVERLAY_ALPHA,
    PALETTE_HEX,
    _hex_rgb,
)
from .store import ComponentStore


# ── RGBA overlay builder ──────────────────────────────────────────────────────

def _paint_fp(out: np.ndarray, store: 'ComponentStore', i: int,
              rgb: tuple, alpha: float) -> None:
    """Max-composite footprint i (normalised to its peak) into the
    (d1, d2, 4) float RGBA image `out`, touching only its own pixels."""
    idx, val = store.fp_pixels(i)
    if idx.size == 0:
        return
    peak = float(val.max())
    if peak < 1e-9:
        return
    d1 = store.dims[0]
    rr, cc = idx % d1, idx // d1
    am = (val / peak) * alpha
    for ch, v in enumerate((rgb[0] * am, rgb[1] * am, rgb[2] * am, am)):
        cur = out[rr, cc, ch]
        out[rr, cc, ch] = np.maximum(cur, v)


def _build_overlay(store: ComponentStore,
                   sel: list, pair) -> np.ndarray:
    """(d1, d2, 4) RGBA float32 composite of the selected / pair footprints."""
    d1, d2  = store.dims
    out     = np.zeros((d1, d2, 4), dtype=np.float32)
    pair_set = set(pair) if pair else set()
    alpha = OVERLAY_ALPHA["selected"]
    for i in dict.fromkeys(list(sel) + list(pair_set)):   # ordered, unique
        if not (0 <= i < store.n) or not store.is_visible(i):
            continue
        if i in pair_set:
            hex_c = CORR_A if i == pair[0] else CORR_B
        else:
            hex_c = PALETTE_HEX[sel.index(i) % len(PALETTE_HEX)]
        _paint_fp(out, store, i, _hex_rgb(hex_c), alpha)
    return np.clip(out, 0, 1)


def _build_roi_overlay(store: ComponentStore,
                       hide_excluded: bool = False) -> np.ndarray:
    """
    Return (d1, d2, 4) RGBA composite of *every* footprint in dim cyan.

    This is the persistent "all ROIs" base layer.  It is rebuilt from the
    store on each mutation (merge / delete / undo / redo), so deleted or
    merged components visibly disappear / appear in the cell viewer.
    """
    d1, d2 = store.dims
    out    = np.zeros((d1, d2, 4), dtype=np.float32)
    inc_rgb, exc_rgb = _hex_rgb(CYAN), _hex_rgb(EXCLUDED)
    use_appr = store.has_approval         # events or roi_filter seeded Incl
    for i in range(store.n):
        if not store.is_visible(i):
            continue
        excluded = use_appr and not store.approved[i]
        if excluded and hide_excluded:
            continue
        _paint_fp(out, store, i, exc_rgb if excluded else inc_rgb,
                  OVERLAY_ALPHA["excluded" if excluded else "all"])
    return np.clip(out, 0, 1)


__all__ = ['_paint_fp', '_build_overlay', '_build_roi_overlay']
