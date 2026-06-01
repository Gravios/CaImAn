"""Anatomical seeding utilities for census-complete CNMF.

This module decouples *cell detection* (computer vision on a summary image)
from *trace extraction* (CNMF).  The motivation: CNMF / corr_pnr only find
cells that produce calcium transients during the recording.  Silent cells are
invisible to activity-based extraction by construction.  To obtain a complete
census — and a count of inactive cells alongside active ones — we detect cells
anatomically (e.g. Cellpose-SAM on a mean / max-over-mean projection), convert
the resulting masks to a seed matrix ``Ain``, and seed CNMF with them.  CNMF
then assigns a temporal trace to *every* seeded ROI; the active / inactive
split is made afterwards by component evaluation (see ``cnmf_runner`` census
mode), so silent cells are *relabelled*, never discarded.

Design notes
------------
* ``masks_to_Ain`` is the load-bearing primitive and carries no heavy
  dependency.  It accepts masks from *any* source — Cellpose, Cellpose-SAM,
  StarDist, suite2p ``stat``, a manual ROI editor, or a hand-painted label
  image — which keeps the segmenter swappable.
* ``segment_anatomical`` lazily imports ``cellpose`` and targets Cellpose 4.x
  (the Cellpose-SAM model).  It is a convenience wrapper only; if cellpose is
  not installed the rest of the module still works on pre-computed masks.
* Pixel flattening is **Fortran order** throughout, matching CaImAn's
  ``A`` convention (``np.reshape(..., order='F')``).  ``dims`` is
  ``(d1, d2) = (height, width)`` as returned by ``load_memmap``.

References
----------
Cellpose-SAM: Pachitariu, Rariden & Stringer (2025), bioRxiv.
suite2p ``anatomical_only`` mode: Cellpose on max-projection / mean image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, Sequence[np.ndarray]]


# ── Summary projections ───────────────────────────────────────────────────────

def summary_projection(images: np.ndarray, kind: str = "max_div_mean",
                       eps: float = 1e-6) -> np.ndarray:
    """Collapse a ``(T, d1, d2)`` movie to a 2-D image for segmentation.

    Parameters
    ----------
    images
        Movie array, time on axis 0.  A memmap view is fine; reduction is
        done in chunks so the whole movie is never materialised.
    kind
        ``"mean"``, ``"max"``, ``"std"`` or ``"max_div_mean"`` (default).
        ``max_div_mean`` is the suite2p anatomical input — it suppresses the
        static-bright-structure bias of a plain max projection and tends to
        give Cellpose the cleanest soma boundaries.
    eps
        Floor added to the mean before division to avoid blow-up in dark
        pixels.

    Returns
    -------
    np.ndarray
        ``(d1, d2)`` float32 image.
    """
    if images.ndim != 3:
        raise ValueError(f"expected (T, d1, d2) movie, got shape {images.shape}")

    T = images.shape[0]
    chunk = max(1, min(T, 2000))

    if kind in ("mean", "max_div_mean", "std"):
        acc = np.zeros(images.shape[1:], dtype=np.float64)
        acc_sq = np.zeros_like(acc) if kind == "std" else None
        for i in range(0, T, chunk):
            blk = np.asarray(images[i:i + chunk], dtype=np.float64)
            acc += blk.sum(axis=0)
            if acc_sq is not None:
                acc_sq += (blk * blk).sum(axis=0)
        mean = acc / T
        if kind == "mean":
            return mean.astype(np.float32)
        if kind == "std":
            var = np.maximum(acc_sq / T - mean * mean, 0.0)
            return np.sqrt(var).astype(np.float32)

    if kind in ("max", "max_div_mean"):
        mx = np.full(images.shape[1:], -np.inf, dtype=np.float64)
        for i in range(0, T, chunk):
            blk = np.asarray(images[i:i + chunk], dtype=np.float64)
            np.maximum(mx, blk.max(axis=0), out=mx)
        if kind == "max":
            return mx.astype(np.float32)
        return (mx / (mean + eps)).astype(np.float32)  # type: ignore[name-defined]

    raise ValueError(f"unknown projection kind: {kind!r}")


# ── Masks → seed matrix ───────────────────────────────────────────────────────

def _iter_masks(masks: ArrayLike, dims: Optional[tuple]):
    """Yield 2-D boolean footprints from any supported mask container.

    Accepts a 2-D integer label image, a 3-D boolean / integer stack
    ``(K, d1, d2)``, or a sequence of 2-D boolean arrays.
    """
    if isinstance(masks, np.ndarray) and masks.ndim == 2 and not masks.dtype == bool:
        # Integer label image: 0 = background, 1..N = cell ids.
        labels = np.unique(masks)
        labels = labels[labels != 0]
        for lab in labels:
            yield masks == lab
    elif isinstance(masks, np.ndarray) and masks.ndim == 3:
        for k in range(masks.shape[0]):
            yield masks[k].astype(bool)
    elif isinstance(masks, np.ndarray) and masks.ndim == 2 and masks.dtype == bool:
        yield masks
    else:  # sequence of 2-D arrays
        for m in masks:
            m = np.asarray(m)
            if m.ndim != 2:
                raise ValueError(f"mask in sequence has ndim {m.ndim}, expected 2")
            yield m.astype(bool)


def masks_to_Ain(masks: ArrayLike,
                 dims: Optional[tuple] = None,
                 normalize: bool = True,
                 min_pixels: int = 4,
                 dtype=np.float32) -> sparse.csc_matrix:
    """Convert anatomical masks to a CaImAn seed matrix ``Ain``.

    Parameters
    ----------
    masks
        One of: a 2-D integer label image ``(d1, d2)`` (Cellpose / suite2p
        style, 0 = background); a 3-D boolean/integer stack ``(K, d1, d2)``;
        or a sequence of 2-D boolean masks.
    dims
        ``(d1, d2)`` spatial dimensions.  Required only to validate against
        the mask shape; inferred from a 2-D / 3-D array if omitted.
    normalize
        L2-normalise each column, matching the output convention of
        ``greedyROI`` so the spatial/temporal updates start on the same
        footing as a blind init.  Set ``False`` to keep raw binary supports.
    min_pixels
        Drop masks smaller than this (segmentation speckle).  Set to 0 to
        keep everything.
    dtype
        Output dtype (default float32).

    Returns
    -------
    scipy.sparse.csc_matrix
        ``(d1*d2, K)`` matrix with pixels flattened in **F order**, ready to
        pass to ``CNMF(..., Ain=Ain)``.
    """
    footprints = list(_iter_masks(masks, dims))
    if not footprints:
        raise ValueError("no masks found — empty label image?")

    d1, d2 = footprints[0].shape
    if dims is not None and tuple(dims) != (d1, d2):
        raise ValueError(f"mask shape {(d1, d2)} != dims {tuple(dims)}")

    npix = d1 * d2
    cols = []
    kept = 0
    for fp in footprints:
        if fp.shape != (d1, d2):
            raise ValueError(f"inconsistent mask shape {fp.shape} != {(d1, d2)}")
        idx = np.flatnonzero(fp.ravel(order="F"))
        if idx.size < min_pixels:
            continue
        data = np.ones(idx.size, dtype=dtype)
        if normalize:
            data /= np.sqrt(idx.size)  # L2 norm of a binary column = sqrt(n)
        cols.append(sparse.csc_matrix(
            (data, (idx, np.zeros(idx.size, dtype=np.intp))),
            shape=(npix, 1), dtype=dtype))
        kept += 1

    if not cols:
        raise ValueError(
            f"all {len(footprints)} masks were smaller than min_pixels={min_pixels}")

    Ain = sparse.hstack(cols, format="csc").astype(dtype)
    logger.info("masks_to_Ain: %d masks → Ain %s (dropped %d < %d px)",
                kept, Ain.shape, len(footprints) - kept, min_pixels)
    return Ain


# ── Complete seed (A, C, b, f) ────────────────────────────────────────────────

def _randomized_svd(M: np.ndarray, rank: int, n_oversamples: int = 6,
                    seed: int = 0):
    """Truncated SVD via Halko et al. randomized range finder (numpy only)."""
    rng = np.random.RandomState(seed)
    n = M.shape[1]
    r = min(rank + n_oversamples, n)
    Q, _ = np.linalg.qr(M @ rng.standard_normal((n, r)).astype(M.dtype))
    B = Q.T @ M
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :rank], S[:rank], Vt[:rank]


def complete_seed(Ain: sparse.spmatrix, Yr: np.ndarray, nb: int = 2,
                  t_sub: int = 2000, dtype=np.float32):
    """Build a complete CNMF seed ``(Ain, Cin, b_in, f_in)`` from footprints.

    The single-FOV seeded path in ``cnmf.fit`` skips ``initialize()`` when
    ``A`` is provided and immediately runs ``update_spatial(use_init=True)``,
    which requires ``C`` and a background ``(b, f)`` to already exist.  This
    helper supplies coarse but consistent values that CNMF then refines:

    * ``Cin`` — projection of the movie onto each (L2-normalised) footprint,
      i.e. ``Ain.T @ Yr``, clipped to non-negative.
    * ``b_in`` / ``f_in`` — rank-``nb`` randomized-SVD background of the
      residual ``Yr - Ain @ Cin``, estimated on a temporal subsample for
      bounded memory and then projected over the full time axis.

    Parameters
    ----------
    Ain
        ``(npix, K)`` sparse seed matrix from :func:`masks_to_Ain`.
    Yr
        ``(npix, T)`` movie (memmap is fine; read in time chunks).
    nb
        Number of global background components (matches ``init/nb``).
    t_sub
        Frames used for the background spatial basis estimate.

    Returns
    -------
    (Cin, b_in, f_in)
        ``Cin`` ``(K, T)``, ``b_in`` ``(npix, nb)``, ``f_in`` ``(nb, T)``.
    """
    Ain = Ain.tocsc().astype(dtype)
    npix, K = Ain.shape
    T = Yr.shape[1]
    if Yr.shape[0] != npix:
        raise ValueError(f"Yr pixels {Yr.shape[0]} != Ain pixels {npix}")

    chunk = max(1, min(T, 4000))
    AT = Ain.T  # (K, npix)

    # Cin = Ain^T Yr, chunked over time.
    Cin = np.empty((K, T), dtype=dtype)
    for i in range(0, T, chunk):
        Cin[:, i:i + chunk] = AT @ np.asarray(Yr[:, i:i + chunk], dtype=dtype)
    _cin_raw_min, _cin_raw_max = float(Cin.min()), float(Cin.max())
    # Mean of Yr over the union of footprint pixels (sampled) — if this is ~0
    # the movie is empty at the seeds (alignment/movie bug); if it is ~baseline
    # but pre-clip Cin is <=0 the footprints carry no positive signal.
    _fp_pix = np.unique(Ain.indices)
    _fp_sample = _fp_pix[:min(_fp_pix.size, 4096)]
    _yr_fp_mean = float(np.asarray(Yr[_fp_sample, :], dtype=np.float64).mean()) \
        if _fp_sample.size else float("nan")
    np.clip(Cin, 0, None, out=Cin)
    logger.info("complete_seed: pre-clip Cin range [%.4g, %.4g]; mean Yr over "
                "%d footprint px = %.4g", _cin_raw_min, _cin_raw_max,
                _fp_pix.size, _yr_fp_mean)

    # Residual on a temporal subsample → randomized-SVD spatial background basis.
    sub = np.linspace(0, T - 1, num=min(T, t_sub), dtype=int)
    Ys = np.asarray(Yr[:, sub], dtype=dtype)
    Rs = Ys - (Ain @ Cin[:, sub])
    nb_eff = max(1, min(nb, Rs.shape[1]))
    b_in, _s, _vt = _randomized_svd(Rs, nb_eff)
    b_in = np.ascontiguousarray(b_in, dtype=dtype)      # (npix, nb_eff)

    # f_in = b^T Yr - (b^T Ain) Cin, chunked over time → (nb, T).
    bT_A = (b_in.T @ Ain)                                # (nb, K)
    f_in = np.empty((nb_eff, T), dtype=dtype)
    for i in range(0, T, chunk):
        bt_y = b_in.T @ np.asarray(Yr[:, i:i + chunk], dtype=dtype)
        f_in[:, i:i + chunk] = bt_y - bT_A @ Cin[:, i:i + chunk]

    logger.info("complete_seed: Cin %s, b_in %s, f_in %s (nb=%d)",
                Cin.shape, b_in.shape, f_in.shape, nb_eff)
    _cfin = np.isfinite(Cin).all()
    logger.info("complete_seed: Cin range [%.4g, %.4g] per-row sums "
                "min=%.4g (zero rows: %d / %d); finite: Cin=%s b_in=%s f_in=%s",
                float(np.nanmin(Cin)), float(np.nanmax(Cin)),
                float(Cin.sum(axis=1).min()), int((Cin.sum(axis=1) == 0).sum()),
                Cin.shape[0], _cfin, bool(np.isfinite(b_in).all()),
                bool(np.isfinite(f_in).all()))
    if not _cfin:
        logger.warning("complete_seed: Cin contains non-finite values — the "
                       "detrended movie likely has NaN/Inf; seeded components "
                       "will be eliminated as empty/nan in update_spatial.")
    return Cin, b_in, f_in


# ── Label-image loading ───────────────────────────────────────────────────────

def load_label_image(path: Union[str, Path]) -> np.ndarray:
    """Load a 2-D label image from ``.npy`` / ``_seg.npy`` / ``.tif`` / ``.png``.

    Recognises Cellpose ``*_seg.npy`` dictionaries (extracts the ``masks``
    key).  Returns an integer label image ``(d1, d2)``.
    """
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.item()
        if isinstance(obj, dict):  # Cellpose _seg.npy
            if "masks" not in obj:
                raise KeyError(f"{path} is a dict without a 'masks' key")
            return np.asarray(obj["masks"])
        return np.asarray(obj)
    if suf in (".tif", ".tiff"):
        import tifffile
        return np.asarray(tifffile.imread(str(path)))
    if suf in (".png", ".bmp"):
        import imageio.v3 as iio
        return np.asarray(iio.imread(str(path)))
    raise ValueError(f"unsupported label-image format: {suf}")


# ── Optional Cellpose-SAM segmentation ────────────────────────────────────────

def segment_anatomical(projection: np.ndarray,
                       diameter: Optional[float] = None,
                       flow_threshold: float = 0.4,
                       cellprob_threshold: float = 0.0,
                       gpu: bool = True,
                       pretrained_model: Optional[str] = None,
                       normalize: bool = True) -> np.ndarray:
    """Segment a 2-D projection with Cellpose-SAM.

    Thin convenience wrapper around Cellpose 4.x (the Cellpose-SAM model).
    Cellpose is imported lazily, so this is the *only* function in the module
    that requires it — everything downstream works on the returned label
    image, which can equally come from the Cellpose GUI, StarDist, or suite2p.

    Parameters
    ----------
    projection
        ``(d1, d2)`` summary image, e.g. from :func:`summary_projection`.
    diameter
        Expected cell diameter in pixels.  ``None`` lets the model estimate
        it.  For your data (soma ~15-25 px) passing ~20 is a good prior.
    flow_threshold, cellprob_threshold
        Standard Cellpose post-processing thresholds.
    gpu
        Use the GPU model.
    pretrained_model
        Path to a fine-tuned model, or ``None`` for the default Cellpose-SAM
        weights.
    normalize
        Percentile-normalise the image before inference (recommended).

    Returns
    -------
    np.ndarray
        Integer label image ``(d1, d2)``.
    """
    try:
        from cellpose import models
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "cellpose is required for segment_anatomical(); install with "
            "`pip install cellpose` (>=4.0 for Cellpose-SAM). You can also "
            "skip this function and pass pre-computed masks to masks_to_Ain()."
        ) from exc

    img = np.asarray(projection, dtype=np.float32)
    # Cellpose 4.x: CellposeModel loads the SAM-based model by default.
    if pretrained_model is not None:
        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    else:
        model = models.CellposeModel(gpu=gpu)

    eval_kwargs = dict(flow_threshold=flow_threshold,
                       cellprob_threshold=cellprob_threshold,
                       normalize=normalize)
    if diameter is not None:
        eval_kwargs["diameter"] = diameter

    out = model.eval(img, **eval_kwargs)
    masks = out[0] if isinstance(out, (tuple, list)) else out
    n = int(masks.max())
    logger.info("segment_anatomical: Cellpose-SAM found %d cells", n)
    return np.asarray(masks).astype(np.int32)
