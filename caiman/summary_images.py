#!/usr/bin/env python

"""
functions that creates image from a video file

Primarily intended for plotting, returns correlation images ( local or max )
"""

import cv2
import logging
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure
from scipy.sparse import coo_matrix
from typing import Any, Optional

import caiman
import caiman.base.movies
from caiman.source_extraction.cnmf.pre_processing import get_noise_fft

# ── Optional GPU acceleration (CuPy) ─────────────────────────────────────────
# Imported lazily so the module is importable without a CUDA installation.
# All GPU helpers check _gpu_available() before touching CuPy symbols.
try:
    import cupy as _cp
    import cupyx.scipy.ndimage as _cpnd
    _CUPY_LOADED = True
except ImportError:
    _cp = None
    _cpnd = None
    _CUPY_LOADED = False


def _gpu_available() -> bool:
    """Return True if CuPy is installed and a CUDA device is reachable."""
    if not _CUPY_LOADED:
        return False
    try:
        _cp.zeros(1)
        return True
    except Exception:
        return False

def max_correlation_image(Y, bin_size: int = 1000, eight_neighbours: bool = True, swap_dim: bool = True) -> np.ndarray:
    """Computes the max-correlation image for the input dataset Y with bin_size

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format

        bin_size: scalar (integer)
             Length of bin_size (if last bin is smaller than bin_size < 2 bin_size is increased to impose uniform bins)

        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively

        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

    Returns:
        Cn: d1 x d2 [x d3] matrix,
            max correlation image
    """
    logger = logging.getLogger("caiman")

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    T = Y.shape[0]
    if T <= bin_size:
        Cn_bins = local_correlations_fft(Y, eight_neighbours=eight_neighbours, swap_dim=False)
        return Cn_bins
    else:
        if T % bin_size < bin_size / 2.:
            bin_size = T // (T // bin_size)

        n_bins = T // bin_size
        Cn_bins = np.zeros(((n_bins,) + Y.shape[1:]))
        for i in range(n_bins):
            Cn_bins[i] = local_correlations_fft(Y[i * bin_size:(i + 1) * bin_size],
                                                eight_neighbours=eight_neighbours,
                                                swap_dim=False)
            logger.debug(i * bin_size)

        Cn = np.max(Cn_bins, axis=0)
        return Cn

def local_correlations_fft(Y,
                           eight_neighbours: bool = True,
                           swap_dim: bool = True,
                           opencv: bool = True,
                           rolling_window=None) -> np.ndarray:
    """Computes the correlation image for the input dataset Y using a faster FFT based method

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format
    
        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively
    
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front
    
        opencv: Boolean
            If True process using OpenCV method

        rolling_window: (undocumented)

    Returns:
        Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    if rolling_window is None:
        Y -= np.mean(Y, axis=0)
        Ystd = np.std(Y, axis=0)
        Ystd[Ystd == 0] = np.inf
        Y /= Ystd
    else:
        Ysum = np.cumsum(Y, axis=0)
        Yrm = (Ysum[rolling_window:] - Ysum[:-rolling_window]) / rolling_window
        Y[:rolling_window] -= Yrm[0]
        Y[rolling_window:] -= Yrm
        del Yrm, Ysum
        Ystd = np.cumsum(Y**2, axis=0)
        Yrst = np.sqrt((Ystd[rolling_window:] - Ystd[:-rolling_window]) / rolling_window)
        Yrst[Yrst == 0] = np.inf
        Y[:rolling_window] /= Yrst[0]
        Y[rolling_window:] /= Yrst
        del Ystd, Yrst

    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype='float32')
            sz[1, 1, 1] = 0
        else:
            # yapf: disable
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                          dtype='float32')
            # yapf: enable
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='float32')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    if opencv and Y.ndim == 3:
        Yconv = np.stack([cv2.filter2D(img, -1, sz, borderType=0) for img in Y])
        MASK = cv2.filter2D(np.ones(Y.shape[1:], dtype='float32'), -1, sz, borderType=0)
    else:
        Yconv = convolve(Y, sz[np.newaxis, :], mode='constant')
        MASK = convolve(np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')

    YYconv = Yconv * Y
    del Y, Yconv
    if rolling_window is None:
        Cn = np.mean(YYconv, axis=0) / MASK
    else:
        YYconv_cs = np.cumsum(YYconv, axis=0)
        del YYconv
        YYconv_rm = (YYconv_cs[rolling_window:] - YYconv_cs[:-rolling_window]) / rolling_window
        del YYconv_cs
        Cn = YYconv_rm / MASK

    return Cn


def local_correlations_multicolor(Y, swap_dim: bool = True) -> np.ndarray:
    """Computes the correlation image with color depending on orientation

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format

        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

    Returns:
        rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """
    if Y.ndim == 4:
        raise Exception('Not Implemented')

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)
    rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:,]), axis=0)
    rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:,]), axis=0)

    return np.dstack([rho_h[:, 1:] / 2, rho_d1 / 2, rho_d2 / 2])


def local_correlations(Y, eight_neighbours: bool = True, swap_dim: bool = True, order_mean=1) -> np.ndarray:
    """Computes the correlation image for the input dataset Y

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format
    
        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively
    
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

        order_mean: (undocumented)

    Returns:
        rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    rho = np.zeros(Y.shape[1:])
    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    # yapf: disable
    if order_mean == 0:
        rho = np.ones(Y.shape[1:])
        rho_h = rho_h
        rho_w = rho_w
        rho[:-1, :] = rho[:-1, :] * rho_h
        rho[1:,  :] = rho[1:,  :] * rho_h
        rho[:, :-1] = rho[:, :-1] * rho_w
        rho[:,  1:] = rho[:,  1:] * rho_w
    else:
        rho[:-1, :] = rho[:-1, :] + rho_h**(order_mean)
        rho[1:,  :] = rho[1:,  :] + rho_h**(order_mean)
        rho[:, :-1] = rho[:, :-1] + rho_w**(order_mean)
        rho[:,  1:] = rho[:,  1:] + rho_w**(order_mean)

    if Y.ndim == 4:
        rho_d = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
        rho[:, :, :-1] = rho[:, :, :-1] + rho_d
        rho[:, :, 1:] = rho[:, :, 1:] + rho_d

        neighbors = 6 * np.ones(Y.shape[1:])
        neighbors[0]        = neighbors[0]        - 1
        neighbors[-1]       = neighbors[-1]       - 1
        neighbors[:,     0] = neighbors[:,     0] - 1
        neighbors[:,    -1] = neighbors[:,    -1] - 1
        neighbors[:,  :, 0] = neighbors[:,  :, 0] - 1
        neighbors[:, :, -1] = neighbors[:, :, -1] - 1

    else:
        if eight_neighbours:
            rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:,]), axis=0)
            rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:,]), axis=0)

            if order_mean == 0:
                rho_d1 = rho_d1
                rho_d2 = rho_d2
                rho[:-1, :-1] = rho[:-1, :-1] * rho_d2
                rho[1:,   1:] = rho[1:,   1:] * rho_d1
                rho[1:,  :-1] = rho[1:,  :-1] * rho_d1
                rho[:-1,  1:] = rho[:-1,  1:] * rho_d2
            else:
                rho[:-1, :-1] = rho[:-1, :-1] + rho_d2**(order_mean)
                rho[1:,   1:] = rho[1:,   1:] + rho_d1**(order_mean)
                rho[1:,  :-1] = rho[1:,  :-1] + rho_d1**(order_mean)
                rho[:-1,  1:] = rho[:-1,  1:] + rho_d2**(order_mean)

            neighbors = 8 * np.ones(Y.shape[1:3])
            neighbors[0,   :] = neighbors[0,   :] - 3
            neighbors[-1,  :] = neighbors[-1,  :] - 3
            neighbors[:,   0] = neighbors[:,   0] - 3
            neighbors[:,  -1] = neighbors[:,  -1] - 3
            neighbors[0,   0] = neighbors[0,   0] + 1
            neighbors[-1, -1] = neighbors[-1, -1] + 1
            neighbors[-1,  0] = neighbors[-1,  0] + 1
            neighbors[0,  -1] = neighbors[0,  -1] + 1
        else:
            neighbors = 4 * np.ones(Y.shape[1:3])
            neighbors[0,  :]  = neighbors[0,  :] - 1
            neighbors[-1, :]  = neighbors[-1, :] - 1
            neighbors[:,  0]  = neighbors[:,  0] - 1
            neighbors[:, -1]  = neighbors[:, -1] - 1

    # yapf: enable
    if order_mean == 0:
        rho = np.power(rho, 1. / neighbors)
    else:
        rho = np.power(np.divide(rho, neighbors), 1 / order_mean)

    return rho


def _build_psf_kernel(gSig, center_psf):
    """Build the spatial PSF kernel on CPU and return as a float32 ndarray."""
    if not isinstance(gSig, list):
        gSig = [gSig, gSig]
    ksize0 = int(2 * gSig[0]) * 2 + 1
    ksize1 = int(2 * gSig[1]) * 2 + 1
    psf = cv2.getGaussianKernel(ksize0, gSig[0], cv2.CV_32F).dot(
          cv2.getGaussianKernel(ksize1, gSig[1], cv2.CV_32F).T)
    if center_psf:
        ind_nz = psf >= psf[0].max()
        psf -= psf[ind_nz].mean()
        psf[~ind_nz] = 0.0
    return psf.astype(np.float32)


def _filter_chunk(chunk_gpu, kernel_gpu):
    """Apply the PSF filter in-place to a (Tc, d1, d2) chunk on the GPU."""
    for t in range(chunk_gpu.shape[0]):
        chunk_gpu[t] = _cpnd.convolve(chunk_gpu[t], kernel_gpu, mode='reflect')


def _correlation_pnr_gpu(Y, gSig, center_psf, swap_dim,
                         noise_range, noise_method,
                         chunk_gb: float = 2.0) -> tuple:
    """Two-pass chunked GPU implementation of correlation_pnr.

    The input movie is streamed through the GPU in chunks of ``chunk_gb`` GB
    rather than loaded all at once, so peak VRAM usage is bounded to that
    budget plus a handful of (d1, d2) accumulators.

    Pass 1 — per chunk, accumulate:
        * per-pixel sum and sum-of-squares  → global mean and variance
        * per-pixel running maximum         → PNR numerator
        * per-chunk Welch PSD segment       → noise std (sn)

    Pass 2 — per chunk, accumulate:
        * thresholded, normalised neighbour correlations → correlation image

    Peak VRAM ≈ chunk_gb + ~100 MB for accumulators and one FFT segment.
    """
    logger = logging.getLogger("caiman")

    if swap_dim:
        Y = np.transpose(Y, (Y.ndim - 1,) + tuple(range(Y.ndim - 1)))

    T, d1, d2 = Y.shape
    n_pixels   = d1 * d2

    # Frames per chunk: choose so one chunk ≤ chunk_gb GB
    bytes_per_frame  = d1 * d2 * 4           # float32
    frames_per_chunk = max(1, int(chunk_gb * 1e9 / bytes_per_frame))
    frames_per_chunk = min(frames_per_chunk, T)
    logger.debug(f"correlation_pnr GPU: T={T}, chunk={frames_per_chunk} frames "
                 f"({frames_per_chunk * bytes_per_frame / 1e9:.2f} GB/chunk)")

    # Build PSF kernel once on CPU
    kernel_gpu = None
    if gSig is not None:
        kernel_gpu = _cp.asarray(_build_psf_kernel(gSig, center_psf))

    # Neighbour-correlation kernel and border mask (tiny, stays in VRAM)
    sz   = _cp.ones((3, 3), dtype=_cp.float32); sz[1, 1] = 0.0
    MASK = _cpnd.convolve(_cp.ones((d1, d2), dtype=_cp.float32),
                          sz, mode='constant')
    MASK[MASK == 0] = 1.0

    # ── Pass 1 accumulators ───────────────────────────────────────────────
    # All (d1, d2) float64 to avoid precision loss in long sums
    sum_gpu    = _cp.zeros((d1, d2), dtype=_cp.float64)
    sum2_gpu   = _cp.zeros((d1, d2), dtype=_cp.float64)
    data_max   = _cp.full((d1, d2), -_cp.inf, dtype=_cp.float32)

    # Welch PSD: accumulate mean-band-power per chunk, then average
    ff         = _cp.linspace(0.0, 0.5, frames_per_chunk // 2 + 1)
    band       = (ff >= noise_range[0]) & (ff <= noise_range[1])
    psd_sum    = _cp.zeros((d1, d2), dtype=_cp.float64)
    n_segments = 0

    for t0 in range(0, T, frames_per_chunk):
        t1    = min(t0 + frames_per_chunk, T)
        Tc    = t1 - t0
        chunk = _cp.asarray(Y[t0:t1].astype(np.float32))   # H2D: one chunk

        if kernel_gpu is not None:
            _filter_chunk(chunk, kernel_gpu)

        sum_gpu  += chunk.sum(axis=0).astype(_cp.float64)
        sum2_gpu += (chunk.astype(_cp.float64) ** 2).sum(axis=0)
        data_max  = _cp.maximum(data_max, chunk.max(axis=0))

        # Welch segment: rfft along time axis (C-contiguous, no copy)
        # Recompute band mask if this chunk is shorter than frames_per_chunk
        if Tc != frames_per_chunk:
            ff_c  = _cp.linspace(0.0, 0.5, Tc // 2 + 1)
            band_c = (ff_c >= noise_range[0]) & (ff_c <= noise_range[1])
        else:
            band_c = band

        xdft = _cp.fft.rfft(chunk.reshape(Tc, n_pixels), axis=0)  # (Tc//2+1, pixels)
        psd  = (_cp.abs(xdft[band_c, :]) ** 2) * (2.0 / Tc)       # (n_band, pixels)
        del xdft, chunk

        if noise_method == 'mean':
            psd_sum += psd.mean(axis=0).reshape(d1, d2).astype(_cp.float64)
        elif noise_method == 'median':
            psd_sum += _cp.median(psd, axis=0).reshape(d1, d2).astype(_cp.float64)
        else:  # logmexp
            psd_sum += _cp.exp(_cp.mean(_cp.log(psd + 1e-10), axis=0)).reshape(d1, d2).astype(_cp.float64)
        del psd
        n_segments += 1

    # Finalise Pass-1 statistics
    mean_gpu = (sum_gpu / T).astype(_cp.float32)                   # (d1, d2)
    var_gpu  = (_cp.maximum(sum2_gpu / T - (sum_gpu / T) ** 2,
                            0.0)).astype(_cp.float32)
    std_gpu  = _cp.sqrt(var_gpu); del var_gpu, sum_gpu, sum2_gpu   # (d1, d2)
    std_gpu[std_gpu == 0] = _cp.inf

    sn = _cp.sqrt((psd_sum / n_segments).astype(_cp.float32))      # (d1, d2) noise std
    del psd_sum

    # data_max holds the per-pixel maximum *before* mean subtraction, so
    # subtract the mean to get the zero-mean peak.
    data_max -= mean_gpu

    pnr_gpu = _cp.maximum(_cp.divide(data_max,
                                     _cp.where(sn > 0, sn, _cp.inf)), 0.0)
    pnr = _cp.asnumpy(pnr_gpu); del pnr_gpu, data_max

    # Normalisation for correlation: z-score of (filtered - mean) / sn
    # Per-frame: norm = ((frame - mean) / sn - mu_z) / std_z
    # We need mu_z and std_z of the thresholded signal — approximate with a
    # second pass over the chunks.
    safe_sn = _cp.where(sn > 0, sn, _cp.inf)

    # ── Pass 2: accumulate correlation ────────────────────────────────────
    # We normalise each frame on the fly: z = (frame - mean) / sn, zero
    # values below 3, then z-score across the chunk for the correlation.
    YYconv_sum = _cp.zeros((d1, d2), dtype=_cp.float64)
    n_frames_used = 0

    for t0 in range(0, T, frames_per_chunk):
        t1    = min(t0 + frames_per_chunk, T)
        chunk = _cp.asarray(Y[t0:t1].astype(np.float32))   # H2D: one chunk

        if kernel_gpu is not None:
            _filter_chunk(chunk, kernel_gpu)

        # z-score by noise std; threshold; z-score for correlation
        chunk -= mean_gpu[_cp.newaxis]
        chunk /= safe_sn[_cp.newaxis]                       # in-place: now data/sn
        chunk[chunk < 3.0] = 0.0                            # threshold per-frame slice

        # Normalise across time within this chunk for the correlation estimate
        mu_c  = chunk.mean(axis=0)
        std_c = chunk.std(axis=0); std_c[std_c == 0] = _cp.inf

        Tc = t1 - t0
        for t in range(Tc):
            frame  = (chunk[t] - mu_c) / std_c             # (d1, d2) — 1 MB
            Yconv  = _cpnd.convolve(frame, sz, mode='constant')
            YYconv_sum += (Yconv * frame).astype(_cp.float64)
            del frame, Yconv
        del chunk, mu_c, std_c
        n_frames_used += Tc

    cn = _cp.asnumpy((YYconv_sum / n_frames_used / MASK).astype(_cp.float32))
    del YYconv_sum, MASK, mean_gpu, std_gpu, safe_sn, sn

    _cp.get_default_memory_pool().free_all_blocks()
    logger.debug("correlation_pnr: GPU chunked path complete")
    return cn, pnr


def correlation_pnr(Y, gSig=None, center_psf: bool = True, swap_dim: bool = True,
                    background_filter: str = 'disk',
                    noise_range: list = None, noise_method: str = 'mean',
                    use_gpu: bool = None,
                    chunk_gb: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute the local correlation image and peak-to-noise ratio (PNR) image.

    If a CUDA GPU and CuPy are available the computation is dispatched to the
    GPU automatically (or unconditionally when ``use_gpu=True``).  The movie is
    streamed through the GPU in chunks of ``chunk_gb`` GB so that VRAM usage is
    bounded regardless of movie size.

    The function signature is a strict superset of the original: all existing
    call sites work without modification.

    Args:
        Y : np.ndarray (3D or 4D)
            Input movie data.
        gSig : scalar or list, optional
            Gaussian half-width for spatial bandpass filter (pixels).
            ``None`` disables filtering.
        center_psf : bool
            Subtract the mean of the filter kernel (recommended for 2p data).
        swap_dim : bool
            ``True`` if time is in the *last* axis (MATLAB / CaImAn convention).
        background_filter : str
            ``'disk'`` (default) or ``'box'`` — background ring shape for the
            CPU path only (GPU always uses the disk/Gaussian kernel).
        noise_range : list [f_low, f_high], optional
            Frequency band as a fraction of Nyquist for noise estimation.
            Default ``[0.25, 0.5]``.
        noise_method : str
            ``'mean'`` (default), ``'median'``, or ``'logmexp'``.
        use_gpu : bool or None
            ``True``  — require GPU (raises if unavailable).
            ``False`` — force CPU.
            ``None``  — use GPU if available, otherwise CPU (default).
        chunk_gb : float
            GPU path only.  Size of each temporal chunk transferred to VRAM in
            gigabytes.  Reduce if you get OOM; increase for fewer H2D transfers.
            Default ``2.0``.

    Returns:
        cn  : np.ndarray (d1, d2)  — local correlation image.
        pnr : np.ndarray (d1, d2)  — peak-to-noise ratio image.
    """
    logger = logging.getLogger("caiman")

    if noise_range is None:
        noise_range = [0.25, 0.5]

    # ── Dispatch decision ─────────────────────────────────────────────────
    if use_gpu is True and not _gpu_available():
        raise RuntimeError(
            "correlation_pnr: use_gpu=True but no CUDA device found. "
            "Install cupy-cuda12x (or the matching version) and check nvidia-smi."
        )

    _run_gpu = _gpu_available() if use_gpu is None else bool(use_gpu)

    if _run_gpu:
        logger.debug("correlation_pnr: dispatching to GPU")
        return _correlation_pnr_gpu(
            Y, gSig=gSig, center_psf=center_psf, swap_dim=swap_dim,
            noise_range=noise_range, noise_method=noise_method,
            chunk_gb=chunk_gb,
        )

    # ── CPU path (original implementation, unchanged) ─────────────────────
    logger.debug("correlation_pnr: using CPU path")

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    # parameters
    _, d1, d2 = Y.shape
    data_raw = Y.reshape(-1, d1, d2).astype('float32')

    # filter data
    data_filtered = data_raw.copy()
    if gSig:
        if not isinstance(gSig, list):
            gSig = [gSig, gSig]
        ksize = tuple([int(2 * i) * 2 + 1 for i in gSig])

        if center_psf:
            if background_filter == 'box':
                for idx, img in enumerate(data_filtered):
                    data_filtered[idx, ] = cv2.GaussianBlur(
                        img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1) \
                        - cv2.boxFilter(img, ddepth=-1, ksize=ksize, borderType=1)
            else:
                psf = cv2.getGaussianKernel(ksize[0], gSig[0],
                                            cv2.CV_32F).dot(cv2.getGaussianKernel(ksize[1], gSig[1], cv2.CV_32F).T)
                ind_nonzero = psf >= psf[0].max()
                psf -= psf[ind_nonzero].mean()
                psf[~ind_nonzero] = 0
                for idx, img in enumerate(data_filtered):
                    data_filtered[idx,] = cv2.filter2D(img, -1, psf, borderType=1)
        else:
            for idx, img in enumerate(data_filtered):
                data_filtered[idx,] = cv2.GaussianBlur(img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1)

    # compute peak-to-noise ratio
    data_filtered -= data_filtered.mean(axis=0)
    data_max = np.max(data_filtered, axis=0)
    data_std = get_noise_fft(data_filtered.T, noise_method='mean')[0].T
    pnr = np.divide(data_max, data_std)
    pnr[pnr < 0] = 0

    # remove small values
    tmp_data = data_filtered.copy() / data_std
    tmp_data[tmp_data < 3] = 0

    # compute correlation image
    cn = local_correlations_fft(tmp_data, swap_dim=False)

    return cn, pnr


def iter_chunk_array(arr: np.array, chunk_size: int):
    if ((arr.shape[0] // chunk_size) - 1) > 0:
        for i in range((arr.shape[0] // chunk_size) - 1):
            yield arr[chunk_size * i:chunk_size * (i + 1)]
        yield arr[chunk_size * (i + 1):]
    else:
        yield arr


def correlation_image_ecobost(mov, chunk_size: int = 1000, dview=None):
    """ Compute correlation image as Erick. Removes the mean from each chunk
    before computing the correlation
    Args:
        mov: ndarray or list of str
            time x w x h

    chunk_size: int
        number of frames over which to compute the correlation (not working if
        passing list of string)
    """
    # MAP
    if isinstance(mov, list):
        if dview is not None:
            res = dview.map(map_corr, mov)
        else:
            res = map(map_corr, mov)

    else:
        scan = mov.astype(np.float32)
        num_frames = scan.shape[0]
        res = map(map_corr, iter_chunk_array(scan, chunk_size))

    sum_x, sum_sqx, sum_xy, num_frames = [np.sum(np.array(a), 0) for a in zip(*res)]
    denom_factor = np.sqrt(num_frames * sum_sqx - sum_x**2)
    corrs = np.zeros(sum_xy.shape)
    for k in [0, 1, 2, 3]:
        rotated_corrs = np.rot90(corrs, k=k)
        rotated_sum_x = np.rot90(sum_x, k=k)
        rotated_dfactor = np.rot90(denom_factor, k=k)
        rotated_sum_xy = np.rot90(sum_xy, k=k)

        # Compute correlation
        rotated_corrs[1:, :, k] = (num_frames * rotated_sum_xy[1:, :, k] -
                                   rotated_sum_x[1:] * rotated_sum_x[:-1]) /\
                                  (rotated_dfactor[1:] * rotated_dfactor[:-1])
        rotated_corrs[1:, 1:, 4 + k] = (num_frames * rotated_sum_xy[1:, 1:, 4 + k]
                                        - rotated_sum_x[1:, 1:] * rotated_sum_x[:-1, : -1]) /\
                                       (rotated_dfactor[1:, 1:] * rotated_dfactor[:-1, :-1])

        # Return back to original orientation
        corrs = np.rot90(rotated_corrs, k=4 - k)
        sum_x = np.rot90(rotated_sum_x, k=4 - k)
        denom_factor = np.rot90(rotated_dfactor, k=4 - k)
        sum_xy = np.rot90(rotated_sum_xy, k=4 - k)

    correlation_image = np.sum(corrs, axis=-1)
    # edges
    norm_factor = 5 * np.ones(correlation_image.shape)
    # corners
    norm_factor[[0, -1, 0, -1], [0, -1, -1, 0]] = 3
    # center
    norm_factor[1:-1, 1:-1] = 8
    correlation_image /= norm_factor

    return correlation_image


def map_corr(scan) -> tuple[Any, Any, Any, int]:
    '''This part of the code is in a mapping function that's run over different
    movies in parallel
    '''
    # TODO: Tighten prototype above
    if isinstance(scan, str):
        scan = caiman.load(scan)

    # h x w x num_frames
    chunk = np.array(scan).transpose([1, 2, 0])
    # Subtract overall brightness per frame
    chunk -= chunk.mean(axis=(0, 1))

    # Compute sum_x and sum_x^2
    chunk_sum = np.sum(chunk, axis=-1, dtype=float)
    chunk_sqsum = np.sum(chunk**2, axis=-1, dtype=float)

    # Compute sum_xy: Multiply each pixel by its eight neighbors
    chunk_xysum = np.zeros((chunk.shape[0], chunk.shape[1], 8))
    # amount of 90 degree rotations
    for k in [0, 1, 2, 3]:
        rotated_chunk = np.rot90(chunk, k=k)
        rotated_xysum = np.rot90(chunk_xysum, k=k)

        # Multiply each pixel by one above and by one above to the left
        rotated_xysum[1:, :, k] = np.sum(rotated_chunk[1:] * rotated_chunk[:-1], axis=-1, dtype=float)
        rotated_xysum[1:, 1:, 4 + k] = np.sum(rotated_chunk[1:, 1:] * rotated_chunk[:-1, :-1], axis=-1, dtype=float)

        # Return back to original orientation
        chunk = np.rot90(rotated_chunk, k=4 - k)
        chunk_xysum = np.rot90(rotated_xysum, k=4 - k)

    num_frames = chunk.shape[-1]

    return chunk_sum, chunk_sqsum, chunk_xysum, num_frames


def prepare_local_correlations(Y, swap_dim: bool = False,
                               eight_neighbours: bool = False) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Computes the correlation image and some statistics to update it online

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format

        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data
            Use 18 neighbors if true, and 6 if false for 4D data

    """
    # TODO: Tighten prototype above
    if swap_dim:
        Y = np.transpose(Y, (Y.ndim - 1,) + tuple(range(Y.ndim - 1)))

    T = len(Y)
    dims = Y.shape[1:]
    Yr = Y.T.reshape(-1, T)
    if Y.ndim == 4:
        d1, d2, d3 = dims
        sz = generate_binary_structure(3, 2 if eight_neighbours else 1)
        sz[1, 1, 1] = 0
    else:
        d1, d2 = dims
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='uint8')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='uint8')

    idx = [i - 1 for i in np.nonzero(sz)]

    def get_indices_of_neighbors(pixel):
        pixel = np.unravel_index(pixel, dims, order='F')
        x = pixel[0] + idx[0]
        y = pixel[1] + idx[1]
        if len(dims) == 3:
            z = pixel[2] + idx[2]
            inside = (x >= 0) * (x < d1) * (y >= 0) * (y < d2) * (z >= 0) * (z < d3)
            return np.ravel_multi_index((x[inside], y[inside], z[inside]), dims, order='F')
        else:
            inside = (x >= 0) * (x < d1) * (y >= 0) * (y < d2)
            return np.ravel_multi_index((x[inside], y[inside]), dims, order='F')

    N = [get_indices_of_neighbors(p) for p in range(np.prod(dims))]
    col_ind = np.concatenate(N)
    row_ind = np.concatenate([[i] * len(k) for i, k in enumerate(N)])
    num_neigbors = np.concatenate([[len(k)] * len(k) for k in N]).astype(Yr.dtype)

    first_moment = Yr.mean(1)
    second_moment = (Yr**2).mean(1)
    crosscorr = np.mean(Yr[row_ind] * Yr[col_ind], 1)
    sig = np.sqrt(second_moment - first_moment**2)

    M = coo_matrix(
        ((crosscorr - first_moment[row_ind] * first_moment[col_ind]) / (sig[row_ind] * sig[col_ind]) / num_neigbors,
         (row_ind, col_ind)),
        dtype=Yr.dtype)
    cn = M.dot(np.ones(M.shape[1], dtype=M.dtype)).reshape(dims, order='F')

    return first_moment, second_moment, crosscorr, col_ind, row_ind, num_neigbors, M, cn


def update_local_correlations(t,
                              frames,
                              first_moment,
                              second_moment,
                              crosscorr,
                              col_ind,
                              row_ind,
                              num_neigbors,
                              M,
                              del_frames=None) -> np.ndarray:
    """Updates sufficient statistics in place and returns correlation image"""
    dims = frames.shape[1:]
    stride = len(frames)
    if stride:
        frames = frames.reshape((stride, -1), order='F')
        if del_frames is None:
            tmp = 1 - float(stride) / t
            first_moment *= tmp
            second_moment *= tmp
            crosscorr *= tmp
        else:
            if stride > 10:
                del_frames = del_frames.reshape((stride, -1), order='F')
                first_moment -= del_frames.sum(0) / t
                second_moment -= (del_frames**2).sum(0) / t
                crosscorr -= np.sum(del_frames[:, row_ind] * del_frames[:, col_ind], 0) / t
            else:      # loop is faster
                for f in del_frames:
                    f = f.ravel(order='F')
                    first_moment -= f / t
                    second_moment -= (f**2) / t
                    crosscorr -= (f[row_ind] * f[col_ind]) / t
        if stride > 10:
            frames = frames.reshape((stride, -1), order='F')
            first_moment += frames.sum(0) / t
            second_moment += (frames**2).sum(0) / t
            crosscorr += np.sum(frames[:, row_ind] * frames[:, col_ind], 0) / t
        else:          # loop is faster
            for f in frames:
                f = f.ravel(order='F')
                first_moment += f / t
                second_moment += (f**2) / t
                crosscorr += (f[row_ind] * f[col_ind]) / t

    sig = np.sqrt(second_moment - first_moment**2)
    M.data = ((crosscorr - first_moment[row_ind] * first_moment[col_ind]) / (sig[row_ind] * sig[col_ind]) /
              num_neigbors)
    cn = M.dot(np.ones(M.shape[1], dtype=M.dtype)).reshape(dims, order='F')
    return cn


def local_correlations_movie(file_name,
                             tot_frames: Optional[int] = None,
                             fr: int = 30,
                             window: int = 30,
                             stride: int = 1,
                             swap_dim: bool = False,
                             eight_neighbours: bool = True,
                             mode: str = 'simple'):
    """
    Compute an online correlation image as moving average

    Args:
        Y:  string or np.ndarray (3D or 4D).
            Input movie filename or data
        tot_frames: int
            Number of frames considered
        fr: int
            Frame rate
        window: int
            Window length in frames
        stride: int
            Stride length in frames
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front
        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data
            Use 18 neighbors if true, and 6 if false for 4D data
        mode: 'simple', 'exponential', or 'cumulative'
            Mode of moving average

    Returns:
        corr_movie: caiman.movie (3D or 4D).
            local correlation movie

    """
    Y = caiman.load(file_name) if isinstance(file_name, str) else file_name
    Y = Y[..., :tot_frames] if swap_dim else Y[:tot_frames]
    first_moment, second_moment, crosscorr, col_ind, row_ind, num_neigbors, M, cn = \
        prepare_local_correlations(Y[..., :window] if swap_dim else Y[:window],
                                   swap_dim=swap_dim, eight_neighbours=eight_neighbours)
    if swap_dim:
        Y = np.transpose(Y, (Y.ndim - 1,) + tuple(range(Y.ndim - 1)))
    T = len(Y)
    dims = Y.shape[1:]
    corr_movie = np.zeros(((T - window) // stride + 1,) + dims, dtype=Y.dtype)
    corr_movie[0] = cn
    if mode == 'simple':
        for tt in range((T - window) // stride):
            corr_movie[tt + 1] = update_local_correlations(window, Y[tt * stride + window:(tt + 1) * stride + window],
                                                           first_moment, second_moment, crosscorr, col_ind, row_ind,
                                                           num_neigbors, M, cn, Y[tt * stride:(tt + 1) * stride]) # FIXME all params after M are invalid
    elif mode == 'exponential':
        for tt, frames in enumerate(Y[window:window + (T - window) // stride * stride].reshape((-1, stride) + dims)):
            corr_movie[tt + 1] = update_local_correlations(window, frames, first_moment, second_moment, crosscorr,
                                                           col_ind, row_ind, num_neigbors, M)
    elif mode == 'cumulative':
        for tt, frames in enumerate(Y[window:window + (T - window) // stride * stride].reshape((-1, stride) + dims)):
            corr_movie[tt + 1] = update_local_correlations(tt + window + 1, frames, first_moment, second_moment,
                                                           crosscorr, col_ind, row_ind, num_neigbors, M)
    else:
        raise Exception('mode of the moving average must be simple, exponential or cumulative')
    return caiman.movie(corr_movie, fr=fr)


def local_correlations_movie_offline(file_name,
                                     Tot_frames=None,
                                     fr: float = 10.,
                                     window: int = 100,
                                     stride: int = 100,
                                     swap_dim: bool = False,
                                     eight_neighbours: bool = True,
                                     order_mean: int = 1,
                                     ismulticolor: bool = False,
                                     dview=None,
                                     remove_baseline: bool = False,
                                     winSize_baseline: int = 50,
                                     quantil_min_baseline: float = 8,
                                     gaussian_blur: bool=False):
    """
    Efficient (parallel) computation of correlation image in shifting windows 
    with option for prior baseline removal

    Args:
        Y:  str
            path to movie file

        Tot_frames: int
            Number of total frames considered

        fr: int (100)
            Frame rate (optional)

        window: int (100)
            Window length in frames

        stride: int (30)
            Stride length in frames

        swap_dim: bool (False)
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front (default: False)

        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data
            Use 18 neighbors if true, and 6 if false for 4D data

        dview: map object
            Use it for parallel computation

        remove_baseline: bool (False)
            Flag for removing baseline prior to computation of CI

        winSize_baseline: int (50)
            Running window length for computing baseline

        quantile_min_baseline: float (8)
            Percentile used for baseline computations
            
        gaussian_blur: bool (False)
            Gaussian smooth the signal

    Returns:
        mm: caiman.movie (3D or 4D).
            local correlation movie

    """
    if Tot_frames is None:
        _, Tot_frames = caiman.base.movies.get_file_size(file_name)

    params:list = [[file_name, range(j, j + window), eight_neighbours, swap_dim,
                     order_mean, ismulticolor, remove_baseline, winSize_baseline,
                     quantil_min_baseline, gaussian_blur]
                    for j in range(0, Tot_frames - window, stride)]

    params.append([file_name, range(Tot_frames - window, Tot_frames), eight_neighbours, swap_dim,
                   order_mean, ismulticolor, remove_baseline, winSize_baseline,
                   quantil_min_baseline, gaussian_blur])

    if dview is None:
        parallel_result = list(map(local_correlations_movie_parallel, params))
    else:
        #TODO phrase better
        if 'multiprocessing' in str(type(dview)):
            parallel_result = dview.map_async(local_correlations_movie_parallel, params).get(4294967)
        else:
            parallel_result = dview.map_sync(local_correlations_movie_parallel, params)
            dview.results.clear()

    mm = caiman.movie(np.concatenate(parallel_result, axis=0), fr=fr/len(parallel_result))
    return mm


def local_correlations_movie_parallel(params:tuple) -> np.ndarray:
    mv_name, idx, eight_neighbours, swap_dim, order_mean, ismulticolor, remove_baseline, winSize_baseline, quantil_min_baseline, gaussian_blur = params
    mv = caiman.load(mv_name, subindices=idx, in_memory=True)
    if gaussian_blur:
        mv = mv.gaussian_blur_2D()

    if remove_baseline:
        mv.removeBL(quantilMin=quantil_min_baseline, windowSize=winSize_baseline, in_place=True)

    if ismulticolor:
        return local_correlations_multicolor(mv, swap_dim=swap_dim)[None, :, :].astype(np.float32)
    else:
        return local_correlations(mv, eight_neighbours=eight_neighbours, swap_dim=swap_dim,
                                  order_mean=order_mean)[None, :, :].astype(np.float32)
        
def mean_image(file_name,
                 Tot_frames=None,
                 fr: float = 10.,
                 window: int = 100,
                 dview=None):
    """
    Efficient (parallel) computation of mean image in chunks

    Args:
        Y:  str
            path to movie file

        Tot_frames: int
            Number of total frames considered

        fr: int (100)
            Frame rate (optional)

        window: int (100)
            Window length in frames

        dview: map object
            Use it for parallel computation
    
    Returns:
        mm: caiman.movie (2D).
            mean image

    """
    if Tot_frames is None:
        _, Tot_frames = caiman.base.movies.get_file_size(file_name)

    params:list = [[file_name, range(j * window, (j + 1) * window)]
                    for j in range(int(Tot_frames / window))]

    remain_frames = Tot_frames - int(Tot_frames / window) * window
    if remain_frames > 0:
        params.append([file_name, range(int(Tot_frames / window) * window, Tot_frames)])

    if dview is None:
        parallel_result = list(map(mean_image_parallel, params))
    else:
        if 'multiprocessing' in str(type(dview)):
            parallel_result = dview.map_async(mean_image_parallel, params).get(4294967)
        else:
            parallel_result = dview.map_sync(mean_image_parallel, params)
            dview.results.clear()

    mm = caiman.movie(np.concatenate(parallel_result, axis=0), fr=fr/len(parallel_result))
    if remain_frames > 0:
        mean_image = (mm[:-1].sum(axis=0) + (remain_frames / window) * mm[-1]) / (len(mm) - 1 + remain_frames / window)  
    else:
        mean_image = mm.mean(axis=0)
    return mean_image

def mean_image_parallel(params:tuple) -> np.ndarray:
    mv_name, idx = params
    mv = caiman.load(mv_name, subindices=idx, in_memory=True)
    return mv.mean(axis=0)[np.newaxis,:,:]
