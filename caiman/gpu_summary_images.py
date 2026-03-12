"""
gpu_summary_images.py
=====================
GPU-accelerated drop-in replacement for caiman.summary_images.correlation_pnr.

All operations run on the GPU via CuPy:
  - Spatial PSF filtering        : cupyx.scipy.ndimage.convolve (batched over T)
  - Noise estimation              : cp.fft.rfft over all pixels simultaneously
                                    (replaces 262,144 serial cv2.dft calls on CPU)
  - Local correlation image       : cp.ndimage.uniform_filter + elementwise reduce

Falls back to the CPU implementation if CuPy is unavailable.

Usage
-----
    from gpu_summary_images import correlation_pnr_gpu
    cn, pnr = correlation_pnr_gpu(images[::5], gSig=gSig[0], swap_dim=False)
"""

from __future__ import annotations
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpnd
    _CUPY_OK = True
except ImportError:
    _CUPY_OK = False


def _gpu_available() -> bool:
    if not _CUPY_OK:
        return False
    try:
        cp.zeros(1)
        return True
    except Exception:
        return False


def _local_correlations_gpu(Y_gpu: "cp.ndarray") -> "cp.ndarray":
    """
    Compute per-pixel correlation with 8-neighbours.
    Y_gpu: (T, d1, d2) float32, already zero-mean and unit-variance along axis 0.
    Returns: (d1, d2) float32 correlation image.
    """
    # Neighbour sum kernel (3x3, centre=0)
    sz = cp.ones((3, 3), dtype=cp.float32)
    sz[1, 1] = 0.0

    # Apply 3×3 neighbour-sum filter to every frame
    # cupyx.scipy.ndimage.convolve doesn't batch natively — use uniform_filter trick:
    # sum of 8 neighbours = (3×3 box sum) - centre pixel
    # box_sum = uniform_filter(img, size=3) * 9
    Yconv = cp.stack([
        cpnd.uniform_filter(img, size=3) * 9.0 - img
        for img in Y_gpu          # loop over T frames; each is (d1,d2) on GPU
    ])                             # (T, d1, d2)

    MASK = 8.0                     # 8 neighbours per interior pixel (border pixels
                                   # have fewer, but uniform_filter handles edges)

    YYconv = Yconv * Y_gpu         # (T, d1, d2) elementwise
    cn = cp.mean(YYconv, axis=0) / MASK   # (d1, d2)
    return cn


def correlation_pnr_gpu(
    Y,
    gSig: float | None = None,
    center_psf: bool = True,
    swap_dim: bool = True,
    noise_range: list[float] | None = None,
    noise_method: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation image and peak-to-noise ratio image on the GPU.

    Parameters
    ----------
    Y : np.ndarray, shape (T, d1, d2)  or  (d1, d2, T) if swap_dim=True
    gSig : Gaussian half-width for spatial bandpass filter (same unit as pixels)
    center_psf : subtract mean of filter kernel (recommended for 2p data)
    swap_dim : True if time is in the last axis (CaImAn convention for raw arrays)
    noise_range : [f_low, f_high] as fraction of Nyquist; default [0.25, 0.5]
    noise_method : 'mean' | 'median' | 'logmexp'

    Returns
    -------
    cn  : np.ndarray (d1, d2) — local correlation image
    pnr : np.ndarray (d1, d2) — peak-to-noise ratio image
    """
    if noise_range is None:
        noise_range = [0.25, 0.5]

    if not _gpu_available():
        # Graceful fallback
        from caiman.summary_images import correlation_pnr
        return correlation_pnr(Y, gSig=gSig, center_psf=center_psf,
                               swap_dim=swap_dim)

    # ── move to GPU ────────────────────────────────────────────────────────
    if swap_dim:
        Y = np.transpose(Y, (Y.ndim - 1,) + tuple(range(Y.ndim - 1)))

    T, d1, d2 = Y.shape
    data_gpu = cp.asarray(Y.reshape(T, d1, d2).astype(np.float32))  # one H2D transfer

    # ── spatial bandpass filter (PSF centering) ────────────────────────────
    if gSig is not None:
        import cv2
        ksize = int(2 * gSig) * 2 + 1

        if center_psf:
            # Build centered PSF kernel once on CPU, transfer to GPU
            psf_cpu = cv2.getGaussianKernel(ksize, gSig, cv2.CV_32F)
            psf_cpu = psf_cpu.dot(psf_cpu.T)
            ind_nz = psf_cpu >= psf_cpu[0].max()
            psf_cpu -= psf_cpu[ind_nz].mean()
            psf_cpu[~ind_nz] = 0.0
            kernel = cp.asarray(psf_cpu)                    # (ksize, ksize) on GPU
        else:
            psf_cpu = cv2.getGaussianKernel(ksize, gSig, cv2.CV_32F)
            psf_cpu = psf_cpu.dot(psf_cpu.T)
            kernel = cp.asarray(psf_cpu)

        # Batch convolution: convolve each frame independently
        # cupyx.scipy.ndimage.convolve works on a single 2D array;
        # we vectorise over the time axis with a loop — still GPU, each call
        # is a cuDNN-backed FFT convolution on a (d1,d2) tile.
        for t in range(T):
            data_gpu[t] = cpnd.convolve(data_gpu[t], kernel, mode='reflect')

    # ── zero-mean along time ───────────────────────────────────────────────
    data_gpu -= data_gpu.mean(axis=0, keepdims=True)

    # ── peak fluorescence ──────────────────────────────────────────────────
    data_max = data_gpu.max(axis=0)                         # (d1, d2)

    # ── noise estimation via batched rfft ─────────────────────────────────
    # CPU path: cv2.dft called once per pixel = 262,144 serial calls for 512×512.
    # GPU path: single cp.fft.rfft over the time axis across ALL pixels at once.
    xdft = cp.fft.rfft(data_gpu.reshape(T, -1).T, axis=-1)  # (pixels, T//2+1)
    ff = cp.linspace(0, 0.5, xdft.shape[-1])
    band = (ff >= noise_range[0]) & (ff <= noise_range[1])
    psd = (cp.abs(xdft[:, band]) ** 2) * (2.0 / T)         # (pixels, n_freqs)

    if noise_method == "mean":
        sn = psd.mean(axis=-1)
    elif noise_method == "median":
        sn = cp.median(psd, axis=-1)
    else:  # logmexp
        sn = cp.exp(cp.mean(cp.log(psd + 1e-10), axis=-1))

    sn = cp.sqrt(sn).reshape(d1, d2)                        # (d1, d2) std estimate

    # ── PNR image ─────────────────────────────────────────────────────────
    pnr_gpu = cp.divide(data_max, cp.where(sn > 0, sn, cp.inf))
    pnr_gpu = cp.maximum(pnr_gpu, 0.0)

    # ── threshold and compute local correlation ────────────────────────────
    tmp_data = data_gpu / cp.where(sn > 0, sn, cp.inf)[cp.newaxis, ...]
    tmp_data[tmp_data < 3.0] = 0.0

    # Normalise for correlation computation
    mu  = tmp_data.mean(axis=0, keepdims=True)
    std = tmp_data.std(axis=0, keepdims=True)
    std[std == 0] = cp.inf
    tmp_norm = (tmp_data - mu) / std

    cn_gpu = _local_correlations_gpu(tmp_norm)

    # ── transfer results back to CPU ───────────────────────────────────────
    cn  = cp.asnumpy(cn_gpu)
    pnr = cp.asnumpy(pnr_gpu)

    # free VRAM
    del data_gpu, tmp_data, tmp_norm, xdft, psd, sn, pnr_gpu, cn_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return cn, pnr
