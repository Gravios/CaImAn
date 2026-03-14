#!/usr/bin/env python

""" A set of pre-processing operations in the input dataset:

1. Interpolation of missing data
2. Identification of saturated pixels
3. Estimation of noise level for each imaged voxel
4. Estimation of global time constants

"""

import logging
import numpy as np
import scipy
from scipy.linalg import toeplitz
import shutil
import tempfile

import caiman.mmapping


def interpolate_missing_data(Y):
    """
    Interpolate any missing data using nearest neighbor interpolation.
    Missing data is identified as entries with values NaN

    Args:
        Y   np.ndarray (3D)
            movie, raw data in 3D format (d1 x d2 x T)

    Returns:
        Y   np.ndarray (3D)
            movie, data with interpolated entries (d1 x d2 x T)
        coordinate list
            list of interpolated coordinates

    Raises:
        Exception 'The algorithm has not been tested with missing values (NaNs). Remove NaNs and rerun the algorithm.'
    """
    logger = logging.getLogger("caiman")
    coor = []
    logger.info('Checking for missing data entries (NaN)')
    if np.any(np.isnan(Y)):
        logger.info('Interpolating missing data')
        for idx, row in enumerate(Y):
            nans = np.where(np.isnan(row))[0]
            n_nans = np.where(~np.isnan(row))[0]
            coor.append((idx, nans))
            Y[idx, nans] = np.interp(nans, n_nans, row[n_nans])
        raise Exception(
            'The algorithm has not been tested with missing values (NaNs). Remove NaNs and rerun the algorithm.')

    return Y, coor

def find_unsaturated_pixels(Y, saturationValue=None, saturationThreshold=0.9, saturationTime=0.005):
    """Identifies the saturated pixels that are saturated and returns the ones that are not.
    A pixel is defined as saturated if its observed fluorescence is above
    saturationThreshold*saturationValue at least saturationTime fraction of the time.

    Args:
        Y: np.ndarray
            input movie data, either 2D or 3D with time in the last axis

        saturationValue: scalar (optional)
            Saturation level, default value the lowest power of 2 larger than max(Y)

        saturationThreshold: scalar between 0 and 1 (optional)
            Fraction of saturationValue above which the fluorescence is considered to
            be in the saturated region. Default value 0.9

        saturationTime: scalar between 0 and 1 (optional)
            Fraction of time that pixel needs to be in the saturated
            region to be considered saturated. Default: 0.005

    Returns:
        normalPixels:   nd.array
            list of unsaturated pixels
    """
    if saturationValue is None:
        saturationValue = np.power(2, np.ceil(np.log2(np.max(Y)))) - 1

    Ysat = (Y >= saturationThreshold * saturationValue)
    pix = np.mean(Ysat, Y.ndim - 1).flatten('F') > saturationTime
    normalPixels = np.where(pix)

    return normalPixels

def get_noise_welch(Y, noise_range=[0.25, 0.5], noise_method='logmexp',
                    max_num_samples_fft=3072):
    """Estimate the noise level for each pixel by averaging the power spectral density.

    Args:
        Y: np.ndarray
            Input movie data with time in the last axis

        noise_range: np.ndarray [2 x 1] between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
            default: [0.25,0.5]

        noise method: string
            method of averaging the noise.
            Choices:
                'mean': Mean
                'median': Median
                'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
        sn: np.ndarray
            Noise level for each pixel
    """
    T = Y.shape[-1]
    if T > max_num_samples_fft:
        Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1],
                            Y[..., int(T // 2 - max_num_samples_fft / 3 / 2):
                            int(T // 2 + max_num_samples_fft / 3 / 2)],
                            Y[..., -max_num_samples_fft // 3:]), axis=-1)
        T = Y.shape[-1]
    ff, Pxx = scipy.signal.welch(Y)
    Pxx = Pxx[..., (ff >= noise_range[0]) & (ff <= noise_range[1])]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx, -1) / 2),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx, -1) / 2),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(Pxx / 2), -1)))
    }[noise_method](Pxx)
    return sn


def _get_noise_fft_gpu(Y, noise_range, noise_method, max_num_samples_fft):
    """GPU-accelerated noise estimation via CuPy batched rfft.

    Replaces the CPU path for large inputs (pixels x T) where:
      - np.concatenate would allocate a 3+ GB dense array on CPU RAM
      - The cv2.dft loop iterates 262,144 times serially

    Strategy:
      1. Build temporal sub-sample indices on CPU (~3072 frames)
      2. Transfer only the sub-sampled columns to GPU (~3.2 GB H2D, once)
      3. Run cp.fft.rfft across all pixels in a single batched kernel
      4. Compute PSD and aggregate (mean/median/logmexp) on GPU
      5. Pull back only the result vector — (pixels,) float32, ~1 MB D2H

    Returns sn (np.ndarray, CPU) and psdx=None (no caller uses psdx).
    """
    import cupy as cp
    logger = logging.getLogger("caiman")

    T_orig   = Y.shape[-1]
    n_pixels = int(np.prod(Y.shape[:-1]))

    # ── VRAM pre-check: bail out early if allocation would OOM ─────────────
    # Y_sub (float32) + rfft output (complex64) ≈ n_pixels × T_sub × 12 bytes.
    # CUDA_ERROR_ILLEGAL_ADDRESS occurs when the kernel accesses beyond the
    # allocated region — raise early to trigger the graceful CPU fallback.
    T_sub = min(max_num_samples_fft, T_orig) if max_num_samples_fft else T_orig
    _bytes_needed = n_pixels * T_sub * 12  # float32 + complex64
    try:
        _free, _total = cp.cuda.Device(0).mem_info
        if _bytes_needed > _free * 0.85:
            raise MemoryError(
                f"precheck: need {_bytes_needed/1e9:.1f} GB, only "
                f"{_free/1e9:.1f} GB free on GPU — skipping GPU noise FFT"
            )
    except Exception as _vram_exc:
        raise _vram_exc  # propagate so caller falls back to CPU

    # ── Build temporal sub-sample indices (CPU) ───────────────────────────────
    if T_orig > max_num_samples_fft:
        n3  = max_num_samples_fft // 3
        mid = T_orig // 2
        idx = np.concatenate([
            np.arange(1, n3 + 1),
            np.arange(mid - n3 // 2, mid + n3 // 2),
            np.arange(T_orig - n3, T_orig),
        ])
        T = len(idx)
    else:
        idx = None
        T = T_orig

    # ── Frequency mask ────────────────────────────────────────────────────────
    ff  = np.arange(0, 0.5 + 1.0 / T, 1.0 / T)
    ind = (ff > noise_range[0]) & (ff <= noise_range[1])
    ind = ind[:T // 2 + 1]      # trim to rfft output length
    ind_gpu = cp.asarray(ind)

    # ── Transfer sub-sampled frames to GPU ────────────────────────────────────
    # Index only the ~3072 selected time points — avoids materialising the
    # full (pixels, T_orig) array in CPU RAM before the H2D transfer.
    Y_flat = Y.reshape(n_pixels, T_orig) if Y.ndim > 1 else Y[np.newaxis]
    Y_sub  = np.ascontiguousarray(
        Y_flat[:, idx] if idx is not None else Y_flat,
        dtype=np.float32,
    )                                               # (pixels, T_sub) — ~3.2 GB
    Y_gpu  = cp.asarray(Y_sub)
    del Y_sub                                       # free CPU copy immediately

    # ── Batched rfft + PSD ────────────────────────────────────────────────────
    xdft  = cp.fft.rfft(Y_gpu, axis=-1)            # (pixels, T//2+1) complex64
    del Y_gpu
    psdx  = (2.0 / T) * (xdft.real ** 2 + xdft.imag ** 2)
    del xdft
    psdx  = psdx[:, ind_gpu]                       # (pixels, n_freq_bins)

    # ── Aggregate on GPU (mean_psd equivalent) ────────────────────────────────
    if noise_method == 'mean':
        sn_gpu = cp.sqrt(cp.mean(psdx / 2.0, axis=-1))
    elif noise_method == 'median':
        sn_gpu = cp.sqrt(cp.median(psdx / 2.0, axis=-1))
    else:   # logmexp
        sn_gpu = cp.sqrt(cp.exp(cp.mean(cp.log(psdx / 2.0 + 1e-10), axis=-1)))
    del psdx

    # ── Pull result to CPU ────────────────────────────────────────────────────
    sn = cp.asnumpy(sn_gpu).reshape(Y.shape[:-1])
    del sn_gpu
    logger.debug(f"get_noise_fft GPU: sn shape={sn.shape}, T_sub={T}, n_freq={ind.sum()}")
    return sn, None   # psdx=None; no caller uses it


def get_noise_fft(Y, noise_range=[0.25, 0.5], noise_method='logmexp', max_num_samples_fft=3072,
                  opencv=True):
    """Estimate the noise level for each pixel by averaging the power spectral density.

    Args:
        Y: np.ndarray
            Input movie data with time in the last axis

        noise_range: np.ndarray [2 x 1] between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
            default: [0.25,0.5]

        noise method: string
            method of averaging the noise.
            Choices:
                'mean': Mean
                'median': Median
                'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
        sn: np.ndarray
            Noise level for each pixel
        psdx: np.ndarray or None
            Power spectral density values (None on GPU path; no caller uses this)
    """
    T        = Y.shape[-1]
    n_pixels = int(np.prod(Y.shape[:-1]))

    # ── GPU path ──────────────────────────────────────────────────────────────
    # Use when the input is large enough that the CPU concatenation cost
    # (3+ GB dense alloc) and the serial cv2.dft loop (262k iterations)
    # are both significant.  For small inputs (patch workers, single pixels)
    # H2D overhead dominates and the CPU path is faster.
    if n_pixels > 4096 and T > max_num_samples_fft:
        try:
            return _get_noise_fft_gpu(Y, noise_range, noise_method, max_num_samples_fft)
        except Exception as _gpu_exc:
            logging.getLogger("caiman").warning(
                f"get_noise_fft: GPU path failed ({_gpu_exc}); falling back to CPU"
            )

    # ── CPU path (original) ───────────────────────────────────────────────────
    if T > max_num_samples_fft:
        Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1],
                            Y[..., int(T // 2 - max_num_samples_fft / 3 / 2)
                                          :int(T // 2 + max_num_samples_fft / 3 / 2)],
                            Y[..., -max_num_samples_fft // 3:]), axis=-1)
        T = Y.shape[-1]

    # we create a map of what is the noise on the FFT space
    ff = np.arange(0, 0.5 + 1. / T, 1. / T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1, ind2)
    # we compute the mean of the noise spectral density s
    if Y.ndim > 1:
        if opencv:
            import cv2
            try:
                cv2.setNumThreads(0)
            except:
                pass
            psdx_list = []
            for y in Y.reshape(-1, T):
                dft = cv2.dft(y, flags=cv2.DFT_COMPLEX_OUTPUT).squeeze()[
                    :len(ind)][ind]
                psdx_list.append(np.sum(1. / T * dft * dft, 1))
            psdx = np.reshape(psdx_list, Y.shape[:-1] + (-1,))
        else:
            xdft = np.fft.rfft(Y, axis=-1)
            xdft = xdft[..., ind[:xdft.shape[-1]]]
            psdx = 1. / T * abs(xdft)**2
        psdx *= 2
        sn = mean_psd(psdx, method=noise_method)

    else:
        xdft = np.fliplr(np.fft.rfft(Y))
        psdx = 1. / T * (xdft**2)
        psdx[1:] *= 2
        sn = mean_psd(psdx[ind[:psdx.shape[0]]], method=noise_method)

    return sn, psdx


def get_noise_fft_parallel(Y, n_pixels_per_process=100, dview=None, **kwargs):
    """parallel version of get_noise_fft.

    Args:
        Y: ndarray
            input movie (n_pixels x Time). Can be also memory mapped file.

        n_processes: [optional] int
            number of processes/threads to use concurrently

        n_pixels_per_process: [optional] int
            number of pixels to be simultaneously processed by each process

        backend: [optional] string
            the type of concurrency to be employed. only 'multithreading' for the moment

        **kwargs: [optional] dict
            all the parameters passed to get_noise_fft

    Returns:
        sn: ndarray(double)
            noise associated to each pixel
    """
    folder = tempfile.mkdtemp()

    # Pre-allocate a writeable shared memory map as a container for the
    # results of the parallel computation
    pixel_groups = list(
        range(0, Y.shape[0] - n_pixels_per_process + 1, n_pixels_per_process))

    if isinstance(Y, np.memmap):  # if input file is already memory mapped then find the filename
        Y_name = Y.filename

    else:
        if dview is not None:
            raise Exception('parallel processing requires memory mapped files')
        Y_name = Y

    argsin = [(Y_name, i, n_pixels_per_process, kwargs) for i in pixel_groups]
    pixels_remaining = Y.shape[0] % n_pixels_per_process
    if pixels_remaining > 0:
        argsin.append(
            (Y_name, Y.shape[0] - pixels_remaining, pixels_remaining, kwargs))

    if dview is None:
        print('Single Thread')
        results = list(map(fft_psd_multithreading, argsin))

    else:
        if 'multiprocessing' in str(type(dview)):
            results = dview.map_async(
                fft_psd_multithreading, argsin).get(4294967)
        else:
            ne = len(dview)
            print(('Running on %d engines.' % (ne)))
            if dview.client.profile == 'default':
                results = dview.map_sync(fft_psd_multithreading, argsin)

            else:
                print(('PROFILE:' + dview.client.profile))
                results = dview.map_sync(fft_psd_multithreading, argsin)

    _, _, psx_ = results[0]
    sn_s = np.zeros(Y.shape[0])
    psx_s = np.zeros((Y.shape[0], psx_.shape[-1]))
    for idx, sn, psx_ in results:
        sn_s[idx] = sn
        psx_s[idx, :] = psx_

    sn_s = np.array(sn_s)
    psx_s = np.array(psx_s)

    try:
        shutil.rmtree(folder)

    except:
        print(("Failed to delete: " + folder))
        raise

    return sn_s, psx_s

def fft_psd_parallel(Y, sn_s, i, num_pixels, **kwargs):
    """helper function to parallelize get_noise_fft

    Args:
        Y: ndarray
                input movie (n_pixels x Time), can be also memory mapped file
        sn_s: ndarray (memory mapped)
            file where to store the results of computation.
        i: int
            pixel index start
        num_pixels: int
            number of pixel to select starting from i
        **kwargs: dict
            arguments to be passed to get_noise_fft

     Returns:
        idx: list
            list of the computed pixels
        res: ndarray(double)
            noise associated to each pixel
        psx: ndarray
            position of those pixels
    """
    idxs = list(range(i, i + num_pixels))
    res = get_noise_fft(Y[idxs], **kwargs)
    sn_s[idxs] = res

def fft_psd_multithreading(args):
    """helper function to parallelize get_noise_fft

    Args:
        Y: ndarray
            input movie (n_pixels x Time), can be also memory mapped file
        sn_s: ndarray (memory mapped)
            file where to store the results of computation.
        i: int
            pixel index start
        num_pixels: int
            number of pixel to select starting from i
        **kwargs: dict
            arguments to be passed to get_noise_fft

    Returns:
        idx: list
            list of the computed pixels
        res: ndarray(double)
            noise associated to each pixel
        psx: ndarray
            position of those pixels
    """

    (Y, i, num_pixels, kwargs) = args
    if isinstance(Y, str):
        Y, _, _ = caiman.mmapping.load_memmap(Y)

    idxs = list(range(i, i + num_pixels))
    #print(len(idxs))
    res, psx = get_noise_fft(Y[idxs], **kwargs)

    return (idxs, res, psx)

def mean_psd(y, method='logmexp'):
    """
    Averaging the PSD

    Args:
        y: np.ndarray
             PSD values

        method: string
            method of averaging the noise.
            Choices:
             'mean': Mean
             'median': Median
             'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
        mp: array
            mean psd
    """

    if method == 'mean':
        mp = np.sqrt(np.mean(y / 2, axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(y / 2, axis=-1))
    else:
        mp = np.log((y + 1e-10) / 2)
        mp = np.mean(mp, axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)

    return mp

def estimate_time_constant(Y, sn, p=None, lags=5, include_noise=False, pixels=None):
    """
    Estimating global time constants for the dataset Y through the autocovariance function (optional).
    The function is no longer used in the standard setting of the algorithm since every trace has its own
    time constant.

    Args:
        Y: np.ndarray (2D)
            input movie data with time in the last axis

        p: positive integer
            order of AR process, default: 2

        lags: positive integer
            number of lags in the past to consider for determining time constants. Default 5

        include_noise: Boolean
            Flag to include pre-estimated noise value when determining time constants. Default: False

        pixels: np.ndarray
            Restrict estimation to these pixels (e.g., remove saturated pixels). Default: All pixels

    Returns:
        g:  np.ndarray (p x 1)
            Discrete time constants
    """
    if p is None:
        raise Exception("You need to define p")
    if pixels is None:
        pixels = np.arange(np.size(Y) // Y.shape[-1])

    npx = len(pixels)
    lags += p
    XC = np.zeros((npx, 2 * lags + 1))
    for j in range(npx):
        XC[j, :] = np.squeeze(axcov(np.squeeze(Y[pixels[j], :]), lags))

    gv = np.zeros(npx * lags)
    if not include_noise:
        XC = XC[:, np.arange(lags - 1, -1, -1)]
        lags -= p

    A = np.zeros((npx * lags, p))
    for i in range(npx):
        if not include_noise:
            A[i * lags + np.arange(lags), :] = toeplitz(np.squeeze(XC[i, np.arange(
                p - 1, p + lags - 1)]), np.squeeze(XC[i, np.arange(p - 1, -1, -1)]))
        else:
            A[i * lags + np.arange(lags), :] = toeplitz(np.squeeze(XC[i, lags + np.arange(
                lags)]), np.squeeze(XC[i, lags + np.arange(p)])) - (sn[i]**2) * np.eye(lags, p)
            gv[i * lags + np.arange(lags)] = np.squeeze(XC[i, lags + 1:])

    if not include_noise:
        gv = XC[:, p:].T
        gv = np.squeeze(np.reshape(gv, (np.size(gv), 1), order='F'))

    g = np.dot(np.linalg.pinv(A), gv)

    return g


def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag

    Args:
        data : array
            Array containing fluorescence data

        maxlag : int
            Number of lags to use in autocovariance calculation

    Returns:
        axcov : array
            Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(xcov / T)

def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).

    Args:
        value : int

    Returns:
        exponent : int
    """

    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent


def preprocess_data(Y, sn=None, dview=None, n_pixels_per_process=100,
                    noise_range=[0.25, 0.5], noise_method='logmexp',
                    compute_g=False, p=2, lags=5, include_noise=False,
                    pixels=None, max_num_samples_fft=3000, check_nan=True):
    """
    Performs the pre-processing operations described above.

    Args:
        Y: ndarray
            input movie (n_pixels x Time). Can be also memory mapped file.

        n_processes: [optional] int
            number of processes/threads to use concurrently

        n_pixels_per_process: [optional] int
            number of pixels to be simultaneously processed by each process

        p: positive integer
            order of AR process, default: 2

        lags: positive integer
            number of lags in the past to consider for determining time constants. Default 5

        include_noise: Boolean
            Flag to include pre-estimated noise value when determining time constants. Default: False

        noise_range: np.ndarray [2 x 1] between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
            default: [0.25,0.5]

        noise method: string
            method of averaging the noise.
            Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
        Y: ndarray
             movie preprocessed (n_pixels x Time). Can be also memory mapped file.
        g:  np.ndarray (p x 1)
            Discrete time constants
        psx: ndarray
            position of those pixels
        sn_s: ndarray (memory mapped)
            file where to store the results of computation.
    """

    if check_nan:
        Y, coor = interpolate_missing_data(Y)

    if sn is None:
        if dview is None:
            sn, psx = get_noise_fft(Y, noise_range=noise_range, noise_method=noise_method,
                                    max_num_samples_fft=max_num_samples_fft)
        else:
            sn, psx = get_noise_fft_parallel(Y, n_pixels_per_process=n_pixels_per_process, dview=dview,
                                             noise_range=noise_range, noise_method=noise_method,
                                             max_num_samples_fft=max_num_samples_fft)
    else:
        psx = None

    if compute_g:
        g = estimate_time_constant(Y, sn, p=p, lags=lags,
                                   include_noise=include_noise, pixels=pixels)
    else:
        g = None

    # psx  # no need to keep psx in memory as long a we don't use it elsewhere
    return Y, sn, g, None
