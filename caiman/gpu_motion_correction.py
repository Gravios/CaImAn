"""
gpu_motion_correction.py
========================
GPU-accelerated drop-in replacement for CaImAn's multiprocessing motion
correction dispatch, using CuPy.

Installation
------------
    pip install cupy-cuda12x      # CUDA 12.x  (RTX 5070 Ti)
    pip install cupy-cuda11x      # CUDA 11.x

If CuPy is absent, ``gpu_available()`` returns False.

Architecture — Rigid MC
-----------------------
CPU path: T frames → N chunks → N subprocesses → sequential per-frame FFTs.
GPU path: T frames → GPU batches of B → cuFFT batch → single stream, no
multiprocessing overhead.

Steps per batch of B frames:
  1. H2D transfer of B frames
  2. cuFFT batch fft2(B frames)            — one kernel
  3. × conj(template_freq)                 — one kernel
  4. cuFFT batch ifft2 → B cross-corrs     — one kernel
  5. Masked batched argmax → B coarse shts — one kernel
  6. Batched upsampled DFT (2× cuBLAS GEMM) → sub-pixel shifts
  7. Phase-ramp × IFFT batch → B corrected — two kernels
  8. D2H + write to mmap

Architecture — PW-Rigid MC
--------------------------
For each batch of B frames:
  1. Extract P patches per frame → (B·P, ph, pw) GPU tensor
  2. cuFFT batch all B·P patches
  3. × conj(template_patch_freqs)  (pre-computed once)
  4. cuFFT batch ifft2 → B·P cross-correlations
  5. Batched argmax → (B, P, 2) patch shifts
  6. Build (B, H, W) shift field via GPU resize
  7. cupyx.scipy.ndimage.map_coordinates → B corrected frames
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from numpy.fft import ifftshift

logger = logging.getLogger("caiman")

# ── Optional CuPy import ─────────────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpnd
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def gpu_available() -> bool:
    """Return True when CuPy is installed and at least one CUDA device exists."""
    if not _CUPY_AVAILABLE:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _require_gpu(fn_name: str) -> None:
    if not _CUPY_AVAILABLE:
        raise RuntimeError(
            f"{fn_name} requires CuPy.\n"
            "Install with:  pip install cupy-cuda12x  (CUDA 12.x)"
        )
    if not gpu_available():
        raise RuntimeError(f"{fn_name}: no CUDA device found.")


# ── VRAM auto-sizing ─────────────────────────────────────────────────────────

def _auto_batch_size(frame_shape: tuple, vram_fraction: float = 0.40) -> int:
    """Largest batch that fits in *vram_fraction* of free VRAM."""
    if not _CUPY_AVAILABLE:
        return 256
    try:
        free, _ = cp.cuda.runtime.memGetInfo()
        usable = int(free * vram_fraction)
        # 3 complex64 buffers (src, products, cross_corr) + 1 float32 output
        bytes_per_frame = int(np.prod(frame_shape)) * (8 * 3 + 4)
        return max(32, min(usable // bytes_per_frame, 2000))
    except Exception:
        return 256


# ── Frequency-grid cache ─────────────────────────────────────────────────────

class _FreqGrids:
    __slots__ = ("Nr_grid", "Nc_grid")

    def __init__(self, H: int, W: int):
        Nr = cp.asarray(
            ifftshift(np.arange(-np.fix(H / 2.0), np.ceil(H / 2.0))).astype(np.float32)
        )
        Nc = cp.asarray(
            ifftshift(np.arange(-np.fix(W / 2.0), np.ceil(W / 2.0))).astype(np.float32)
        )
        self.Nc_grid, self.Nr_grid = cp.meshgrid(Nc, Nr)   # both (H, W)


_GRID_CACHE: dict[tuple, _FreqGrids] = {}


def _grids(H: int, W: int) -> _FreqGrids:
    key = (H, W)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = _FreqGrids(H, W)
    return _GRID_CACHE[key]


# ── Batched upsampled DFT ────────────────────────────────────────────────────

def _upsampled_dft_batch(
    data: "cp.ndarray",          # (N, H, W) complex64
    region_size: int,
    upsample_factor: float,
    axis_offsets: "cp.ndarray",  # (N, 2) float64
) -> "cp.ndarray":               # (N, R, R) complex64
    """
    Batched upsampled DFT via two cuBLAS batched GEMMs.
    Equivalent to _upsampled_dft(data[n], …) for each n, vectorised.
    """
    N, H, W = data.shape
    R  = int(region_size)
    uf = upsample_factor
    r_idx = cp.arange(R, dtype=cp.float64)  # (R,)

    # Row kernel  (N, R, H)
    h_ax = cp.asarray(
        ifftshift(np.arange(H)).astype(np.float64) - float(np.floor(H // 2))
    )
    r_off = r_idx[None, :] - axis_offsets[:, 0:1]                   # (N, R)
    row_k = cp.exp(
        (-1j * 2.0 * np.pi / (H * uf)) * r_off[:, :, None] * h_ax[None, None, :]
    ).astype(cp.complex64)                                           # (N, R, H)

    # Col kernel  (N, W, R)
    w_ax = cp.asarray(
        ifftshift(np.arange(W)).astype(np.float64) - float(np.floor(W // 2))
    )
    c_off = r_idx[None, :] - axis_offsets[:, 1:2]                   # (N, R)
    col_k = cp.exp(
        (-1j * 2.0 * np.pi / (W * uf)) * w_ax[None, :, None] * c_off[:, None, :]
    ).astype(cp.complex64)                                           # (N, W, R)

    # (N, R, H) @ (N, H, W) → (N, R, W)  →  @ (N, W, R) → (N, R, R)
    return cp.matmul(cp.matmul(row_k, data), col_k)


# ── Batched phase-correlation registration ───────────────────────────────────

def _batch_register(
    frames_gpu: "cp.ndarray",          # (N, H, W) float32
    template_freq: "cp.ndarray",       # (H, W) or (N, H, W) complex64
    upsample_factor: int,
    max_shifts: tuple,
    shifts_lb: Optional[np.ndarray] = None,
    shifts_ub: Optional[np.ndarray] = None,
) -> tuple["cp.ndarray", "cp.ndarray", "cp.ndarray"]:
    """
    Batched phase-correlation.

    Returns
    -------
    shifts      : (N, 2) float64 on GPU  — (row, col) shift per item
    src_freqs   : (N, H, W) complex64 on GPU
    diffphases  : (N,) float64 on GPU
    """
    N, H, W = frames_gpu.shape

    src_freqs = cp.fft.fft2(frames_gpu.astype(cp.float32)).astype(cp.complex64)

    if template_freq.ndim == 2:
        products = src_freqs * cp.conj(template_freq)[None, :, :]
    else:
        products = src_freqs * cp.conj(template_freq)               # per-patch

    cross = cp.fft.ifft2(products).astype(cp.complex64)
    mag   = cp.abs(cross)

    # Mask out-of-range shifts
    if shifts_lb is not None and shifts_ub is not None:
        lb, ub = np.asarray(shifts_lb), np.asarray(shifts_ub)
        if lb[0] < 0 and ub[0] >= 0:
            mag[:, ub[0]:lb[0], :] = 0.0
        else:
            if lb[0] > 0: mag[:, :lb[0], :] = 0.0
            mag[:, ub[0]:, :] = 0.0
        if lb[1] < 0 and ub[1] >= 0:
            mag[:, :, ub[1]:lb[1]] = 0.0
        else:
            if lb[1] > 0: mag[:, :, :lb[1]] = 0.0
            mag[:, :, ub[1]:] = 0.0
    else:
        mag[:, max_shifts[0]:-max_shifts[0], :] = 0.0
        mag[:, :, max_shifts[1]:-max_shifts[1]] = 0.0

    flat   = cp.argmax(mag.reshape(N, -1), axis=1)
    row_pk = flat // W
    col_pk = flat %  W
    shifts = cp.stack([row_pk, col_pk], axis=1).astype(cp.float64)

    mid = cp.array([H // 2, W // 2], dtype=cp.float64)
    shifts[:, 0] = cp.where(shifts[:, 0] > mid[0], shifts[:, 0] - H, shifts[:, 0])
    shifts[:, 1] = cp.where(shifts[:, 1] > mid[1], shifts[:, 1] - W, shifts[:, 1])

    if upsample_factor == 1:
        CCmax     = cross[cp.arange(N), row_pk, col_pk]
        diffphase = cp.arctan2(CCmax.imag.astype(cp.float64), CCmax.real.astype(cp.float64))
        return shifts, src_freqs, diffphase

    # Sub-pixel via batched upsampled DFT
    uf           = float(upsample_factor)
    shifts_c     = cp.round(shifts * uf) / uf
    region_size  = int(np.ceil(uf * 1.5))
    dftshift     = float(np.fix(region_size / 2.0))
    norm         = float(H * W) * uf ** 2
    offsets      = dftshift - shifts_c * uf

    cc_ups = _upsampled_dft_batch(products.conj(), region_size, uf, offsets).conj() / norm

    R      = region_size
    fine_f = cp.argmax(cp.abs(cc_ups).reshape(N, -1), axis=1)
    fine_r = (fine_f // R).astype(cp.float64) - dftshift
    fine_c = (fine_f %  R).astype(cp.float64) - dftshift
    shifts = shifts_c + cp.stack([fine_r, fine_c], axis=1) / uf

    ri    = cp.clip((fine_f // R).astype(cp.int64), 0, R - 1)
    ci    = cp.clip((fine_f %  R).astype(cp.int64), 0, R - 1)
    CCmax = cc_ups[cp.arange(N), ri, ci]
    diffphase = cp.arctan2(CCmax.imag.astype(cp.float64), CCmax.real.astype(cp.float64))

    return shifts, src_freqs, diffphase


# ── Apply shifts in Fourier domain ───────────────────────────────────────────

def _batch_apply_dft(
    src_freqs: "cp.ndarray",    # (N, H, W) complex64
    shifts: "cp.ndarray",       # (N, 2) float64
    diffphases: "cp.ndarray",   # (N,) float64
    border_nan: Union[bool, str],
) -> "cp.ndarray":              # (N, H, W) float32
    N, H, W = src_freqs.shape
    g  = _grids(H, W)
    sh = shifts[:, 0, None, None]
    sc = shifts[:, 1, None, None]
    Nr = g.Nr_grid[None]
    Nc = g.Nc_grid[None]

    ramp = cp.exp(
        (1j * 2.0 * np.pi) * (-sh * Nr / H - sc * Nc / W)
    ).astype(cp.complex64)
    dp   = cp.exp(1j * diffphases.astype(cp.float32))[:, None, None]

    imgs = cp.real(cp.fft.ifft2(src_freqs * ramp * dp)).astype(cp.float32)

    if border_nan is not False:
        shifts_np = cp.asnumpy(shifts)
        for n in range(N):
            sh_n, sc_n = float(shifts_np[n, 0]), float(shifts_np[n, 1])
            mxh = int(np.ceil( max(0.0,  sh_n))); mnh = int(np.floor(min(0.0, sh_n)))
            mxw = int(np.ceil( max(0.0,  sc_n))); mnw = int(np.floor(min(0.0, sc_n)))
            if border_nan is True:
                if mxh > 0: imgs[n, :mxh, :]  = cp.nan
                if mnh < 0: imgs[n, mnh:, :]  = cp.nan
                if mxw > 0: imgs[n, :, :mxw]  = cp.nan
                if mnw < 0: imgs[n, :, mnw:]   = cp.nan
            elif border_nan == 'min':
                fill = float(cp.nanmin(imgs[n]))
                if mxh > 0: imgs[n, :mxh, :]  = fill
                if mnh < 0: imgs[n, mnh:, :]  = fill
                if mxw > 0: imgs[n, :, :mxw]  = fill
                if mnw < 0: imgs[n, :, mnw:]   = fill
            elif border_nan == 'copy':
                if mxh > 0: imgs[n, :mxh, :]  = imgs[n, mxh:mxh+1, :]
                if mnh < 0: imgs[n, mnh:, :]   = imgs[n, mnh-1:mnh, :]
                if mxw > 0: imgs[n, :, :mxw]  = imgs[n, :, mxw:mxw+1]
                if mnw < 0: imgs[n, :, mnw:]   = imgs[n, :, mnw-1:mnw]
    return imgs


# ── Apply shifts via GPU warp (opencv path) ───────────────────────────────────

def _batch_warp(
    frames_gpu: "cp.ndarray",   # (N, H, W) float32
    shifts: "cp.ndarray",       # (N, 2) float64
    border_nan: Union[bool, str],
) -> "cp.ndarray":
    N = frames_gpu.shape[0]
    out = cp.empty_like(frames_gpu)
    shifts_np = cp.asnumpy(shifts)
    for n in range(N):
        sh, sc = float(shifts_np[n, 0]), float(shifts_np[n, 1])
        mode = 'nearest' if border_nan == 'copy' else 'constant'
        out[n] = cpnd.shift(frames_gpu[n], shift=(sh, sc), mode=mode, cval=0.0, order=3)
        if border_nan is True:
            mxh = int(np.ceil( max(0.0, sh))); mnh = int(np.floor(min(0.0, sh)))
            mxw = int(np.ceil( max(0.0, sc))); mnw = int(np.floor(min(0.0, sc)))
            if mxh > 0: out[n, :mxh, :]  = cp.nan
            if mnh < 0: out[n, mnh:, :]  = cp.nan
            if mxw > 0: out[n, :, :mxw]  = cp.nan
            if mnw < 0: out[n, :, mnw:]   = cp.nan
        elif border_nan == 'min':
            fill = float(cp.nanmin(out[n]))
            mxh = int(np.ceil( max(0.0, sh))); mnh = int(np.floor(min(0.0, sh)))
            mxw = int(np.ceil( max(0.0, sc))); mnw = int(np.floor(min(0.0, sc)))
            if mxh > 0: out[n, :mxh, :]  = fill
            if mnh < 0: out[n, mnh:, :]  = fill
            if mxw > 0: out[n, :, :mxw]  = fill
            if mnw < 0: out[n, :, mnw:]   = fill
    return out


# ── High-pass filter on GPU ───────────────────────────────────────────────────

def _highpass_batch(frames_gpu: "cp.ndarray", gSig_filt: tuple) -> "cp.ndarray":
    import cv2
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker   = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T).astype(np.float32)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz]  = 0.0
    ker_g = cp.asarray(ker2D)
    out   = cp.empty_like(frames_gpu)
    for n in range(frames_gpu.shape[0]):
        out[n] = cpnd.convolve(frames_gpu[n], ker_g, mode='mirror')
    return out


# ── PW-Rigid: GPU displacement-field warp ─────────────────────────────────────

def _pwrigid_warp_gpu(
    frame_gpu: "cp.ndarray",        # (H, W) float32
    shift_row: np.ndarray,          # (gr, gc) — row shifts on coarse grid
    shift_col: np.ndarray,          # (gr, gc) — col shifts on coarse grid
    border_nan: Union[bool, str],
    shifts_interpolate: bool,
    patch_centers_orig: tuple,
    newstrides: tuple,
    newoverlaps: tuple,
) -> "cp.ndarray":                  # (H, W) float32
    """Upsample coarse shift field and apply via map_coordinates on GPU."""
    import cv2
    H, W = frame_gpu.shape

    if shifts_interpolate:
        from caiman.motion_correction import get_patch_centers, interpolate_shifts
        pc_new = get_patch_centers((H, W), newoverlaps, newstrides)
        sy = interpolate_shifts(shift_row, patch_centers_orig, pc_new).astype(np.float32)
        sx = interpolate_shifts(shift_col, patch_centers_orig, pc_new).astype(np.float32)
    else:
        sy = cv2.resize(shift_row.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        sx = cv2.resize(shift_col.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)

    rows = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    cols = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))
    src_r = cp.asarray(rows + sy)
    src_c = cp.asarray(cols + sx)

    mode   = 'nearest' if border_nan == 'copy' else 'constant'
    result = cpnd.map_coordinates(frame_gpu, [src_r, src_c], order=3, mode=mode, cval=0.0)

    if border_nan is True:
        ms = int(np.ceil(max(np.nanmax(np.abs(sy)), np.nanmax(np.abs(sx)))))
        if ms > 0:
            result[:ms, :]  = cp.nan; result[-ms:, :] = cp.nan
            result[:, :ms]  = cp.nan; result[:, -ms:] = cp.nan
    elif border_nan == 'min':
        fill = float(cp.nanmin(result))
        ms = int(np.ceil(max(np.nanmax(np.abs(sy)), np.nanmax(np.abs(sx)))))
        if ms > 0:
            result[:ms, :]  = fill; result[-ms:, :] = fill
            result[:, :ms]  = fill; result[:, -ms:] = fill
    return result


# ── Frame loading ─────────────────────────────────────────────────────────────

def _load_frames(source, idxs: np.ndarray, var_name_hdf5: str, is3D: bool) -> np.ndarray:
    """
    Load a batch of frames by index from any source.

    caiman.load()'s tiff branch has two separate code paths depending on
    whether subindices is a *list* or not:
      - list  → per-dimension index spec: [time_idx, row_idx, col_idx]
      - other → used directly as fancy/slice index on the time axis

    We must therefore pass a numpy array, never a plain Python list.
    """
    from caiman.shared_memory_utils import ShmHandle, attach_shared_frames
    import caiman
    if isinstance(source, ShmHandle):
        return np.ascontiguousarray(attach_shared_frames(source, list(idxs)))
    if isinstance(source, np.ndarray):
        return source[idxs]
    # Pass as numpy array (not list) so caiman.load uses single-axis fancy indexing
    idx_arr = np.asarray(idxs)
    return np.asarray(caiman.load(source, subindices=idx_arr, var_name_hdf5=var_name_hdf5))


# ── Main entry point ──────────────────────────────────────────────────────────

def motion_correction_piecewise_gpu(
    fname,
    idxs_list: list,
    template: np.ndarray,
    shape_mov: tuple,
    fname_tot: Optional[str],
    max_shifts: tuple,
    strides: Optional[tuple],
    overlaps: Optional[tuple],
    max_deviation_rigid: int,
    upsample_factor_grid: int,
    add_to_movie: float,
    nonneg_movie: bool,
    gSig_filt,
    border_nan: Union[bool, str],
    is3D: bool,
    indices: tuple,
    shifts_opencv: bool,
    shifts_interpolate: bool,
    upsample_factor_fft: int = 10,
    gpu_batch_size: Optional[int] = None,
    var_name_hdf5: str = 'mov',
) -> list:
    """
    GPU-batched motion correction replacing the multiprocessing dispatch.

    Processes every frame in ``idxs_list`` on the GPU (no subprocesses),
    writes corrected frames to ``fname_tot``, and returns results in the
    **exact same format** as the CPU ``tile_and_correct_wrapper`` so that all
    downstream shift-parsing is unchanged.

    Returns
    -------
    list of (shift_info, idxs, mean_template) — one entry per original split.
    """
    _require_gpu("motion_correction_piecewise_gpu")
    import caiman.mmapping

    rigid_mode = (strides is None or max_deviation_rigid == 0)
    H, W = int(template.shape[0]), int(template.shape[1])

    # ── Template FFT ──────────────────────────────────────────────────────
    tmpl_np = template.astype(np.float32)
    if gSig_filt is not None:
        from caiman.motion_correction import high_pass_filter_space
        tmpl_np = high_pass_filter_space(tmpl_np, gSig_filt)
    tmpl_gpu  = cp.asarray(tmpl_np + float(add_to_movie))
    tmpl_freq = cp.fft.fft2(tmpl_gpu).astype(cp.complex64)
    _grids(H, W)  # warm cache

    # ── PW-Rigid: pre-compute template patch FFTs & grid geometry ─────────
    patch_corners      = None
    patch_shape        = None
    tmpl_patch_freq    = None
    patch_centers_orig = None
    newstrides_eff     = None
    newoverlaps_eff    = None

    if not rigid_mode:
        from caiman.motion_correction import sliding_window, get_patch_centers
        patches_list = []
        corners_list = []
        for xind, yind, xstart, ystart, patch in sliding_window(tmpl_np, overlaps, strides):
            patches_list.append(patch)
            corners_list.append((xstart, ystart))
        patches_np      = np.stack(patches_list).astype(np.float32)
        patch_shape     = patches_np.shape[1:]
        patch_corners   = np.array(corners_list)                     # (P, 2)
        tmpl_patch_freq = cp.fft.fft2(cp.asarray(patches_np)).astype(cp.complex64)
        patch_centers_orig = get_patch_centers((H, W), overlaps, strides)
        newstrides_eff  = tuple(
            np.round(np.divide(strides, upsample_factor_grid)).astype(int)
        )
        newoverlaps_eff = overlaps

    # ── Output mmap ───────────────────────────────────────────────────────
    if fname_tot is not None:
        out_mmap = np.memmap(
            fname_tot, mode='r+', dtype=np.float32,
            shape=caiman.mmapping.prepare_shape(shape_mov), order='F'
        )
    else:
        out_mmap = None

    # ── Flatten index list ────────────────────────────────────────────────
    all_idxs = np.concatenate([np.asarray(ix) for ix in idxs_list])
    T_total  = len(all_idxs)

    if gpu_batch_size is None:
        gpu_batch_size = _auto_batch_size((H, W))

    logger.info(
        f"GPU MC: {'rigid' if rigid_mode else 'pw-rigid'}, "
        f"frames={T_total}, batch={gpu_batch_size}, shape=({H},{W})"
    )

    # Per-frame accumulated results
    all_shifts      = np.zeros((T_total, 2), dtype=np.float64)
    pw_frame_shifts = []   # (P, 2) per frame — pw-rigid only

    # Per-chunk running mean for template update.
    # We accumulate (sum, count) on the fly so no corrected frames need to
    # be retained in RAM after being written to the mmap — this keeps peak
    # RAM at O(batch) instead of O(T).
    chunk_sum   = [np.zeros((H, W), dtype=np.float64) for _ in idxs_list]
    chunk_count = [0] * len(idxs_list)

    # Build a flat frame→chunk index map so we can credit frames to their
    # chunk without an O(chunks) search per frame.
    frame_to_chunk = {}
    for ci, chunk_idxs in enumerate(idxs_list):
        for fi in chunk_idxs:
            frame_to_chunk[int(fi)] = ci

    # ── Process batches ───────────────────────────────────────────────────
    bias = np.float32(add_to_movie) if nonneg_movie else np.float32(0)

    for bs in range(0, T_total, gpu_batch_size):
        be   = min(bs + gpu_batch_size, T_total)
        bidx = all_idxs[bs:be]
        B    = len(bidx)

        frames_np  = _load_frames(fname, bidx, var_name_hdf5, is3D)
        frames_np  = frames_np[(slice(None),) + indices].astype(np.float32)
        frames_gpu = cp.asarray(frames_np + float(add_to_movie))     # (B, H, W)

        frames_fft = _highpass_batch(frames_gpu, gSig_filt) if gSig_filt is not None else frames_gpu

        # ── Rigid shifts ──────────────────────────────────────────────────
        shifts_g, src_freqs_g, dphase_g = _batch_register(
            frames_fft, tmpl_freq, upsample_factor_fft, max_shifts
        )
        all_shifts[bs:be] = cp.asnumpy(shifts_g)
        neg_sh = -shifts_g

        if rigid_mode:
            # Apply shifts
            if gSig_filt is not None:
                sf = cp.fft.fft2(frames_gpu.astype(cp.complex64))
                corrected = _batch_apply_dft(sf, neg_sh, dphase_g, border_nan)
            elif shifts_opencv:
                corrected = _batch_warp(frames_gpu, neg_sh, border_nan)
            else:
                corrected = _batch_apply_dft(src_freqs_g, neg_sh, dphase_g, border_nan)

            corrected_np = cp.asnumpy(corrected)   # (B, H, W) — only buffer we keep

            # Write to mmap and accumulate chunk means; then discard frame data
            if out_mmap is not None:
                for li, gfi in enumerate(bidx):
                    out_mmap[:, gfi] = corrected_np[li].reshape(-1, order='F') + bias
            for li, gfi in enumerate(bidx):
                ci = frame_to_chunk.get(int(gfi))
                if ci is not None:
                    chunk_sum[ci]   += corrected_np[li].astype(np.float64)
                    chunk_count[ci] += 1
            del corrected_np  # free immediately — not stored beyond this batch

        else:
            # ── PW-Rigid: per-frame, batch patch FFTs ──────────────────────
            ph, pw   = patch_shape
            P        = len(patch_corners)
            rigid_np = cp.asnumpy(shifts_g)                          # (B, 2)

            for bi in range(B):
                frame_src = frames_gpu[bi]                           # (H, W)

                frame_filt_bi = frames_fft[bi]
                patches_bi = cp.stack([
                    frame_filt_bi[r:r+ph, c:c+pw]
                    for r, c in patch_corners
                ])                                                   # (P, ph, pw)

                lb = np.ceil( rigid_np[bi] - max_deviation_rigid).astype(int)
                ub = np.floor(rigid_np[bi] + max_deviation_rigid).astype(int)

                pshifts_g, _, pdiffs_g = _batch_register(
                    patches_bi, tmpl_patch_freq,
                    upsample_factor_fft, max_shifts,
                    shifts_lb=lb, shifts_ub=ub
                )
                pshifts_np = cp.asnumpy(pshifts_g)                  # (P, 2)
                pw_frame_shifts.append(pshifts_np)

                rows_u = sorted(set(int(r) for r, c in patch_corners))
                cols_u = sorted(set(int(c) for r, c in patch_corners))
                gr, gc = len(rows_u), len(cols_u)
                shift_row = pshifts_np[:, 0].reshape(gr, gc)
                shift_col = pshifts_np[:, 1].reshape(gr, gc)

                apply_frame = frame_src if gSig_filt is None else cp.asarray(
                    frames_np[bi] + float(add_to_movie)
                )
                corrected_f  = _pwrigid_warp_gpu(
                    apply_frame, shift_row, shift_col,
                    border_nan, shifts_interpolate,
                    patch_centers_orig, newstrides_eff, newoverlaps_eff
                )
                corrected_np = cp.asnumpy(corrected_f)
                gfi = int(bidx[bi])
                if out_mmap is not None:
                    out_mmap[:, gfi] = corrected_np.reshape(-1, order='F') + bias
                ci = frame_to_chunk.get(gfi)
                if ci is not None:
                    chunk_sum[ci]   += corrected_np.astype(np.float64)
                    chunk_count[ci] += 1
                del corrected_np  # free immediately

        logger.debug(f"  GPU batch {bs}–{be}/{T_total}")

    # Flush
    if out_mmap is not None:
        out_mmap.flush()
        del out_mmap

    # ── Assemble results ──────────────────────────────────────────────────
    results   = []
    frame_cur = 0
    for ci, chunk_idxs in enumerate(idxs_list):
        n  = len(chunk_idxs)
        ix = np.asarray(chunk_idxs)

        if rigid_mode:
            shift_info = [
                [(-float(all_shifts[frame_cur+i, 0]),
                  -float(all_shifts[frame_cur+i, 1])), None, None]
                for i in range(n)
            ]
        else:
            shift_info = []
            for i in range(n):
                ps = pw_frame_shifts[frame_cur + i]
                total_sh = [(-float(ps[p, 0]), -float(ps[p, 1])) for p in range(len(patch_corners))]
                shift_info.append([total_sh, None, None])

        # Mean corrected frame from running sum (no frame retention needed)
        cnt = chunk_count[ci]
        if cnt > 0:
            mean_tmpl = (chunk_sum[ci] / cnt).astype(np.float32)
        else:
            mean_tmpl = np.zeros((H, W), dtype=np.float32)
        nan_mask = np.isnan(mean_tmpl)
        if nan_mask.any():
            finite_min = float(np.nanmin(mean_tmpl)) if not np.all(nan_mask) else 0.0
            mean_tmpl[nan_mask] = finite_min

        results.append((shift_info, list(ix), mean_tmpl))
        frame_cur += n

    return results
