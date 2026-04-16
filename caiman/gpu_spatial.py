"""GPU-accelerated spatial update for CNMF.

Core idea
---------
The standard per-pixel NNLS in spatial.py solves for each pixel i:

    a_i = argmin ||Y_i - C_local.T @ a||²   s.t. a >= 0

where Y_i is a T-vector and C_local is (K_local × T).
The dominant cost is the O(K_local × T) system, solved 262,144 times.

If we precompute:
    YC = Y @ Cf_sc.T            (d × K) — via GPU tiled matmul
    CC = Cf_sc_c @ Cf_sc_c.T   (K × K) — CPU, tiny

then each per-pixel solve reduces to a (K_local × K_local) gram system.

Scaling conventions (must match regression_ipyparallel exactly)
--------------------------------------------------------------
LassoLars path:
  - StandardScaler(with_mean=False) scales each row of C by its std(ddof=0)
  - LassoLars(fit_intercept=True) then centers both X and y internally
  - So: Cf_sc  = Cf / std(Cf, axis=1)
          Cf_sc_c = Cf_sc - Cf_sc.mean(axis=1)   (column-wise, i.e. per component)
          Y_c     = Y    - Y.mean(axis=1, keepdims=True)
          Gram    = Cf_sc_c @ Cf_sc_c.T   (no /T — sklearn passes raw, uses n_samples)
          Xy_i    = Y_c[i] @ Cf_sc_c.T
  - Returned coef_ is in StandardScaler-scaled space (not original).

NNLS_L0 path:
  - Uses C directly (no scaling), Cholesky factorisation of C @ C.T.
"""

import logging
import numpy as np
import scipy.optimize
import scipy.linalg

logger = logging.getLogger("caiman")

def _vram_budget_gb() -> float:
    """Return a safe VRAM budget: 85% of free device memory, floored at 1 GB."""
    try:
        import cupy as cp
        free, total = cp.cuda.Device(0).mem_info
        return max(1.0, free * 0.85 / 2**30)
    except Exception:
        return 12.0  # conservative fallback if CuPy unavailable



def _gpu_available() -> bool:
    try:
        import cupy as cp
        cp.array([0], dtype=np.float32)
        return True
    except Exception:
        return False


def precompute_YC_gpu(Y, C, tile_pixels: int | None = None) -> np.ndarray:
    """Compute YC = Y @ C.T on GPU in tiles.

    Parameters
    ----------
    Y : array-like (d, T) float32, may be mmap
    C : np.ndarray (K, T) float32

    Returns
    -------
    YC : np.ndarray (d, K) float32 on CPU RAM
    """
    import cupy as cp

    d, T = Y.shape[0], Y.shape[1]
    K    = C.shape[0]

    if tile_pixels is None:
        vram_bytes  = _vram_budget_gb() * 2**30
        C_bytes     = K * T * 4
        avail_bytes = vram_bytes - C_bytes
        tile_pixels = max(1, int(avail_bytes // (T * 4 + K * 4)))
        tile_pixels = min(tile_pixels, d)

    logger.info(
        f"precompute_YC_gpu: d={d} T={T} K={K} "
        f"tile={tile_pixels}px  passes={-(-d // tile_pixels)}"
    )

    C_gpu = cp.asarray(C, dtype=cp.float32)   # (K, T) on GPU
    YC    = np.empty((d, K), dtype=np.float32)

    for p0 in range(0, d, tile_pixels):
        p1     = min(p0 + tile_pixels, d)
        Y_tile = cp.asarray(np.asarray(Y[p0:p1]), dtype=cp.float32)  # (tile, T)
        YC[p0:p1] = cp.asnumpy(Y_tile @ C_gpu.T)                     # (tile, K)
        del Y_tile

    del C_gpu
    return YC


def _precompute_gram_lasso(Cf):
    """Precompute scaled gram matrix for lasso_lars path."""
    scale      = np.std(Cf, axis=1, ddof=0).astype(np.float32)
    scale      = np.where(scale == 0, 1.0, scale)
    Cf_sc      = (Cf / scale[:, None]).astype(np.float32)
    col_mean   = Cf_sc.mean(axis=1).astype(np.float32)
    Cf_sc_c    = (Cf_sc - col_mean[:, None]).astype(np.float32)
    CC_gram    = (Cf_sc_c @ Cf_sc_c.T).astype(np.float64)
    return Cf_sc_c, scale, col_mean, CC_gram


def _precompute_gram_nnls(Cf):
    """Precompute gram matrix for nnls_L0 path."""
    CC   = (Cf @ Cf.T).astype(np.float64)
    L    = np.linalg.cholesky(CC + 1e-8 * np.eye(CC.shape[0]))
    return CC, L


# ── Module-level shared state for parallel pixel workers ─────────────────────
# Populated before dispatching; workers read via closure (serial) or
# module global (fork). Cleared immediately after.
_PIXEL_STATE: dict = {}


def _solve_pixel_chunk(px_range: tuple) -> tuple:
    """Solve NNLS gram system for pixels [px0, px1).

    Reads shared precomputed arrays from module-level _PIXEL_STATE.
    Returns (data, rows, cols) lists for sparse A_ assembly.
    """
    from sklearn.linear_model._least_angle import lars_path_gram as _lpg

    px0, px1    = px_range
    s           = _PIXEL_STATE
    method_ls   = s['method_ls']
    ind2_       = s['ind2_']
    sn          = s['sn']
    nr          = s['nr']
    T           = s['T']
    YC_raw      = s['YC_raw']

    data: list = []
    rows: list = []
    cols: list = []

    if method_ls == 'lasso_lars':
        CC_gram     = s['CC_gram']
        cct         = s['cct']
        Y_row_means = s['Y_row_means']
        Cf_sc_c_sum = s['Cf_sc_c_sum']

        for px in range(px0, px1):
            local_idx = ind2_[px]
            if len(local_idx) == 0 or sn[px] <= 0:
                continue
            Xy_i   = (YC_raw[px, local_idx].astype(np.float64)
                      - float(Y_row_means[px]) * Cf_sc_c_sum[local_idx].astype(np.float64))
            Gram_i = CC_gram[np.ix_(local_idx, local_idx)]
            local_nr = local_idx[local_idx < nr]
            lam = (0.0 if len(local_nr) == 0 else
                   0.5 * float(sn[px]) * float(np.sqrt(np.max(cct[local_nr]))) / T)
            try:
                _, _, coefs = _lpg(Xy=Xy_i, Gram=Gram_i, n_samples=T,
                                   alpha_min=lam, method='lasso',
                                   positive=True, copy_Gram=False, copy_X=False)
                a = coefs[:, -1].astype(np.float32)
            except Exception:
                try:
                    L_i = np.linalg.cholesky(Gram_i + 1e-8 * np.eye(len(local_idx)))
                    rhs = scipy.linalg.solve_triangular(L_i, Xy_i, lower=True)
                    a, _ = scipy.optimize.nnls(L_i.T, rhs)
                    a = a.astype(np.float32)
                except Exception:
                    continue
            nz = np.where(np.maximum(a, 0) > 0)[0]
            data.extend(a[nz].tolist())
            rows.extend([px] * len(nz))
            cols.extend(local_idx[nz].tolist())

    else:  # nnls_L0
        CC_nnls = s['CC_nnls']

        for px in range(px0, px1):
            local_idx = ind2_[px]
            if len(local_idx) == 0 or sn[px] <= 0:
                continue
            YC_i = YC_raw[px, local_idx].astype(np.float64)
            CC_i = CC_nnls[np.ix_(local_idx, local_idx)]
            try:
                L_i     = np.linalg.cholesky(CC_i + 1e-8 * np.eye(len(local_idx)))
                rhs     = scipy.linalg.solve_triangular(L_i, YC_i, lower=True)
                a_sc, _ = scipy.optimize.nnls(L_i.T, rhs)
            except Exception:
                continue
            noise = float(sn[px])**2 * T
            RSS   = float(np.dot(YC_i - CC_i @ a_sc, YC_i - CC_i @ a_sc))
            if RSS <= noise:
                while True:
                    eliminate = []
                    nz_idx = np.where(a_sc[:-1] > 0)[0]
                    for i in nz_idx:
                        mask = a_sc > 0
                        mask[i] = False
                        CC_masked = CC_i * mask[:, None] * mask[None, :]
                        try:
                            L_m  = np.linalg.cholesky(CC_masked + 1e-8 * np.eye(len(local_idx)))
                            r_m  = scipy.linalg.solve_triangular(L_m, YC_i * mask, lower=True)
                            a_try, rss_try = scipy.optimize.nnls(L_m.T, r_m)
                        except Exception:
                            continue
                        if rss_try * rss_try < noise:
                            eliminate.append((i, rss_try))
                    if not eliminate:
                        break
                    a_sc[min(eliminate, key=lambda x: x[1])[0]] = 0.0
            a = a_sc.astype(np.float32)
            nz = np.where(np.maximum(a, 0) > 0)[0]
            data.extend(a[nz].tolist())
            rows.extend([px] * len(nz))
            cols.extend(local_idx[nz].tolist())

    return data, rows, cols


def update_spatial_gpu(Y, Cf, f, ind2_, sn, nr, d, T, nb,
                       method_ls='lasso_lars', cct=None,
                       n_pixels_per_process=128,
                       dview=None) -> tuple:
    """GPU-accelerated replacement for the regression_ipyparallel loop.

    Parameters
    ----------
    Y    : (d, T) movie, mmap or ndarray
    Cf   : (K_total, T) all temporal components including background (C vstack f)
    f    : (nb, T) background temporal (used only for lambda computation)
    ind2_: list[array] of length d — local component indices per pixel
    sn   : (d,) noise std
    nr   : int  number of neuron components
    d, T : int
    nb   : int  number of background components
    method_ls : 'lasso_lars' | 'nnls_L0'
    cct  : (nr,) diagonal of C @ C.T (for lambda; if None, computed from Cf)
    dview : multiprocessing.Pool or None — parallelises per-pixel gram solves

    Returns
    -------
    data, rows, cols : lists for building sparse A_
    """
    # ── Gram precomputation (GPU) ─────────────────────────────────────────────
    if method_ls == 'lasso_lars':
        Cf_sc_c, Cf_scale, Cf_col_mean, CC_gram = _precompute_gram_lasso(Cf)
        logger.info("gpu_spatial: computing Y @ Cf_sc_c.T on GPU")
        YC_raw      = precompute_YC_gpu(Y, Cf_sc_c.astype(np.float32))
        Cf_sc_c_sum = Cf_sc_c.sum(axis=1).astype(np.float32)
        if cct is None:
            cct = np.sum(Cf[:nr]**2, axis=1).astype(np.float32)
        Y_row_means = np.asarray(Y).mean(axis=1).astype(np.float32)
    else:
        logger.info("gpu_spatial: computing Y @ Cf.T on GPU (nnls_L0 path)")
        Cf_sc_c = None
        CC_nnls, L_nnls = _precompute_gram_nnls(Cf)
        YC_raw = precompute_YC_gpu(Y, Cf.astype(np.float32))
        if cct is None:
            cct = np.sum(Cf[:nr]**2, axis=1).astype(np.float32)

    # ── Per-pixel gram solve (parallel via dview, or serial fallback) ─────────
    logger.info("gpu_spatial: running per-pixel gram solves")

    global _PIXEL_STATE
    _PIXEL_STATE = {
        'method_ls': method_ls, 'ind2_': ind2_, 'sn': sn,
        'nr': nr, 'T': T, 'YC_raw': YC_raw, 'cct': cct,
    }
    if method_ls == 'lasso_lars':
        _PIXEL_STATE.update({'CC_gram': CC_gram, 'Y_row_means': Y_row_means,
                             'Cf_sc_c_sum': Cf_sc_c_sum})
    else:
        _PIXEL_STATE['CC_nnls'] = CC_nnls

    n_workers = getattr(dview, '_processes', 1) if dview is not None else 1
    chunk = max(1, d // n_workers)
    ranges = [(i, min(i + chunk, d)) for i in range(0, d, chunk)]

    if dview is not None and 'multiprocessing' in str(type(dview)):
        results = dview.map_async(_solve_pixel_chunk, ranges).get(4294967)
    else:
        results = list(map(_solve_pixel_chunk, ranges))

    _PIXEL_STATE.clear()

    data: list = []
    rows: list = []
    cols: list = []
    for d_chunk, r_chunk, c_chunk in results:
        data.extend(d_chunk)
        rows.extend(r_chunk)
        cols.extend(c_chunk)

    return data, rows, cols
