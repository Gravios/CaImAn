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
        tile_pixels = max(1, int((vram_bytes - C_bytes) / (T * 4)))
        tile_pixels = min(tile_pixels, d)

    n_tiles = int(np.ceil(d / tile_pixels))
    logger.info(f"gpu_spatial.precompute_YC_gpu: {d}×{K} in {n_tiles} tiles of {tile_pixels}px")

    C_gpu = cp.asarray(np.ascontiguousarray(C, dtype=np.float32))
    YC    = np.empty((d, K), dtype=np.float32)

    for i in range(n_tiles):
        s = i * tile_pixels
        e = min(s + tile_pixels, d)
        tile_cpu = np.ascontiguousarray(Y[s:e, :], dtype=np.float32)
        tile_gpu = cp.asarray(tile_cpu);  del tile_cpu
        YC[s:e]  = cp.asnumpy(tile_gpu @ C_gpu.T);  del tile_gpu

    del C_gpu
    return YC


def _precompute_gram_lasso(Cf):
    """Precompute gram matrices for the LassoLars path.

    Replicates: make_pipeline(StandardScaler(with_mean=False),
                              LassoLars(fit_intercept=True))

    StandardScaler scales each feature (row of Cf) by std(ddof=0).
    LassoLars fit_intercept=True centers both X and y.

    Returns
    -------
    Cf_sc_c   : (K, T) float64  — scaled and column-centered C
    Cf_scale  : (K,)  float32  — per-row std used for scaling
    Cf_col_mean : (K,) float64  — per-row mean of Cf_sc (for centering Y rows)
    CC_gram   : (K, K) float64  — Cf_sc_c @ Cf_sc_c.T  (raw, no /T)
    """
    # Step 1: scale each component trace by its std (matching StandardScaler)
    Cf_scale = np.std(Cf, axis=1).astype(np.float32)        # (K,)
    Cf_scale  = np.where(Cf_scale < 1e-10, 1.0, Cf_scale)
    Cf_sc     = (Cf / Cf_scale[:, None]).astype(np.float64) # (K, T), each row std≈1

    # Step 2: center each row (LassoLars fit_intercept centers X)
    Cf_col_mean = Cf_sc.mean(axis=1)                        # (K,)
    Cf_sc_c     = Cf_sc - Cf_col_mean[:, None]              # (K, T) centered

    # Step 3: precompute Gram
    CC_gram = Cf_sc_c @ Cf_sc_c.T                           # (K, K)

    return Cf_sc_c, Cf_scale, Cf_col_mean, CC_gram


def _precompute_gram_nnls(Cf):
    """Precompute gram matrices for the NNLS_L0 path (no scaling needed).

    Returns
    -------
    CC   : (K, K) float64  — Cf @ Cf.T
    L    : (K, K) float64  — lower Cholesky of CC + eps*I
    """
    CC = (Cf.astype(np.float64)) @ (Cf.astype(np.float64)).T
    L  = np.linalg.cholesky(CC + 1e-8 * np.eye(CC.shape[0]))
    return CC, L


def update_spatial_gpu(Y, Cf, f, ind2_, sn, nr, d, T, nb,
                       method_ls='lasso_lars', cct=None,
                       n_pixels_per_process=128) -> tuple:
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

    Returns
    -------
    data, rows, cols : lists for building sparse A_
    """
    from sklearn.linear_model._least_angle import lars_path_gram

    K_total = Cf.shape[0]

    # ── Gram precomputation ───────────────────────────────────────────────────
    if method_ls == 'lasso_lars':
        Cf_sc_c, Cf_scale, Cf_col_mean, CC_gram = _precompute_gram_lasso(Cf)

        # YC_centered = Y @ Cf_sc_c.T
        # But Y rows also need centering: y_c = y - mean(y)
        # And: y_c @ Cf_sc_c.T = y @ Cf_sc_c.T - mean(y) * Cf_sc_c.sum(axis=1)
        # = YC_raw - Y_row_means[:, None] * Cf_sc_c_sum
        # We'll compute YC_raw on GPU, then subtract the mean correction on CPU.
        logger.info("gpu_spatial: computing Y @ Cf_sc_c.T on GPU")
        YC_raw    = precompute_YC_gpu(Y, Cf_sc_c.astype(np.float32))    # (d, K)
        Cf_sc_c_sum = Cf_sc_c.sum(axis=1).astype(np.float32)            # (K,)

        # cct in original (unscaled) space for lambda computation
        if cct is None:
            cct = np.sum(Cf[:nr]**2, axis=1).astype(np.float32)

    else:  # nnls_L0
        logger.info("gpu_spatial: computing Y @ Cf.T on GPU (nnls_L0 path)")
        Cf_sc_c   = None
        CC_nnls, L_nnls = _precompute_gram_nnls(Cf)
        YC_raw    = precompute_YC_gpu(Y, Cf.astype(np.float32))          # (d, K)
        if cct is None:
            cct = np.sum(Cf[:nr]**2, axis=1).astype(np.float32)

    # ── Per-pixel gram solve ──────────────────────────────────────────────────
    logger.info("gpu_spatial: running per-pixel gram solves")

    data: list = []
    rows: list = []
    cols: list = []

    # Precompute Y row means (needed for lasso y-centering correction)
    if method_ls == 'lasso_lars':
        Y_row_means = np.asarray(Y).mean(axis=1).astype(np.float32)    # (d,)

    for px in range(d):
        local_idx = ind2_[px]
        if len(local_idx) == 0 or sn[px] <= 0:
            continue

        if method_ls == 'lasso_lars':
            # Xy_i = y_c @ Cf_sc_c_local.T
            #      = YC_raw[px, local] - Y_row_means[px] * Cf_sc_c_sum[local]
            Xy_i = (YC_raw[px, local_idx].astype(np.float64)
                    - float(Y_row_means[px]) * Cf_sc_c_sum[local_idx].astype(np.float64))
            Gram_i = CC_gram[np.ix_(local_idx, local_idx)]

            # lambda: same formula as regression_ipyparallel
            local_nr = local_idx[local_idx < nr]
            lam = (0.0 if len(local_nr) == 0 else
                   0.5 * float(sn[px]) * float(np.sqrt(np.max(cct[local_nr]))) / T)

            try:
                _, _, coefs = lars_path_gram(
                    Xy=Xy_i, Gram=Gram_i, n_samples=T,
                    alpha_min=lam, method='lasso',
                    positive=True, copy_Gram=False, copy_X=False,
                )
                a = coefs[:, -1].astype(np.float32)
            except Exception:
                # Fallback: nnls on local gram
                try:
                    L_i = np.linalg.cholesky(Gram_i + 1e-8 * np.eye(len(local_idx)))
                    rhs = scipy.linalg.solve_triangular(L_i, Xy_i, lower=True)
                    a, _ = scipy.optimize.nnls(L_i.T, rhs)
                    a = a.astype(np.float32)
                except Exception:
                    continue

        else:  # nnls_L0
            YC_i    = YC_raw[px, local_idx].astype(np.float64)
            CC_i    = CC_nnls[np.ix_(local_idx, local_idx)]
            try:
                L_i     = np.linalg.cholesky(CC_i + 1e-8 * np.eye(len(local_idx)))
                rhs     = scipy.linalg.solve_triangular(L_i, YC_i, lower=True)
                a_sc, _ = scipy.optimize.nnls(L_i.T, rhs)
            except Exception:
                continue

            # L0 pruning (noise in T dimension)
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
                    best_i = min(eliminate, key=lambda x: x[1])[0]
                    a_sc[best_i] = 0.0

            # Unscale: Cf_sc = Cf / C_scale, so a_original = a_sc / C_scale[local]
            # For nnls_L0 we used unscaled Cf, so a is already in original space
            a = a_sc.astype(np.float32)

        nz = np.where(np.maximum(a, 0) > 0)[0]
        data.extend(a[nz].tolist())
        rows.extend([px] * len(nz))
        cols.extend(local_idx[nz].tolist())

    return data, rows, cols
