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
          Gram    = Cf_sc_c @ Cf_sc_c.T   (no /T -- sklearn passes raw, uses n_samples)
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
    try:
        import cupy as cp
        free, _ = cp.cuda.Device(0).mem_info
        return max(1.0, free * 0.85 / 2**30)
    except Exception:
        return 12.0


def _gpu_available() -> bool:
    try:
        import cupy as cp
        cp.array([0], dtype=np.float32)
        return True
    except Exception:
        return False


def precompute_YC_gpu(Y, C, tile_pixels=None):
    import cupy as cp
    d, T = Y.shape[0], Y.shape[1]
    K    = C.shape[0]
    if tile_pixels is None:
        vram_bytes  = _vram_budget_gb() * 2**30
        C_bytes     = K * T * 4
        avail_bytes = vram_bytes - C_bytes
        tile_pixels = max(1, min(int(avail_bytes // (T * 4 + K * 4)), d))
    logger.info(f"precompute_YC_gpu: d={d} T={T} K={K} tile={tile_pixels}px  passes={-(-d//tile_pixels)}")
    C_gpu = cp.asarray(C, dtype=cp.float32)
    YC    = np.empty((d, K), dtype=np.float32)
    for p0 in range(0, d, tile_pixels):
        p1 = min(p0 + tile_pixels, d)
        Y_tile = cp.asarray(np.asarray(Y[p0:p1]), dtype=cp.float32)
        YC[p0:p1] = cp.asnumpy(Y_tile @ C_gpu.T)
        del Y_tile
    del C_gpu
    return YC


def _precompute_gram_lasso(Cf):
    scale    = np.std(Cf, axis=1, ddof=0).astype(np.float32)
    scale    = np.where(scale == 0, 1.0, scale)
    Cf_sc    = (Cf / scale[:, None]).astype(np.float32)
    col_mean = Cf_sc.mean(axis=1).astype(np.float32)
    Cf_sc_c  = (Cf_sc - col_mean[:, None]).astype(np.float32)
    CC_gram  = (Cf_sc_c @ Cf_sc_c.T).astype(np.float64)
    return Cf_sc_c, scale, col_mean, CC_gram


def _precompute_gram_nnls(Cf):
    CC = (Cf @ Cf.T).astype(np.float64)
    L  = np.linalg.cholesky(CC + 1e-8 * np.eye(CC.shape[0]))
    return CC, L


def _solve_pixel_chunk(args):
    """Solve NNLS gram system for a contiguous pixel range.

    All precomputed arrays are passed explicitly so this works with
    spawn workers (no shared module state).
    """
    from sklearn.linear_model._least_angle import lars_path_gram as _lpg

    (px0, px1, method_ls, ind2_slice, sn_slice, nr, T,
     YC_slice, cct, extra) = args

    data: list = []
    rows: list = []
    cols: list = []

    if method_ls == 'lasso_lars':
        CC_gram, Y_row_means_slice, Cf_sc_c_sum = extra

        for i, px in enumerate(range(px0, px1)):
            local_idx = ind2_slice[i]
            if len(local_idx) == 0 or sn_slice[i] <= 0:
                continue
            Xy_i   = (YC_slice[i, local_idx].astype(np.float64)
                      - float(Y_row_means_slice[i]) * Cf_sc_c_sum[local_idx].astype(np.float64))
            Gram_i = CC_gram[np.ix_(local_idx, local_idx)]
            local_nr = local_idx[local_idx < nr]
            lam = (0.0 if len(local_nr) == 0 else
                   0.5 * float(sn_slice[i]) * float(np.sqrt(np.max(cct[local_nr]))) / T)
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
        CC_nnls = extra

        for i, px in enumerate(range(px0, px1)):
            local_idx = ind2_slice[i]
            if len(local_idx) == 0 or sn_slice[i] <= 0:
                continue
            YC_i = YC_slice[i, local_idx].astype(np.float64)
            CC_i = CC_nnls[np.ix_(local_idx, local_idx)]
            try:
                L_i     = np.linalg.cholesky(CC_i + 1e-8 * np.eye(len(local_idx)))
                rhs     = scipy.linalg.solve_triangular(L_i, YC_i, lower=True)
                a_sc, _ = scipy.optimize.nnls(L_i.T, rhs)
            except Exception:
                continue
            noise = float(sn_slice[i])**2 * T
            RSS   = float(np.dot(YC_i - CC_i @ a_sc, YC_i - CC_i @ a_sc))
            if RSS <= noise:
                while True:
                    eliminate = []
                    for j in np.where(a_sc[:-1] > 0)[0]:
                        mask = a_sc > 0
                        mask[j] = False
                        CC_m = CC_i * mask[:, None] * mask[None, :]
                        try:
                            L_m  = np.linalg.cholesky(CC_m + 1e-8 * np.eye(len(local_idx)))
                            r_m  = scipy.linalg.solve_triangular(L_m, YC_i * mask, lower=True)
                            a_t, rss_t = scipy.optimize.nnls(L_m.T, r_m)
                        except Exception:
                            continue
                        if rss_t * rss_t < noise:
                            eliminate.append((j, rss_t))
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
                       n_pixels_per_process=128, dview=None):
    """GPU-accelerated spatial update with parallel CPU gram solves.

    GPU computes Y @ Cf.T; per-pixel NNLS solves are distributed across
    workers via dview (or run serially if dview is None).
    """
    # ── Gram precomputation (GPU) ─────────────────────────────────────────────
    if method_ls == 'lasso_lars':
        Cf_sc_c, _, _, CC_gram = _precompute_gram_lasso(Cf)
        logger.info("gpu_spatial: computing Y @ Cf_sc_c.T on GPU")
        YC_raw      = precompute_YC_gpu(Y, Cf_sc_c.astype(np.float32))
        Cf_sc_c_sum = Cf_sc_c.sum(axis=1).astype(np.float32)
        if cct is None:
            cct = np.sum(Cf[:nr]**2, axis=1).astype(np.float32)
        Y_row_means = np.asarray(Y).mean(axis=1).astype(np.float32)
        extra_shared = (CC_gram, Y_row_means, Cf_sc_c_sum)
    else:
        logger.info("gpu_spatial: computing Y @ Cf.T on GPU (nnls_L0 path)")
        CC_nnls, _ = _precompute_gram_nnls(Cf)
        YC_raw = precompute_YC_gpu(Y, Cf.astype(np.float32))
        if cct is None:
            cct = np.sum(Cf[:nr]**2, axis=1).astype(np.float32)
        extra_shared = CC_nnls

    # ── Build work items — each contains its own data slice ──────────────────
    # Passing data as args (not module globals) is required for spawn workers.
    logger.info("gpu_spatial: running per-pixel gram solves")
    n_workers = getattr(dview, '_processes', 1) if dview is not None else 1
    chunk     = max(1, d // n_workers)
    work_items = []
    for p0 in range(0, d, chunk):
        p1 = min(p0 + chunk, d)
        work_items.append((
            p0, p1, method_ls,
            ind2_[p0:p1],           # list slice
            sn[p0:p1],              # (chunk,)
            nr, T,
            YC_raw[p0:p1],          # (chunk, K) — primary data per chunk
            cct,                    # (nr,) — shared, small
            extra_shared,           # gram + lasso helpers — shared, small
        ))

    if dview is not None and 'multiprocessing' in str(type(dview)):
        results = dview.map_async(_solve_pixel_chunk, work_items).get(4294967)
    else:
        results = list(map(_solve_pixel_chunk, work_items))

    data: list = []
    rows: list = []
    cols: list = []
    for d_c, r_c, c_c in results:
        data.extend(d_c)
        rows.extend(r_c)
        cols.extend(c_c)
    return data, rows, cols
