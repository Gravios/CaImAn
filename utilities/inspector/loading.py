"""Loading: results .hdf5, stacks, session discovery, backgrounds, demo data."""

import os
import sys

import numpy as np

from .constants import TRACE_DENOISED, TRACE_DFF
from .deps import _ed, _rf
from .store import ComponentStore, _has_contrast, _to_sparse


def _load_image_file(path: str) -> np.ndarray:
    """Load a background image from .npy/.npz/.tif/.tiff/.png/.jpg.

    Returns the raw array (2-D, or 3-D stack to be collapsed by the store).
    Stacks are returned as-is so the caller can max-project them.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        with np.load(path) as z:
            for key in ("projection", "img", "image", "Cn", "arr_0"):
                if key in z:
                    return z[key]
            return z[z.files[0]]
    if ext in (".tif", ".tiff"):
        if len(_tif_shape(path)) <= 2:
            import tifffile
            return tifffile.imread(path)
        return _stack_projections(path)["stack max"]   # streamed, not imread
    if ext in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        return np.asarray(Image.open(path).convert("F"))
    raise ValueError(f"unsupported background format: {ext}")


TIF_EXTS = (".tif", ".tiff")
SIBLING_STACK_MAX_BYTES = 2 << 30   # auto-project sibling stacks only up to 2 GiB


def _tif_shape(path: str) -> tuple:
    """Shape of the first series of a TIFF, read from the header only."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        return tuple(tf.series[0].shape)


def _stack_projections(path: str, chunk: int = 256) -> dict:
    """Mean and max projections of a TIFF stack, streamed frame by frame.

    Never holds more than ``chunk`` frames in memory, so multi-GB BigTIFF
    movies work.  Leading axes (time, z, channel) are all reduced; the last
    two are (H, W).  A 2-D TIFF returns the image as both projections.
    """
    import tifffile
    shape = _tif_shape(path)
    if len(shape) <= 2:
        img = np.asarray(tifffile.imread(path), dtype=np.float32)
        return {"stack mean": img, "stack max": img}
    H, W = shape[-2:]
    acc = np.zeros((H, W), dtype=np.float64)
    mx  = np.full((H, W), -np.inf, dtype=np.float32)
    n   = 0
    try:                                        # contiguous: memmap + chunks
        arr = tifffile.memmap(path, mode="r").reshape(-1, H, W)
        for s in range(0, arr.shape[0], chunk):
            blk = np.asarray(arr[s:s + chunk], dtype=np.float32)
            acc += blk.sum(axis=0, dtype=np.float64)
            np.maximum(mx, blk.max(axis=0), out=mx)
            n += blk.shape[0]
    except Exception:                           # compressed / tiled: per page
        acc[:] = 0; mx[:] = -np.inf; n = 0
        with tifffile.TiffFile(path) as tf:
            for page in tf.pages:
                fr = np.asarray(page.asarray(), dtype=np.float32).reshape(-1, H, W)
                acc += fr.sum(axis=0, dtype=np.float64)
                np.maximum(mx, fr.max(axis=0), out=mx)
                n += fr.shape[0]
    if n == 0:
        raise ValueError(f"no frames in {path}")
    return {"stack mean": (acc / n).astype(np.float32), "stack max": mx}


def resolve_results_for_stack(stack_path: str) -> str:
    """Find the CNMF results file that belongs to a raw ``.tif`` stack.

    Looks in the stack's folder, then its parent, for (in order)
    ``<stem>_results.hdf5``, ``<stem>_caiman_results.hdf5``, then any
    ``<stem>*.hdf5`` / ``<stem>*.h5``; originals are preferred over
    ``*_curated`` copies.  Raises FileNotFoundError listing where it looked.
    """
    import glob
    stem   = os.path.splitext(os.path.basename(stack_path))[0]
    folder = _os_dirname(stack_path)
    dirs   = [folder, os.path.dirname(folder)]
    for d in dirs:                              # exact names win in either dir
        for name in (f"{stem}_results.hdf5", f"{stem}_caiman_results.hdf5"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    for d in dirs:
        cands = sorted(glob.glob(os.path.join(glob.escape(d), glob.escape(stem) + "*.hdf5"))
                       + glob.glob(os.path.join(glob.escape(d), glob.escape(stem) + "*.h5")))
        cands.sort(key=lambda p: os.path.splitext(p)[0].endswith("_curated"))
        if cands:
            return cands[0]
    raise FileNotFoundError(
        f"no CNMF results for stack '{os.path.basename(stack_path)}' — looked for "
        f"{stem}_results.hdf5 / {stem}*.hdf5 in {dirs[0]} and {dirs[1]}. "
        f"Pass the results file instead, or use --results.")


def load_any(path: str, results: str = None) -> 'ComponentStore':
    """Open a results file, or a raw ``.tif`` stack plus its results.

    For a stack, the matching results file is found with
    :func:`resolve_results_for_stack` (or taken from ``results``), and the
    stack's streamed mean / max projections are added as backgrounds with
    the mean made active, so the actual cells are visible.
    """
    if os.path.isdir(path):
        res_path, tif = discover_session(path, results)
        return load_any(tif, res_path) if tif else load_from_hdf5(res_path)
    if os.path.splitext(path)[1].lower() not in TIF_EXTS:
        return load_from_hdf5(results or path)
    res_path = results or resolve_results_for_stack(path)
    print(f"Stack {os.path.basename(path)} -> results {res_path}")
    store = load_from_hdf5(res_path)
    # The sibling scan may already have projected this very stack.
    if store.bg_sources.get("stack mean") != os.path.abspath(path):
        for name, img in _stack_projections(path).items():
            store.add_background(name, img)
            store.bg_sources[name] = os.path.abspath(path)
    store.set_background("stack mean")
    return store


RESULTS_SUFFIX = "_results.hdf5"


def discover_session(folder: str, results: str = None):
    """Find a session's results file and raw stack inside ``folder``.

    The session base name is the folder's own name (``<dir>/``), so a
    session directory laid out as::

        <base>/<base>_results.hdf5
        <base>/<base>.tif

    is found from inside it.  Results: ``<base>_results.hdf5``, else the one
    ``*_results.hdf5`` in the folder (``*_results_curated.hdf5`` copies don't
    match the pattern).  Stack: ``<base>.tif`` / ``<base>.tiff``, or None.
    ``results`` overrides the results lookup.  Returns ``(results, tif)``.
    """
    import glob
    folder = os.path.abspath(folder)
    base   = os.path.basename(folder)
    tif = next((os.path.join(folder, base + e) for e in TIF_EXTS
                if os.path.isfile(os.path.join(folder, base + e))), None)
    if results:
        return results, tif
    exact = os.path.join(folder, base + RESULTS_SUFFIX)
    if os.path.isfile(exact):
        return exact, tif
    cands = sorted(glob.glob(os.path.join(glob.escape(folder), "*" + RESULTS_SUFFIX)))
    if len(cands) == 1:
        return cands[0], tif
    if cands:
        raise FileNotFoundError(
            f"{len(cands)} *{RESULTS_SUFFIX} files in {folder} and none is "
            f"{base}{RESULTS_SUFFIX}: "
            + ", ".join(os.path.basename(c) for c in cands)
            + " — pass one explicitly.")
    raise FileNotFoundError(
        f"no {base}{RESULTS_SUFFIX} (or any *{RESULTS_SUFFIX}) in {folder}")


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_from_hdf5(path: str, all_components: bool = True) -> ComponentStore:
    """Load a CNMF results file — every row of the estimates.

    Rows outside ``idx_components`` (CaImAn-rejected; in census mode the
    inactive cells) are loaded too and flagged ``accepted = False`` (Eval
    column), so Save never drops them.  ``all_components`` is kept for
    callers; loading a subset is no longer supported because saving it
    would lose the other rows."""
    import os as _os
    # Ensure CAIMAN_TEMP is set to a writable directory before load_CNMF
    # calls fn_relocated.  If the current value is missing or unwritable,
    # fall back to the directory that contains the HDF5 file.
    _ct = _os.environ.get("CAIMAN_TEMP", "")
    if not (_ct and _os.path.isdir(_ct) and _os.access(_ct, _os.W_OK)):
        _fallback = _os.path.abspath(_os.path.dirname(path)) or "."
        _os.environ["CAIMAN_TEMP"] = _fallback

    from caiman.source_extraction.cnmf.cnmf import load_CNMF
    cnm  = load_CNMF(path)
    est  = cnm.estimates
    dims = cnm.dims if (cnm.dims and all(d > 0 for d in cnm.dims)) else est.dims

    A = _to_sparse(est.A)
    C = est.C
    K = A.shape[1]

    # accepted / rejected (census: active / inactive) from idx_components
    accepted = None
    idx = getattr(est, "idx_components", None)
    if idx is not None:
        idx = np.asarray(idx, dtype=np.int64).ravel()
        accepted = np.zeros(K, dtype=bool)
        accepted[idx[(idx >= 0) & (idx < K)]] = True

    def _sel_rows(arr):
        """An optional (K, T) trace array as float32, or None."""
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        return None if arr.size == 0 else arr

    snr  = getattr(est, "SNR_comp",  None)
    rval = getattr(est, "r_values",  None)
    cnn  = getattr(est, "cnn_preds", None)

    # Build the trace registry: denoised C is canonical; add ΔF/F, the
    # residual-added raw trace, and deconvolved spikes when available and
    # shape-compatible.  Display-only — the dropdown switches between them.
    Cd     = np.asarray(C, dtype=np.float32)
    traces = {TRACE_DENOISED: Cd}
    F_dff  = _sel_rows(getattr(est, "F_dff", None))
    YrA    = _sel_rows(getattr(est, "YrA",   None))
    S      = _sel_rows(getattr(est, "S",     None))
    if F_dff is not None and F_dff.shape == Cd.shape:
        traces[TRACE_DFF] = F_dff
    if YrA is not None and YrA.shape == Cd.shape:
        traces["C + residual (raw)"] = Cd + YrA
    if S is not None and S.shape == Cd.shape and np.any(S):
        traces["S (deconvolved)"] = S

    _Cn = getattr(est, "Cn", None)
    Cn  = (_Cn if _Cn is not None else
           A.max(axis=1).toarray().ravel().reshape(dims, order="F"))
    pdata = getattr(getattr(cnm, "params", None), "data", None) or {}
    store = ComponentStore(A, C, Cn, dims,
                           SNR_comp=snr, r_values=rval, cnn_preds=cnn,
                           cnm_obj=cnm, fpath=path, traces=traces,
                           orig_idx=np.arange(K), accepted=accepted,
                           fr=_opt_float(pdata.get("fr")),
                           decay_time=_opt_float(pdata.get("decay_time")))
    _attach_roi_filter(store, path)        # seeds Incl from `passed` …
    _attach_stroh_results(store, path)     # … unless event approvals exist
    _attach_sibling_backgrounds(store, path)
    return store


def _opt_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def _attach_roi_filter(store: 'ComponentStore', path: str):
    """Load a ``roi_filter`` group if present, mapped through orig_idx.

    Include flags start from the group's saved manual approvals, else from
    the filter's ``passed`` verdict.  Event approvals (stroh_results), read
    afterwards, take precedence."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "roi_filter" not in f:
                return
            fields, params, appr = _rf().read_group(f["roi_filter"])
    except Exception as exc:
        print(f"[roi_filter] could not read roi_filter: {exc}", file=sys.stderr)
        return
    if not fields:
        return
    K  = int(next(iter(fields.values())).size)
    oi = store.orig_idx
    if not (oi.size and (oi >= 0).all() and oi.max() < K):
        print(f"[roi_filter] {K} rows; cannot map onto {store.n} shown "
              "components — ignored", file=sys.stderr)
        return
    store.qc        = _rf().take(fields, oi)
    store.qc_params = params
    base = appr if appr is not None else fields.get("passed")
    if base is not None:
        store.approved = np.asarray(base, bool)[oi].copy()


def _attach_stroh_results(store: 'ComponentStore', path: str):
    """Load an SLab ``stroh_results`` group (events + approvals) if present."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "stroh_results" not in f:
                return
            res, appr = _ed().read_stroh_results(f["stroh_results"])
    except Exception as exc:
        print(f"[events] could not read stroh_results: {exc}", file=sys.stderr)
        return
    if res is None:
        print("[events] stroh_results too old (< v7.1) — ignored", file=sys.stderr)
    elif res.binary.shape[1] != store.C.shape[1]:
        print("[events] stroh_results length differs from C — ignored", file=sys.stderr)
    elif not store.attach_events(res, appr):
        print(f"[events] stroh_results has {res.n} rows; cannot map onto "
              f"{store.n} shown components — ignored", file=sys.stderr)


def _attach_sibling_backgrounds(store: 'ComponentStore', path: str):
    """Register data-projection images sitting next to the results file.

    Companion images share the *session prefix* with the results file
    (e.g. ``<session>_mean.npy`` next to ``<session>_results.hdf5``); a small
    token vocabulary is also accepted.  Candidates must be 2-D and match the
    FOV (enforced by the store's coercion).  If the correlation image is flat
    — e.g. denoised data saturates Cn — the best-ranked projection is
    auto-activated so the viewer shows the actual cells.
    """
    import glob
    folder = _os_dirname(path)
    prefix = _session_prefix(path)
    tokens = ("projection", "percentile", "mean", "median",
              "max", "pnr", "summary", "anat", "template", "_cn")
    exts   = (".npy", ".npz", ".tif", ".tiff", ".png", ".jpg", ".jpeg")

    found = []   # (label, lowername)
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        if os.path.abspath(f) == os.path.abspath(path):
            continue
        low       = os.path.basename(f).lower()
        name_stem = os.path.splitext(os.path.basename(f))[0]
        if not low.endswith(exts):
            continue
        shares_prefix = bool(prefix) and name_stem.lower().startswith(prefix.lower())
        has_token     = any(tok in low for tok in tokens)
        if not (shares_prefix or has_token):
            continue
        try:
            label = name_stem
            if shares_prefix:
                label = name_stem[len(prefix):].lstrip("._- ") or name_stem
            # SLab rule: the raw stack <session>.tif next to the results is the
            # anatomical reference.  Project it (streamed) unless it is huge.
            if (low.endswith(TIF_EXTS) and name_stem == prefix
                    and len(_tif_shape(f)) > 2):
                if os.path.getsize(f) > SIBLING_STACK_MAX_BYTES:
                    print(f"[bg] skipping {os.path.basename(f)} "
                          f"({os.path.getsize(f) / 2**30:.1f} GiB); "
                          f"pass it with --background or open the .tif",
                          file=sys.stderr)
                    continue
                for pname, img in _stack_projections(f).items():
                    store.add_background(pname, img)
                    store.bg_sources[pname] = os.path.abspath(f)
                    found.append((pname, "mean" if "mean" in pname else "max"))
                continue
            img = _load_image_file(f)
            if np.asarray(img).squeeze().ndim != 2:   # other stacks: skip
                continue
            store.add_background(label, img)           # coercion dim-checks
            found.append((label, low))
        except Exception:
            continue

    if found and not _has_contrast(store.backgrounds.get("correlation")):
        def _rank(item):
            low = item[1]
            for r, tok in enumerate(tokens):
                if tok in low:
                    return r
            return len(tokens)
        store.set_background(sorted(found, key=_rank)[0][0])


def _session_prefix(path: str) -> str:
    """``<session>`` from ``<session>[_caiman][_results][_curated].hdf5``.

    Trailing results/cnmf/caiman/curated tails are stripped repeatedly, so
    ``<session>_caiman_results_curated`` → ``<session>``.
    """
    import re as _re
    prefix, prev = os.path.splitext(os.path.basename(path))[0], None
    while prefix != prev:
        prev   = prefix
        prefix = _re.sub(r'[._-]?(results?|cnmf?|cnm|caiman|estimates?|curated)$',
                         '', prefix, flags=_re.I)
    return prefix


def _os_dirname(path: str) -> str:
    return os.path.abspath(os.path.dirname(path)) or "."


# ── Demo data ─────────────────────────────────────────────────────────────────

def _make_demo_store() -> ComponentStore:
    print("[demo]  No file specified — running with synthetic data.")
    print("[demo]  Injected duplicate pair: C003 ≈ C005")
    rng  = np.random.default_rng(42)
    dims = (100, 100)
    K, T = 35, 2000
    d    = dims[0] * dims[1]

    A = np.zeros((d, K), dtype=np.float32)
    for k in range(K):
        cy, cx = rng.integers(8, 92, size=2)
        y, x   = np.ogrid[:dims[0], :dims[1]]
        sigma  = rng.uniform(4, 9)
        mask   = (y - cy) ** 2 + (x - cx) ** 2 < sigma ** 2
        A[mask.ravel(order="F"), k] = rng.uniform(
            0.5, 1.0, mask.sum()).astype(np.float32)

    C    = rng.random((K, T)).astype(np.float32)
    C[5] = C[3] + rng.random(T).astype(np.float32) * 0.03   # duplicate pair
    snr  = np.abs(rng.normal(3.0, 1.5, K)).astype(np.float32)
    rval = np.clip(rng.normal(0.7, 0.2, K), -1, 1).astype(np.float32)
    cnn  = np.clip(rng.normal(0.75, 0.15, K), 0, 1).astype(np.float32)
    Cn   = A.max(axis=1).reshape(dims, order="F")

    return ComponentStore(A, C, Cn, dims,
                          SNR_comp=snr, r_values=rval, cnn_preds=cnn)


__all__ = ['_load_image_file', 'TIF_EXTS', 'SIBLING_STACK_MAX_BYTES', '_tif_shape', '_stack_projections', 'resolve_results_for_stack', 'load_any', 'RESULTS_SUFFIX', 'discover_session', 'load_from_hdf5', '_opt_float', '_attach_roi_filter', '_attach_stroh_results', '_attach_sibling_backgrounds', '_session_prefix', '_os_dirname', '_make_demo_store']
