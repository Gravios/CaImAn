"""
xcorr_correction.py — Bidirectional line-scan phase correction
==============================================================

Resonant and galvo-resonant scanners acquire alternating rows in opposite
directions.  Even rows are scanned left→right; odd rows right→left.  A
mechanical phase delay between the two directions introduces a consistent
column offset (typically 0–8 px) that appears as a "comb" artefact: the
image looks doubled or cross-hatched when magnified.

This module estimates the optimal circular column shift for odd rows by
cross-correlating the column profiles of the even-row and odd-row mean
sub-images over a search range ±max_shift, then applies the correction to
every frame and writes a new TIFF with the suffix ``_Xcorrected.tif``.

Algorithm
---------
1.  Compute a temporal mean projection over up to ``n_frames`` evenly-spaced
    frames (robust to shot noise and transient artefacts).
2.  From the mean frame, compute:
      - ``even_profile`` = mean of all even rows (column profile)
      - ``odd_profile``  = mean of all odd rows
3.  Evaluate ``dot(even_profile, roll(odd_profile, -k))`` for each integer
    lag *k* in ``[-max_shift, +max_shift]``.  The lag that maximises this
    is the column shift of odd rows relative to even rows.
4.  Apply ``np.roll(frame[1::2, :], shift, axis=1)`` to every frame.

GPU acceleration
----------------
If CuPy is available (``use_gpu=True``, the default), steps 1 and 4 are
accelerated:

- **Mean projection**: frames are loaded in chunks, transferred to the GPU,
  accumulated as float32, then transferred back once — identical pattern to
  the precompute_corr_pnr_filtered_fov step.
- **Frame correction**: each output chunk is corrected on-GPU with
  ``cp.roll`` and written back as the original dtype via ``cp.asnumpy``.
  This avoids redundant host-side copies for large (>512 px wide) FOVs.

CPU fallback is used automatically when CuPy is unavailable or raises an
exception.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core algorithm  (runs on CPU ndarray — called after GPU mean is done)
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_row_shift(
    mean_frame: np.ndarray,
    max_shift: int,
) -> int:
    """Return the integer column shift to apply to odd rows to align them.

    Evaluates ``dot(even_profile, roll(odd_profile, -k))`` for each integer
    lag *k* in ``[-max_shift, +max_shift]`` and returns the *k* that
    maximises the dot product.

    Positive → roll odd rows right; negative → roll left.
    """
    even_profile = mean_frame[0::2, :].mean(axis=0).astype(np.float64)
    odd_profile  = mean_frame[1::2, :].mean(axis=0).astype(np.float64)
    even_profile -= even_profile.mean()
    odd_profile  -= odd_profile.mean()

    lags   = np.arange(-max_shift, max_shift + 1)
    scores = np.array([
        np.dot(even_profile, np.roll(odd_profile, -k))
        for k in lags
    ])
    return int(lags[np.argmax(scores)])


# ──────────────────────────────────────────────────────────────────────────────
# GPU helpers
# ──────────────────────────────────────────────────────────────────────────────

def _try_import_cupy():
    """Return the cupy module or None if unavailable."""
    try:
        import cupy as cp
        cp.zeros(1)   # trigger lazy init / device check
        return cp
    except Exception:
        return None


def _mean_projection_gpu(tif, idx_est, rows, cols, cp) -> np.ndarray:
    """Compute a float32 mean projection on the GPU.

    Pages are read from *tif* (already open TiffFile) in chunks of
    ~256 MB, transferred to the GPU as a batch, and accumulated.
    A single D2H transfer returns the final mean.
    """
    n = len(idx_est)
    accum_gpu = cp.zeros((rows, cols), dtype=cp.float32)

    chunk = max(1, min(n, (256 * 2**20) // max(1, rows * cols * 4)))
    for start in range(0, n, chunk):
        batch_idx = idx_est[start:start + chunk]
        cpu_block = np.stack(
            [tif.pages[int(i)].asarray() for i in batch_idx],
            axis=0,
        ).astype(np.float32)            # (B, rows, cols)
        gpu_block  = cp.asarray(cpu_block)
        accum_gpu += gpu_block.sum(axis=0)
        del gpu_block, cpu_block

    return cp.asnumpy(accum_gpu / n)    # single D2H


def _apply_correction_gpu(
    tif_in,
    writer,
    n_pages: int,
    rows: int,
    cols: int,
    dtype,
    shift: int,
    cp,
    log,
) -> None:
    """Apply the row shift to all frames on the GPU and stream to writer."""
    bytes_per_frame = rows * cols * np.dtype(dtype).itemsize
    chunk = max(1, min(n_pages, (256 * 2**20) // max(1, bytes_per_frame)))

    for start in range(0, n_pages, chunk):
        end       = min(start + chunk, n_pages)
        cpu_block = np.stack(
            [tif_in.pages[i].asarray() for i in range(start, end)],
            axis=0,
        ).astype(np.float32)                        # (B, rows, cols)

        gpu_block = cp.asarray(cpu_block)
        if shift != 0:
            gpu_block[:, 1::2, :] = cp.roll(
                gpu_block[:, 1::2, :], shift, axis=2
            )
        corrected_cpu = cp.asnumpy(gpu_block).astype(dtype)
        del gpu_block, cpu_block

        for fi in range(corrected_cpu.shape[0]):
            writer.write(corrected_cpu[fi], contiguous=True)

        log.debug(f"xcorr_correction (GPU): frames {start}–{end-1}")


# ──────────────────────────────────────────────────────────────────────────────
# CPU fallback helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mean_projection_cpu(tif, idx_est, rows, cols) -> np.ndarray:
    accum = np.zeros((rows, cols), dtype=np.float64)
    for i in idx_est:
        pg = tif.pages[int(i)].asarray()
        if pg.ndim > 2:
            pg = pg.mean(axis=tuple(range(pg.ndim - 2)))
        accum += pg.astype(np.float64)
    return (accum / len(idx_est)).astype(np.float32)


def _apply_correction_cpu(tif_in, writer, n_pages, rows, cols, dtype, shift, log):
    bytes_per_frame = rows * cols * np.dtype(dtype).itemsize
    chunk = max(1, min(n_pages, (256 * 2**20) // max(1, bytes_per_frame)))

    for start in range(0, n_pages, chunk):
        end   = min(start + chunk, n_pages)
        block = np.stack(
            [tif_in.pages[i].asarray() for i in range(start, end)],
            axis=0,
        )
        for fi in range(block.shape[0]):
            frame_in = block[fi]
            if frame_in.ndim > 2:
                frame_in = frame_in.mean(
                    axis=tuple(range(frame_in.ndim - 2))
                ).astype(dtype)
            out = frame_in.copy()
            if shift != 0:
                out[1::2, :] = np.roll(frame_in[1::2, :], shift, axis=1)
            writer.write(out, contiguous=True)

        log.debug(f"xcorr_correction (CPU): frames {start}–{end-1}")


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def correct_line_scan(
    src_tif:   str | os.PathLike,
    max_shift: int = 16,
    n_frames:  int = 500,
    use_gpu:   bool = True,
    logger:    Optional[logging.Logger] = None,
    overwrite: bool = False,
) -> str:
    """Estimate and apply a bidirectional line-scan phase correction.

    Reads *src_tif*, estimates the column offset between even and odd rows
    from a mean projection of up to *n_frames* evenly-spaced frames, applies
    a circular column shift to all odd rows in every frame, and writes the
    result to ``<stem>_Xcorrected.tif`` in the same directory.

    Parameters
    ----------
    src_tif   : str or Path   Input TIFF (single- or multi-page).
    max_shift : int           Maximum column shift to search (±). Default 16.
    n_frames  : int           Frames used for shift estimation.
                              0 = use all frames.  Default 500.
    use_gpu   : bool          Use CuPy for mean projection and frame
                              correction when available.  Default True.
    logger    : Logger        Uses the module logger if not provided.
    overwrite : bool          If False (default) and the output file already
                              exists, skip writing and return its path.

    Returns
    -------
    str   Absolute path of the written (or pre-existing) ``_Xcorrected.tif``.

    Raises
    ------
    FileNotFoundError   If *src_tif* does not exist.
    ValueError          If the file contains fewer than 2 rows.
    """
    log = logger or _log
    src = Path(src_tif).resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    out = src.with_name(src.stem + "_Xcorrected.tif")
    if out.exists() and not overwrite:
        log.info(f"xcorr_correction: output already exists, skipping — {out.name}")
        return str(out)

    log.info(f"xcorr_correction: reading {src.name}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    with tifffile.TiffFile(str(src)) as tif:
        n_pages = len(tif.pages)
        if n_pages == 0:
            raise ValueError(f"{src}: no pages found")
        first = tif.pages[0].asarray()
        if first.ndim == 2:
            rows, cols = first.shape
        elif first.ndim == 3:
            rows, cols = first.shape[-2], first.shape[-1]
        else:
            raise ValueError(f"Unexpected frame shape: {first.shape}")
        if rows < 2:
            raise ValueError(
                f"{src}: only {rows} row(s) — cannot apply row correction"
            )
        dtype = first.dtype
        log.info(
            f"xcorr_correction: {n_pages} frames  {rows}×{cols} px  dtype={dtype}"
        )

        # Frame indices for mean estimation
        if n_frames <= 0 or n_frames >= n_pages:
            idx_est = np.arange(n_pages)
        else:
            idx_est = np.linspace(0, n_pages - 1, n_frames, dtype=int)

        # ── Mean projection ───────────────────────────────────────────────────
        cp = _try_import_cupy() if use_gpu else None
        if cp is not None:
            try:
                log.info(
                    f"xcorr_correction: building mean projection on GPU "
                    f"({len(idx_est)} frames)"
                )
                mean_proj = _mean_projection_gpu(tif, idx_est, rows, cols, cp)
            except Exception as exc:
                log.warning(
                    f"xcorr_correction: GPU mean failed ({exc}); falling back to CPU"
                )
                cp = None
                mean_proj = _mean_projection_cpu(tif, idx_est, rows, cols)
        else:
            log.info(
                f"xcorr_correction: building mean projection on CPU "
                f"({len(idx_est)} frames)"
            )
            mean_proj = _mean_projection_cpu(tif, idx_est, rows, cols)

    # ── Shift estimation (always CPU — tiny computation on (rows, cols) array) ─
    shift = _estimate_row_shift(mean_proj, max_shift)
    backend = "GPU" if cp is not None else "CPU"
    log.info(
        f"xcorr_correction: odd-row shift = {shift:+d} px "
        f"(search ±{max_shift} px,  backend={backend})"
    )
    if shift == 0:
        log.info("xcorr_correction: shift is zero — writing pass-through copy")

    # ── Apply correction ──────────────────────────────────────────────────────
    # Write to a temporary file and atomically rename on success.  This
    # serves two purposes:
    #   (1) A mid-write crash leaves <out>.tmp behind, not a partial <out>
    #       that the next run's "skip if exists" check would mistake for
    #       valid output.
    #   (2) The GPU→CPU fallback must close-and-reopen the writer (see
    #       below) — using a .tmp path means the partial GPU writes never
    #       reach <out>.
    log.info(f"xcorr_correction: writing → {out.name}")
    bytes_per_frame = rows * cols * np.dtype(dtype).itemsize
    bigtiff = (n_pages * bytes_per_frame) > 2 ** 31

    out_tmp = out.with_suffix(out.suffix + ".tmp")
    if out_tmp.exists():
        out_tmp.unlink()   # orphan from a prior crash

    def _open_writer():
        return tifffile.TiffWriter(str(out_tmp), bigtiff=bigtiff)

    with tifffile.TiffFile(str(src)) as tif_in:
        writer = _open_writer()
        try:
            if cp is not None:
                try:
                    _apply_correction_gpu(
                        tif_in, writer, n_pages, rows, cols, dtype, shift, cp, log
                    )
                except Exception as exc:
                    log.warning(
                        f"xcorr_correction: GPU correction failed ({exc}); "
                        f"falling back to CPU"
                    )
                    # The GPU path may have written some pages before
                    # failing.  Restarting the CPU loop from 0 on the
                    # SAME writer would leave those GPU pages in the
                    # final file, producing K*chunk extra duplicated
                    # frames at the start of the output.  Close the
                    # writer, delete the partial .tmp, and reopen fresh.
                    writer.close()
                    if out_tmp.exists():
                        out_tmp.unlink()
                    writer = _open_writer()
                    _apply_correction_cpu(
                        tif_in, writer, n_pages, rows, cols, dtype, shift, log
                    )
            else:
                _apply_correction_cpu(
                    tif_in, writer, n_pages, rows, cols, dtype, shift, log
                )
        finally:
            writer.close()

    # Atomic rename — observers either see no <out> or the complete file.
    os.replace(str(out_tmp), str(out))

    log.info(f"xcorr_correction: done — {out}")
    return str(out)
