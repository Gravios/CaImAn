#!/usr/bin/env python
"""Regression tests for the periphery AST-audit repair series.

Tests two behavioural fixes:

1. ``test_xcorr_correction_gpu_fallback_produces_correct_page_count``
       HIGH — when the GPU correction path fails after partial writes,
       the CPU fallback must NOT produce a TIFF with more pages than
       the source.  Pre-fix, K*chunk GPU pages remained in the file
       and the CPU restart added n_pages more.

2. ``test_timing_module_optional_resolves``
       MEDIUM — get_type_hints(write_report) must succeed.  Pre-fix,
       Optional was used in the signature but not imported, so the
       call raised NameError.

No CaImAn runtime needed.
"""

import os
import sys
import tempfile
import typing
from pathlib import Path
import importlib

import numpy as np
import pytest

import caiman.utils.xcorr_correction as xcorr_mod
import caiman.utils.timing as timing_mod


# ── 1. xcorr_correction: GPU fallback page-count correctness ────────────────

def test_xcorr_correction_gpu_fallback_produces_correct_page_count(
    tmp_path, monkeypatch
):
    """If GPU correction succeeds for k chunks then fails, the resulting
    output TIFF must have exactly n_pages — NOT n_pages + the partial
    GPU pages."""
    import tifffile

    n_pages = 32
    rows, cols = 8, 16
    src = tmp_path / "src.tif"

    # Write a source TIFF with n_pages frames, each a known constant
    with tifffile.TiffWriter(str(src)) as tw:
        for i in range(n_pages):
            tw.write(np.full((rows, cols), i, dtype=np.uint16), contiguous=True)

    # Force CuPy detection to succeed (so the GPU path is selected),
    # then monkey-patch _apply_correction_gpu to write a few chunks
    # and then raise.
    class _FakeCp:
        pass
    monkeypatch.setattr(xcorr_mod, "_try_import_cupy", lambda: _FakeCp())

    # Mean projection: just return zeros (shift will be 0)
    monkeypatch.setattr(
        xcorr_mod, "_mean_projection_gpu",
        lambda tif, idx, r, c, cp: np.zeros((r, c), dtype=np.float32),
    )

    # GPU "applies correction" — writes 12 of the 32 pages, then dies
    def fake_gpu(tif_in, writer, n_pages, rows, cols, dtype, shift, cp, log):
        for i in range(12):   # K*chunk = 12 successful page writes
            frame = tif_in.pages[i].asarray()
            writer.write(frame, contiguous=True)
        raise RuntimeError("simulated GPU failure after 12 page writes")

    monkeypatch.setattr(xcorr_mod, "_apply_correction_gpu", fake_gpu)

    out_path = xcorr_mod.correct_line_scan(str(src), use_gpu=True)

    # Final output must have exactly n_pages, not 12 + n_pages = 44
    with tifffile.TiffFile(out_path) as tf:
        assert len(tf.pages) == n_pages, (
            f"xcorr output has {len(tf.pages)} pages; expected {n_pages} "
            f"(GPU partial writes leaked into final file)"
        )
        # Frames must match source 1:1 (CPU pass identity since shift=0)
        for i in range(n_pages):
            np.testing.assert_array_equal(
                tf.pages[i].asarray(),
                np.full((rows, cols), i, dtype=np.uint16),
                err_msg=f"frame {i} mismatches source",
            )

    # .tmp must not survive
    tmp = Path(out_path).with_suffix(Path(out_path).suffix + ".tmp")
    assert not tmp.exists(), f".tmp file orphaned: {tmp}"


def test_xcorr_correction_skip_does_not_rerun_on_clean_output(tmp_path):
    """Existing valid output is reused (the cheap fast-path)."""
    import tifffile

    src = tmp_path / "src.tif"
    out = tmp_path / "src_Xcorrected.tif"

    rows, cols, n = 4, 8, 3
    with tifffile.TiffWriter(str(src)) as tw:
        for i in range(n):
            tw.write(np.full((rows, cols), i, dtype=np.uint16), contiguous=True)
    # Pre-create the output so the skip-if-exists path triggers
    with tifffile.TiffWriter(str(out)) as tw:
        for i in range(n):
            tw.write(np.full((rows, cols), 99, dtype=np.uint16), contiguous=True)

    result = xcorr_mod.correct_line_scan(str(src), overwrite=False)
    assert Path(result) == out
    # File untouched (still contains the 99s, not the source content)
    with tifffile.TiffFile(result) as tf:
        np.testing.assert_array_equal(
            tf.pages[0].asarray(), np.full((rows, cols), 99, dtype=np.uint16),
        )


# ── 2. timing.write_report: Optional resolves ────────────────────────────────

def test_timing_module_optional_resolves():
    """get_type_hints(write_report) must succeed — i.e. every type name
    in the signature is importable from the module's namespace."""
    # Reload to ensure we test the current source, not a stale import
    importlib.reload(timing_mod)
    hints = typing.get_type_hints(timing_mod.write_report)
    assert "extra" in hints
    # The Optional[dict] in the signature should resolve to a Union with NoneType
    extra_hint = hints["extra"]
    args = typing.get_args(extra_hint)
    assert type(None) in args, \
        f"extra annotation should include None; got args={args}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
