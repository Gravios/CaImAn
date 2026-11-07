"""Event tables export and blinded inter-rater review files."""

import os
from typing import TYPE_CHECKING

import numpy as np

from .deps import _ed, _rf

if TYPE_CHECKING:                      # annotations only
    from .store import ComponentStore


# ── Event exports / blinded review (no Qt; called by the window) ─────────────

def export_event_tables(store: 'ComponentStore', peak_path: str) -> list:
    """Write SLab PeakStats + ROIStats tables for the *included* components.

    ``peak_path`` names the PeakStats file (.xlsx or .csv); ROIStats goes
    next to it with ``PeakStats`` → ``ROIStats`` in the name.  ``.xlsx``
    needs pandas + openpyxl and falls back to ``.csv``.  ROIStats lists every
    included ROI, including those with 0 events (SLab dropped them).
    Returns the paths written.
    """
    ed  = _ed()
    ev  = store.events
    if ev is None:
        raise RuntimeError("no events to export")
    rows  = np.flatnonzero(store.approved)
    peaks = ev.peak_table(rows)
    peaks["label"] = np.array([store.labels[r] for r in peaks["roi"]], dtype=object)
    rois  = ed.roi_stats(ev, store.approved, ev.binary.shape[1],
                         labels=store.labels, SNR_comp=store.SNR_comp,
                         cnn_preds=store.cnn_preds)
    if store.qc is not None:                 # ROI-filter fields, same rows
        rr = np.asarray(rois["ROI"], dtype=np.int64)
        for k, v in store.qc.items():
            col = np.asarray(v)[rr]
            rois[f"qc_{k}"] = col.astype(int) if col.dtype == np.bool_ else col
        if "fail_reasons" in store.qc:
            rois["qc_fail"] = np.array(
                [",".join(_rf().decode_reasons(m)) for m in store.qc["fail_reasons"][rr]],
                dtype=object)
    folder, base = os.path.split(peak_path)
    stem, ext = os.path.splitext(base)
    roi_stem = (stem.replace("PeakStats", "ROIStats") if "PeakStats" in stem
                else stem + "_ROIStats")
    ext = ext.lower() if ext.lower() in (".xlsx", ".csv") else ".xlsx"
    pd = None
    if ext == ".xlsx":
        import importlib.util
        if importlib.util.find_spec("openpyxl") is not None:
            try:
                import pandas as pd
            except ImportError:
                pd = None
        if pd is None:
            ext = ".csv"
    out = []
    for table, st in ((peaks, stem), (rois, roi_stem)):
        p = os.path.join(folder, st + ext)
        if ext == ".xlsx":
            pd.DataFrame(table).to_excel(p, index=False)
        else:
            import csv
            cols = list(table)
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(cols)
                for k in range(len(table[cols[0]]) if cols else 0):
                    w.writerow([table[c][k] for c in cols])
        out.append(p)
    return out


def blind_plan(files, rater: str):
    """Split results files for a blinded review by ``rater``.

    Returns ``(unrated, rated, skipped)``; files without a ``stroh_results``
    group (no events yet) or unreadable files are ``skipped``.
    """
    import h5py
    unrated, rated, skipped = [], [], []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                if "stroh_results" not in f:
                    skipped.append(p)
                elif "SLabRandomization" in f and rater in f["SLabRandomization"]:
                    rated.append(p)
                else:
                    unrated.append(p)
        except Exception:
            skipped.append(p)
    return unrated, rated, skipped


def write_blind_rating(path: str, rater: str, store: 'ComponentStore',
                       file_list) -> None:
    """Append ``SLabRandomization/<rater>`` to the results file — the SLab
    inter-rater layout (stroh_results fields + file_list + timestamp attr).
    Only that group is written; estimates are untouched.  Requires a store
    loaded with ``all_components=True`` and no merges/deletes, so rows are
    the file's estimates rows."""
    import h5py
    from datetime import datetime
    if not np.array_equal(store.orig_idx, np.arange(store.n)):
        raise RuntimeError("ratings must index the file's full estimates")
    with h5py.File(path, "a") as f:
        g = _ed().write_stroh_results(f, store.events, store.approved,
                                      group=f"SLabRandomization/{rater}")
        g.create_dataset("file_list",
                         data=[str(p).encode("utf-8") for p in file_list])
        g.attrs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


__all__ = ['export_event_tables', 'blind_plan', 'write_blind_rating']
