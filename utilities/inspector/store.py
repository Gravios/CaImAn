"""ComponentStore — the row-aligned component model (footprints, traces,
metrics, flags), its edits / undo, derived caches and save.  Qt-free."""

import os
import sys
from collections import deque

import numpy as np
import scipy.sparse

from .constants import HISTORY_MAXLEN, TRACE_DENOISED, TRACE_DFF
from .deps import _ed, _ek, _rf


class ComponentStore:
    """
    Mutable container for CNMF component data: one row per component in
    every row-aligned field (footprints A as CSC float32 (d × K), traces,
    labels, metrics, hidden / include / accepted flags, events, qc).
    """

    def __init__(self, A, C, Cn, dims,
                 SNR_comp=None, r_values=None, cnn_preds=None,
                 cnm_obj=None, fpath=None, traces=None,
                 orig_idx=None, fr=None, decay_time=None, accepted=None):
        # Footprints: (d, K) CSC float32, pixels in Fortran order of dims.
        # Columns are only ever replaced (never written in place), so undo
        # snapshots can share them.
        self._A = _to_sparse(A)
        self.dims = dims
        K = self._A.shape[1]
        self._derived = {}      # corr / centroids / peaks / pixel lookup / kinetics
        self.kinetics_params = None   # event_kinetics params (None → defaults)

        # ── CaImAn evaluation (census: active / inactive) ────────────────────
        # accepted   : row was in estimates.idx_components (census: active);
        #              rejected rows (idx_components_bad, census: inactive)
        #              are loaded, shown and saved like any other row
        # eval_known : the file carried idx_components (else all accepted)
        # hide_rejected : view option — hide rejected rows (still saved)
        self.eval_known = accepted is not None
        self.accepted = (np.ones(K, dtype=bool) if accepted is None
                         else np.asarray(accepted, dtype=bool).copy())
        self.hide_rejected = False

        # ── Event detection / curation (SLab stroh_results) ──────────────────
        # events   : EventResults aligned to rows, or None until detected/loaded
        # approved : manual include flag (SLab "manually_approved"); unlike
        #            delete it is non-destructive and is saved alongside A/C
        # orig_idx : row in the source file's estimates (-1 for merged rows),
        #            used to map file-level stroh_results onto the shown subset
        self.events   = None
        self.approved = np.ones(K, dtype=bool)
        # qc       : roi_filter fields {name: (K,) array} aligned to rows, or
        #            None when the file has no roi_filter group
        self.qc        = None
        self.qc_params = {}
        self.orig_idx = (np.arange(K) if orig_idx is None
                         else np.asarray(orig_idx, dtype=np.int64))
        self.fr         = fr
        self.decay_time = decay_time

        # ── Temporal traces (display registry) ───────────────────────────────
        # Every entry is a (K, T) array.  TRACE_DENOISED is the canonical
        # trace written back on save; the others (ΔF/F, raw, deconvolved) are
        # for display only and are selected via the trace dropdown.  self.C
        # always mirrors the denoised entry so save / correlation / merge math
        # are unaffected by the display choice.
        if traces is None:
            traces = {TRACE_DENOISED: np.asarray(C, dtype=np.float32)}
        self.traces = {k: np.asarray(v, dtype=np.float32) for k, v in traces.items()}
        if TRACE_DENOISED not in self.traces:
            self.traces[TRACE_DENOISED] = np.asarray(C, dtype=np.float32)
        self.disp_key = TRACE_DENOISED
        self.C = self.traces[TRACE_DENOISED]

        # ── Background images ────────────────────────────────────────────────
        # `Cn` is the *active* greyscale shown behind the footprints.  A named
        # registry holds every available image so the user can switch live
        # (correlation image, footprint max-projection, a loaded data
        # projection, etc).  self.Cn always mirrors the active entry so the
        # rest of the code can keep reading store.Cn unchanged.
        self.backgrounds: dict = {}
        self.bg_sources:  dict = {}    # background name -> source file path
        self.bg_key = None
        self.add_background("correlation",
                            np.asarray(Cn, dtype=np.float32), make_active=True)
        fp_max = (self._A.max(axis=1).toarray().ravel()
                  .reshape(dims, order="F").astype(np.float32))
        self.add_background("footprint max", fp_max)
        # If the correlation image is (near-)flat — e.g. SUPPORT-denoised data
        # saturates Cn to ~1.0 — fall back to the footprint projection, which
        # at least reveals where the components sit.
        if not _has_contrast(self.Cn):
            self.set_background("footprint max")

        self.SNR_comp  = _opt_arr(SNR_comp)
        self.r_values  = _opt_arr(r_values)
        self.cnn_preds = _opt_arr(cnn_preds)

        self.cnm_obj = cnm_obj
        self.fpath   = fpath
        self.labels  = [f"C{i:03d}" for i in range(self._A.shape[1])]
        self.hidden  = np.zeros(self._A.shape[1], dtype=bool)  # per-component visibility

        # History stacks — each entry is (op_label: str, snapshot: dict)
        self._undo_stack: deque = deque(maxlen=HISTORY_MAXLEN)
        self._redo_stack: deque = deque(maxlen=HISTORY_MAXLEN)

    # ── Background images ──────────────────────────────────────────────────────

    def _coerce_bg(self, img) -> np.ndarray:
        """Coerce an arbitrary image to a 2-D float32 array matching dims.

        A 3-D stack (frames, h, w) is collapsed to a max-projection, so a raw
        movie file can be dropped in directly as an 'actual cells' background.
        A transposed image is auto-corrected.
        """
        a = np.asarray(img, dtype=np.float32).squeeze()
        if a.ndim == 3:
            a = a.max(axis=0)
        if a.ndim != 2:
            raise ValueError(f"background must be 2-D (got shape {a.shape})")
        if a.shape == tuple(self.dims):
            return np.ascontiguousarray(a)
        if a.shape == tuple(self.dims)[::-1]:
            return np.ascontiguousarray(a.T)
        raise ValueError(
            f"background shape {a.shape} does not match dims {tuple(self.dims)}")

    def add_background(self, name: str, img, make_active: bool = False) -> str:
        """Register a named background image (overwriting any same-named one)."""
        self.backgrounds[name] = self._coerce_bg(img)
        if make_active or self.bg_key is None:
            self.set_background(name)
        return name

    def set_background(self, name: str):
        if name not in self.backgrounds:
            raise KeyError(name)
        self.bg_key = name
        self.Cn = self.backgrounds[name]

    def background_names(self) -> list:
        return list(self.backgrounds.keys())

    # ── Temporal trace selection (display only) ────────────────────────────────

    def trace_names(self) -> list:
        return list(self.traces.keys())

    def set_trace(self, name: str):
        if name not in self.traces:
            raise KeyError(name)
        self.disp_key = name

    def disp_trace(self, i):
        """The currently displayed trace for component i."""
        return self.traces[self.disp_key][i]

    @property
    def disp_T(self) -> int:
        return self.traces[self.disp_key].shape[1]

    # ── Per-component visibility (view only, not saved) ────────────────────────

    def is_hidden(self, i) -> bool:
        return bool(self.hidden[i])

    def is_visible(self, i) -> bool:
        """Drawn / clickable: not hidden, and not a hidden-away rejected row."""
        return not (self.hidden[i] or (self.hide_rejected and not self.accepted[i]))

    def visible_mask(self) -> np.ndarray:
        """(K,) bool — :meth:`is_visible` for every row."""
        return ~(self.hidden | (self.hide_rejected & ~self.accepted))

    @property
    def n_accepted(self) -> int:
        return int(self.accepted.sum())

    def toggle_hidden(self, indices):
        for i in indices:
            self.hidden[i] = not self.hidden[i]

    def show_all(self):
        """Make every component visible again."""
        self.hidden[:] = False

    @property
    def n_hidden(self) -> int:
        return int(self.hidden.sum())


    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n(self):
        return self._A.shape[1]

    # ── Derived, cached per component set ─────────────────────────────────────
    # Kept in self._derived; merge / delete update them incrementally,
    # undo / redo restore them with the snapshot, anything else drops them.

    def _cached(self, key, build):
        v = self._derived.get(key)
        if v is None:
            v = self._derived[key] = build()
        return v

    @property
    def corr(self) -> np.ndarray:
        """(K, K) masked Pearson of the denoised traces; diagonal NaN."""
        return self._cached("corr", lambda: _active_pearson_corr(self.C))

    @property
    def centroids(self) -> np.ndarray:
        """(K, 2) footprint-weighted (row, col); FOV centre if empty."""
        return self._cached("centroids", lambda: _centroids(self._A, self.dims))

    @property
    def dist(self) -> np.ndarray:
        """(K, K) centroid distance (px); diagonal NaN."""
        def build():
            c = self.centroids.astype(np.float64)
            d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)).astype(np.float32)
            np.fill_diagonal(d, np.nan)
            return d
        return self._cached("dist", build)

    @property
    def peaks(self) -> np.ndarray:
        """(K,) footprint maxima."""
        return self._cached("peaks",
                            lambda: self._A.max(axis=0).toarray().ravel())

    # ── Event kinetics (diagnostic; approval unchanged) ─────────────────────

    @property
    def kinetics(self):
        """Per-row :func:`event_kinetics.validate_row` dicts, or None without
        events / ΔF/F.  Cached; merge / delete / undo keep it in step."""
        if self.events is None or self.dff is None:
            return None
        return self._cached("kinetics", lambda: _ek().validate(
            self.kinetics_trace, self.events, self.kinetics_params))

    RAW_TRACE = "C + residual (raw)"

    @property
    def kinetics_source(self) -> str:
        """``"raw"`` — C + YrA rescaled to ΔF/F (independent of the model's
        kinetics) — when the residual is loaded, else ``"dff"``."""
        return "raw" if self.RAW_TRACE in self.traces else "dff"

    @property
    def kinetics_trace(self):
        """(K, T) trace the kinetic check runs on (see kinetics_source)."""
        if self.dff is None:
            return None
        if self.kinetics_source == "raw":
            return self._cached("kin_trace", lambda: _ek().raw_dff(
                self.C, self.traces[self.RAW_TRACE] - self.C, self.dff))
        return self.dff

    @property
    def kinetics_circular(self) -> bool:
        """True when the check has to use a ΔF/F that is C rescaled (no
        residual in the file): every event then passes by construction."""
        if self.kinetics_source == "raw" or self.dff is None:
            return False
        r = self._cached("dff_C_r", lambda: _ek().dff_tracks_C(self.dff, self.C))
        return bool(r > 0.99)

    @property
    def kinetics_summary(self):
        """(K,) arrays of event_kinetics.SUMMARY_FIELDS, or None."""
        rows = self.kinetics
        return None if rows is None else _ek().summary_table(rows)

    def set_kinetics_params(self, params) -> bool:
        """Adopt new kinetics criteria; True if they changed (cache dropped)."""
        params = None if params is None else dict(params)
        if params == self.kinetics_params:
            return False
        self.kinetics_params = params
        self._derived.pop("kinetics", None)
        return True

    def _kinetics_row(self, i):
        ev = self.events
        return _ek().validate_row(self.kinetics_trace[i], ev.peaks[i], ev.fr,
                                  ev.decay_time, self.kinetics_params)

    def invalidate_corr(self):
        """Drop every cached derived quantity."""
        self._derived = {}

    def components_at(self, row: int, col: int):
        """(components, values) with a non-zero footprint at pixel (row, col)."""
        csr = self._cached("csr", self._A.tocsr)
        p = int(row) + int(col) * int(self.dims[0])
        a, b = csr.indptr[p], csr.indptr[p + 1]
        return csr.indices[a:b], csr.data[a:b]

    # ── History ───────────────────────────────────────────────────────────────

    def _capture(self) -> dict:
        """Snapshot of all mutable component state.

        Footprints, traces, metrics, events, qc and the derived caches are
        only ever replaced by merge / delete / ROI filter / detection, never
        written in place, so the snapshot holds references to them (no copy).
        Only the flags edited in place (hidden, approved) are copied."""
        return {
            "A":         self._A,
            "traces":    dict(self.traces),
            "labels":    list(self.labels),
            "hidden":    self.hidden.copy(),
            "SNR_comp":  self.SNR_comp,
            "r_values":  self.r_values,
            "cnn_preds": self.cnn_preds,
            "events":    self.events,
            "approved":  self.approved.copy(),
            "accepted":  self.accepted,
            "orig_idx":  self.orig_idx,
            "qc":        dict(self.qc) if self.qc is not None else None,
            "derived":   dict(self._derived),
        }

    def _restore(self, snap: dict):
        """Replace mutable state from a snapshot."""
        self._A        = snap["A"]
        self.traces    = snap["traces"]
        self.C         = self.traces[TRACE_DENOISED]
        self.labels    = snap["labels"]
        self.hidden    = snap.get("hidden",
                                  np.zeros(self._A.shape[1], dtype=bool))
        self.SNR_comp  = snap["SNR_comp"]
        self.r_values  = snap["r_values"]
        self.cnn_preds = snap["cnn_preds"]
        self.events    = snap["events"]
        self.approved  = snap["approved"]
        self.accepted  = snap.get("accepted", np.ones(self._A.shape[1], dtype=bool))
        self.orig_idx  = snap["orig_idx"]
        self.qc        = snap.get("qc")
        self._derived  = dict(snap.get("derived", {}))

    def _push_undo(self, op: str):
        """
        Call this BEFORE any destructive mutation.
        Saves the current state onto the undo stack and clears redo history,
        since a new branch has been created.
        """
        self._undo_stack.append((op, self._capture()))
        self._redo_stack.clear()

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_description(self) -> str:
        if not self._undo_stack:
            return "Nothing to undo"
        op, _ = self._undo_stack[-1]
        n = len(self._undo_stack)
        return f"Undo {op}  ({n} step{'s' if n > 1 else ''} available)"

    @property
    def redo_description(self) -> str:
        if not self._redo_stack:
            return "Nothing to redo"
        op, _ = self._redo_stack[-1]
        n = len(self._redo_stack)
        return f"Redo {op}  ({n} step{'s' if n > 1 else ''} available)"

    def undo(self) -> str:
        """
        Restore the state before the last destructive operation.
        The current state is pushed onto the redo stack so it can be
        re-applied.  Returns a human-readable status string.
        """
        if not self._undo_stack:
            return "Nothing to undo"
        op, snap = self._undo_stack.pop()
        self._redo_stack.append((op, self._capture()))
        self._restore(snap)
        n_left = len(self._undo_stack)
        suffix = f"  ({n_left} undo step{'s' if n_left != 1 else ''} left)" if n_left else ""
        return f"Undid {op}{suffix}"

    def redo(self) -> str:
        """
        Re-apply the most recently undone operation.
        The current state is pushed back onto the undo stack.
        Returns a human-readable status string.
        """
        if not self._redo_stack:
            return "Nothing to redo"
        op, snap = self._redo_stack.pop()
        self._undo_stack.append((op, self._capture()))
        self._restore(snap)
        n_left = len(self._redo_stack)
        suffix = f"  ({n_left} redo step{'s' if n_left != 1 else ''} left)" if n_left else ""
        return f"Redid {op}{suffix}"

    # ── Per-component helpers ─────────────────────────────────────────────────

    def fp_pixels(self, i):
        """(flat Fortran-order pixel indices, values) of footprint i."""
        a, b = self._A.indptr[i], self._A.indptr[i + 1]
        return self._A.indices[a:b], self._A.data[a:b]

    def footprint(self, i) -> np.ndarray:
        """Footprint i as a dense (d1, d2) image."""
        out = np.zeros(self._A.shape[0], dtype=np.float32)
        idx, val = self.fp_pixels(i)
        out[idx] = val
        return out.reshape(self.dims, order="F")

    def max_a(self, i):   return float(self.peaks[i])
    def peak_df(self, i): return float(self.C[i].max() - self.C[i].min())
    def snr(self, i):     return float(self.SNR_comp[i])  if self.SNR_comp  is not None else float("nan")
    def rval(self, i):    return float(self.r_values[i])  if self.r_values  is not None else float("nan")
    def cnn(self, i):     return float(self.cnn_preds[i]) if self.cnn_preds is not None else float("nan")

    # ── Mutation ──────────────────────────────────────────────────────────────

    def merge(self, indices):
        """
        Merge components at `indices` into one, appended at the end.

        Footprint and denoised trace are a rank-1 non-negative fit of the
        constituents' summed signal Σᵢ aᵢcᵢᵀ (:func:`_merge_fit`), so the
        merged unit reproduces what the parts explained together.  Other
        linear traces combine with the same weights; ΔF/F is rebuilt from
        the merged signal over the merged baseline (:func:`_merge_dff`).
        SNR / r-value / CNN can't be recomputed here and become NaN
        (unknown); event detection and the ROI filter skip unknown metrics.
        Returns the new component's index.
        """
        self._push_undo("merge")
        idx = sorted(indices)
        new_a, beta = _merge_fit(self._A[:, idx], self.C[idx])
        new_label = "+".join(self.labels[i] for i in idx)

        keep = [j for j in range(self.n) if j not in set(idx)]
        old, self._derived = self._derived, {}
        A_parts = self._A[:, idx]
        self._A     = _to_sparse(scipy.sparse.hstack([self._A[:, keep], new_a]))
        for name, arr in self.traces.items():
            if name == TRACE_DFF:
                new_row = _merge_dff(arr[idx], self.C[idx], A_parts, new_a, beta)
            else:
                new_row = beta @ arr[idx]
            self.traces[name] = np.vstack([arr[keep], new_row[None, :].astype(arr.dtype)])
        self.C      = self.traces[TRACE_DENOISED]
        self.labels = [self.labels[j] for j in keep] + [new_label]
        # Merged component is hidden only if every constituent was hidden.
        self.hidden = np.append(self.hidden[keep], bool(self.hidden[idx].all()))

        for attr in ("SNR_comp", "r_values", "cnn_preds"):   # unknown for a merge
            arr = getattr(self, attr)
            if arr is not None:
                setattr(self, attr, np.append(arr[keep], np.nan))
        # Merged component is included if any constituent was; it has no
        # source row.  Its events are re-detected on the merged traces.
        self.approved = np.append(self.approved[keep], bool(self.approved[idx].any()))
        # accepted (census: active) if any constituent was
        self.accepted = np.append(self.accepted[keep], bool(self.accepted[idx].any()))
        self.orig_idx = np.append(self.orig_idx[keep], -1)
        if self.events is not None:
            self.events = self.events.take(keep).append(self._detect_row(self.n - 1))
        if self.qc is not None:
            self.qc = _rf().append(_rf().take(self.qc, keep), self._qc_row(self.n - 1))
        self._derived = self._derived_after(old, keep, merged=True)
        return self.n - 1

    def delete(self, indices):
        self._push_undo("delete")
        keep = [j for j in range(self.n) if j not in set(indices)]
        old, self._derived = self._derived, {}
        self._A     = self._A[:, keep]
        for name, arr in self.traces.items():
            self.traces[name] = arr[keep]
        self.C      = self.traces[TRACE_DENOISED]
        self.labels = [self.labels[j] for j in keep]
        self.hidden = self.hidden[keep]
        for attr in ("SNR_comp", "r_values", "cnn_preds"):
            arr = getattr(self, attr)
            if arr is not None:
                setattr(self, attr, arr[keep])
        self.approved = self.approved[keep]
        self.accepted = self.accepted[keep]
        self.orig_idx = self.orig_idx[keep]
        if self.events is not None:
            self.events = self.events.take(keep)
        if self.qc is not None:
            self.qc = _rf().take(self.qc, keep)
        self._derived = self._derived_after(old, keep, merged=False)

    def _derived_after(self, old: dict, keep, merged: bool) -> dict:
        """Carry cached corr / centroids / peaks over a delete (rows `keep`)
        or a merge (rows `keep` + one new last row) without recomputing
        the K×K matrices.  Distances and the pixel lookup are rebuilt."""
        keep = np.asarray(keep, dtype=np.int64)
        new  = {}
        if old.get("corr") is not None:
            sub = old["corr"][np.ix_(keep, keep)]
            if merged:
                k   = self.n - 1
                row = _active_pearson_row(self.C, k)
                mat = np.empty((k + 1, k + 1), dtype=np.float32)
                mat[:k, :k] = sub
                mat[k, :] = row; mat[:, k] = row
                sub = mat
            new["corr"] = sub
        if old.get("kinetics") is not None and self.events is not None \
                and self.dff is not None:
            ks = [old["kinetics"][j] for j in keep]
            if merged:
                ks.append(self._kinetics_row(self.n - 1))
            new["kinetics"] = ks
        for key, fn in (("centroids", _centroids),
                        ("peaks", lambda A, d: A.max(axis=0).toarray().ravel())):
            if old.get(key) is not None:
                v = old[key][keep]
                if merged:
                    v = np.concatenate([v, fn(self._A[:, -1:], self.dims)])
                new[key] = v
        return new

    # ── ROI filter ────────────────────────────────────────────────────────────

    def _qc_row(self, i):
        """roi_filter fields for row i alone (a merged component), computed
        with the file's filter parameters.  Fields the row's data cannot
        provide are filled as unknown (NaN / 0 / False)."""
        rf = _rf()
        try:
            one = rf.evaluate(
                self._A[:, i:i + 1], self.dims,
                SNR_comp=self._metric_row("SNR_comp", i),
                r_values=self._metric_row("r_values", i),
                cnn_preds=self._metric_row("cnn_preds", i),
                F_dff=None if self.dff is None else self.dff[i:i + 1],
                C=self.C[i:i + 1],
                params=rf.resolve_params(self.qc_params))
        except Exception as exc:
            print(f"[roi_filter] merged row not evaluated: {exc}", file=sys.stderr)
            one = {}
        row = {}
        for k, v in self.qc.items():
            if k in one:
                row[k] = np.asarray(one[k], dtype=v.dtype)
            else:
                fill = (np.nan if np.issubdtype(v.dtype, np.floating)
                        else False if v.dtype == np.bool_ else 0)
                row[k] = np.array([fill], dtype=v.dtype)
        return row

    def _metric_row(self, attr: str, i: int):
        """``attr[i:i+1]``, or None when absent or NaN (a merged row) so the
        criterion is skipped instead of failing."""
        arr = getattr(self, attr)
        if arr is None or not np.isfinite(arr[i]):
            return None
        return arr[i:i + 1]

    def run_roi_filter(self, cfg: dict, set_incl: bool = False) -> dict:
        """Evaluate every row with the ROI filter (undoable).

        ``cfg`` is a roi_filter section (preferences); nulls are filled from
        the loaded run's CaImAn params, then from the store (fr, decay
        time).  ``set_incl`` resets the include flags to the QC verdict.
        Returns the summary."""
        rf = _rf()
        p  = self.resolve_filter_params(cfg)
        fields = rf.evaluate(self._A, self.dims,
                             SNR_comp=self.SNR_comp, r_values=self.r_values,
                             cnn_preds=self.cnn_preds, F_dff=self.dff, C=self.C,
                             params=p)
        self._push_undo("ROI filter")
        self.qc, self.qc_params = fields, p
        if set_incl:
            self.approved = np.asarray(fields["passed"], bool).copy()
        return rf.summarize(fields)

    def resolve_filter_params(self, cfg: dict) -> dict:
        rf = _rf()
        cfg = dict(cfg)
        g = cfg.get("gSig")
        if g is not None and np.ndim(g) == 0:
            cfg["gSig"] = [float(g), float(g)]
        p = rf.resolve_params(cfg, getattr(self.cnm_obj, "params", None))
        if p.get("fr") is None and self.fr:
            p["fr"] = float(self.fr)
        if p.get("decay_time") is None and self.decay_time:
            p["decay_time"] = float(self.decay_time)
        return p

    @property
    def has_approval(self) -> bool:
        """Include flags exist once events are detected or a roi_filter
        group seeded them."""
        return self.events is not None or self.qc is not None

    # ── Events / approval ─────────────────────────────────────────────────────

    @property
    def dff(self):
        """ΔF/F traces (K, T) or None — the input event detection needs."""
        return self.traces.get(TRACE_DFF)

    @property
    def can_detect(self) -> bool:
        return self.dff is not None

    def detect_events(self, fr, decay_time, progress=None, **kw):
        """Run SLab event detection on every row; approval := auto-approval
        (undoable).  Rows with an unknown (NaN) SNR / CNN — merged
        components — are detected without that criterion."""
        if not self.can_detect:
            raise RuntimeError("No ΔF/F traces (estimates.F_dff) in this file — "
                               "run cnm.estimates.detrend_df_f() first.")
        ed = _ed()
        res = ed.detect_events(self.dff, self.C, fr, decay_time,
                               self.SNR_comp, self.cnn_preds, progress=progress, **kw)
        unknown = np.zeros(self.n, bool)
        for arr in (self.SNR_comp, self.cnn_preds):
            if arr is not None:
                unknown |= ~np.isfinite(arr)
        for r in np.flatnonzero(unknown):           # re-run without NaN criteria
            one = ed.detect_events(
                self.dff[r:r + 1], self.C[r:r + 1], fr, decay_time,
                self._metric_row("SNR_comp", r), self._metric_row("cnn_preds", r), **kw)
            for f in ("binary", "filtered", "thresholds", "frequencies", "auto_approved"):
                getattr(res, f)[r] = getattr(one, f)[0]
            res.peaks[r] = one.peaks[0]
        self._push_undo("detect events")
        self._derived.pop("kinetics", None)
        self.events = res
        self.approved = self.events.auto_approved.copy()
        self.fr, self.decay_time = float(fr), float(decay_time)

    def _detect_row(self, i):
        """EventResults for row i alone, using the current events' settings
        (a blank row if ΔF/F is unavailable)."""
        ev = self.events
        if self.can_detect:
            p = ev.params
            one = _ed().detect_events(
                self.dff[i:i + 1], self.C[i:i + 1], ev.fr, ev.decay_time,
                self._metric_row("SNR_comp", i), self._metric_row("cnn_preds", i),
                threshold=ev.threshold, min_snr=p.get("min_snr", 3.0),
                min_cnn=p.get("min_cnn", 0.15), lowpass_hz=p.get("lowpass_hz"))
            one.params = dict(p)
            return one
        blank = ev.take([0]) if ev.n else ev
        blank.binary[:] = 0; blank.filtered[:] = 0
        blank.thresholds[:] = 0; blank.frequencies[:] = 0
        blank.peaks = [np.zeros((0, 4))]; blank.auto_approved[:] = False
        return blank

    def toggle_approved(self, indices):
        for i in indices:
            self.approved[i] = not self.approved[i]

    @property
    def n_approved(self) -> int:
        return int(self.approved.sum())

    def attach_events(self, res, approved):
        """Adopt file-level stroh_results, mapped through orig_idx.

        Rows whose source index is outside the stored results (or merged
        rows) are re-detected when possible.  Returns False if the stored
        results don't fit this file (row count mismatch)."""
        oi = self.orig_idx
        self._derived.pop("kinetics", None)
        if res.n == self.n and np.array_equal(oi, np.arange(self.n)):
            self.events, self.approved = res, np.asarray(approved, bool).copy()
        elif oi.size and oi.max() < res.n:
            ok = oi >= 0
            if not ok.all():
                return False
            self.events = res.take(oi)
            self.approved = np.asarray(approved, bool)[oi].copy()
        else:
            return False
        if not np.isfinite(self.events.fr) and self.fr:
            self.events.fr = float(self.fr)
        if not np.isfinite(self.events.decay_time) and self.decay_time:
            self.events.decay_time = float(self.decay_time)
        self.fr = self.events.fr
        self.decay_time = self.events.decay_time
        return True

    # ── Save ──────────────────────────────────────────────────────────────────

    @staticmethod
    def curated_path(src_path: str) -> str:
        """Insert ``_curated`` before the extension so the original is never
        overwritten.  Idempotent: calling twice gives the same result.

        Examples::

            results.hdf5          -> results_curated.hdf5
            results_curated.hdf5  -> results_curated.hdf5
        """
        base, ext = os.path.splitext(src_path)
        if base.endswith("_curated"):
            return src_path
        return base + "_curated" + ext

    def save(self, path: str):
        """Write the curated A/C back via cnm_obj.save(), which uses
        CaImAn's own fn_relocated + save_dict_to_hdf5 serialisation path.

        Derived per-component fields (S, R, g, …) are nulled out because they
        are invalid after manual edits and must be recomputed; F_dff and the
        residual YrA are written from the edited traces.
        Every row is written; idx_components / idx_components_bad keep the
        accepted / rejected split (census: active / inactive, and the census
        dict is updated).  A provenance entry is appended.
        """
        import time as _time
        if self.cnm_obj is None:
            raise RuntimeError("No CaImAn object available -- cannot save.")

        est = self.cnm_obj.estimates
        K   = self.n

        # Spatial / temporal
        est.A  = self._A.copy()
        est.C  = self.C
        est.nr = K

        # Null stale K-indexed derived fields.  ΔF/F is carried through
        # merge/delete in the trace registry (merged rows = mean of their
        # constituents' ΔF/F); it is written back rather than nulled because
        # event detection and SLabAnalysisApp both need it.
        dff = self.dff
        for field in ("S", "YrA", "R", "F_dff",
                      "g", "bl", "c1", "neurons_sn", "lam",
                      "coordinates", "A_thr"):
            if getattr(est, field, None) is not None:
                setattr(est, field, None)
        if dff is not None:
            est.F_dff = dff
        # The residual stays (merged rows: the parts' β-weighted residual),
        # so the raw trace C + YrA — and the kinetic check — survive Save.
        if self.RAW_TRACE in self.traces:
            est.YrA = self.traces[self.RAW_TRACE] - self.C

        # Quality metrics
        est.SNR_comp  = self.SNR_comp
        est.r_values  = self.r_values
        est.cnn_preds = self.cnn_preds

        # Every row is written; the accepted / rejected split (census:
        # active / inactive) is kept in idx_components / idx_components_bad.
        acc = self.accepted if self.eval_known else np.ones(K, dtype=bool)
        est.idx_components     = np.flatnonzero(acc).astype(int)
        est.idx_components_bad = np.flatnonzero(~acc).astype(int)
        if isinstance(getattr(est, "census", None), dict):
            est.census = {
                "n_total":      K,
                "n_active":     int(acc.sum()),
                "n_inactive":   int((~acc).sum()),
                "idx_active":   np.flatnonzero(acc).tolist(),
                "idx_inactive": np.flatnonzero(~acc).tolist(),
            }

        # Append provenance record
        if hasattr(self.cnm_obj, "provenance"):
            self.cnm_obj.provenance.append({
                "event":        "curated",
                "time":         int(_time.time()),
                "description":  "Manual curation via cnmf_inspector",
                "n_components": K,
                "n_accepted":   int(acc.sum()),
                "n_approved":   self.n_approved,
                "n_merged_dff_rows": int((self.orig_idx < 0).sum()),
                "source_file":  self.fpath or "",
                "save_path":    path,
            })

        # Delegate to CaImAn's own serialiser
        self.cnm_obj.save(path)
        self.fpath = path
        self.orig_idx = np.arange(K)          # rows now index the saved file

        # Events + approvals as an SLab stroh_results group, rows = saved rows;
        # roi_filter fields (merged rows re-evaluated) with the include flags
        if self.events is not None or self.qc is not None:
            import h5py
            with h5py.File(path, "a") as f:
                if self.events is not None:
                    _ed().write_stroh_results(f, self.events, self.approved)
                if self.qc is not None:
                    _rf().write_group(f, self.qc, self.qc_params,
                                      manually_approved=self.approved)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_sparse(A) -> scipy.sparse.csc_matrix:
    """(d, K) footprints as canonical CSC float32 (sorted, no explicit 0)."""
    A = scipy.sparse.csc_matrix(A, dtype=np.float32)
    A.eliminate_zeros()
    A.sort_indices()
    return A


def _merge_fit(A_parts, C_parts, n_iter: int = 30):
    """Rank-1 non-negative fit a·cᵀ of Y = Σᵢ aᵢcᵢᵀ (never formed).

    Returns ``(a, beta)``: the merged footprint (d × 1 sparse), scaled to the
    largest constituent's norm, and weights with c = beta @ C_parts
    (betaᵢ = a·aᵢ / ‖a‖²) — exact for the final trace update.  Falls back to
    the pixel-wise max footprint and the mean trace when the parts carry
    no signal.
    """
    A_parts = scipy.sparse.csc_matrix(A_parts, dtype=np.float64)
    Cp = np.asarray(C_parts, dtype=np.float64)
    m = A_parts.shape[1]
    norms = np.sqrt(np.asarray(A_parts.multiply(A_parts).sum(axis=0)).ravel())
    cmax = np.maximum(Cp.max(axis=1), 0)
    if not (norms.max() > 0 and cmax.max() > 0):
        return A_parts.max(axis=1).astype(np.float32), np.full(m, 1.0 / m)
    G = (A_parts.T @ A_parts).toarray()             # m × m footprint Gram
    w = cmax.copy()                                 # a = A_parts @ w
    for _ in range(n_iter):
        aa = w @ G @ w
        beta = (G @ w) / aa                         # c = beta @ Cp
        c = np.maximum(beta @ Cp, 0)
        cc = c @ c
        if cc <= 0:
            break
        w_new = (Cp @ c) / cc                       # a = A_parts @ w_new (≥ 0)
        if np.allclose(w_new, w, rtol=1e-10, atol=0):
            w = w_new
            break
        w = w_new
    a = np.asarray(A_parts @ w).ravel()
    a = np.maximum(a, 0)
    scale = norms.max() / np.linalg.norm(a)         # footprint at constituent scale
    a *= scale
    beta = (A_parts.T @ a) / (a @ a)                # c = beta @ Cp for this a
    return scipy.sparse.csc_matrix(a[:, None].astype(np.float32)), np.asarray(beta).ravel()


def _merge_dff(dff_parts, C_parts, A_parts, a_new, beta) -> np.ndarray:
    """ΔF/F of a merged unit from its parts.

    Each part's baseline F0ᵢ is fitted from ΔF/Fᵢ ≈ cᵢ / F0ᵢ, turned into a
    per-pixel background b0ᵢ = F0ᵢ‖aᵢ‖² / Σaᵢ (CaImAn's projection of a
    locally flat background), and the merged baseline is the projection of
    that background onto the merged footprint.  The merged signal
    Σᵢ βᵢ F0ᵢ ΔF/Fᵢ keeps each part's residual.  Parts without a usable
    baseline fall back to the β-weighted mean of their ΔF/F.
    """
    D  = np.asarray(dff_parts, dtype=np.float64)
    Cp = np.asarray(C_parts, dtype=np.float64)
    A_parts = scipy.sparse.csc_matrix(A_parts, dtype=np.float64)
    a = np.asarray(a_new.toarray() if scipy.sparse.issparse(a_new) else a_new,
                   dtype=np.float64).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        F0 = (Cp * Cp).sum(1) / (Cp * D).sum(1)
    sum_a = np.asarray(A_parts.sum(axis=0)).ravel()
    nrm2  = np.asarray(A_parts.multiply(A_parts).sum(axis=0)).ravel()
    ok = np.isfinite(F0) & (F0 > 0) & (sum_a > 0) & (a @ a > 0)
    if not ok.all():
        wb = np.where(beta > 0, beta, 0)
        return (wb @ D) / wb.sum() if wb.sum() > 0 else D.mean(0)
    b0 = F0 * nrm2 / sum_a                          # per-pixel background
    b0_bar = (b0 * sum_a).sum() / sum_a.sum()
    F0_new = b0_bar * a.sum() / (a @ a)
    return (beta * F0) @ D / F0_new


def _centroids(A, dims) -> np.ndarray:
    """(K, 2) footprint-weighted (row, col) of CSC footprints (Fortran
    pixel order); the FOV centre for an empty footprint."""
    d1, d2 = int(dims[0]), int(dims[1])
    p = np.arange(A.shape[0])
    total = np.asarray(A.sum(axis=0)).ravel().astype(np.float64)
    cy = np.asarray(A.T @ (p % d1).astype(np.float64)).ravel()
    cx = np.asarray(A.T @ (p // d1).astype(np.float64)).ravel()
    ok = total > 1e-9
    out = np.empty((A.shape[1], 2), dtype=np.float32)
    out[:, 0] = np.where(ok, cy / np.where(ok, total, 1), d1 / 2.0)
    out[:, 1] = np.where(ok, cx / np.where(ok, total, 1), d2 / 2.0)
    return out


def _opt_arr(x):
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    return arr if arr.size > 0 else None   # treat empty arrays as absent


def _has_contrast(img, eps: float = 1e-6) -> bool:
    """True if an image has usable greyscale contrast.

    Denoised data can saturate the correlation image to a near-constant
    value (Cn ~ 1.0); such an image makes a useless background.
    """
    a = np.asarray(img, dtype=np.float32)
    if a.size == 0 or not np.isfinite(a).any():
        return False
    finite = a[np.isfinite(a)]
    ptp = float(finite.max() - finite.min())
    return ptp > eps and float(finite.std()) > eps


# ── Correlation helper ───────────────────────────────────────────────────────

def _active_pearson_corr(C: np.ndarray,
                         active_thr: float = 0.0) -> np.ndarray:
    """
    Pairwise Pearson correlation computed only on *active* frames.

    For each pair (i, j) the active frame mask is:

        mask_ij = (C[i] > active_thr) | (C[j] > active_thr)

    Pearson r is then evaluated on C[i][mask_ij] vs C[j][mask_ij].
    If fewer than 3 active frames exist for a pair, that cell is set
    to NaN rather than returning a spurious value.

    Why not plain Pearson on all frames?
    ─────────────────────────────────────
    CaImAn traces have long zero-valued (or near-zero) baseline epochs.
    Including those frames inflates the apparent correlation between any
    two cells that happen to share a quiet period, and suppresses the
    true co-activation signal relative to the noise.

    Why not Spearman?
    ─────────────────
    With sparse, non-negative traces, the baseline zeros all receive the
    same average rank.  If one trace has a true zero baseline while
    another has small noise at baseline, the zero-rank plateau is broken
    in one trace but not the other, drastically lowering Spearman r even
    between near-duplicate components.  Masked Pearson on active frames
    avoids this by simply ignoring the silent epochs entirely.

    Parameters
    ----------
    C          : (K, T) float32 trace matrix
    active_thr : threshold above which a frame is considered active.
                 Defaults to 0.0, matching CaImAn's non-negative OASIS
                 output (traces are exactly 0 during silent epochs).

    Returns
    -------
    (K, K) float32 symmetric matrix; diagonal is NaN.
    """
    return _masked_pearson(C, C, active_thr)


def _masked_pearson(X: np.ndarray, Y: np.ndarray,
                    active_thr: float = 0.0,
                    block_bytes: float = 64e6) -> np.ndarray:
    """(Kx, Ky) Pearson r of X[i] vs Y[j] over frames where either is
    active (> active_thr); NaN with < 3 such frames or a flat trace.

    Every masked sum splits into frames where x is active plus frames
    where only y is (Σₘ xy = (xa)·y + (x − xa)·(yb)), so a block of the
    matrix is seven matrix products instead of a Python pair loop.  Rows
    are processed in blocks so the float64 work arrays stay near
    ``block_bytes`` whatever K × T is; when X is Y only the upper block
    triangle is computed and the diagonal is NaN.
    """
    Kx, T = X.shape
    Ky    = Y.shape[0]
    out   = np.full((Kx, Ky), np.nan, dtype=np.float32)
    if Kx == 0 or Ky == 0 or T < 3:
        return out
    same = X is Y
    step = int(max(16, block_bytes // (6 * 8 * T)))   # ~6 (rows × T) f64 arrays

    def parts(M, lo, hi):
        """Per-block work arrays: mask, values on / off it, squares."""
        m  = np.asarray(M[lo:hi], dtype=np.float64)
        a  = (M[lo:hi] > active_thr).astype(np.float64)
        on = m * a
        off = m - on
        return a, on, off, (on * m).sum(1), off * m, on.sum(1), m

    for ylo in range(0, Ky, step):
        yhi = min(ylo + step, Ky)
        b, ya, yo, syy_on, yo2, sy_on, y = parts(Y, ylo, yhi)
        for xlo in range(0, (yhi if same else Kx), step):
            xhi = min(xlo + step, Kx)
            if same and xlo == ylo:
                a, xa, xo, sxx_on, xo2, sx_on, x = b, ya, yo, syy_on, yo2, sy_on, y
            else:
                a, xa, xo, sxx_on, xo2, sx_on, x = parts(X, xlo, xhi)
            # Σ over a ∨ b = Σ over a  +  Σ over b where a is off
            n   = a.sum(1)[:, None] + (1 - a) @ b.T
            sx  = sx_on[:, None] + xo @ b.T
            sy  = sy_on[None, :] + a @ yo.T
            sxx = sxx_on[:, None] + xo2 @ b.T
            syy = syy_on[None, :] + a @ yo2.T
            sxy = xa @ y.T + xo @ ya.T
            with np.errstate(divide="ignore", invalid="ignore"):
                vx  = sxx - sx * sx / n
                vy  = syy - sy * sy / n
                r   = (sxy - sx * sy / n) / np.sqrt(vx * vy)
            bad = (n < 2.5) | (vx < 1e-24) | (vy < 1e-24) | ~np.isfinite(r)
            r = np.clip(np.where(bad, np.nan, r), -1.0, 1.0).astype(np.float32)
            out[xlo:xhi, ylo:yhi] = r
            if same:
                out[ylo:yhi, xlo:xhi] = r.T
    if same and Kx == Ky:
        np.fill_diagonal(out, np.nan)
    return out


def _active_pearson_row(C: np.ndarray, k: int,
                        active_thr: float = 0.0) -> np.ndarray:
    """Row k of :func:`_active_pearson_corr` (same masking), NaN at k."""
    row = _masked_pearson(C[k:k + 1], C, active_thr)[0]
    row[k] = np.nan
    return row


__all__ = ['ComponentStore', '_to_sparse', '_centroids', '_opt_arr', '_has_contrast', '_active_pearson_corr', '_masked_pearson', '_active_pearson_row']
