"""InspectorWindow: ROI filter, event detection, export, session info, batch detection and blinded review."""

import os
import sys

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
)

from .deps import _ed, _ek, _rf
from .prefs import kinetics_prefs, roi_filter_prefs
from .loading import _os_dirname, _session_prefix, load_from_hdf5
from .review import blind_plan, export_event_tables, write_blind_rating
from .dialogs import RunFilterDialog


class _AnalysisMixin:
    """Methods of :class:`~utilities.inspector.window.InspectorWindow`."""

    def _run_filter_dialog(self):
        if self._blind_block():
            return
        s = self.store
        dlg = RunFilterDialog(self, s)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        sel = self._selected_rows()
        try:
            summ = s.run_roi_filter(roi_filter_prefs(), set_incl=dlg.set_incl)
        except Exception as exc:
            QMessageBox.critical(self, "Run ROI filter", str(exc))
            return
        self._full_refresh()
        self._reselect(sel)
        msg = f"ROI filter: {summ.get('n_passed', 0)}/{summ['n_total']} passed"
        if "n_active" in summ:
            msg += f", {summ['n_active']} active"
        fails = {k: v for k, v in summ.get("n_fail", {}).items() if v}
        if fails:
            msg += "  (fails: " + ", ".join(f"{k} {v}" for k, v in fails.items()) + ")"
        self.statusBar().showMessage(msg + "  —  Save to keep")

    def _blind_block(self) -> bool:
        if self._blind is not None:
            self.statusBar().showMessage(
                "Blinded review in progress — Analysis ▸ End blinded review first")
            return True
        return False

    def run_detection(self, fr, decay_time, progress=None, **kw):
        """Detect events on the current store and refresh, keeping the
        selection.  Raises on failure (callers report it)."""
        sel = self._selected_rows()
        self.store.detect_events(fr, decay_time, progress=progress, **kw)
        self._full_refresh()
        self._reselect(sel)
        ev = self.store.events
        rule = ""
        if ev.auto_slab is not None:
            rule = f" (SLab {int(ev.auto_slab.sum())}; kinetics rule)"
        elif self.store.kinetics_rule_skipped:
            rule = (" — ⚠ kinetics rule not applied: no residual and ΔF/F ≈ C, "
                    "the check would be circular")
        self.statusBar().showMessage(
            f"Events: {sum(len(p) for p in ev.peaks)} in {ev.n} components; "
            f"{int(ev.auto_approved.sum())} auto-approved / included{rule}")

    def _detect_events_dialog(self):
        if self._blind_block():                # the rating must use the file's events
            return
        s = self.store
        if not s.can_detect:
            QMessageBox.warning(
                self, "Detect events",
                "This file has no ΔF/F traces (estimates.F_dff).\n"
                "Run cnm.estimates.detrend_df_f() in the pipeline first.")
            return
        if s.has_approval and QMessageBox.question(
                self, "Detect events",
                ("Existing events and include flags will be replaced "
                 "(Edit ▸ Undo restores them)."
                 if s.events is not None else
                 "Include flags (seeded from the ROI filter) will be replaced "
                 "by event auto-approval (Edit ▸ Undo restores them).")
                + "\nContinue?"
                ) != QMessageBox.StandardButton.Yes:
            return
        ev, prm = s.events, (s.events.params if s.events else {})
        dlg  = QDialog(self)
        dlg.setWindowTitle("Detect events")
        form = QFormLayout(dlg)

        def spin(val, lo, hi, step, dec, suffix=""):
            w = QDoubleSpinBox(dlg)
            w.setRange(lo, hi); w.setDecimals(dec); w.setSingleStep(step)
            w.setValue(val); w.setSuffix(suffix)
            return w
        sp_fr  = spin(s.fr or 30.0, 0.1, 1000.0, 0.1, 2, " Hz")
        sp_dec = spin(s.decay_time or 0.4, 0.0, 5.0, 0.01, 3, " s")
        sp_thr = spin(ev.threshold if ev else 0.2, 0.01, 2.0, 0.01, 3)
        sp_snr = spin(prm.get("min_snr", 3.0), 0.0, 100.0, 0.5, 2)
        sp_cnn = spin(prm.get("min_cnn", 0.15), 0.0, 1.0, 0.05, 2)
        sp_lp  = spin(prm.get("lowpass_hz") or 0.0, 0.0, 500.0, 0.5, 2, " Hz")
        sp_lp.setSpecialValueText("SLab (0.974·Nyquist)")
        kin_cfg = kinetics_prefs()
        cb_kin = QCheckBox("Require valid kinetics", dlg)
        cb_kin.setChecked(bool(kin_cfg.get("approve_on_kinetics")))
        cb_kin.setToolTip(
            "Auto-approve only components SLab approves that also have "
            f"≥ {kin_cfg.get('min_valid_events', 1)} kinetically valid event(s) on "
            "the raw trace (criteria: Preferences ▸ Event kinetics).  Off = SLab.")
        for label, w, tip in (
                ("Imaging frequency", sp_fr, "Frame rate; sets refractory "
                 "distance and events/min."),
                ("Indicator decay time", sp_dec, "Minimum peak spacing = "
                 "decay time × frame rate."),
                ("Max threshold (ΔF/F)", sp_thr, "Cap on the noise-derived "
                 "threshold min(10·noise, this)."),
                ("Min SNR", sp_snr, "Components below are skipped and "
                 "excluded."),
                ("Min CNN", sp_cnn, "Auto-exclude below (ignored when the "
                 "file has no CNN scores)."),
                ("Low-pass cutoff", sp_lp, "0 = SLab's normalised cutoff "
                 "(bit-identical to SLabAnalysisApp).")):
            w.setToolTip(tip)
            form.addRow(label, w)
        form.addRow("Approval", cb_kin)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        prog = QProgressDialog("Detecting events…", None, 0, s.n, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(300)

        def cb(i, n):
            prog.setMaximum(n); prog.setValue(i)
            QApplication.processEvents()
        try:
            self.run_detection(sp_fr.value(), sp_dec.value(), progress=cb,
                               threshold=sp_thr.value(), min_snr=sp_snr.value(),
                               min_cnn=sp_cnn.value(),
                               lowpass_hz=sp_lp.value() or None,
                               kinetics=dict(kin_cfg, approve_on_kinetics=cb_kin.isChecked()))
        except Exception as exc:
            QMessageBox.critical(self, "Detect events", str(exc))
        finally:
            prog.close()

    def _export_events(self):
        s = self.store
        if s.events is None:
            self.statusBar().showMessage("No events to export — detect first")
            return
        folder = _os_dirname(s.fpath) if s.fpath else os.getcwd()
        prefix = _session_prefix(s.fpath) if s.fpath else "inspector"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export event stats (PeakStats; ROIStats is written alongside)",
            os.path.join(folder, f"{prefix}_PeakStats.xlsx"),
            "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        try:
            out = export_event_tables(s, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            return
        self.statusBar().showMessage(
            "Exported " + ", ".join(os.path.basename(p) for p in out)
            + f"  ({s.n_approved} included components)")

    def _session_info(self):
        s, ev = self.store, self.store.events
        snr = (s.SNR_comp[np.isfinite(s.SNR_comp)]
               if s.SNR_comp is not None else np.array([]))
        T   = s.C.shape[1]
        fr  = ev.fr if ev is not None else s.fr
        dur = f"{T} frames" + (f" ({T / fr / 60:.1f} min)" if fr else "")
        name = ("(hidden in blinded review)" if self._blind is not None
                else (s.fpath or "(demo)"))
        lines = [
            f"File:  {name}",
            f"Components:  {s.n}   (hidden {s.n_hidden})",
            ("CaImAn evaluation:  "
             f"{s.n_accepted} accepted / {s.n - s.n_accepted} rejected "
             "(census: active / inactive)") if s.eval_known and self._blind is None
            else "CaImAn evaluation:  " + ("hidden" if self._blind else "n/a"),
            f"Recording length:  {dur}",
            "SNR min / mean / max:  " + (
                f"{snr.min():.2f} / {snr.mean():.2f} / {snr.max():.2f}"
                if snr.size else "n/a"),
            f"Imaging frequency:  {fr if fr else 'n/a'}",
            f"Indicator decay time:  "
            f"{(ev.decay_time if ev is not None else s.decay_time) or 'n/a'}",
        ]
        if s.qc is not None and self._blind is None:
            qc = s.qc
            line = "ROI filter:  "
            if "passed" in qc:
                line += f"{int(qc['passed'].sum())}/{s.n} passed"
            if "active" in qc:
                line += f",  {int(qc['active'].sum())} active"
            lines.append(line)
            if "fail_reasons" in qc:
                rf = _rf()
                fails = {n: int(((qc["fail_reasons"] & b) != 0).sum())
                         for n, b in rf.REASONS.items()}
                fails = {n: c for n, c in fails.items() if c}
                if fails:
                    lines.append("   fails:  " + ",  ".join(
                        f"{n} {c}" for n, c in fails.items()))
            if ev is None:
                lines.append(f"Included:  {s.n_approved}")
        if ev is not None and s.dff is not None:
            lines.append("Kinetics trace:  " + (
                "C + residual (raw), on the ΔF/F scale" if s.kinetics_source == "raw"
                else "ΔF/F ≈ C (no residual) — circular, every event passes"
                if s.kinetics_circular else "ΔF/F (file has no residual YrA)"))
        if ev is not None:
            lines.append("Approval rule:  " + _ek().describe_rule(ev.params))
            if ev.auto_slab is not None:
                lines.append(f"SLab auto-approved:  {int(ev.auto_slab.sum())}")
            lines += [
                f"Auto-approved:  {int(ev.auto_approved.sum())}",
                f"Included (manual):  {s.n_approved}",
                f"Events:  {sum(len(p) for p in ev.peaks)}",
                f"Detection threshold cap:  {ev.threshold}",
                f"Analysis version:  {ev.version}",
            ]
        else:
            lines.append("Analysis version:  not analysed")
        QMessageBox.information(self, "Session info", "\n".join(lines))

    def _batch_detect(self):
        if self._blind_block():
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder of results files",
                                                  self._start_dir())
        if not folder:
            return
        self._remember_dir(folder)
        ed    = _ed()
        files = ed.find_results_files(folder)
        if not files:
            QMessageBox.information(self, "Batch detect", "No .hdf5/.h5 files found.")
            return
        if len(files) <= 10:
            listing = "\n".join(files)
        else:
            from collections import Counter
            listing = "\n".join(f"{d}: {n} files" for d, n in
                                 Counter(os.path.dirname(f) for f in files).items())
        ans = QMessageBox.question(
            self, "Batch detect events",
            f"{len(files)} files:\n\n{listing}\n\n"
            "stroh_results is written into each file (estimates untouched).\n"
            + ("Auto-approval requires valid kinetics (Preferences ▸ Event "
               "kinetics).\n" if kinetics_prefs().get("approve_on_kinetics") else "")
            +
            "Overwrite files that already have results?\n"
            "Yes = overwrite · No = skip those · Cancel = abort",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Cancel:
            return
        kin = kinetics_prefs()
        kin = kin if kin.get("approve_on_kinetics") else None
        prog = QProgressDialog("Detecting events…", "Cancel", 0, len(files), self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        log = []
        for k, f in enumerate(files):
            if prog.wasCanceled():
                break
            prog.setLabelText(f"{os.path.basename(f)}  ({k + 1}/{len(files)})")
            QApplication.processEvents()
            try:
                msg = ed.detect_events_in_hdf5(
                    f, overwrite=(ans == QMessageBox.StandardButton.Yes), kinetics=kin)
            except Exception as exc:
                msg = f"ERROR {exc}"
            log.append(f"{os.path.relpath(f, folder)}: {msg}")
            prog.setValue(k + 1)
        prog.close()
        box = QMessageBox(self)
        box.setWindowTitle("Batch detect events")
        box.setText(f"Processed {len(log)}/{len(files)} files.")
        box.setDetailedText("\n".join(log))
        if self.store.fpath and os.path.abspath(self.store.fpath) in map(os.path.abspath, files):
            box.setInformativeText("The open file was processed on disk — "
                                   "reopen it to see its new events.")
        box.exec()

    # ── Blinded review (SLab Randomization Mode) ──────────────────────────────

    def start_blinded(self, rater: str, files: list):
        """Begin a blinded review of ``files`` (already filtered/shuffled)."""
        self._blind = dict(rater=rater, files=list(files), i=0)
        self.act_hide_rejected.setEnabled(False)   # every row is rated
        self.table.blinded = True                  # no QC / Eval / kinetics
        self.event_view.blinded = True
        self.trace_view.blinded = True
        self.act_blind_next.setVisible(True)
        self.act_blind_end.setEnabled(True)
        self.act_blind.setEnabled(False)
        self._blind_open(0)

    def _start_blinded(self):
        import random
        import re as _re
        rater, ok = QInputDialog.getText(self, "Blinded review", "Rater name:")
        rater = _re.sub(r"[^A-Z0-9_-]", "", (rater or "").strip().upper())
        if not ok or not rater:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder to review",
                                                  self._start_dir())
        if not folder:
            return
        self._remember_dir(folder)
        unrated, rated, skipped = blind_plan(_ed().find_results_files(folder), rater)
        files = list(unrated)
        if rated and QMessageBox.question(
                self, "Blinded review",
                f"{len(rated)} file(s) were already rated by {rater}.\n"
                "Rate them again as well?") == QMessageBox.StandardButton.Yes:
            files += rated
        if not files:
            QMessageBox.information(
                self, "Blinded review",
                "Nothing to review." + (
                    f"\n{len(skipped)} file(s) have no events yet — run "
                    "Analysis ▸ Batch detect events first." if skipped else ""))
            return
        random.shuffle(files)
        if skipped:
            self.statusBar().showMessage(
                f"{len(skipped)} file(s) without events skipped")
        self.start_blinded(rater, files)

    def _blind_open(self, i: int):
        b = self._blind
        while i < len(b["files"]):
            try:
                store = load_from_hdf5(b["files"][i], all_components=True)
                if store.events is not None:
                    # Start from the algorithm's verdict, not whatever manual
                    # curation the file carries (manually_approved).
                    store.approved = store.events.auto_approved.copy()
                    break
                err = "events could not be mapped"
            except Exception as exc:
                err = str(exc)
            print(f"[blind] skipping file {i + 1}: {err}", file=sys.stderr)
            i += 1
        else:
            self._finish_blinded()
            return
        b["i"] = i
        self._swap_store(store)
        self.setWindowTitle(f"Blinded review — file {i + 1} of "
                            f"{len(b['files'])} — rater {b['rater']}")
        self.statusBar().showMessage(
            f"File {i + 1}/{len(b['files'])}: {store.n} components, "
            f"{store.n_approved} included — I toggles, Ctrl+Return saves")

    def _blind_save_next(self):
        b = self._blind
        if b is None:
            return
        try:
            write_blind_rating(b["files"][b["i"]], b["rater"], self.store, b["files"])
        except Exception as exc:
            QMessageBox.critical(self, "Save rating", str(exc))
            return
        if b["i"] + 1 >= len(b["files"]):
            self._finish_blinded()
        else:
            self._blind_open(b["i"] + 1)

    def _finish_blinded(self):
        n = len(self._blind["files"]) if self._blind else 0
        self._end_blinded()
        QMessageBox.information(self, "Blinded review",
                                f"Finished — {n} file(s) in this session.")

    def _end_blinded(self):
        self._blind = None
        self.table.blinded = False
        self.event_view.blinded = False
        self.trace_view.blinded = False
        self.act_hide_rejected.setEnabled(True)
        self.store.hide_rejected = self.act_hide_rejected.isChecked()
        self.corr_view.redraw(); self.dist_view.redraw()
        self.act_blind_next.setVisible(False)
        self.act_blind_end.setEnabled(False)
        self.act_blind.setEnabled(True)
        self.setWindowTitle("CaImAn Component Inspector")
        self._full_refresh()


__all__ = ['_AnalysisMixin']
