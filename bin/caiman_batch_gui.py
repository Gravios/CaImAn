#!/usr/bin/env python3
"""
caiman_batch_gui.py — tabbed batch-processing GUI for the session pipeline.

Point it at the *subjects* level of a data tree:

    python bin/caiman_batch_gui.py /data/source/ms2p/strohA/strohA-sa

Tab 1 · Select
    Foldable tree of subjects -> trials. A "trial" is any directory D that
    holds its own stack ``<D.name>.tif`` (i.e. session == D.name == tif stem,
    matching batch_sessions' channel-subdir convention). Drag a trial leaf to
    the batch list to queue it; drag a subject node to queue all of its
    trials. (Double-click or the Add button do the same.) Confirm to continue.

Tab 2 · Parameters
    The pipeline JSON template shown as collapsible lists; edit any value.
    "Apply" writes the (edited) JSON template and copies the Python template
    next to each queued trial's tif, named ``<session>_pipeline.{py,json}``.

Tab 3 · Run
    The list of trials to process. "Run" launches them sequentially. The
    current trial gets a light-blue block behind its name; a trial that
    errors gets a light-grey block and strike-through text. Errors are
    skipped and the next trial runs.
"""
from __future__ import annotations

import sys
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QListWidget, QListWidgetItem, QSplitter, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QFileDialog, QMessageBox,
    QStyledItemDelegate, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QBrush, QFont

_UROLE = Qt.ItemDataRole.UserRole
_ORIG_ROLE = Qt.ItemDataRole.UserRole + 1   # stores a leaf's original JSON value

# highlight colours (spec)
COL_RUNNING = QColor(179, 217, 255)   # light blue — current trial
COL_ERROR   = QColor(210, 210, 210)   # light grey — errored trial
COL_DONE    = QColor(210, 240, 210)   # light green — completed OK
COL_CLEAR   = QColor(0, 0, 0, 0)      # transparent — pending


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Trial:
    subject: str
    name: str            # session stem == directory name == tif stem
    directory: Path      # dir holding the tif + where pipeline files go
    tif: Path

    @property
    def pipeline_py(self) -> Path:
        return self.directory / f"{self.name}_pipeline.py"

    @property
    def pipeline_json(self) -> Path:
        return self.directory / f"{self.name}_pipeline.json"


def discover_subjects(root: Path) -> list[tuple[str, list[Trial]]]:
    """Return [(subject_name, [Trial, ...]), ...] for a subjects-level dir.

    Subjects are the immediate subdirectories of *root*. A trial is any
    directory D anywhere below a subject that contains ``<D.name>.tif`` (the
    raw stack — derived tifs like ``*_Xcorrected.tif`` have stem != dir name
    and are ignored).
    """
    root = Path(root)
    out: list[tuple[str, list[Trial]]] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        trials: list[Trial] = []
        for tif in sorted(subject_dir.rglob("*.tif")):
            d = tif.parent
            if tif.stem == d.name:                       # raw session stack
                trials.append(Trial(subject_dir.name, d.name, d, tif))
        if trials:
            out.append((subject_dir.name, trials))
    return out


# ── Tab 1 · tree + batch list ─────────────────────────────────────────────────

class SubjectTree(QTreeWidget):
    """Drag-source tree of subjects -> trials."""

    def __init__(self):
        super().__init__()
        self.setHeaderLabels(["Subject / Trial"])
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)

    def populate(self, discovered):
        self.clear()
        for subject, trials in discovered:
            s_item = QTreeWidgetItem([f"{subject}  ({len(trials)})"])
            self.addTopLevelItem(s_item)
            for tr in trials:
                leaf = QTreeWidgetItem([tr.name])
                leaf.setData(0, _UROLE, str(tr.directory))
                s_item.addChild(leaf)
        self.expandToDepth(0)

    @staticmethod
    def item_dirs(item) -> list[str]:
        """Trial directories represented by an item (leaf -> itself,
        subject -> all children)."""
        d = item.data(0, _UROLE)
        if d:
            return [d]
        return [item.child(i).data(0, _UROLE)
                for i in range(item.childCount())
                if item.child(i).data(0, _UROLE)]

    def mimeData(self, items):
        dirs: list[str] = []
        for it in items:
            for d in self.item_dirs(it):
                if d not in dirs:
                    dirs.append(d)
        md = super().mimeData(items)
        md.setText("\n".join(dirs))
        return md


class BatchList(QListWidget):
    """Drop-target list of queued trials (unique by directory)."""

    def __init__(self, resolver):
        super().__init__()
        self._resolver = resolver          # dir(str) -> Trial
        self._queued: set[str] = set()
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def add_dirs(self, dirs):
        added = 0
        for d in dirs:
            if not d or d in self._queued:
                continue
            tr = self._resolver(d)
            if tr is None:
                continue
            self._queued.add(d)
            it = QListWidgetItem(f"{tr.subject}  /  {tr.name}")
            it.setData(_UROLE, d)
            self.addItem(it)
            added += 1
        return added

    def remove_selected(self):
        for it in self.selectedItems():
            self._queued.discard(it.data(_UROLE))
            self.takeItem(self.row(it))

    def clear_all(self):
        self._queued.clear()
        self.clear()

    def trials(self) -> list[Trial]:
        return [self._resolver(self.item(i).data(_UROLE))
                for i in range(self.count())]

    # accept drops carrying newline-joined trial dirs
    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasText() else e.ignore()

    def dragMoveEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasText() else e.ignore()

    def dropEvent(self, e):
        text = e.mimeData().text()
        if text:
            self.add_dirs(text.split("\n"))
            e.acceptProposedAction()
        else:
            e.ignore()


# ── Tab 2 · collapsible JSON editor ───────────────────────────────────────────

class _Col0ReadOnly(QStyledItemDelegate):
    """Make the key column non-editable; only values (col 1) can be edited."""
    def createEditor(self, parent, option, index):
        if index.column() == 0:
            return None
        return super().createEditor(parent, option, index)


def _value_to_text(v):
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _coerce(text, original):
    """Coerce edited text back to the original value's JSON type."""
    if isinstance(original, bool):
        return text.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(original, int) and not isinstance(original, bool):
        try:
            return int(text)
        except ValueError:
            return original
    if isinstance(original, float):
        try:
            return float(text)
        except ValueError:
            return original
    if isinstance(original, list):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return original
    if original is None:
        t = text.strip()
        if t.lower() in ("", "null", "none"):
            return None
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            return t
    return text


class JsonEditor(QWidget):
    """Two-column collapsible tree editor for the pipeline JSON template."""

    def __init__(self):
        super().__init__()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Parameter", "Value"])
        self.tree.setColumnWidth(0, 320)
        self.tree.setItemDelegateForColumn(0, _Col0ReadOnly(self.tree))
        self.tree.setAlternatingRowColors(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tree)

    def load(self, data: dict):
        self.tree.clear()
        for k, v in data.items():
            self.tree.addTopLevelItem(self._make_item(k, v))
        self.tree.collapseAll()               # start collapsed (collapsible)

    def _make_item(self, key, value):
        it = QTreeWidgetItem([str(key), ""])
        if isinstance(value, dict):
            for k, v in value.items():
                it.addChild(self._make_item(k, v))
        else:
            it.setText(1, _value_to_text(value))
            # Store the original value on the item itself — QTreeWidgetItem is
            # unhashable, so it cannot be used as a dict key.
            it.setData(1, _ORIG_ROLE, value)
            editable = not str(key).startswith("_comment")
            if editable:
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
            else:
                it.setForeground(0, QBrush(QColor(120, 120, 120)))
                it.setForeground(1, QBrush(QColor(120, 120, 120)))
        return it

    def to_dict(self) -> dict:
        return {self.tree.topLevelItem(i).text(0): self._read(self.tree.topLevelItem(i))
                for i in range(self.tree.topLevelItemCount())}

    def _read(self, item):
        if item.childCount():
            return {item.child(i).text(0): self._read(item.child(i))
                    for i in range(item.childCount())}
        key = item.text(0)
        orig = item.data(1, _ORIG_ROLE)
        if key.startswith("_comment"):
            return orig
        return _coerce(item.text(1), orig)


# ── Tab 3 · runner ────────────────────────────────────────────────────────────

class BatchRunner(QThread):
    started_trial  = pyqtSignal(int)
    finished_trial = pyqtSignal(int, bool, str)
    log            = pyqtSignal(str)
    all_done       = pyqtSignal()

    def __init__(self, trials: list[Trial]):
        super().__init__()
        self.trials = trials
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        for i, tr in enumerate(self.trials):
            if self._abort:
                break
            self.started_trial.emit(i)
            self.log.emit(f"\n=== [{i + 1}/{len(self.trials)}] {tr.name} ===")
            ok, msg = self._run_one(tr)
            self.finished_trial.emit(i, ok, msg)   # continue regardless (skip errors)
        self.all_done.emit()

    def _run_one(self, tr: Trial):
        if not tr.pipeline_py.exists():
            return False, "pipeline .py missing (run Apply first)"
        try:
            proc = subprocess.Popen(
                [sys.executable, str(tr.pipeline_py)],
                cwd=str(tr.directory),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:                # stream output live
                self.log.emit(line.rstrip())
            rc = proc.wait()
            return (rc == 0), ("" if rc == 0 else f"exit code {rc}")
        except Exception as exc:                    # noqa: BLE001 — report + skip
            return False, str(exc)


# ── Main window ───────────────────────────────────────────────────────────────

class BatchGUI(QMainWindow):
    def __init__(self, root: Path, template_dir: Path):
        super().__init__()
        self.setWindowTitle(f"CaImAn batch — {root}")
        self.resize(1100, 720)
        self.template_dir = template_dir
        self._by_dir: dict[str, Trial] = {}
        self._runner: BatchRunner | None = None

        discovered = discover_subjects(root)
        for _, trials in discovered:
            for tr in trials:
                self._by_dir[str(tr.directory)] = tr

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_select_tab(discovered), "1 · Select")
        self.tabs.addTab(self._build_params_tab(), "2 · Parameters")
        self.tabs.addTab(self._build_run_tab(), "3 · Run")
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

        n = sum(len(t) for _, t in discovered)
        self.statusBar().showMessage(
            f"{len(discovered)} subjects, {n} trials found under {root}")
        if n == 0:
            QMessageBox.warning(self, "No trials",
                                f"No <name>.tif stacks found under:\n{root}")

    # -- Tab 1 --------------------------------------------------------------
    def _build_select_tab(self, discovered):
        self.tree = SubjectTree()
        self.tree.populate(discovered)
        self.tree.itemDoubleClicked.connect(
            lambda it, _c: self.batch.add_dirs(SubjectTree.item_dirs(it)))

        self.batch = BatchList(self._by_dir.get)

        add_btn   = QPushButton("Add selected →")
        add_btn.clicked.connect(self._add_selected)
        rm_btn    = QPushButton("Remove")
        rm_btn.clicked.connect(self.batch.remove_selected)
        clr_btn   = QPushButton("Clear")
        clr_btn.clicked.connect(self.batch.clear_all)

        mid = QVBoxLayout()
        mid.addStretch()
        for b in (add_btn, rm_btn, clr_btn):
            mid.addWidget(b)
        mid.addStretch()
        mid_w = QWidget(); mid_w.setLayout(mid)

        split = QSplitter()
        left = QWidget(); lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Subjects · trials (drag to the batch list)"))
        lv.addWidget(self.tree)
        right = QWidget(); rv = QVBoxLayout(right)
        rv.addWidget(QLabel("Batch list"))
        rv.addWidget(self.batch)
        split.addWidget(left); split.addWidget(mid_w); split.addWidget(right)
        split.setSizes([460, 90, 460])

        self.confirm_btn = QPushButton("Confirm selection →")
        self.confirm_btn.clicked.connect(self._confirm_selection)

        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(split)
        row = QHBoxLayout(); row.addStretch(); row.addWidget(self.confirm_btn)
        v.addLayout(row)
        return w

    def _add_selected(self):
        dirs = []
        for it in self.tree.selectedItems():
            dirs += SubjectTree.item_dirs(it)
        self.batch.add_dirs(dirs)

    def _confirm_selection(self):
        self._trials = self.batch.trials()
        if not self._trials:
            QMessageBox.information(self, "Empty", "The batch list is empty.")
            return
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(f"{len(self._trials)} trials queued")

    # -- Tab 2 --------------------------------------------------------------
    def _build_params_tab(self):
        self.editor = JsonEditor()
        self._tpl_json = self.template_dir / "template_pipeline.json"
        self._tpl_py   = self.template_dir / "template_pipeline.py"
        try:
            self.editor.load(json.loads(self._tpl_json.read_text(encoding="utf-8")))
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.warning(self, "Template", f"Could not load template:\n{exc}")

        load_btn = QPushButton("Load template JSON…")
        load_btn.clicked.connect(self._load_template)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_templates)

        self.tpl_label = QLabel(str(self._tpl_json))
        w = QWidget(); v = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Template:")); top.addWidget(self.tpl_label, 1)
        top.addWidget(load_btn)
        v.addLayout(top)
        v.addWidget(self.editor)
        row = QHBoxLayout(); row.addStretch(); row.addWidget(apply_btn)
        v.addLayout(row)
        return w

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load template JSON", str(self.template_dir), "JSON (*.json)")
        if not path:
            return
        p = Path(path)
        try:
            self.editor.load(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.critical(self, "Load error", str(exc)); return
        self._tpl_json = p
        cand = p.with_name(p.name.replace("_pipeline.json", "_pipeline.py")
                           .replace(".json", ".py"))
        if cand.exists():
            self._tpl_py = cand
        self.tpl_label.setText(str(self._tpl_json))

    def _apply_templates(self):
        try:
            data = self.editor.to_dict()
            json_text = json.dumps(data, indent=2, ensure_ascii=False)
            py_text = self._tpl_py.read_text(encoding="utf-8")
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.critical(self, "Apply error", str(exc)); return

        written = 0
        for tr in self._trials:
            tr.pipeline_json.write_text(json_text, encoding="utf-8")
            tr.pipeline_py.write_text(py_text, encoding="utf-8")
            written += 1

        self._populate_run_list()
        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(2)
        self.statusBar().showMessage(
            f"Applied templates to {written} trials")

    # -- Tab 3 --------------------------------------------------------------
    def _build_run_tab(self):
        self.run_list = QListWidget()
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._start_run)

        split = QSplitter(Qt.Orientation.Vertical)
        top = QWidget(); tv = QVBoxLayout(top)
        tv.addWidget(QLabel("Trials to process")); tv.addWidget(self.run_list)
        bot = QWidget(); bv = QVBoxLayout(bot)
        bv.addWidget(QLabel("Log")); bv.addWidget(self.log_view)
        split.addWidget(top); split.addWidget(bot); split.setSizes([420, 260])

        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(split)
        row = QHBoxLayout(); row.addStretch(); row.addWidget(self.run_btn)
        v.addLayout(row)
        return w

    def _populate_run_list(self):
        self.run_list.clear()
        for tr in self._trials:
            self.run_list.addItem(QListWidgetItem(f"{tr.subject}  /  {tr.name}"))

    def _set_row(self, idx, colour, strike=False):
        it = self.run_list.item(idx)
        if it is None:
            return
        it.setBackground(QBrush(colour))
        f = it.font(); f.setStrikeOut(strike); it.setFont(f)

    def _start_run(self):
        if self._runner and self._runner.isRunning():
            return
        for i in range(self.run_list.count()):
            self._set_row(i, COL_CLEAR, strike=False)
        self.run_btn.setEnabled(False)
        self._errors = 0
        self._runner = BatchRunner(self._trials)
        self._runner.started_trial.connect(
            lambda i: self._set_row(i, COL_RUNNING))
        self._runner.finished_trial.connect(self._on_trial_finished)
        self._runner.log.connect(self.log_view.append)
        self._runner.all_done.connect(self._on_all_done)
        self._runner.start()

    def _on_trial_finished(self, idx, ok, msg):
        if ok:
            self._set_row(idx, COL_DONE, strike=False)
        else:
            self._set_row(idx, COL_ERROR, strike=True)
            self._errors += 1
            self.log_view.append(f"  ERROR: {msg} — skipping")

    def _on_all_done(self):
        self.run_btn.setEnabled(True)
        n = self.run_list.count()
        self.statusBar().showMessage(
            f"Done: {n - self._errors}/{n} succeeded, {self._errors} errored")


# ── Entry point ───────────────────────────────────────────────────────────────

def _template_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "utilities" / "pipelines"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication(sys.argv)

    root = Path(argv[0]).resolve() if argv else None
    if root is None:
        chosen = QFileDialog.getExistingDirectory(None, "Select subjects-level directory")
        if not chosen:
            return 0
        root = Path(chosen)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    win = BatchGUI(root, _template_dir())
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
