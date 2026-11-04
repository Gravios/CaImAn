#!/usr/bin/env python3
"""
stack_sessions_gui.py — edit session naming and Trial YAML, then stack.

Point it at a directory holding stack-sessions session directories:

    stack-sessions-gui /data/source/strohA/.../strohA-ia-000000-20150709
    stack-sessions-gui --prefix strohA-ia      # current directory, filtered

Left: every subdirectory whose name matches the session schema
(``<source>-<user>-<subject>-<date>-<trial>_<time>-<mag>-<experiment>``),
with YAML / stacked-channel / raw-frame status.  Multi-select is supported.

Tab · Naming
    One field per naming element.  With several sessions selected, only the
    fields you change are applied to all of them.  "Rename…" previews every
    path that changes (the session dir, its YAML, channel subdirs and
    stacks; raw frames are untouched), warns about pipeline artefacts that
    embed old absolute paths, then renames with rollback on failure.
    Optionally re-derives the name-based YAML fields afterwards.

Tab · Recording
    The session's Trial YAML as an editable tree.  Add or remove fields and
    groups, fill missing keys from the template, infer null values from the
    name / frame filename / OME header, or re-sync the name-derived fields.
    "Save" writes this session; "Apply to all selected" writes only the
    edits you made (not the whole file) into every selected session's YAML.
    Every save keeps the previous file as ``<name>.yaml.bak``.

Tab · Template
    ``utilities/pipelines/template_acquisition.yaml`` — the skeleton every
    new Trial YAML starts from.  Add fields here to make them standard.

Tab · Stack
    Runs ``stack-sessions`` on the selected sessions (or every session
    matching the prefix) with live output.  Dry run is on by default.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from PyQt6.QtCore import QProcess, QProcessEnvironment, QRegularExpression, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QFontDatabase, QRegularExpressionValidator, QTextCursor,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QSplitter,
    QStyledItemDelegate, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from utilities import session_meta as sm
from utilities.rename_stem import FIELDS, build_stem

_ORIG_ROLE = Qt.ItemDataRole.UserRole          # leaf: original Python value
_KIND_ROLE = Qt.ItemDataRole.UserRole + 1      # "dict" | "leaf"
_NEW_ROLE  = Qt.ItemDataRole.UserRole + 2      # key added in this edit (key editable)

COL_ERROR = QColor(200, 40, 40)
COL_NULL  = QColor(140, 140, 140)
COL_OK    = QColor(30, 130, 60)


def _mono() -> QFont:
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)


def _ask(parent, title, text, detail="") -> bool:
    box = QMessageBox(QMessageBox.Icon.Question, title, text,
                      QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, parent)
    if detail:
        box.setDetailedText(detail)
    return box.exec() == QMessageBox.StandardButton.Ok


def _fmt_path(p) -> str:
    return ".".join(str(k) for k in p)


# ── YAML tree editor ──────────────────────────────────────────────────────────

class _CellGuard(QStyledItemDelegate):
    """Values of leaves are editable; keys only for fields added in this edit."""

    def createEditor(self, parent, option, index):
        if index.column() == 0 and not index.data(_NEW_ROLE):
            return None
        if index.column() == 1 and index.data(_KIND_ROLE) == "dict":
            return None
        return super().createEditor(parent, option, index)


class YamlEditor(QWidget):
    """Editable tree view of a nested dict.  ``to_doc()`` reads it back."""

    modifiedChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig: dict = {}
        self._orig_leaves: dict = {}
        self._modified = False
        self._building = False

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Field", "Value"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setStretchLastSection(True)
        self.tree.setItemDelegate(_CellGuard(self.tree))
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                  | QAbstractItemView.EditTrigger.EditKeyPressed
                                  | QAbstractItemView.EditTrigger.SelectedClicked)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemChanged.connect(self._on_item_changed)

        b_field = QPushButton("Add field")
        b_group = QPushButton("Add group")
        b_del   = QPushButton("Remove")
        b_exp   = QPushButton("Expand all")
        b_field.setToolTip("Add a value under the selected group (or next to the selected field)")
        b_group.setToolTip("Add a nested group, e.g. a new region or setting with value/units")
        b_field.clicked.connect(lambda: self._add(group=False))
        b_group.clicked.connect(lambda: self._add(group=True))
        b_del.clicked.connect(self._remove)
        b_exp.clicked.connect(self.tree.expandAll)

        row = QHBoxLayout()
        for b in (b_field, b_group, b_del):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(b_exp)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row)
        lay.addWidget(self.tree)

    # -- loading -----------------------------------------------------------

    def load(self, doc: dict, original: dict | None = None) -> None:
        """Show *doc*; edits are highlighted against *original* (default: *doc*)."""
        expanded = self._expanded_paths() if self.tree.topLevelItemCount() else None
        self._orig = copy.deepcopy(doc if original is None else original)
        self._orig_leaves = sm.leaves(self._orig)
        self._building = True
        self.tree.clear()
        self._fill(self.tree.invisibleRootItem(), doc, ())
        self._building = False
        if expanded is None:
            self.tree.expandAll()
        else:
            self._restore_expanded(expanded)
        self._refresh_marks()

    def original(self) -> dict:
        return copy.deepcopy(self._orig)

    def _fill(self, parent: QTreeWidgetItem, d: dict, prefix: tuple) -> None:
        for k, v in d.items():
            self._make_item(parent, k, v, prefix + (k,), new=False)

    def _make_item(self, parent, key, value, path, *, new: bool) -> QTreeWidgetItem:
        it = QTreeWidgetItem(parent, [str(key), ""])
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
        it.setData(0, _NEW_ROLE, new)
        if isinstance(value, dict):
            it.setData(0, _KIND_ROLE, "dict")
            it.setData(1, _KIND_ROLE, "dict")
            f = it.font(0); f.setBold(True); it.setFont(0, f)
            for k, v in value.items():
                self._make_item(it, k, v, path + (k,), new=new)
        else:
            it.setData(0, _KIND_ROLE, "leaf")
            it.setData(1, _KIND_ROLE, "leaf")
            it.setData(1, _ORIG_ROLE, value)
            it.setText(1, sm.value_to_text(value))
        return it

    # -- editing -----------------------------------------------------------

    def _add(self, *, group: bool) -> None:
        sel = self.tree.currentItem()
        if sel is None:
            parent = self.tree.invisibleRootItem()
        elif sel.data(0, _KIND_ROLE) == "dict":
            parent = sel
        else:
            parent = sel.parent() or self.tree.invisibleRootItem()
        existing = {parent.child(i).text(0) for i in range(parent.childCount())}
        base = "new_group" if group else "new_field"
        key, n = base, 1
        while key in existing:
            n += 1
            key = f"{base}_{n}"
        value = {"value": None, "units": None} if group else None
        self._building = True
        it = self._make_item(parent, key, value, (), new=True)
        self._building = False
        parent.setExpanded(True)
        it.setExpanded(True)
        self.tree.setCurrentItem(it)
        self.tree.editItem(it, 0)
        self._refresh_marks()

    def _remove(self) -> None:
        it = self.tree.currentItem()
        if it is None:
            return
        (it.parent() or self.tree.invisibleRootItem()).removeChild(it)
        self._refresh_marks()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if not self._building:
            self._refresh_marks()

    # -- reading back ------------------------------------------------------

    def _item_path(self, it: QTreeWidgetItem) -> tuple:
        keys = []
        while it is not None:
            keys.append(self._key_of(it))
            it = it.parent()
        return tuple(reversed(keys))

    def _key_of(self, it: QTreeWidgetItem):
        """Keys keep their original type (YAML may use int keys)."""
        text = it.text(0)
        if not it.data(0, _NEW_ROLE):
            parent = it.parent()
            src = self._orig if parent is None else sm.get_path(self._orig, self._item_path(parent))
            if isinstance(src, dict):
                for k in src:
                    if str(k) == text:
                        return k
        return text

    def _read(self, parent: QTreeWidgetItem, errors: list) -> dict:
        out: dict = {}
        for i in range(parent.childCount()):
            it = parent.child(i)
            key = self._key_of(it)
            if key == "" or (isinstance(key, str) and not key.strip()):
                errors.append((it, "empty field name"))
                continue
            if key in out:
                errors.append((it, f"duplicate field '{key}'"))
                continue
            if it.data(0, _KIND_ROLE) == "dict":
                out[key] = self._read(it, errors)
            else:
                try:
                    out[key] = sm.coerce(it.text(1), it.data(1, _ORIG_ROLE))
                except ValueError as exc:
                    errors.append((it, str(exc)))
        return out

    def to_doc(self) -> dict:
        errors: list = []
        doc = self._read(self.tree.invisibleRootItem(), errors)
        if errors:
            it, msg = errors[0]
            raise ValueError(f"{_fmt_path(self._item_path(it))}: {msg}")
        return doc

    def is_modified(self) -> bool:
        return self._modified

    # -- highlighting ------------------------------------------------------

    def _refresh_marks(self) -> None:
        self._building = True
        bad = False

        def walk(parent):
            nonlocal bad
            for i in range(parent.childCount()):
                it = parent.child(i)
                path = self._item_path(it)
                if it.data(0, _KIND_ROLE) == "dict":
                    walk(it)
                    self._style(it, changed=bool(it.data(0, _NEW_ROLE)), error=None, null=False)
                    continue
                try:
                    v = sm.coerce(it.text(1), it.data(1, _ORIG_ROLE))
                    err = None
                except ValueError as exc:
                    v, err, bad = None, str(exc), True
                o = self._orig_leaves.get(path, sm.DELETED)
                changed = o is sm.DELETED or o != v or type(o) is not type(v)
                self._style(it, changed=changed, error=err, null=(err is None and v is None))
        walk(self.tree.invisibleRootItem())
        self._building = False

        try:
            mod = bad or sm.diff(self._orig, self.to_doc()) != []
        except ValueError:
            mod = True
        if mod != self._modified:
            self._modified = mod
            self.modifiedChanged.emit(mod)

    @staticmethod
    def _style(it, *, changed, error, null) -> None:
        for col in (0, 1):
            f = it.font(col)
            f.setBold(bool(changed) or (col == 0 and it.data(0, _KIND_ROLE) == "dict"))
            f.setItalic(bool(null) and col == 1)
            it.setFont(col, f)
        if error:
            it.setForeground(1, QBrush(COL_ERROR))
            it.setToolTip(1, error)
        else:
            it.setForeground(1, QBrush(COL_NULL) if null else QBrush())
            it.setToolTip(1, "")

    def _expanded_paths(self) -> set:
        out = set()

        def walk(parent):
            for i in range(parent.childCount()):
                it = parent.child(i)
                if it.isExpanded():
                    out.add(self._item_path(it))
                walk(it)
        walk(self.tree.invisibleRootItem())
        return out

    def _restore_expanded(self, paths: set) -> None:
        def walk(parent):
            for i in range(parent.childCount()):
                it = parent.child(i)
                p = self._item_path(it)
                it.setExpanded(p in paths or not any(q[:len(p)] == p for q in self._orig_leaves))
                walk(it)
        walk(self.tree.invisibleRootItem())


# ── Recording tab ─────────────────────────────────────────────────────────────

class RecordingTab(QWidget):
    saved = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sessions: list[sm.Session] = []
        self._current: sm.Session | None = None
        self._creating = False

        self.header = QLabel("No session selected")
        self.header.setWordWrap(True)
        self.editor = YamlEditor()
        self.editor.modifiedChanged.connect(self._update_buttons)

        self.b_tmpl  = QPushButton("Fill missing from template")
        self.b_infer = QPushButton("Infer empty values")
        self.b_sync  = QPushButton("Sync from name")
        self.b_tmpl.setToolTip("Add template fields this YAML lacks (existing values untouched)")
        self.b_infer.setToolTip("Fill null values from the session name, frame filename (FA/LP/depth) "
                                "and OME header — what stack-sessions would write")
        self.b_sync.setToolTip("Overwrite datetime, trial id, condition, subject id and "
                               "magnification with the values encoded in the session name")
        self.b_tmpl.clicked.connect(self._fill_template)
        self.b_infer.clicked.connect(self._infer)
        self.b_sync.clicked.connect(self._sync)

        self.b_revert = QPushButton("Revert")
        self.b_save   = QPushButton("Save")
        self.b_all    = QPushButton("Apply edits to all selected")
        self.b_all.setToolTip("Write only the fields you changed here into every selected session's YAML")
        self.b_revert.clicked.connect(self.revert)
        self.b_save.clicked.connect(self.save)
        self.b_all.clicked.connect(self._save_all)

        tools = QHBoxLayout()
        for b in (self.b_tmpl, self.b_infer, self.b_sync):
            tools.addWidget(b)
        tools.addStretch(1)
        actions = QHBoxLayout()
        actions.addWidget(self.b_revert)
        actions.addStretch(1)
        actions.addWidget(self.b_all)
        actions.addWidget(self.b_save)

        lay = QVBoxLayout(self)
        lay.addWidget(self.header)
        lay.addLayout(tools)
        lay.addWidget(self.editor, 1)
        lay.addLayout(actions)
        self._update_buttons()

    def is_modified(self) -> bool:
        return self._current is not None and self.editor.is_modified()

    def set_sessions(self, sessions: list[sm.Session], current: sm.Session | None) -> None:
        self._sessions, self._current = sessions, current
        self.revert()

    def revert(self) -> None:
        s = self._current
        if s is None:
            self.header.setText("No session selected")
            self.editor.load({})
            self._update_buttons()
            return
        try:
            doc = sm.load_yaml(s.yaml_path)
        except Exception as exc:
            QMessageBox.critical(self, "YAML", f"Cannot read {s.yaml_path.name}:\n{exc}")
            doc = {}
        self._creating = not s.yaml_path.exists()
        if not self._creating:
            self.header.setText(f"<b>{s.name}</b><br>{s.yaml_path.name}")
            self.editor.load(doc)
        else:
            self.header.setText(f"<b>{s.name}</b><br><i>No YAML yet — showing the template; "
                                f"\"Create YAML\" writes {s.yaml_path.name}</i>")
            self.editor.load(sm.load_yaml(sm.template_path()))
        self._update_buttons()

    def _update_buttons(self, *_):
        has = self._current is not None
        mod = self.is_modified()
        creating = has and self._creating
        for b in (self.b_tmpl, self.b_infer, self.b_sync):
            b.setEnabled(has)
        self.b_save.setText("Create YAML" if creating else "Save")
        self.b_save.setEnabled(has and (mod or creating))
        self.b_revert.setEnabled(has and mod)
        n = len(self._sessions)
        self.b_all.setVisible(n > 1)
        self.b_all.setEnabled(has and mod and n > 1)
        self.b_all.setText(f"Apply edits to all {n} selected")

    def _edit(self, fn, what: str) -> None:
        try:
            doc = self.editor.to_doc()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        try:
            changed = fn(doc)
        except Exception as exc:
            QMessageBox.critical(self, what, f"{what} failed:\n{exc}")
            return
        self.editor.load(doc, original=self.editor.original())
        win = self.window()
        if hasattr(win, "statusBar"):
            win.statusBar().showMessage(
                f"{what}: {len(changed)} field(s) changed" if changed else f"{what}: nothing to change", 6000)

    def _fill_template(self):
        self._edit(lambda d: sm.merge_missing(d, sm.load_yaml(sm.template_path())), "Fill from template")

    def _infer(self):
        self._edit(lambda d: sm.fill_nulls(d, sm.infer_doc(self._current)), "Infer")

    def _sync(self):
        self._edit(lambda d: sm.sync_from_name(d, self._current.name), "Sync from name")

    def save(self) -> bool:
        if self._current is None:
            return True
        try:
            doc = self.editor.to_doc()
            sm.save_yaml(self._current.yaml_path, doc)
        except Exception as exc:
            QMessageBox.critical(self, "Save", str(exc))
            return False
        self.revert()
        self.saved.emit()
        return True

    def _save_all(self) -> None:
        try:
            doc = self.editor.to_doc()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        changes = sm.diff(self.editor.original(), doc)
        if not changes:
            return
        lines = [f"{_fmt_path(p)}  →  "
                 + ("(removed)" if v is sm.DELETED else sm.value_to_text(v)) for p, v in changes]
        if not _ask(self, "Apply to selected",
                    f"Write {len(changes)} edited field(s) into {len(self._sessions)} session YAML file(s)?\n"
                    "Missing YAML files are created from the template.\n"
                    "Previous versions are kept as .yaml.bak.",
                    "\n".join(lines)):
            return
        template = sm.load_yaml(sm.template_path())
        failed = []
        for s in self._sessions:
            try:
                d = sm.load_yaml(s.yaml_path) if s.yaml_path.exists() else copy.deepcopy(template)
                sm.apply_changes(d, changes)
                sm.save_yaml(s.yaml_path, d)
            except Exception as exc:
                failed.append(f"{s.name}: {exc}")
        if failed:
            QMessageBox.warning(self, "Apply to selected", "Some files failed:\n\n" + "\n".join(failed))
        self.revert()
        self.saved.emit()


# ── Template tab ──────────────────────────────────────────────────────────────

class TemplateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = sm.template_path()
        info = QLabel(f"<b>{self.path.name}</b> — skeleton for every new Trial YAML "
                      f"(stack-sessions and \"Fill missing from template\").<br>"
                      f"<span style='color:gray'>{self.path}</span>")
        info.setWordWrap(True)
        self.editor = YamlEditor()
        self.editor.modifiedChanged.connect(self._update)
        self.b_revert = QPushButton("Revert")
        self.b_save = QPushButton("Save template")
        self.b_revert.clicked.connect(self.revert)
        self.b_save.clicked.connect(self.save)
        row = QHBoxLayout()
        row.addWidget(self.b_revert)
        row.addStretch(1)
        row.addWidget(self.b_save)
        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.editor, 1)
        lay.addLayout(row)
        self.revert()

    def is_modified(self) -> bool:
        return self.editor.is_modified()

    def _update(self, *_):
        self.b_save.setEnabled(self.is_modified())
        self.b_revert.setEnabled(self.is_modified())

    def revert(self):
        try:
            self.editor.load(sm.load_yaml(self.path))
        except Exception as exc:
            QMessageBox.critical(self, "Template", str(exc))
        self._update()

    def save(self) -> bool:
        try:
            sm.save_yaml(self.path, self.editor.to_doc())
        except Exception as exc:
            QMessageBox.critical(self, "Template", str(exc))
            return False
        self.revert()
        return True


# ── Naming tab ────────────────────────────────────────────────────────────────

class NamingTab(QWidget):
    renamed = pyqtSignal(list)          # new session names

    MULTI = "<multiple>"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sessions: list[sm.Session] = []
        self._initial: dict[str, str] = {}
        self.edits: dict[str, QLineEdit] = {}
        self.errs: dict[str, QLabel] = {}

        form = QFormLayout()
        for f in FIELDS:
            e = QLineEdit()
            e.setValidator(QRegularExpressionValidator(QRegularExpression(sm.FIELD_PATTERNS[f]), e))
            e.textChanged.connect(self._preview)
            err = QLabel()
            err.setStyleSheet("color: rgb(200,40,40)")
            row = QHBoxLayout()
            row.addWidget(e, 1)
            row.addWidget(err)
            form.addRow(sm.FIELD_LABELS[f], row)
            self.edits[f], self.errs[f] = e, err

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(_mono())
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self.sync = QCheckBox("After renaming, update the YAML fields derived from the name "
                              "(datetime, trial id, condition, subject id, magnification)")
        self.sync.setChecked(True)

        self.b_revert = QPushButton("Revert")
        self.b_rename = QPushButton("Rename…")
        self.b_revert.clicked.connect(lambda: self.set_sessions(self._sessions))
        self.b_rename.clicked.connect(self._rename)
        row = QHBoxLayout()
        row.addWidget(self.b_revert)
        row.addStretch(1)
        row.addWidget(self.b_rename)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("Preview"))
        lay.addWidget(self.preview, 1)
        lay.addWidget(self.sync)
        lay.addLayout(row)
        self.set_sessions([])

    def set_sessions(self, sessions: list[sm.Session]) -> None:
        self._sessions = sessions
        for f, e in self.edits.items():
            vals = {s.parts[f] for s in sessions}
            e.blockSignals(True)
            if len(vals) == 1:
                e.setText(vals.pop())
                e.setPlaceholderText("")
            else:
                e.clear()
                e.setPlaceholderText(self.MULTI if sessions else "")
            e.setEnabled(bool(sessions))
            e.blockSignals(False)
            self._initial[f] = e.text()
        self._preview()

    def edited_fields(self) -> dict[str, str]:
        out = {}
        for f, e in self.edits.items():
            t = e.text()
            if t != self._initial.get(f, "") and (t or not e.placeholderText()):
                out[f] = t
        return out

    def _new_parts(self, s: sm.Session) -> dict:
        return {**s.parts, **self.edited_fields()}

    def _preview(self, *_) -> None:
        edits = self.edited_fields()
        any_err = False
        for f in FIELDS:
            self.errs[f].clear()
        lines = []
        for s in self._sessions:
            parts = self._new_parts(s)
            errs = sm.validate_parts(parts)
            for f, msg in errs.items():
                if f in edits or len(self._sessions) == 1:
                    self.errs[f].setText(msg)
            if errs:
                any_err = True
                lines.append(f"  {s.name}\n✗ " + ", ".join(f"{k}: {v}" for k, v in errs.items()))
            else:
                new = build_stem(parts)
                lines.append(f"  {s.name}" if new == s.name else f"  {s.name}\n→ {new}")
        self.preview.setPlainText("\n".join(lines))
        changing = any(build_stem(self._new_parts(s)) != s.name for s in self._sessions) if not any_err else False
        self.b_rename.setEnabled(bool(edits) and changing and not any_err)
        self.b_revert.setEnabled(bool(edits))

    def _rename(self) -> None:
        win = self.window()
        if hasattr(win, "confirm_unsaved") and not win.confirm_unsaved(template=False):
            return
        targets = [(s, build_stem(self._new_parts(s))) for s in self._sessions]
        targets = [(s, n) for s, n in targets if n != s.name]
        new_names = [n for _, n in targets]
        if len(set(new_names)) != len(new_names):
            QMessageBox.warning(self, "Rename", "Two sessions would get the same name.")
            return
        plans, downstream = [], []
        try:
            for s, n in targets:
                plans.append((s, n, sm.plan_rename(s.path, n)))
                downstream += sm.downstream_files(s.path)
        except FileExistsError as exc:
            QMessageBox.warning(self, "Rename", str(exc))
            return
        total = sum(len(p) for _, _, p in plans)
        detail = "\n".join(f"{o.relative_to(s.path.parent)}\n  → {nw.name}"
                           for s, _, plan in plans for o, nw in plan)
        warn = ""
        if downstream:
            warn = (f"\n\n⚠ {len(downstream)} pipeline file(s) (_pipeline.json, _results.hdf5, "
                    "*.mmap) embed absolute paths that will NOT be updated; re-run new-session "
                    "for these sessions afterwards.")
            detail += "\n\nNot rewritten:\n" + "\n".join(str(p) for p in downstream)
        if not _ask(self, "Rename",
                    f"Rename {len(plans)} session(s) — {total} path(s) in total?{warn}", detail):
            return
        done, failed = [], []
        for s, n, plan in plans:
            try:
                sm.apply_rename(plan)
            except OSError as exc:
                failed.append(f"{s.name}: {exc} (rolled back)")
                continue
            done.append(n)
            if self.sync.isChecked():
                yp = s.path.with_name(n) / f"{n}.yaml"
                if yp.exists():
                    try:
                        d = sm.load_yaml(yp)
                        if sm.sync_from_name(d, n):
                            sm.save_yaml(yp, d)
                    except Exception as exc:
                        failed.append(f"{n}: renamed, but YAML sync failed: {exc}")
        if failed:
            QMessageBox.warning(self, "Rename", "\n".join(failed))
        unchanged = [s.name for s in self._sessions if s.name not in {t.name for t, _ in targets}]
        self.renamed.emit(done + unchanged)


# ── Stack tab ─────────────────────────────────────────────────────────────────

class StackTab(QWidget):
    finished = pyqtSignal()

    def __init__(self, main: "MainWindow"):
        super().__init__(main)
        self.main = main
        self.proc: QProcess | None = None

        self.r_sel = QRadioButton()
        self.r_all = QRadioButton("All sessions matching the prefix")
        self.r_sel.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.r_sel)
        grp.addButton(self.r_all)
        self.dry = QCheckBox("Dry run (preview only)")
        self.dry.setChecked(True)
        self.delete = QCheckBox("Delete source frames after each successful stack")
        self.incomplete = QCheckBox("Allow incomplete (fewer frames than SizeT, contiguous from 0)")

        self.b_run = QPushButton("Run stack-sessions")
        self.b_stop = QPushButton("Stop")
        self.b_stop.setEnabled(False)
        self.b_run.clicked.connect(self.run)
        self.b_stop.clicked.connect(self.stop)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(_mono())
        self.log.setMaximumBlockCount(20000)

        opts = QVBoxLayout()
        for w in (self.r_sel, self.r_all, self.dry, self.delete, self.incomplete):
            opts.addWidget(w)
        row = QHBoxLayout()
        row.addWidget(self.b_run)
        row.addWidget(self.b_stop)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addLayout(opts)
        lay.addLayout(row)
        lay.addWidget(self.log, 1)
        self.set_selected(0)

    def set_selected(self, n: int) -> None:
        self.r_sel.setText(f"Selected sessions ({n})")

    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning

    def build_args(self) -> list[str] | None:
        parent = self.main.parent_dir()
        if parent is None:
            return None
        args = ["-u", "-m", "utilities.stack_sessions", "--parent", str(parent)]
        if self.r_sel.isChecked():
            names = [s.name for s in self.main.selected_sessions()]
            if not names:
                QMessageBox.information(self, "Stack", "Select one or more sessions first.")
                return None
            for n in names:
                args += ["--session", n]
        else:
            prefix = self.main.prefix().rstrip("-")
            if not prefix:
                QMessageBox.information(self, "Stack", "Enter a prefix (e.g. strohA-ia) first.")
                return None
            args += ["--prefix", prefix]
        if self.dry.isChecked():
            args.append("--dry-run")
        if self.delete.isChecked():
            args.append("--delete-sources")
        if self.incomplete.isChecked():
            args.append("--allow-incomplete")
        return args

    def run(self) -> None:
        if self.running():
            return
        args = self.build_args()
        if args is None:
            return
        if self.delete.isChecked() and not self.dry.isChecked():
            if not _ask(self, "Delete source frames",
                        "Source frame TIFFs are deleted permanently after each channel is "
                        "stacked successfully. Continue?"):
                return
        if not self.main.confirm_unsaved():
            return
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", os.pathsep.join(filter(None, [str(_REPO), env.value("PYTHONPATH")])))
        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(self.main.parent_dir()))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.errorOccurred.connect(self._error)
        self.log.appendPlainText(f"$ {Path(sys.executable).name} {' '.join(args)}\n")
        self.b_run.setEnabled(False)
        self.b_stop.setEnabled(True)
        self.proc.start(sys.executable, args)

    def stop(self) -> None:
        if self.running():
            self.proc.kill()

    def _read(self) -> None:
        data = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(data)
        self.log.ensureCursorVisible()

    def _error(self, err) -> None:
        if err == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(f"\nFailed to start {sys.executable}")
            self._done(-1, None)

    def _done(self, code, _status) -> None:
        self.log.appendPlainText(f"\n[exit {code}]\n")
        self.b_run.setEnabled(True)
        self.b_stop.setEnabled(False)
        self.finished.emit()


# ── Main window ───────────────────────────────────────────────────────────────

class _SessionItem(QTreeWidgetItem):
    raw_frames = 0

    def __lt__(self, other):
        if self.treeWidget() is not None and self.treeWidget().sortColumn() == 3:
            return self.raw_frames < getattr(other, "raw_frames", 0)
        return super().__lt__(other)


class MainWindow(QMainWindow):
    def __init__(self, parent_dir: Path | None, prefix: str = ""):
        super().__init__()
        self.setWindowTitle("Stack sessions — naming & recording metadata")
        self.resize(1400, 850)
        self._sessions: dict[str, sm.Session] = {}
        self._shown: list[str] = []

        self.e_parent = QLineEdit(str(parent_dir) if parent_dir else "")
        self.e_parent.returnPressed.connect(lambda: self.refresh())
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._browse)
        self.e_prefix = QLineEdit(prefix)
        self.e_prefix.setPlaceholderText("all")
        self.e_prefix.setMaximumWidth(180)
        self.e_prefix.returnPressed.connect(lambda: self.refresh())
        b_refresh = QPushButton("Refresh")
        b_refresh.clicked.connect(lambda: self.refresh())
        top = QHBoxLayout()
        top.addWidget(QLabel("Parent:"))
        top.addWidget(self.e_parent, 1)
        top.addWidget(b_browse)
        top.addWidget(QLabel("Prefix:"))
        top.addWidget(self.e_prefix)
        top.addWidget(b_refresh)

        self.list = QTreeWidget()
        self.list.setHeaderLabels(["Session", "YAML", "Stacked", "Raw frames"])
        self.list.setRootIsDecorated(False)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setSortingEnabled(True)
        self.list.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.list.header().setStretchLastSection(False)
        for c in (0, 1, 2, 3):
            self.list.header().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.list.itemSelectionChanged.connect(self._on_selection)

        self.naming = NamingTab()
        self.recording = RecordingTab()
        self.template = TemplateTab()
        self.stack = StackTab(self)
        self.naming.renamed.connect(lambda names: self.refresh(select=names))
        self.recording.saved.connect(lambda: self.refresh(select=self._shown))
        self.stack.finished.connect(lambda: self.refresh(select=self._shown))
        self.recording.editor.modifiedChanged.connect(lambda m: self._mark_tab(1, "Recording", m))
        self.template.editor.modifiedChanged.connect(lambda m: self._mark_tab(2, "Template", m))

        self.tabs = QTabWidget()
        self.tabs.addTab(self.naming, "Naming")
        self.tabs.addTab(self.recording, "Recording")
        self.tabs.addTab(self.template, "Template")
        self.tabs.addTab(self.stack, "Stack")

        split = QSplitter()
        split.addWidget(self.list)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([620, 780])

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        self.setCentralWidget(central)
        self.statusBar()

        if parent_dir:
            self.refresh()

    # -- accessors used by tabs --------------------------------------------

    def parent_dir(self) -> Path | None:
        p = Path(self.e_parent.text().strip()).expanduser()
        if not self.e_parent.text().strip() or not p.is_dir():
            QMessageBox.warning(self, "Parent", f"Not a directory: {p}")
            return None
        return p.resolve()

    def prefix(self) -> str:
        return self.e_prefix.text().strip()

    def selected_sessions(self) -> list[sm.Session]:
        return [self._sessions[n] for n in self._shown if n in self._sessions]

    # -- list --------------------------------------------------------------

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Directory containing session folders",
                                             self.e_parent.text() or str(Path.cwd()))
        if d:
            self.e_parent.setText(d)
            self.refresh()

    def refresh(self, select: list[str] | None = None) -> None:
        if select is None:
            if not self.confirm_unsaved(template=False):
                return
            select = self._shown
        parent = self.parent_dir()
        if parent is None:
            return
        try:
            sessions = sm.discover_sessions(parent, self.prefix() or None)
        except OSError as exc:
            QMessageBox.critical(self, "Refresh", str(exc))
            return
        self._sessions = {s.name: s for s in sessions}
        self.list.blockSignals(True)
        self.list.setSortingEnabled(False)
        self.list.clear()
        for s in sessions:
            stacked = " ".join(f"C{c}" for c in s.stacked) or "—"
            if s.partial:
                stacked += "  ⚠ partial"
            it = _SessionItem([s.name, "✓" if s.yaml_path.exists() else "—",
                               stacked, str(s.raw_frames) if s.raw_frames else "—"])
            it.raw_frames = s.raw_frames
            if s.yaml_path.exists():
                it.setForeground(1, QBrush(COL_OK))
            if s.partial:
                it.setForeground(2, QBrush(COL_ERROR))
            self.list.addTopLevelItem(it)
            if s.name in (select or []):
                it.setSelected(True)
        self.list.setSortingEnabled(True)
        self.list.blockSignals(False)
        self.statusBar().showMessage(f"{len(sessions)} session(s) in {parent}", 6000)
        self._apply_selection()

    def _names_selected(self) -> list[str]:
        return sorted(it.text(0) for it in self.list.selectedItems())

    def _on_selection(self) -> None:
        if self._names_selected() == self._shown:
            return
        if not self.confirm_unsaved(template=False):
            # restore the previous selection without re-entering
            self.list.blockSignals(True)
            for i in range(self.list.topLevelItemCount()):
                it = self.list.topLevelItem(i)
                it.setSelected(it.text(0) in self._shown)
            self.list.blockSignals(False)
            return
        self._apply_selection()

    def _apply_selection(self) -> None:
        self._shown = [n for n in self._names_selected() if n in self._sessions]
        sel = self.selected_sessions()
        cur = self.list.currentItem()
        current = self._sessions.get(cur.text(0)) if cur is not None and cur.isSelected() else None
        if current is None and sel:
            current = sel[0]
        self.naming.set_sessions(sel)
        self.recording.set_sessions(sel, current)
        self.stack.set_selected(len(sel))

    def _mark_tab(self, idx: int, name: str, modified: bool) -> None:
        self.tabs.setTabText(idx, f"{name} *" if modified else name)

    # -- unsaved-change guard ----------------------------------------------

    def confirm_unsaved(self, template: bool = True) -> bool:
        pending = [("Recording", self.recording)] if self.recording.is_modified() else []
        if template and self.template.is_modified():
            pending.append(("Template", self.template))
        for label, tab in pending:
            r = QMessageBox.question(
                self, "Unsaved changes", f"Save {label.lower()} changes first?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Cancel:
                return False
            if r == QMessageBox.StandardButton.Save and not tab.save():
                return False
            if r == QMessageBox.StandardButton.Discard:
                tab.revert()
        return True

    def closeEvent(self, ev) -> None:
        if self.stack.running() and not _ask(self, "Quit", "stack-sessions is still running. Kill it and quit?"):
            ev.ignore()
            return
        if not self.confirm_unsaved():
            ev.ignore()
            return
        self.stack.stop()
        ev.accept()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Edit stack-sessions naming and Trial YAML metadata.")
    p.add_argument("--parent", default=None,
                   help="Directory containing session folders (default: current directory)")
    p.add_argument("--prefix", default="", help="Only list sessions starting with this prefix")
    p.add_argument("path", nargs="*", help="Same as --parent")
    args = p.parse_args(argv)
    parent = args.parent or (args.path[0] if args.path else None) or str(Path.cwd())

    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = MainWindow(Path(parent).expanduser().resolve(), args.prefix)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
