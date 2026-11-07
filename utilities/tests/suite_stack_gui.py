"""Offscreen test for utilities/session_meta.py + bin/stack_sessions_gui.py.
"""
from _harness import REPO, light_root, ok, require, stub_caiman  # noqa: F401
require('numpy', 'tifffile', 'yaml', 'PyQt6.QtWidgets')
import importlib.util, os, shutil, sys, tempfile, types
from pathlib import Path

stub_caiman()
STUB = light_root()

import numpy as np, tifffile, yaml
from datetime import datetime
from utilities import session_meta as sm
from utilities import stack_sessions as ss

H, W, N = 8, 10, 6


def ome_xml(n):
    return ('<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            f'<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="{W}" SizeY="{H}" '
            f'SizeZ="1" SizeC="1" SizeT="{n}" TimeIncrement="0.033333" TimeIncrementUnit="s" '
            'PhysicalSizeX="0.5" PhysicalSizeY="0.5"><Channel ID="Channel:0:0" SamplesPerPixel="1"/></Pixels></Image></OME>')


def make(parent, name):
    d = parent / name; d.mkdir()
    for t in range(N):
        tifffile.imwrite(d / f"tl1-FA2-LP11_5-120um-spont25x_C00_t{t:04d}.ome.tif",
                         np.full((H, W), t, np.uint16), description=ome_xml(N) if t == 0 else None, metadata=None)
    return d


# ── session_meta unit tests ───────────────────────────────────────────────────
def unit():
    c = sm.coerce
    assert c("null") is None and c("") is None and c("~") is None
    assert c("000070") == "000070", "leading zeros must stay a string (YAML would give octal 56)"
    assert c("no") == "no" and c("true") is True
    assert c("12") == 12 and c("12", 1.5) == 12.0 and isinstance(c("12", 1.5), float)
    assert c("1.5e3") == 1500.0 and c("-0.25") == -0.25
    assert c("[1, 2]") == [1, 2]
    assert c("42", "x") == "42", "string stays string"
    assert c("2015-07-09 14:00:00", datetime(2000, 1, 1)) == datetime(2015, 7, 9, 14)
    try:
        c("[1, 2"); raise AssertionError
    except ValueError:
        pass
    ok("coerce")

    old = {"a": {"b": 1, "c": {"d": 2}}, "e": 3, "f": [1]}
    new = {"a": {"b": 5, "x": {"y": None}}, "f": [1, 2], "g": "new"}
    ch = sm.diff(old, new)
    other = {"a": {"b": 0, "c": {"d": 9}, "keep": 1}, "e": 0, "other": True}
    sm.apply_changes(other, ch)
    assert other == {"a": {"b": 5, "keep": 1, "x": {"y": None}}, "other": True, "f": [1, 2], "g": "new"}, other
    assert sm.diff(new, new) == []
    assert sm.diff({"a": 1}, {"a": 1.0}) != [], "int→float is a change"
    ok("diff / apply_changes (only edits propagate)")

    d = {"a": {"b": 1}}
    assert sm.merge_missing(d, {"a": {"b": 9, "c": None}, "z": {"w": 1}}) == [("a", "c"), ("z",)]
    assert d == {"a": {"b": 1, "c": None}, "z": {"w": 1}}
    d = {"a": {"b": None, "c": 3}}
    assert sm.fill_nulls(d, {"a": {"b": 7, "c": 8, "n": None}, "q": 1}) == [("a", "b"), ("q",)]
    assert d == {"a": {"b": 7, "c": 3}, "q": 1}
    ok("merge_missing / fill_nulls")

    good = sm.parse_stem("strohA-ia-000000-20150709-TL001_134044-25x-spont")
    assert sm.validate_parts(good) == {}
    assert "date" in sm.validate_parts({**good, "date": "20151340"})
    assert "time" in sm.validate_parts({**good, "time": "256000"})
    assert "userId" in sm.validate_parts({**good, "userId": "i-a"})
    assert "subjectId" in sm.validate_parts({**good, "subjectId": ""})
    ok("validate_parts")


def rename_tests(root):
    d = make(root, "strohA-ia-000001-20150709-TL001_134044-25x-spont")
    ch = d / f"{d.name}-C00-fc000006"; ch.mkdir()
    (ch / f"{ch.name}.tif").write_bytes(b"x")
    (ch / f"{ch.name}_pipeline.json").write_text("{}")
    (d / f"{d.name}.yaml").write_text("a: 1\n")
    lookalike = d / f"{d.name}2.txt"; lookalike.write_text("")      # not owned: next char is '2'
    new = "strohA-ia-000001-20150709-TL001_134044-40x-spont"
    plan = sm.plan_rename(d, new)
    assert len(plan) == 5, [p[0].name for p in plan]                 # yaml, ch dir, tif, json, session dir
    assert sm.downstream_files(d) == [ch / f"{ch.name}_pipeline.json"]

    # failure on the 3rd rename rolls back the first two
    real, calls = Path.rename, {"n": 0}
    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("boom")
        return real(self, target)
    Path.rename = flaky
    try:
        sm.apply_rename(plan); raise AssertionError
    except OSError:
        pass
    finally:
        Path.rename = real
    assert (d / f"{d.name}.yaml").exists() and (ch / f"{ch.name}.tif").exists() and ch.exists()
    ok("apply_rename rolls back on failure")

    sm.apply_rename(plan)
    nd = root / new
    assert (nd / f"{new}.yaml").exists()
    assert (nd / f"{new}-C00-fc000006" / f"{new}-C00-fc000006.tif").exists()
    assert (nd / f"{d.name}2.txt").exists(), "look-alike name was renamed"
    assert len(list(nd.glob("tl1-*_C00_t*.ome.tif"))) == N, "raw frames touched"
    try:
        sm.plan_rename(nd, new.replace("40x", "25x"))
        make(root, new.replace("40x", "25x"))
        sm.plan_rename(nd, new.replace("40x", "25x")); raise AssertionError
    except FileExistsError:
        pass
    ok("plan_rename / apply_rename (owned-only, collision check)")


def session_flag(root):
    a = make(root, "strohA-ia-000002-20150709-TL001_100000-25x-spont")
    b = make(root, "strohA-ia-000002-20150709-TL002_110000-25x-spont")
    assert ss.main(["--parent", str(root), "--session", a.name]) == 0
    assert (a / f"{a.name}-C00-fc000006" / f"{a.name}-C00-fc000006.tif").exists()
    assert not list(b.glob("*-fc*")), "--session stacked a non-selected session"
    assert ss.main(["--parent", str(root), "--session", "nope"]) == 1
    assert ss.main(["--parent", str(root)]) == 2
    ok("stack_sessions --session")


# ── GUI ───────────────────────────────────────────────────────────────────────
def gui(root):
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])

    tmpl = root / "template_acquisition.yaml"
    shutil.copy(sm.template_path(), tmpl)
    sm.template_path = lambda: tmpl                     # never write the repo template

    spec = importlib.util.spec_from_file_location("gui", REPO / "bin" / "stack_sessions_gui.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    g._ask = lambda *a, **k: True
    answers = []
    QMessageBox.question = staticmethod(lambda *a, **k: answers.pop(0))
    QMessageBox.warning = QMessageBox.critical = QMessageBox.information = staticmethod(lambda *a, **k: None)

    p = root / "gui"; p.mkdir()
    names = [f"strohA-ia-000003-20150709-TL00{i}_12000{i}-25x-spont" for i in (1, 2, 3)]
    for n in names:
        make(p, n)
    w = g.MainWindow(p)
    w.show()

    from PyQt6.QtCore import QItemSelectionModel

    def select(*ns):          # one selection-change event, like a click
        w.list.blockSignals(True)
        w.list.clearSelection()
        for i in range(w.list.topLevelItemCount()):
            it = w.list.topLevelItem(i)
            if it.text(0) in ns:
                w.list.setCurrentItem(it, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
                it.setSelected(True)
        w.list.blockSignals(False)
        w._on_selection()
        app.processEvents()

    assert w.list.topLevelItemCount() == 3
    ok("GUI lists sessions")

    # Recording: no YAML yet → template shown, not "modified"; Create YAML writes it
    select(names[0])
    r = w.recording
    assert not r.is_modified() and r.b_save.text() == "Create YAML" and r.b_save.isEnabled()
    r._infer()
    doc = r.editor.to_doc()
    st = doc["acquisition_system"]["settings"]
    assert st["fa"]["value"] == 2 and st["laserPower"]["value"] == 11.5 and st["depth"]["value"] == -120
    assert st["n_frames"] == N and doc["subject"]["id"] == 3 and doc["experiment"]["id"] == 1
    assert r.is_modified()
    assert r.save()
    y0 = p / names[0] / f"{names[0]}.yaml"
    assert sm.load_yaml(y0)["acquisition_system"]["settings"]["fa"]["value"] == 2
    ok("Recording: infer from name/frame/OME + create YAML")

    # edit a leaf via the tree, add a custom group, save → .bak kept
    def find(editor, *path):
        it = editor.tree.invisibleRootItem()
        for k in path:
            it = next(it.child(i) for i in range(it.childCount()) if it.child(i).text(0) == k)
        return it
    find(r.editor, "subject", "genotype").setText(1, "Thy1-GCaMP6s")
    find(r.editor, "subject", "sex").setText(1, "f")
    r.editor.tree.setCurrentItem(find(r.editor, "regions"))
    r.editor._add(group=True)
    newg = find(r.editor, "regions", "new_group")
    newg.setText(0, "barrel_ctx")
    newg.child(0).setText(1, "jRGECO1a") if newg.child(0).text(0) == "value" else None
    assert r.is_modified()
    assert r.save()
    d0 = sm.load_yaml(y0)
    assert d0["subject"]["genotype"] == "Thy1-GCaMP6s" and d0["regions"]["barrel_ctx"]["value"] == "jRGECO1a"
    assert y0.with_name(y0.name + ".bak").exists()
    ok("Recording: edit leaf, add group, save with .bak")

    # invalid value blocks save
    find(r.editor, "subject", "age").setText(1, "[1, 2")
    assert r.is_modified() and not r.save()
    r.revert()
    assert not r.is_modified()
    ok("Recording: invalid list rejected")

    # Apply-to-all: only edited fields propagate; other session's values survive
    y1 = p / names[1] / f"{names[1]}.yaml"
    sm.save_yaml(y1, {**sm.load_yaml(tmpl), "subject": {**sm.load_yaml(tmpl)["subject"], "sex": "m", "genotype": "WT"}})
    select(*names)
    assert r._current.name == names[2] and not r.b_all.isHidden(), r._current.name
    find(r.editor, "subject", "species").setText(1, "Mus musculus")
    r._save_all()
    for n in names:
        d = sm.load_yaml(p / n / f"{n}.yaml")
        assert d["subject"]["species"] == "Mus musculus", n
    assert sm.load_yaml(y1)["subject"]["sex"] == "m" and sm.load_yaml(y1)["subject"]["genotype"] == "WT"
    assert sm.load_yaml(y0)["subject"]["genotype"] == "Thy1-GCaMP6s"
    ok("Recording: apply edits to all selected (others' values kept, missing YAML created)")

    # unsaved-change guard on selection change: Cancel keeps selection + edits
    select(names[0])
    find(r.editor, "subject", "age").setText(1, "12")
    answers.append(QMessageBox.StandardButton.Cancel)
    select(names[1])
    assert w._shown == [names[0]] and r.is_modified(), w._shown
    answers.append(QMessageBox.StandardButton.Discard)
    select(names[1])
    assert w._shown == [names[1]] and not r.is_modified()
    ok("unsaved-change guard (cancel / discard)")

    # Naming: bulk edit one field across 2 sessions, rename + YAML sync
    select(names[0], names[1])
    nt = w.naming
    assert nt.edits["time"].text() == "" and nt.edits["time"].placeholderText() == nt.MULTI
    assert nt.edits["magnification"].text() == "25x"
    nt.edits["magnification"].setText("40x")
    assert nt.b_rename.isEnabled() and nt.edited_fields() == {"magnification": "40x"}
    nt._rename()
    app.processEvents()
    new0, new1 = (n.replace("25x", "40x") for n in names[:2])
    assert (p / new0).is_dir() and (p / new1).is_dir() and not (p / names[0]).exists()
    assert sm.load_yaml(p / new0 / f"{new0}.yaml")["acquisition_system"]["settings"]["magnification"]["value"] == 40
    assert sm.load_yaml(p / new0 / f"{new0}.yaml")["subject"]["genotype"] == "Thy1-GCaMP6s"
    assert sorted(w._shown) == sorted([new0, new1]), w._shown
    ok("Naming: bulk rename of one field + YAML sync, selection follows")

    nt.edits["date"].setText("2015130")       # incomplete → invalid
    assert not nt.b_rename.isEnabled() and nt.errs["date"].text()
    nt.set_sessions(nt._sessions)
    ok("Naming: invalid date blocks rename")

    # Template tab edits a copy of the template
    t = w.template
    t.editor.tree.setCurrentItem(None)
    t.editor._add(group=False)
    find(t.editor, "new_field").setText(0, "notes")
    assert t.save() and "notes" in sm.load_yaml(tmpl)
    ok("Template: add field + save")

    # Stack tab: real stack-sessions run on the selection (dry run, then live)
    g._REPO = STUB                             # child process uses the light caiman stand-in
    select(names[2])
    stk = w.stack
    args = stk.build_args()
    assert args[-1] == "--dry-run" and args.count("--session") == 1 and names[2] in args

    def run_and_wait():
        loop = QEventLoop()
        stk.finished.connect(loop.quit)
        QTimer.singleShot(60000, loop.quit)
        stk.run(); loop.exec()
        stk.finished.disconnect(loop.quit)
    run_and_wait()
    assert "[exit 0]" in stk.log.toPlainText(), stk.log.toPlainText()[-2000:]
    assert not list((p / names[2]).glob("*-fc*"))
    stk.dry.setChecked(False)
    run_and_wait()
    assert stk.log.toPlainText().count("[exit 0]") == 2, stk.log.toPlainText()[-2000:]
    assert (p / names[2] / f"{names[2]}-C00-fc000006" / f"{names[2]}-C00-fc000006.tif").exists()
    it = next(w.list.topLevelItem(i) for i in range(3) if w.list.topLevelItem(i).text(0) == names[2])
    assert it.text(2) == "C00", it.text(2)
    ok("Stack: dry run + live run via QProcess, list refreshed")

    w.close()


def main():
    root = Path(tempfile.mkdtemp())
    try:
        unit()
        (root / "r").mkdir(); rename_tests(root / "r")
    except Exception:
        shutil.rmtree(root); raise
    try:
        (root / "s").mkdir(); session_flag(root / "s")
        gui(root)
        print("ALL OK")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
