"""rename-sessions: legacy directory names → session schema + Trial YAML."""
from _harness import REPO, ok, require, stub_caiman  # noqa: F401
require("yaml")
stub_caiman()
import contextlib, csv, io, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import yaml

from utilities import rename_sessions as rs
from utilities.rename_stem import parse_stem

root = Path(tempfile.mkdtemp())


def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = rs.main(list(argv))
    return rc, buf.getvalue()


def tree(base: Path) -> list:
    return sorted(str(p.relative_to(base)) for p in base.rglob("*"))


def ydoc(d: Path) -> dict:
    return yaml.safe_load((d / f"{d.name}.yaml").read_text())


S = ("acquisition_system", "settings")


def get(doc, *path):
    for k in path:
        doc = doc[k]
    return doc


try:
    # ── 1. parsing ──────────────────────────────────────────────────────────
    cases = {
        "140225_tl1_fpstim_z90_p1_50_15-56-08":
            ("20140225", "155608", "TL001", "00x", "fpstim", 1.5, None, 90, []),
        "151106_tl1-25xspont-FA1-LP5-104um_13-34-40":
            ("20151106", "133440", "TL001", "25x", "spont", 5.0, 1, 104, []),
        "151001_tl5-vs25x-FA1-LP1_65-depth116_12-28-06":
            ("20151001", "122806", "TL005", "25x", "vstim", 1.65, 1, 116, []),
        "150113_tl1-LP3_3-FA1-spont25x_12-29-19":
            ("20150113", "122919", "TL001", "25x", "spont", 3.3, 1, None, []),
        "150218_tl1-spont-LP4_45-FA2_19-07-52":
            ("20150218", "190752", "TL001", "00x", "spont", 4.45, 2, None, []),
        "141114_P1-90_15-23-51":
            ("20141114", "152351", "TL000", "00x", "default", None, None, None, ["P1", "90"]),
        "20160301_tl12-chrimson-40x_09-05-00":
            ("20160301", "090500", "TL012", "40x", "chrimson", None, None, None, []),
    }
    for name, (date, time, tl, mag, exp, lp, fa, depth, left) in cases.items():
        p = rs.parse_legacy(name)
        got = (p.parts["date"], p.parts["time"], p.parts["trialId"], p.parts["magnification"],
               p.parts["experimentId"], p.laser_power, p.fa, p.depth_um, p.unparsed)
        assert got == (date, time, tl, mag, exp, lp, fa, depth, left), (name, got)
        assert p.name == name
    assert rs.parse_legacy("strohA-ia-000000-20140807-TL001_131248-00x-default") is None
    assert rs.parse_legacy("140231_tl1_12-00-00") is None, "not a calendar date"
    assert rs.parse_legacy("140225_tl1_25-00-00") is None, "not a clock time"
    assert rs.parse_legacy("notes") is None and not rs.is_legacy("strohA-ia-000000")
    ok(f"legacy names parsed ({len(cases)} cases): YYMMDD/YYYYMMDD, HH-MM-SS, tl<N>→TL<NNN>, "
       "25x/25xspont/vs25x, LP5 / LP1_65 / p1_50, FA<N>, 104um / depth116 / z90, "
       "paradigm words; unknown tokens reported")

    # ── 2. tree, dry run ─────────────────────────────────────────────────────
    base = root / "strohA" / "strohA-ia"
    subj = base / "strohA-ia-000000"
    d0225 = subj / "strohA-ia-000000-20140225"
    leg1 = d0225 / "140225_tl1_fpstim_z90_p1_50_15-56-08"
    leg1.mkdir(parents=True)
    (leg1 / "tl1_C00_t0000.ome.tif").write_bytes(b"II*\0")
    (leg1 / f"{leg1.name}_notes.txt").write_text("old-stem file")      # carries the stem
    sto = base / "strohA-ia-STO0647" / "strohA-ia-STO0647-20150113"
    leg2 = sto / "150113_tl1-LP3_3-FA1-spont25x_12-29-19"
    leg2.mkdir(parents=True)
    done = subj / "strohA-ia-000000-20140807" / "strohA-ia-000000-20140807-TL001_131248-00x-default"
    done.mkdir(parents=True)
    (done / "tl1_C00_t0000.ome.tif").write_bytes(b"II*\0")
    before = tree(root)
    rc, out = run("--root", str(base))
    assert rc == 0 and tree(root) == before, "dry run writes nothing"
    assert "-> strohA-ia-000000-20140225-TL001_155608-00x-fpstim" in out
    assert "-> strohA-ia-STO0647-20150113-TL001_122919-25x-spont" in out
    assert "2 legacy directories: 2 to rename" in out and "20140807-TL001" not in out
    ok("dry run: targets from the tree (sourceId-userId-subjectId from the ancestor "
       "folder), already-named sessions ignored, nothing written")

    # ── 3. apply + YAML ──────────────────────────────────────────────────────
    log = root / "rename.tsv"
    rc, out = run("--root", str(base), "--apply", "--log", str(log))
    assert rc == 0, out
    new1 = d0225 / "strohA-ia-000000-20140225-TL001_155608-00x-fpstim"
    new2 = sto / "strohA-ia-STO0647-20150113-TL001_122919-25x-spont"
    assert not leg1.exists() and new1.is_dir() and new2.is_dir()
    assert (new1 / "tl1_C00_t0000.ome.tif").exists(), "raw frames untouched"
    assert (new1 / f"{new1.name}_notes.txt").exists(), "files carrying the old name follow"
    assert parse_stem(new1.name)["experimentId"] == "fpstim"
    y = ydoc(new1)
    assert get(y, "experiment", "datetime") == "2014-02-25 15:56:08"
    assert get(y, "experiment", "id") == 1 and get(y, "experiment", "condition") == "fpstim"
    assert get(y, "experiment", "legacy_name") == leg1.name
    assert get(y, *S, "laserPower", "value") == 1.5 and get(y, *S, "depth", "value") == -90
    assert get(y, *S, "magnification", "value") is None, "00x = unknown"
    assert get(y, *S, "fa", "value") is None and get(y, *S, "sample_rate", "units") == "Hz"
    y2 = ydoc(new2)
    assert get(y2, "subject", "id") == "STO0647" and get(y2, *S, "magnification", "value") == 25
    assert get(y2, "experiment", "condition") == "spontaneous"
    assert get(y2, *S, "laserPower", "value") == 3.3 and get(y2, *S, "fa", "value") == 1
    rows = list(csv.DictReader(open(log), delimiter="\t"))
    assert [Path(r["dst"]).name for r in rows] == [new1.name, new2.name]
    assert [r["yaml"] for r in rows] == ["wrote", "wrote"]
    ok("apply: renamed with the files carrying the old name; Trial YAML from the "
       "template + name + laser power / FA / depth (negative) + legacy_name; logged")

    # stack-sessions later adds OME fields and keeps these
    from caiman.utils.ome_meta import update_yaml
    update_yaml(new1 / f"{new1.name}.yaml",
                {"sample_rate_hz": 30.5, "size_x": 512, "size_y": 512, "size_t": 6000,
                 "physical_size_x": 0.5, "physical_size_y": 0.5})
    y = ydoc(new1)
    assert get(y, *S, "sample_rate", "value") == 30.5 and get(y, *S, "n_frames") == 6000
    assert get(y, *S, "laserPower", "value") == 1.5 and get(y, *S, "depth", "value") == -90
    assert get(y, "experiment", "legacy_name") == leg1.name
    import utilities.stack_sessions as ss
    from utilities.session_name import parse_session_name
    assert ss._parse_session_name is parse_session_name
    ok("stack-sessions' OME update keeps the rename-sessions fields; both share "
       "utilities.session_name")

    # ── 4. undo ──────────────────────────────────────────────────────────────
    rc, out = run("--undo", str(log))
    assert rc == 0 and new1.exists(), "undo dry run changes nothing"
    rc, out = run("--undo", str(log), "--apply")
    assert rc == 0 and leg1.is_dir() and leg2.is_dir() and not new1.exists()
    assert tree(root) == sorted(before + ["rename.tsv"]), sorted(set(tree(root)) ^ set(before))
    ok("--undo LOG restores names and files exactly (removes the YAML it wrote)")

    # ── 5. existing YAML: only nulls filled ──────────────────────────────────
    rc, _ = run("--root", str(base), "--apply")
    y = ydoc(new1); y["acquisition_system"]["settings"]["laserPower"]["value"] = 9.9
    y["experiment"]["condition"] = None; y["subject"]["species"] = "mouse"
    (new1 / f"{new1.name}.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
    msg = rs.write_yaml(new1, rs.parse_legacy(leg1.name))
    y = ydoc(new1)
    assert msg.startswith("yaml: filled") and get(y, "experiment", "condition") == "fpstim"
    assert get(y, *S, "laserPower", "value") == 9.9 and get(y, "subject", "species") == "mouse"
    assert rs.write_yaml(new1, rs.parse_legacy(leg1.name)).endswith("up to date")
    ok("an existing YAML is never overwritten: only null fields are filled")

    # ── 6. refusals, overrides, numbering, confidence ────────────────────────
    t2 = root / "t2" / "lab-xy-M17-20150301"
    a = t2 / "150301_tl2_spont_10-00-00"; a.mkdir(parents=True)
    b = t2 / "150301-tl2-spont-10-00-00"; b.mkdir()               # same target
    c = t2 / "150301_vstim_09-00-00"; c.mkdir()                    # no tl
    e = t2 / "150301_spont_11-30-00"; e.mkdir()                    # no tl
    f = t2 / "150301_tl1_spont_08-00-00"; f.mkdir()
    (t2 / "lab-xy-M17-20150301-TL001_080000-00x-spont").mkdir()    # f's target exists
    g = t2 / "150301_tl3_spont_12-00-00"; g.mkdir()
    (g / "g_pipeline.json").write_text("{}")
    items = {it.src.name: it for it in rs.plan(t2.parent)}
    assert not items[a.name].ok and not items[b.name].ok
    assert "2 directories would become" in items[a.name].problems[0]
    assert "target exists" in items[f.name].problems[0]
    assert items[c.name].dst.name == "lab-xy-M17-20150301-TL000_090000-00x-vstim"
    assert any("pipeline file" in w for w in items[g.name].warnings)
    num = {it.src.name: it for it in rs.plan(t2.parent, number_missing=True)}
    assert num[c.name].dst.name.split("-")[4].startswith("TL004_"), num[c.name].dst.name
    assert num[e.name].dst.name.split("-")[4].startswith("TL005_"), "next free, by time"
    hi = {it.src.name: it for it in rs.plan(t2.parent, min_confidence=4)}
    assert not hi[c.name].ok and "confidence" in hi[c.name].problems[0]
    nop = root / "t3" / "plain"; (nop / "150301_tl1_09-00-00").mkdir(parents=True)
    it = rs.plan(nop)[0]
    assert "no <source>-<user> ancestor" in it.problems[0]
    it = rs.plan(nop, prefix="strohA-sa", subject="000123")[0]
    assert it.ok and it.dst.name == "strohA-sa-000123-20150301-TL001_090000-00x-default"
    rc, out = run("--root", str(t2.parent), "--apply")
    assert a.exists() and b.exists() and f.exists(), "refused ones untouched"
    assert (t2 / "lab-xy-M17-20150301-TL000_090000-00x-vstim").exists()
    ok("refused: shared or existing targets, low confidence, no prefix (until --prefix / "
       "--subject); --number-missing numbers by time skipping used TLs; pipeline files flagged")

    # ── 7. --yaml-only for already named sessions ────────────────────────────
    rc, out = run("--root", str(done.parent), "--yaml-only")
    assert "would write" in out and not (done / f"{done.name}.yaml").exists()
    rc, out = run("--root", str(done.parent), "--yaml-only", "--apply")
    y = ydoc(done)
    assert get(y, "experiment", "condition") == "default" and "legacy_name" not in y["experiment"]
    assert get(y, "experiment", "datetime") == "2014-08-07 13:12:48"
    ok("--yaml-only writes / fills the YAML of already-named sessions")

    # ── 8. no CaImAn needed ──────────────────────────────────────────────────
    t4 = root / "t4" / "strohA-ia-000000-20140225"; (t4 / leg1.name).mkdir(parents=True)
    code = ("import sys; sys.modules['caiman'] = None; sys.path.insert(0, %r)\n"
            "from utilities.rename_sessions import main\n"
            "sys.exit(main(['--root', %r, '--apply', '-q']))") % (str(REPO), str(t4.parent))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "wrote" in r.stdout, (r.stdout, r.stderr[-1500:])
    assert (t4 / "strohA-ia-000000-20140225-TL001_155608-00x-fpstim").is_dir()
    ok("runs without the CaImAn stack (name and YAML code are standard library + PyYAML)")
    print("ALL OK")
finally:
    shutil.rmtree(root)
