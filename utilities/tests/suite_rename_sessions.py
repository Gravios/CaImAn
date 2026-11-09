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
            ("20141114", "152351", "TL000", "00x", "default", 1.0, None, None, ["90"]),
        # 2014 strohA-ms patterns
        "140225_tl8_fpstim_z100_p10_17-08-24":
            ("20140225", "170824", "TL008", "00x", "fpstim", 10.0, None, 100, []),
        "140324_TL1new_4000img_z100_p10_fa2_fpstim_14-42-09":
            ("20140324", "144209", "TL001", "00x", "fpstim", 10.0, 2, 100, ["new"]),
        "140522_tl5_z130_favg1_14-06-21":
            ("20140522", "140621", "TL005", "00x", "default", None, 1, 130, []),
        "140522_3Dscan4_favg2_75_195_14-50-00":
            ("20140522", "145000", "TL004", "00x", "3dscan", None, 2, None, ["75", "195"]),
        "140604_TL1_p25_z125fpstim_16-12-08":
            ("20140604", "161208", "TL001", "00x", "fpstim", 25.0, None, 125, []),
        "140623_tl5_spontan_z130_p15_16-24-05":
            ("20140623", "162405", "TL005", "00x", "spont", 15.0, None, 130, []),
        "140714_tl2_p25sb_z150_favg1_stim_16-33-26":
            ("20140714", "163326", "TL002", "00x", "stim", 25.0, 1, 150, ["sb"]),
        "140808_tl1_stim_cloud25_z130_favg1_12-26-42":
            ("20140808", "122642", "TL001", "00x", "stim", None, 1, 130, ["cloud25"]),
        "140714_zstack1_17-25-32":
            ("20140714", "172532", "TL001", "00x", "zstack", None, None, None, []),
        "140902_tl1fiber_test_p25_sb_z150_14-14-37":
            ("20140902", "141437", "TL001", "00x", "test", 25.0, None, 150, ["fiber", "sb"]),
        "140922_tl4_ispsipawstim10pulses3Hz_z122_13-20-16":
            ("20140922", "132016", "TL004", "00x", "pawstim", None, None, 122, []),
        "140924_tl8fpstimpinzettealle10sipsil_p20_z125_favg1_17-20-12":
            ("20140924", "172012", "TL008", "00x", "fpstim", 20.0, 1, 125, []),
        # 2015 patterns
        "150121_TL15_spontan_z130_p15_25x_favg2_15_4Hz_16-20-42":
            ("20150121", "162042", "TL015", "25x", "spont", 15.0, 2, 130, []),
        "150122_test3_25xobj_option_z165_25x_16-40-51":
            ("20150122", "164051", "TL003", "25x", "test", None, None, 165, ["obj", "option"]),
        "150310_tl2_Upown_favg1_30_8Hz_P24_z120_11-38-21":
            ("20150310", "113821", "TL002", "00x", "updown", 24.0, 1, 120, []),
        "150220_TL1_persUpstate_z120_p6_favg1_30_8Hz_11-13-02":
            ("20150220", "111302", "TL001", "00x", "persup", 6.0, 1, 120, []),
        "150220_morph2_favg6_13-10-57":
            ("20150220", "131057", "TL002", "00x", "morph", None, 6, None, []),
        "150203_tl1_cloud_p75_z110_updown_40x_10-40-41":
            ("20150203", "104041", "TL001", "40x", "updown", 75.0, None, 110, ["cloud"]),
        "20160301_tl12-chrimson-40x_09-05-00":
            ("20160301", "090500", "TL012", "40x", "chrimson", None, None, None, []),
    }
    for name, (date, time, tl, mag, exp, lp, fa, depth, left) in cases.items():
        p = rs.parse_legacy(name)
        got = (p.parts["date"], p.parts["time"], p.parts["trialId"], p.parts["magnification"],
               p.parts["experimentId"], p.laser_power, p.fa, p.depth_um, p.unparsed)
        assert got == (date, time, tl, mag, exp, lp, fa, depth, left), (name, got)
        assert p.name == name
    p = rs.parse_legacy("140922_tl4_ispsipawstim10pulses3Hz_z122_13-20-16")
    assert (p.side, p.protocol) == ("ipsi", ["10pulses3Hz"])
    p = rs.parse_legacy("140924_tl8fpstimpinzettealle10sipsil_p20_z125_favg1_17-20-12")
    assert (p.side, p.protocol) == ("ipsi", ["pinzettealle10s"])
    p = rs.parse_legacy("140527_tl11_favg2_z140_p7_gain61_17-17-15")
    assert (p.gain, p.laser_power, p.fa) == (61, 7.0, 2)
    assert rs.parse_legacy("140324_TL2_4000img_z100_p10_fa2_spont_14-49-45").n_frames == 4000
    assert rs.parse_legacy("150128_TL1_spont_25x_favg1_30.4Hz_p7_z130_14-43-28").sample_rate == 30.4
    assert rs.parse_legacy("150121_TL1_spont_z110_sb_p6_favg2_15_3Hz_25x_14-00-18").sample_rate == 15.3
    assert rs.parse_legacy("140922_tl2_pawstim10pulses3Hz_z125_12-24-06").sample_rate is None, \
        "a rate inside a protocol word stays protocol"
    assert rs.parse_legacy("150122_tl2_spont_sys3_z195_p18_25x_16-08-24").system == 3
    assert rs.parse_legacy("strohA-ia-000000-20140807-TL001_131248-00x-default") is None
    assert rs.parse_legacy("140231_tl1_12-00-00") is None, "not a calendar date"
    assert rs.parse_legacy("140225_tl1_25-00-00") is None, "not a clock time"
    assert rs.parse_legacy("notes") is None and not rs.is_legacy("strohA-ia-000000")
    # trials never exceed MAX_TRIAL: glued digits split off, ambiguity refused
    for name, tl, left in [("140219_tl1200_visstim_13-00-00", "TL001", ["200"]),
                           ("140408_tl120hzvistim_15-17-23", "TL001", ["20hz"]),
                           ("140408_tl220hzvistim_15-28-22", "TL002", ["20hz"])]:
        p = rs.parse_legacy(name)
        assert (p.parts["trialId"], p.parts["experimentId"], p.unparsed, p.doubts) == \
            (tl, "vstim", left, []), (name, p)
    p = rs.parse_legacy("150302_tl25_11-52-17")
    assert p.parts["trialId"] == "TL025" and p.remarks == ["trial 25 > 20"] and not p.doubts
    assert rs.parse_legacy("150101_tl1150_spont_10-00-00").doubts, "1+150 or 11+50"
    assert rs.parse_legacy("151031_tl3p1_25_d99_14-53-03").laser_power == 1.25
    ok(f"legacy names parsed ({len(cases)} cases): YYMMDD/YYYYMMDD, HH-MM-SS, tl<N>→TL<NNN>, "
       "25x/25xspont/vs25x, LP5 / LP1_65 / p1_50, FA<N>, 104um / depth116 / z90, "
       "paradigm words; 2014: glued tl<N>…, p<N>, favg<N>, z<N><word>, 3Dscan<N> / zstack<N>, "
       "gain<N>, <N>img, paw / forepaw protocol + side; 2015: rates 15_4Hz / 30.4Hz, "
       "persUp / UpDown, <paradigm><N> index, sys<N>; anything else kept as notes")

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

    t5 = root / "t5" / "strohA-ms-000000-20140922"
    d5 = t5 / "140922_tl4_ispsipawstim10pulses3Hz_z122_13-20-16"; d5.mkdir(parents=True)
    d6 = t5 / "140922_tl2_p25sb_gain61_4000img_cloud_spont_12-00-00"; d6.mkdir()
    run("--root", str(t5.parent), "--apply")
    y5 = ydoc(t5 / "strohA-ms-000000-20140922-TL004_132016-00x-pawstim")
    assert get(y5, "experiment", "protocol") == "10pulses3Hz" and get(y5, "experiment", "side") == "ipsi"
    assert get(y5, *S, "depth", "value") == -122 and "notes" not in y5["experiment"]
    y6 = ydoc(t5 / "strohA-ms-000000-20140922-TL002_120000-00x-spont")
    assert get(y6, *S, "gain", "value") == 61 and get(y6, *S, "n_frames") == 4000
    assert get(y6, *S, "laserPower", "value") == 25.0
    assert get(y6, "experiment", "notes") == ["sb", "cloud"]
    ok("YAML: protocol, side, gain, requested n_frames; unexplained tokens as experiment.notes")

    # 2015: date folder with a suffix, System<N> folders, wrong date folders
    t6 = root / "t6"
    mf = t6 / "strohA-ms-000000-20150121_2Pexp_micfailure"
    s1 = mf / "System1" / "150121_TL15_spontan_z130_p15_25x_favg2_15_4Hz_16-20-42"
    s3 = mf / "System3" / "150122_tl2_spont_sys3_z195_p18_25x_16-08-24"
    swapped = t6 / "strohA-ms-000000-20150302" / "150203_tl1_cloud_p75_z110_updown_40x_10-40-41"
    bad = t6 / "strohA-ms-000000-20152020" / "150220_morph2_favg6_13-10-57"
    for d in (s1, s3, swapped, bad):
        d.mkdir(parents=True)
    rc, out = run("--root", str(t6))
    assert "4 legacy directories: 4 to rename" in out, out
    assert "strohA-ms-000000-20150302\n    -> strohA-ms-000000-20150203" in out
    assert "20152020 is not a date" in out and "[--fix-date-folders]" in out
    assert "micfailure: sessions from 20150121, 20150122 — left as is" in out
    before6 = tree(t6)
    log6 = root / "r6.tsv"
    rc, out = run("--root", str(t6), "--apply", "--fix-date-folders", "--log", str(log6))
    assert rc == 0, out
    n1 = mf / "System1" / "strohA-ms-000000-20150121-TL015_162042-25x-spont"
    y = ydoc(n1)
    assert get(y, *S, "sample_rate", "value") == 15.4 and get(y, "acquisition_system", "system") == 1
    assert get(y, "experiment", "folder_note") == "2Pexp_micfailure"
    assert get(ydoc(mf / "System3" / "strohA-ms-000000-20150122-TL002_160824-25x-spont"),
               "acquisition_system", "system") == 3
    assert (t6 / "strohA-ms-000000-20150203" / "strohA-ms-000000-20150203-TL001_104041-40x-updown").is_dir()
    assert (t6 / "strohA-ms-000000-20150220" / "strohA-ms-000000-20150220-TL002_131057-00x-morph").is_dir()
    assert mf.is_dir() and not (t6 / "strohA-ms-000000-20150302").exists()
    rows = list(csv.DictReader(open(log6), delimiter="\t"))
    assert [r["confidence"] for r in rows][-2:] == ["folder", "folder"]
    rc, out = run("--undo", str(log6), "--apply")
    assert rc == 0 and tree(t6) == before6, sorted(set(tree(t6)) ^ set(before6))
    ok("2015: date folder suffix kept as folder_note, System<N> / sys<N> → acquisition_system.system, "
       "sample rate from the name; --fix-date-folders renames folders whose sessions agree "
       "(mixed ones reported); undo restores the tree")

    # ── 6. refusals, overrides, numbering, confidence ────────────────────────
    t2 = root / "t2" / "lab-xy-M17-20150301"
    a = t2 / "150301_tl2_spont_10-00-00"; a.mkdir(parents=True)
    b = t2 / "150301-tl2-spont-10-00-00"; b.mkdir()               # same target
    c = t2 / "150301_vstim_09-00-00"; c.mkdir()                    # no tl
    e = t2 / "150301_spont_11-30-00"; e.mkdir()                    # no tl
    f = t2 / "150301_tl1_spont_08-00-00"; f.mkdir()
    (t2 / "lab-xy-M17-20150301-TL001_080000-00x-spont").mkdir()    # f's target exists
    g = t2 / "150301_tl3_spont_12-00-00"; g.mkdir()
    h = t2 / "150301_tl1150_spont_13-00-00"; h.mkdir()               # 1+150 or 11+50
    (g / "g_pipeline.json").write_text("{}")
    items = {it.src.name: it for it in rs.plan(t2.parent)}
    assert not items[a.name].ok and not items[b.name].ok
    assert "2 directories would become" in items[a.name].problems[0]
    assert "target exists" in items[f.name].problems[0]
    assert items[c.name].dst.name == "lab-xy-M17-20150301-TL000_090000-00x-vstim"
    assert any("pipeline file" in w for w in items[g.name].warnings)
    assert not items[h.name].ok and "ambiguous tl1150" in items[h.name].problems[0]
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
       "--subject), ambiguous glued trial; --number-missing numbers by time skipping used TLs; pipeline files flagged")

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
