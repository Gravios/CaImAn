"""new-session recursive stack discovery."""
from _harness import REPO, ok, require, stub_caiman  # noqa: F401
import contextlib, io, os, shutil, sys, tempfile
from pathlib import Path

import importlib.util
spec = importlib.util.spec_from_file_location("new_session", REPO / "utilities" / "new_session.py")
ns = importlib.util.module_from_spec(spec); spec.loader.exec_module(ns)

D = "strohA-ia-000000-20151117"
S = "strohA-ia-000000-20151117-TL001_134746-25x-spont"


def session(date: Path, name: str, fc="007004", raw=3):
    s = date / name; s.mkdir(parents=True)
    for t in range(raw):
        (s / f"tl1-25xspont-FA1-LP4-87um_C00_t{t:010d}.ome.tif").write_bytes(b"")
    stem = f"{name}-C00-fc{fc}"
    ch = s / stem; ch.mkdir()
    (ch / f"{stem}.tif").write_bytes(b"x")
    (ch / f"{stem}.partial.tif").write_bytes(b"")              # leftover: never picked
    (ch / f"{stem}_els__d1_512_d2_512_d3_1_order_F_frames_10.tif").write_bytes(b"")  # MC output
    (s / f"{name}.yaml").write_text("acquisition_system:\n  settings:\n    sample_rate: {value: 28.67, units: Hz}\n")
    return ch, stem


def infer(cwd):
    old = os.getcwd(); os.chdir(cwd)
    try:
        return ns._infer_from_cwd()
    finally:
        os.chdir(old)


def expect_exit(cwd, *needles):
    try:
        infer(cwd); raise AssertionError("expected SystemExit")
    except SystemExit as e:
        for n in needles:
            assert n in str(e), str(e)
        return str(e)


root = Path(tempfile.mkdtemp())
try:
    date = root / D
    ch, stem = session(date, S)

    # date dir, one session → found recursively
    assert infer(date) == (stem, ch), infer(date)
    # session dir and channel dir
    assert infer(date / S) == (stem, ch)
    assert infer(ch) == (stem, ch), "channel dir path unchanged"
    # subject dir two levels up
    assert infer(root) == (stem, ch)
    print("ok  one stack found from channel / session / date / subject dir")

    # full CLI dry run from the date dir with a custom template
    tpl = root / "template.json"
    tpl.write_text((REPO / "utilities/pipelines/template_pipeline.json").read_text())
    buf = io.StringIO(); old = os.getcwd(); os.chdir(date)
    try:
        with contextlib.redirect_stdout(buf):
            rc = ns.main(["--template-json", str(tpl), "--dry-run"])
    finally:
        os.chdir(old)
    out = buf.getvalue()
    assert rc == 0 and f"session={stem}" in out and f"dest={ch}" in out and "Dry run" in out, out[-800:]
    assert not list(ch.glob("*_pipeline.*")), "dry run wrote files"
    print("ok  new-session --template-json --dry-run from the date dir")

    # second session → listed, not guessed
    ch2, stem2 = session(date, S.replace("TL001_134746", "TL002_140000"), fc="006000")
    msg = expect_exit(date, "2 input stacks", stem, stem2, "batch-sessions")
    assert infer(date / S) == (stem, ch), "session dir still unambiguous"
    print("ok  several stacks are listed with a pointer to batch-sessions")

    # --prefix narrows the search
    def infer_p(cwd, prefix):
        old = os.getcwd(); os.chdir(cwd)
        try:
            return ns._infer_from_cwd(prefix)
        finally:
            os.chdir(old)
    assert infer_p(date, f"{D}-TL002") == (stem2, ch2)
    assert infer_p(root, S) == (stem, ch), "full session name from subject dir"
    try:
        infer_p(date, D); raise AssertionError
    except SystemExit as e:
        assert "2 input stacks matching --prefix" in str(e) and "narrow with --prefix" in str(e), str(e)
    try:
        infer_p(date, "nope"); raise AssertionError
    except SystemExit as e:
        assert "matching --prefix 'nope'" in str(e), str(e)
    try:                                   # prefix also applies inside the channel dir
        infer_p(ch, f"{D}-TL002"); raise AssertionError
    except SystemExit as e:
        assert "matching --prefix" in str(e), str(e)
    buf = io.StringIO(); old = os.getcwd(); os.chdir(date)
    try:
        with contextlib.redirect_stdout(buf):
            rc = ns.main(["--prefix", f"{D}-TL002", "--template-json", str(tpl), "--dry-run"])
    finally:
        os.chdir(old)
    assert rc == 0 and f"session={stem2}" in buf.getvalue(), buf.getvalue()[-500:]
    print("ok  --prefix narrows the recursive search (CLI and API)")

    # MSR TL_dir layout (pre-channel subdir)
    msr_root = root / "msr"; tl = msr_root / "TL003_150000-25x-spont"; tl.mkdir(parents=True)
    (tl / "TL003_150000-25x-spont.msr").write_bytes(b"x")
    (tl / "other.msr").write_bytes(b"x")                       # stem != dir: ignored
    assert infer(msr_root) == ("TL003_150000-25x-spont", tl)
    print("ok  MSR TL_dir layout found")

    # nothing below → clear error
    empty = root / "empty"; (empty / "a").mkdir(parents=True)
    (empty / "a" / "frame_C00_t0000000000.ome.tif").write_bytes(b"")
    expect_exit(empty, "no .tif or .msr found in or below", "stack-sessions")
    print("ok  no stack → error mentions stack-sessions")

    # a tif directly in cwd keeps the old single-file behaviour
    flat = root / "flat"; flat.mkdir(); (flat / "anything.tif").write_bytes(b"")
    assert infer(flat) == ("anything", flat)
    (flat / "other.tif").write_bytes(b"")
    expect_exit(flat, "multiple input files")
    print("ok  cwd with one input file unchanged; two still refused")

    # channel dir with side .tif files: the dir-named stack wins
    assert infer(ch) == (stem, ch)
    print("ok  dir-named stack wins over MC side files in the channel dir")
    print("ALL OK")
finally:
    shutil.rmtree(root)
