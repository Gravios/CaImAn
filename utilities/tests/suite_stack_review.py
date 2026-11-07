"""Regression tests for the second stack-sessions review."""
from _harness import REPO, ok, require, stub_caiman  # noqa: F401
require('numpy', 'tifffile', 'yaml')
stub_caiman()
import io, re, shutil, sys, tempfile, contextlib
from pathlib import Path

import numpy as np, tifffile, yaml

import caiman.utils.stack_to_bigtiff as s2b
from caiman.utils import ome_meta
from utilities import stack_sessions as ss

H, W = 6, 7
results = {}
import logging; logging.disable(logging.CRITICAL)


def ome_xml(size_t, size_c=1, tinc=True):
    st = f' SizeT="{size_t}"' if size_t else ""
    ti = ' TimeIncrement="0.033333" TimeIncrementUnit="s"' if tinc else ""
    return ('<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            f'<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="{W}" SizeY="{H}" '
            f'SizeZ="1" SizeC="{size_c}"{st}{ti} PhysicalSizeX="0.5" PhysicalSizeY="0.5">'
            '<Channel ID="Channel:0:0" SamplesPerPixel="1"/></Pixels></Image></OME>')


def make(parent, name, indices, channels=("00",), size_t=None, width=4, names=None, bad=None, master_xml_only_c00=True):
    d = parent / name; d.mkdir()
    size_t = size_t if size_t is not None else len(indices)
    for c in channels:
        for t in indices:
            w = (names or {}).get((c, t), width)
            p = d / f"tl1-FA1-LP11-90um-spont25x_C{c}_t{t:0{w}d}.ome.tif"
            desc = ome_xml(size_t, len(channels)) if (t == indices[0] and (c == channels[0] or not master_xml_only_c00)) else None
            shape = (H + 1, W) if bad == (c, t) else (H, W)
            tifffile.imwrite(p, np.full(shape, t + 1000 * int(c), np.uint16), description=desc, metadata=None)
    return d


def run(parent, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ss.main(["--prefix", "strohA-ia", "--parent", str(parent), *extra])
    return rc, buf.getvalue()


def stack_of(d, c, n):
    stem = f"{d.name}-C{c}-fc{n:06d}"
    p = d / stem / f"{stem}.tif"
    return tifffile.imread(p) if p.exists() else None


def case(name):
    def deco(fn):
        root = Path(tempfile.mkdtemp())
        try:
            fn(root)
            results[name] = "PASS"
        except AssertionError as exc:
            results[name] = f"FAIL  {exc}"
        except BaseException as exc:
            results[name] = f"ERROR {type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(root)
        return fn
    return deco


S = "strohA-ia-000000-20150709-TL001_134044-25x-spont"


@case("1 dry run leaves existing YAML untouched")
def _(root):
    d = make(root, S, list(range(5)))
    y = d / f"{S}.yaml"
    y.write_text("note: hand-edited\nacquisition_system:\n  settings:\n    sample_rate: {value: 99, units: Hz}\n")
    before = y.read_bytes()
    run(root, "--dry-run")
    assert y.read_bytes() == before, "dry run rewrote the YAML"


@case("2 interrupted padding (mixed widths) still stacks in time order")
def _(root):
    n = 12
    # a killed run padded t0 and t9..t11 (glob order is arbitrary); t1..t8 still 4 digits
    d = make(root, S, list(range(n)), names={("00", t): 6 for t in (0, 9, 10, 11)})
    rc, out = run(root)
    a = stack_of(d, "00", n)
    assert a is not None, out[-600:]
    order = [int(a[i, 0, 0]) for i in range(n)]
    assert order == list(range(n)), f"frame order {order}"


@case("3 same index in both widths is refused, not overwritten")
def _(root):
    d = make(root, S, list(range(6)))
    dup = d / "tl1-FA1-LP11-90um-spont25x_C00_t000003.ome.tif"
    tifffile.imwrite(dup, np.full((H, W), 777, np.uint16), metadata=None)
    rc, out = run(root)
    assert rc == 1, f"rc={rc}"
    assert dup.exists() and (d / "tl1-FA1-LP11-90um-spont25x_C00_t0003.ome.tif").exists(), "a duplicate was overwritten"
    assert stack_of(d, "00", 6) is None and stack_of(d, "00", 7) is None, "stacked despite duplicate index"


@case("4 gap in time indices fails the channel")
def _(root):
    d = make(root, S, [0, 1, 2, 4, 5, 6], size_t=7)
    rc, out = run(root)
    assert rc == 1, f"rc={rc}"
    assert not list(d.glob("*-fc*/*.tif")), "stacked despite a missing frame"
    assert "missing" in out.lower() and "3" in out, out[-400:]


@case("5 fewer frames than SizeT fails; --allow-incomplete stacks a contiguous prefix")
def _(root):
    d = make(root, S, list(range(5)), size_t=8)
    rc, out = run(root)
    assert rc == 1 and not list(d.glob("*-fc*/*.tif")), f"rc={rc}: truncated stack written under fc000008"
    rc, out = run(root, "--allow-incomplete")
    assert rc == 0, out[-400:]
    a = stack_of(d, "00", 5)
    assert a is not None and a.shape[0] == 5, "expected fc000005 with 5 frames"


@case("6 update_yaml keeps values/units it has no data for and uses pixel_dtype")
def _(root):
    y = root / "t.yaml"
    y.write_text(yaml.dump({"acquisition_system": {"settings": {
        "sample_rate": {"value": 30.0, "units": "Hz"},
        "frame_size": {"x": 512, "y": 512, "units": "pixels"},
        "n_frames": 1000, "n_channels": 2, "pixel_dtype": "uint16", "gain": {"value": 3}}}}, sort_keys=False))
    px = {"sample_rate_hz": None, "size_x": 512, "size_y": 512, "size_z": 1, "size_t": None, "size_c": None,
          "pixel_type": "uint16", "physical_size_x": None, "physical_size_y": None}
    with contextlib.redirect_stdout(io.StringIO()):
        ome_meta.update_yaml(y, px)
    s = yaml.safe_load(y.read_text())["acquisition_system"]["settings"]
    assert s["sample_rate"] == {"value": 30.0, "units": "Hz"}, s["sample_rate"]
    assert s["frame_size"].get("units") == "pixels", s["frame_size"]
    assert s["n_frames"] == 1000 and s["n_channels"] == 2, (s["n_frames"], s["n_channels"])
    assert "pixel_type" not in s and s["pixel_dtype"] == "uint16", s
    assert s["gain"] == {"value": 3}
    assert "pixel_size" not in s or s["pixel_size"].get("x") is not None, s.get("pixel_size")


@case("7 alphanumeric subject id parses (STO0647)")
def _(root):
    dn = ss._parse_session_name("strohA-ia-STO0647-20150113-TL001_122919-25x-spont")
    assert dn.get("subject_id") == "STO0647", dn
    assert dn.get("experiment_id") == 1 and dn.get("magnification") == 25, dn
    dn = ss._parse_session_name(S)
    assert dn.get("subject_id") == 0 and dn.get("condition") == "spontaneous", dn


@case("8 --delete-sources keeps the OME master while a channel is unstacked")
def _(root):
    n = 6
    d = make(root, S, list(range(n)), channels=("00", "01"), bad=("01", 2))
    rc, out = run(root, "--delete-sources")
    assert rc == 1
    assert stack_of(d, "00", n) is not None
    remaining = sorted(p.name for p in d.glob("*_C00_t*.tif"))
    assert remaining and re.search(r"_C00_t0+\.ome\.tif$", remaining[0]), f"C00 master deleted: {remaining}"
    assert len(list(d.glob("*_C01_t*.tif"))) == n
    # fix C01, rerun: the master is still readable, C01 stacks, then everything is cleaned up
    bad = next(d.glob("*_C01_t*0002.ome.tif"))
    tifffile.imwrite(bad, np.full((H, W), 1002, np.uint16), metadata=None)
    rc, out = run(root, "--delete-sources")
    assert rc == 0, out[-500:]
    assert stack_of(d, "01", n) is not None
    assert not list(d.glob("*_C01_t*.tif")), "C01 sources left after success"
    assert [p.name for p in d.glob("*_C00_t*.tif")] == [remaining[0]], "only the kept C00 master may remain"
    assert "not deleted by this run" in out, "leftover master not reported"


@case("9 stack-bigtiff keeps sources unless --delete-sources")
def _(root):
    d = make(root, S, list(range(4)))
    out = root / "o" / "s.tif"
    sys.argv = ["stack-bigtiff", "-i", str(d), "-o", str(out), "--flat-output", "--compression", "none"]
    with contextlib.redirect_stdout(io.StringIO()):
        s2b.main()
    assert out.exists()
    assert len(list(d.glob("*_t*.tif"))) == 4, "sources deleted without being asked"


@case("10 stack-bigtiff orders 4/5-digit FluoView indices numerically")
def _(root):
    d = root / "f"; d.mkdir()
    for t in (9998, 9999, 10000, 10001):
        tifffile.imwrite(d / f"x_C00_t{t:04d}.ome.tif", np.full((H, W), t, np.uint16), metadata=None)
    out = root / "o" / "s.tif"
    sys.argv = ["stack-bigtiff", "-i", str(d), "-o", str(out), "--flat-output", "--compression", "none", "--keep-sources"]
    with contextlib.redirect_stdout(io.StringIO()):
        s2b.main()
    a = tifffile.imread(out)
    assert [int(a[i, 0, 0]) for i in range(4)] == [9998, 9999, 10000, 10001], [int(a[i, 0, 0]) for i in range(4)]


@case("11 frame-name tokens parse in both lab naming orders (incl. real 2015-11-17 master)")
def _(root):
    real = ss._parse_frame_name("tl1-25xspont-FA1-LP4-87um_C00_t0000.ome.tif")
    assert real == {"fa": 1, "laser_power": 4.0, "depth_um": -87, "magnification": 25}, real
    old = ss._parse_frame_name("tl1-FA1-LP11_5-90um-spont25x_C00_t0000.ome.tif")
    assert old == {"fa": 1, "laser_power": 11.5, "depth_um": -90, "magnification": 25}, old
    assert ss._parse_frame_name("tl1-FA12-LP3_C00_t0000.ome.tif") == {"fa": 12, "laser_power": 3.0}
    assert "fa" not in ss._parse_frame_name("tl1-xFA1-LP3.ome.tif"), "FA inside another token"


@case("12 dry run reports the padding it would do and renames nothing")
def _(root):
    d = make(root, S, list(range(5)))
    before = sorted(p.name for p in d.iterdir())
    rc, out = run(root, "--dry-run")
    assert rc == 0, out[-400:]
    assert "would rename 5 frame file(s)" in out and "10 digits" in out, out[-400:]
    assert sorted(p.name for p in d.iterdir()) == before, "dry run renamed files"


if __name__ == "__main__":

    for k, v in results.items():
        print(f"{v[:4]}  {k}" + (f"\n      {v[6:]}" if not v.startswith("PASS") else ""))
    n_fail = sum(v != "PASS" for v in results.values())
    print("ALL OK" if not n_fail else f"{n_fail} failing")
    sys.exit(1 if n_fail else 0)
