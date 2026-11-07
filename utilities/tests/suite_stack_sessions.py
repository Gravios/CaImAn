"""Synthetic end-to-end test for utilities/stack_sessions.py."""
from _harness import REPO, ok, require, stub_caiman  # noqa: F401
require('numpy', 'tifffile', 'yaml')
stub_caiman()
import shutil, sys, tempfile
from pathlib import Path

import numpy as np
import tifffile
import yaml

import caiman.utils.stack_to_bigtiff as s2b
from utilities import stack_sessions as ss

H, W = 8, 10
PREFIX = "strohA-ia"


def ome_xml(size_t, size_c):
    st = f' SizeT="{size_t}"' if size_t else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" '
        f'Type="uint16" SizeX="{W}" SizeY="{H}" SizeZ="1" SizeC="{size_c}"{st} '
        'TimeIncrement="0.033333" TimeIncrementUnit="s" '
        'PhysicalSizeX="0.5" PhysicalSizeY="0.5">'
        '<Channel ID="Channel:0:0" SamplesPerPixel="1"/></Pixels></Image></OME>'
    )


def make_session(parent, name, n, channels=("00",), size_t=True, bad=None):
    d = parent / name
    d.mkdir()
    for c in channels:
        for t in range(n):
            shape = (H + 1, W) if bad == (c, t) else (H, W)
            arr = np.full(shape, t + 100 * int(c), np.uint16)
            p = d / f"tl1-FA1-LP11-90um-spont25x_C{c}_t{t:04d}.ome.tif"
            desc = ome_xml(n if size_t else None, len(channels)) if (c == channels[0] and t == 0) else None
            tifffile.imwrite(p, arr, description=desc, metadata=None)
    return d


def run(parent, *extra):
    return ss.main(["--prefix", PREFIX, "--parent", str(parent), *extra])


def check_stack(d, c, n):
    stem = f"{d.name}-C{c}-fc{n:06d}"
    out = d / stem / f"{stem}.tif"
    assert out.exists(), out
    a = tifffile.imread(out)
    assert a.shape == (n, H, W), a.shape
    assert [int(a[i, 0, 0]) for i in range(n)] == [i + 100 * int(c) for i in range(n)], "frame order"
    assert not list(d.rglob("*.partial.tif")), "partial left behind"
    return out


def main():
    root = Path(tempfile.mkdtemp())
    try:
        n = 12
        # A. two-channel happy path + yaml + padding
        a = make_session(root, f"{PREFIX}-000000-20150709-TL001_134044-25x-spont", n, ("00", "01"))
        # B. broken channel: one frame with wrong shape in C01
        b = make_session(root, f"{PREFIX}-000000-20150709-TL002_140000-25x-spont", n, ("00", "01"), bad=("01", 7))
        # C. no SizeT in OME header
        c = make_session(root, f"{PREFIX}-000000-20150709-TL003_150000-25x-spont", n, size_t=False)

        assert run(root, "--dry-run") == 0, "dry run exit"
        assert not list(root.rglob("*-fc*.tif")), "dry run wrote stacks"

        rc = run(root)
        assert rc == 1, f"expected 1 (B/C01 broken), got {rc}"
        check_stack(a, "00", n); check_stack(a, "01", n)
        check_stack(b, "00", n)
        stem = f"{b.name}-C01-fc{n:06d}"
        assert not (b / stem / f"{stem}.tif").exists(), "broken channel produced output"
        check_stack(c, "00", n)                       # fc from disk, not fc000000
        assert not list(c.rglob("*fc000000*")), "fc000000 stem"
        import re
        assert all(re.search(r"_t\d{10}\.ome\.tif$", p.name) for p in a.glob("*_t*.tif"))
        assert len(list(a.glob("*_C00_t000000000[0-9].ome.tif"))) == 10, "10-digit padding"
        doc = yaml.safe_load((a / f"{a.name}.yaml").read_text())
        st = doc["acquisition_system"]["settings"]
        assert st["n_frames"] == n and st["n_channels"] == 2 and st["fa"]["value"] == 1
        print("A/B/C ok")

        # D. interrupted write leaves nothing; rerun completes it
        d = make_session(root, f"{PREFIX}-000000-20150709-TL004_160000-25x-spont", n)
        real = s2b.load_frame
        calls = {"n": 0}
        def boom(*args, **kw):
            calls["n"] += 1
            if calls["n"] == 6:
                raise KeyboardInterrupt
            return real(*args, **kw)
        s2b.load_frame = boom
        try:
            run(root)
            raise AssertionError("KeyboardInterrupt should propagate")
        except KeyboardInterrupt:
            pass
        finally:
            s2b.load_frame = real
        assert not list(d.rglob("*-fc*.tif")), "interrupted write left a stack or partial"
        assert run(root) == 1                          # B still broken; D now completes
        check_stack(d, "00", n)
        print("D ok")

        # E. --delete-sources only removes frames for successful channels
        e = make_session(root, f"{PREFIX}-000000-20150709-TL005_170000-25x-spont", n, ("00", "01"), bad=("01", 3))
        run(root, "--delete-sources")
        left = sorted(p.name for p in e.glob("*_C00_t*.tif"))
        assert len(left) == 1 and left[0].endswith("_C00_t0000000000.ome.tif"), f"C00: only the OME master may remain, got {left}"
        assert len(list(e.glob("*_C01_t*.tif"))) == n, "C01 sources deleted despite failure"
        print("E ok")

        # F. stack-bigtiff CLI single-channel is atomic too
        out = root / "cli" / "stack.tif"
        sys.argv = ["stack-bigtiff", "--input", str(a), "--pattern", "*_C00_t*.tif",
                    "--output", str(out), "--flat-output", "--keep-sources",
                    "--preallocate", "--compression", "none"]
        s2b.main()
        assert tifffile.imread(out).shape == (n, H, W)
        assert not list(out.parent.glob("*.partial.tif"))
        print("F ok")
        print("ALL OK")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
