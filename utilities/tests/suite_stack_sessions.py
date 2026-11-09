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

        # G. existing OME stacks <name>_C<NN>.ome.tif -> plain BigTIFF layout
        g_root = root / "stacks"; g_root.mkdir()
        P2 = "stroh-ej"
        src_name = "stroh-ej-000000-20140206-TL001_00115um_012pp_spont"

        def stack_xml(size_t, size_z=1, size_c=1, files=None):
            # files: every plane mapped to IFD 0 of this file (its own UUID)
            td = "".join(f'<TiffData FirstT="{t}" IFD="0" PlaneCount="1">'
                         f'<UUID FileName="{files}">urn:uuid:self</UUID></TiffData>'
                         for t in range(size_t)) if files else ""
            return ('<?xml version="1.0" encoding="UTF-8"?>'
                    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"'
                    ' UUID="urn:uuid:self">'
                    '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" '
                    f'Type="uint16" SizeX="{W}" SizeY="{H}" SizeZ="{size_z}" SizeC="{size_c}" '
                    f'SizeT="{size_t}" TimeIncrement="0.05" TimeIncrementUnit="s" '
                    'PhysicalSizeX="0.5" PhysicalSizeY="0.5">'
                    '<Channel ID="Channel:0:0" SamplesPerPixel="1"/>' + td +
                    '</Pixels></Image></OME>')

        def make_stack(name, n_pages, xml, c="00", bigtiff=True, stem=None):
            d = g_root / name; d.mkdir(exist_ok=True)
            p = d / f"{stem or name}_C{c}.ome.tif"
            with tifffile.TiffWriter(p, bigtiff=bigtiff) as w:
                for t in range(n_pages):
                    w.write(np.full((H, W), t + 100 * int(c), np.uint16),
                            description=xml if t == 0 else None, metadata=None,
                            contiguous=False)
            return d, p

        # G1: the XML maps every plane to IFD 0 of this file, so an OME reader
        # (CaImAn's) returns the right shape with frame 0 repeated, silently;
        # the conversion must not
        g1, p1 = make_stack(f"{P2}-000000-20140206-TL001_000000-00x-spont", n,
                            stack_xml(n, files=f"{src_name}_C00.ome.tif"), stem=src_name)
        with tifffile.TiffFile(p1) as tf:
            trap = tf.asarray()
        truth = np.stack([np.full((H, W), t, np.uint16) for t in range(n)])
        assert trap.shape == (n, H, W) and (trap == truth[0]).all(), \
            "test premise: an OME read of this stack repeats frame 0"
        # G2 SizeZ=2 (volume) / G3 more pages than SizeT / G4 fewer pages
        g2, _ = make_stack(f"{P2}-000000-20140206-TL002_000000-00x-spont", n, stack_xml(n // 2, 2))
        g3, _ = make_stack(f"{P2}-000000-20140206-TL003_000000-00x-spont", n, stack_xml(n - 2))
        g4, _ = make_stack(f"{P2}-000000-20140206-TL004_000000-00x-spont", n - 3, stack_xml(n))
        # G5 classic TIFF; XML says SizeC=2 (this file is one channel of it)
        g5, p5 = make_stack(f"{P2}-000000-20140206-TL005_000000-00x-spont", n,
                            stack_xml(n, size_c=2), bigtiff=False)
        # G6 frames and a stack for the same channel
        g6 = make_session(g_root, f"{P2}-000000-20140206-TL006_000000-00x-spont", n)
        make_stack(g6.name, n, stack_xml(n))
        # G7 two channels, one fine, one refused
        g7, _ = make_stack(f"{P2}-000000-20140206-TL007_000000-00x-spont", n, stack_xml(n), "00")
        make_stack(g7.name, n + 1, stack_xml(n), "01")

        def runp(*extra):
            return ss.main(["--prefix", P2, "--parent", str(g_root), *extra])
        assert runp("--dry-run") == 1                  # G2-G4, G6, G7/C01 refused
        assert not list(g_root.rglob("*-fc*")) and not list(g_root.rglob("*.yaml")) \
            and not list(g_root.rglob("*.ome.xml")), "dry run wrote something"

        assert runp() == 1
        for d in (g1, g5, g7):
            out = check_stack(d, "00", n)
            with tifffile.TiffFile(out) as tf:
                assert tf.is_bigtiff and not tf.is_ome, "plain BigTIFF, no OME-XML"
        assert p1.exists() and p5.exists(), "sources kept without --delete-sources"
        for d in (g2, g3, g4, g6):
            assert not list(d.glob("*-fc*")), f"{d.name}: refused stack converted"
        for d in (g2, g3, g4):
            assert not list(d.glob("*.yaml")) and not list(d.glob("*.ome.xml")), \
                f"{d.name}: YAML / XML written from the header of a refused stack"
        assert not list(g7.glob("*-C01-fc*"))
        doc = yaml.safe_load((g1 / f"{g1.name}.yaml").read_text())
        st = doc["acquisition_system"]["settings"]
        assert (st["n_frames"], st["sample_rate"]["value"], st["depth"]["value"],
                st["n_channels"]) == (n, 20.0, -115, 1), st
        assert (g1 / f"{g1.name}.ome.xml").read_text().startswith("<?xml"), "XML archived"

        assert runp("--allow-incomplete") == 1         # G4 now converts
        check_stack(g4, "00", n - 3)

        # verification failure: partial removed, source kept, nothing finished
        g8, p8 = make_stack(f"{P2}-000000-20140206-TL008_000000-00x-spont", n, stack_xml(n))
        real_eq = s2b.np.array_equal
        s2b.np.array_equal = lambda *a, **k: False
        try:
            runp()
        finally:
            s2b.np.array_equal = real_eq
        assert p8.exists() and not list(g8.glob("*-fc*/*.tif")), "failed verification left a stack"
        assert not list(g8.rglob("*.partial.tif"))

        runp("--delete-sources")                       # rerun: G8 converts, its source goes
        check_stack(g8, "00", n)
        assert not p8.exists(), "--delete-sources removes the converted source stack"
        assert p1.exists(), "an already converted session's stack is left alone"
        assert len(list(g7.glob("*_C01.ome.tif"))) == 1, "refused channel kept"
        print("G ok: OME stacks converted from their pages (OME-XML ignored, archived), "
              "volume / SizeT mismatch / frames+stack refused, --allow-incomplete, "
              "verified before rename, --delete-sources")
        print("ALL OK")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
