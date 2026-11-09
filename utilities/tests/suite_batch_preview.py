"""batch-preview: one random frame per session / channel into <root>/preview/."""
from _harness import REPO, ok, require  # noqa: F401
require("tifffile")
import contextlib, io, shutil, struct, sys, tempfile, zlib
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(REPO))
from utilities import batch_preview as bp  # noqa: E402  (no caiman import)

H, W, N = 12, 16, 9
root = Path(tempfile.mkdtemp()) / "strohA-ia"


def frame(t, ch=0):
    """Frame t: a ramp plus t * 1000 (+ 50000 for C01), so value = identity."""
    return (np.arange(H * W, dtype=np.uint16).reshape(H, W) + t * 1000 + ch * 50000)


def read_png(p: Path) -> np.ndarray:
    b = p.read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", None, None
    while pos < len(b):
        n, = struct.unpack(">I", b[pos:pos + 4]); t = b[pos + 4:pos + 8]; d = b[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", b[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(t + d)
        if t == b"IHDR":
            w, h = struct.unpack(">II", d[:8]); assert d[8:10] == b"\x08\x00"
        elif t == b"IDAT":
            idat += d
        pos += 12 + n
    raw = zlib.decompress(idat)
    rows = [raw[r * (w + 1):(r + 1) * (w + 1)] for r in range(h)]
    assert all(r[0] == 0 for r in rows)
    return np.frombuffer(b"".join(r[1:] for r in rows), np.uint8).reshape(h, w)


def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = bp.main(["--root", str(root), *argv])
    return rc, buf.getvalue()


def snapshot():
    return sorted((str(p.relative_to(root)), p.stat().st_mtime_ns) for p in root.rglob("*")
                  if PREVIEW not in p.relative_to(root).parts)


PREVIEW = "preview"
try:
    # stacked session, two channels (+ kept raw frames, which must not win)
    s1 = root / "strohA-ia-000001" / "strohA-ia-000001-20151117" / "strohA-ia-000001-20151117-TL001_134746-25x-spont"
    for ch in (0, 1):
        stem = f"{s1.name}-C{ch:02d}-fc{N:06d}"
        (s1 / stem).mkdir(parents=True)
        tifffile.imwrite(s1 / stem / f"{stem}.tif", np.stack([frame(t, ch) for t in range(N)]), bigtiff=True)
        (s1 / stem / f"{stem}.partial.tif").write_bytes(b"")                  # leftover
    tifffile.imwrite(s1 / "tl1_C00_t0000000000.ome.tif", np.zeros((H, W), np.uint16))
    # raw frame files only (not stacked yet)
    s2 = root / "strohA-ia-000001" / "strohA-ia-000001-20151118" / "strohA-ia-000001-20151118-TL002_101010-25x-vstim"
    s2.mkdir(parents=True)
    for t in range(N):
        tifffile.imwrite(s2 / f"tl2_C00_t{t:010d}.ome.tif", frame(t))
    # OME stack, other subject; its XML maps every plane to IFD 0 (an OME read = frame 0)
    s3 = root / "strohA-ia-000002" / "strohA-ia-000002-20140206" / "strohA-ia-000002-20140206-TL001_000000-00x-spont"
    s3.mkdir(parents=True)
    xml = ('<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" UUID="urn:uuid:s">'
           f'<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="{W}" SizeY="{H}" '
           f'SizeZ="1" SizeC="1" SizeT="{N}"><Channel ID="Channel:0:0" SamplesPerPixel="1"/>'
           + "".join(f'<TiffData FirstT="{t}" IFD="0" PlaneCount="1"><UUID FileName="x_C00.ome.tif">urn:uuid:s</UUID></TiffData>'
                     for t in range(N)) + "</Pixels></Image></OME>")
    with tifffile.TiffWriter(s3 / "x_C00.ome.tif", bigtiff=True) as w:
        for t in range(N):
            w.write(frame(t), description=xml if t == 0 else None, metadata=None, contiguous=False)
    # legacy-named session dir (not renamed yet)
    s4 = root / "strohA-ia-000002" / "strohA-ia-000002-20140225" / "140225_tl1_fpstim_z90_p1_50_15-56-08"
    s4.mkdir(parents=True)
    for t in range(3):
        tifffile.imwrite(s4 / f"tl1_C00_t{t:010d}.ome.tif", frame(t))
    (root / "strohA-ia-000002" / "notes").mkdir()                          # no data: ignored
    before = snapshot()

    rc, out = run("--dry-run")
    assert rc == 0 and out.count("would preview") == 5 and not (root / PREVIEW).exists(), out
    ok("--dry-run lists 5 previews (4 sessions, C00 + C01 for the stacked one), writes nothing")

    rc, out = run("--seed", "3")
    pv = sorted((root / PREVIEW).iterdir())
    assert rc == 0 and len(pv) == 5 and all(p.is_file() and p.suffix == ".png" for p in pv), out
    assert not [p for p in (root / PREVIEW).iterdir() if p.is_dir()], "flat"
    names = {p.name.rsplit("-t", 1)[0]: p for p in pv}
    assert set(names) == {f"{s1.name}-C00", f"{s1.name}-C01", f"{s2.name}-C00",
                          f"{s3.name}-C00", f"{s4.name}-C00"}, names
    ok("one PNG per session / channel, flat in <root>/preview, named <session>-C<NN>-t<frame>")

    # content: the image is the named frame, contrast-stretched (ramp -> 0..255)
    ramp = bp.to_8bit(frame(0))
    for key, p in names.items():
        img = read_png(p)
        assert img.shape == (H, W) and np.array_equal(img, ramp), key
        assert img.min() == 0 and img.max() == 255
    t3 = int(names[f"{s3.name}-C00"].stem.rsplit("-t", 1)[1])
    with tifffile.TiffFile(s3 / "x_C00.ome.tif", is_ome=False) as tf:
        assert tf.pages[t3].asarray()[0, 0] == t3 * 1000, "named frame is the file's page"
    _, f1 = bp.pick_frame("stack", [s1 / f"{s1.name}-C01-fc{N:06d}" / f"{s1.name}-C01-fc{N:06d}.tif"],
                          bp.random.Random(0))
    _, f0 = bp.pick_frame("frames", sorted(s2.glob("*.tif")), bp.random.Random(0))
    _, fo = bp.pick_frame("ome", [s3 / "x_C00.ome.tif"], bp.random.Random(5))
    idx_o = bp.random.Random(5).randrange(N)
    assert f1[0, 0] >= 50000 and f0[0, 0] % 1000 == 0 and fo[0, 0] == idx_o * 1000, \
        "C01 read from C01; OME stack read by page, not through its XML"
    ok("each PNG is the named frame (stack page / frame file / OME page, OME-XML ignored), "
       "scaled 0.5–99.5 percentile to 0–255; stacked BigTIFF wins over kept raw frames")

    assert snapshot() == before, "nothing outside preview/ changed"
    ok("nothing outside preview/ written or touched")

    first = sorted(p.name for p in (root / PREVIEW).iterdir())
    rc, out = run()
    assert sorted(p.name for p in (root / PREVIEW).iterdir()) == first and "kept 5 existing" in out
    rc, out = run("--overwrite", "--seed", "11")
    again = sorted(p.name for p in (root / PREVIEW).iterdir())
    assert len(again) == 5 and again != first, "new frames, still one per channel"
    a1 = sorted(p.name for p in (root / PREVIEW).iterdir())
    shutil.rmtree(root / PREVIEW); run("--seed", "11")
    assert sorted(p.name for p in (root / PREVIEW).iterdir()) == a1, "--seed reproduces"
    ok("rerun keeps existing previews; --overwrite redraws (one per channel); --seed reproduces")

    shutil.rmtree(root / PREVIEW)
    rc, out = run("--mean", "3", "--channel", "1")
    pv = list((root / PREVIEW).iterdir())
    assert len(pv) == 1 and pv[0].name.startswith(f"{s1.name}-C01-t") and pv[0].name.endswith("-m3.png")
    t = int(pv[0].name.split("-t")[1].split("-")[0])
    assert t <= N - 3
    _, m = bp.pick_frame("stack", [s1 / f"{s1.name}-C01-fc{N:06d}" / f"{s1.name}-C01-fc{N:06d}.tif"],
                         bp.random.Random(2), mean=3)
    assert m.dtype == np.float64 and m[0, 0] % 1000 == 0 and m[0, 0] > 50000, "mean of 3 frames"
    ok("--mean K averages K consecutive frames (name -m<K>); --channel filters")

    bad = root / "strohA-ia-000003" / "d" / "s"; bad.mkdir(parents=True)
    (bad / "y_C00_t0000000000.ome.tif").write_bytes(b"not a tiff")
    rc, out = run("--overwrite")
    assert rc == 1 and "ERROR" in out and "s C00" in out.replace("/", " ")
    ok("an unreadable session is reported (exit 1), the others still written")
    assert bp.main(["--root", str(root / "strohA-ia-000002" / "notes")]) == 1
    print("ALL OK")
finally:
    shutil.rmtree(root.parent, ignore_errors=True)
