"""estimate-params: cell size per objective, patch geometry, sized stages,
and new-session's magnification / frame-rate / section plumbing."""
from _harness import REPO, ok  # noqa: F401
import importlib.util, math, sys

sys.path.insert(0, str(REPO))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


pe = load("params_estimator", "caiman/utils/params_estimator.py")   # numpy only at import
ns = load("new_session", "utilities/new_session.py")

# ── magnification → cell size ────────────────────────────────────────────────
for m, want in [("25x", "25x"), (25, "25x"), ("25", "25x"), (25.0, "25x"), ("16X", "16x"),
                ("00x", None), (0, None), ("", None), (None, None), ("zoom", None)]:
    assert pe.normalize_magnification(m) == want == ns.normalize_magnification(m), m
assert (pe.um_per_px("20x"), pe.um_per_px("16x"), pe.um_per_px("40x")) == (0.8, 1.0, 0.4)
assert pe.cell_size("20x") == pe.cell_size("25x") == (4, 17), "template"
assert pe.cell_size("16x") == (2, 9), "8 µm @ 1.0 µm/px = 8 px diameter"
assert pe.cell_size("40x") == (11, 45), "18 µm @ 0.4 µm/px = 45 px diameter"
assert pe.cell_size(None) is None and pe.cell_size("00x") is None and pe.cell_size("60x") is None
# template convention: gSiz = cell diameter in px (17 px x 0.8 µm/px = 13.6 µm)
for mag, d_um in pe.CELL_DIAMETER_UM.items():
    gSig, gSiz = pe.cell_size(mag)
    assert abs(gSiz * pe.um_per_px(mag) - d_um) <= 2 * pe.um_per_px(mag), (mag, gSiz)
ok("cell size: 20x/25x template (4/17), 16x 2/9 (8 µm), 40x 11/45 (18 µm); 00x / other → measured")

# ── geometry: template rf/stride unless the ring needs more ──────────────────
assert pe.patch_geometry(17) == (20, 10) and pe.patch_geometry(9) == (20, 10)
assert pe.patch_geometry(45) == (46, 23)
for gSiz in range(5, 70, 4):
    rf, stride = pe.patch_geometry(gSiz)
    assert rf > pe._RING_SIZE_FACTOR * gSiz + 3 and rf % 2 == 0 and stride == rf // 2 or (rf, stride) == (20, 10)
    assert rf > pe._RING_SIZE_FACTOR * gSiz, f"ring constraint at gSiz {gSiz}"
ok("rf / stride stay 20 / 10 unless ring_size_factor × gSiz needs more (40x: 46 / 23); "
   "ring constraint holds for every gSiz")

# ── stages that work in pixels ───────────────────────────────────────────────
s4 = pe.sized_stages(4)
assert s4 == {"seeding.radius": 4, "seeding.min_distance": 5, "seeding.smooth_sigma": 1.0,
              "seeding.min_pixels": 8, "seeding.diameter": 15, "roi_filter.edge_px": 10}, s4
s11, s2 = pe.sized_stages(11), pe.sized_stages(2)
assert s11["seeding.radius"] == 11 and s11["seeding.min_pixels"] == 60 and s11["roi_filter.edge_px"] == 28
assert s2["seeding.radius"] == 2 and s2["seeding.min_pixels"] >= 3 and s2["seeding.smooth_sigma"] >= 0.5
ok("seeding radius = gSig; min_distance / smooth_sigma / Cellpose diameter ∝ gSig, "
   "min_pixels ∝ gSig²; roi_filter.edge_px ∝ gSig; 20x = the template's values")

# ── no constant overrides of the template any more ───────────────────────────
src = (REPO / "caiman/utils/params_estimator.py").read_text()
body = src[src.index("    suggestions = {"):src.index('    logger.info(f"  Suggestions')]
for k in ("maxthr", "merge_thr", "ring_size_factor", "tsub", "ssub", "nb_patch", "normalize_init"):
    assert f'"{k}"' not in body, f"estimator still overrides template key {k}"
ok("the estimator no longer overwrites template constants (maxthr, merge_thr, tsub, …)")

# ── new-session: where the magnification comes from ─────────────────────────
S = "strohA-ia-000000-20151016-TL001_143915-25x-spont-C00-fc011170"
assert ns._resolve_magnification("40x", "20x", S) == ("40x", "--magnification")
assert ns._resolve_magnification(None, "16x", S) == ("16x", "YAML")
assert ns._resolve_magnification(None, "0x", S) == ("25x", "session name"), "YAML 0 = unknown"
assert ns._resolve_magnification(None, None, S) == ("25x", "session name")
assert ns._resolve_magnification(None, None, S.replace("-25x-", "-00x-")) == (None, "unknown")
assert ns._resolve_magnification(None, None, "odd_name") == (None, "unknown")
ok("new-session magnification: --magnification > YAML > session name (…-25x-…); 00x = unknown")

sug = {"gSig": [4, 4], "seeding.radius": 4, "roi_filter.edge_px": 10}
assert ns._for_sections(sug, {"cnmf": {}}) == {"gSig": [4, 4]}
assert ns._for_sections(sug, {"cnmf": {}, "seeding": {"enabled": False}, "roi_filter": {}}) == sug
ok("seeding / roi_filter sizes only written where the pipeline JSON has those sections")

nsrc = (REPO / "utilities/new_session.py").read_text()
call = nsrc[nsrc.index("suggestions = estimate_params("):]
call = call[:call.index(")\n")]
for arg in ("magnification_resolved", "gSig_hint     = args.gSig", "fr            ="):
    assert arg in call, f"estimate_params call misses {arg}"
assert "args.magnification," not in call, "the raw CLI flag must not reach the estimator"
ok("estimate_params gets the resolved magnification, the JSON's frame rate and --gSig")
print("ALL OK")
