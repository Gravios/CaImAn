"""Offscreen test: ROI overlay opacity preferences."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
import importlib.util, json, os, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np

root = Path(tempfile.mkdtemp())
PREFS = root / "cfg" / "inspector_prefs.json"
os.environ["CAIMAN_INSPECTOR_PREFS"] = str(PREFS)
insp = load_inspector()
from PyQt6.QtWidgets import QApplication, QDialog
app = QApplication.instance() or QApplication([])


def alpha_max(img):
    a = np.asarray(img)[..., 3].astype(float)
    return a.max() / (255.0 if a.max() > 1.5 else 1.0)


def window():
    w = insp.InspectorWindow(insp._make_demo_store()); w.resize(780, 700); w.show(); app.processEvents()
    return w


try:
    w = window()
    assert insp.OVERLAY_ALPHA == insp.OVERLAY_ALPHA_DEFAULTS
    d = insp.PreferencesDialog(insp.load_prefs(), {}, None, "appearance", w)
    f = d._fields["appearance"]
    assert (f["alpha_selected"].value(), f["alpha_all"].value(), f["alpha_excluded"].value()) == (55, 12, 22)
    w._on_cell_click(2, False); app.processEvents()
    sel0 = alpha_max(w.cell_view._img_ov.image)
    all0 = alpha_max(w.cell_view._img_all.image)
    assert abs(sel0 - 0.55) < 0.02 and abs(all0 - 0.12) < 0.02, (sel0, all0)
    ok("defaults 55 / 12 / 22 % shown in Preferences and drawn as before")

    # mark one ROI excluded so the excluded alpha is used
    s = w.store
    s.qc = {"passed": np.ones(s.n, bool)}; s.approved[:] = True; s.approved[0] = False
    w.cell_view.refresh_rois(); app.processEvents()
    fp0 = s.footprint(0); pk = np.unravel_index(np.argmax(fp0), fp0.shape)
    a_ex = np.asarray(w.cell_view._img_all.image)[pk][3]
    a_ex = a_ex / 255.0 if a_ex > 1.5 else a_ex
    assert abs(a_ex - 0.22) < 0.03, a_ex

    def edit(dlg):
        ff = dlg._fields["appearance"]
        ff["alpha_selected"].set_value(90); ff["alpha_all"].set_value(0); ff["alpha_excluded"].set_value(50)
    insp.PreferencesDialog.exec = lambda self: (edit(self), QDialog.DialogCode.Accepted)[1]
    built = []
    orig_refresh = w._full_refresh
    w._full_refresh = lambda: (built.append(1), orig_refresh())[1]
    assert w._edit_prefs("appearance"); app.processEvents()
    saved = json.loads(PREFS.read_text())["appearance"]
    assert (saved["alpha_selected"], saved["alpha_all"], saved["alpha_excluded"]) == (90, 0, 50)
    assert insp.OVERLAY_ALPHA == {"selected": 0.9, "all": 0.0, "excluded": 0.5}
    assert abs(alpha_max(w.cell_view._img_ov.image) - 0.9) < 0.02
    a_ex = np.asarray(w.cell_view._img_all.image)[pk][3]
    a_ex = a_ex / 255.0 if a_ex > 1.5 else a_ex
    assert abs(a_ex - 0.5) < 0.03, a_ex
    others = np.asarray(w.cell_view._img_all.image).copy()
    fp_mask = np.zeros(fp0.shape, bool)
    fp_mask[fp0 > 0] = True
    assert alpha_max(others[~fp_mask][None]) < 0.01, "other ROIs invisible at 0 %"
    assert not built, "opacity-only change does not rebuild the table"
    assert w.table.selected_comps() == [2]
    ok("saving 90 / 0 / 50 % redraws the overlay live (selection kept, no table rebuild)")
    w.close()

    # invalid values clamp / fall back; startup applies saved values
    PREFS.write_text(json.dumps({"appearance": {"alpha_selected": 150, "alpha_all": "x",
                                                "alpha_excluded": -5}}))
    w = window()
    assert insp.OVERLAY_ALPHA == {"selected": 1.0, "all": 0.12, "excluded": 0.0}, insp.OVERLAY_ALPHA
    w.close()
    PREFS.write_text(json.dumps({"appearance": {"alpha_selected": 30}}))
    w = window(); w._on_cell_click(1, False); app.processEvents()
    assert abs(alpha_max(w.cell_view._img_ov.image) - 0.30) < 0.02
    w.close()
    ok("startup applies the saved opacity; out-of-range / invalid values clamp or fall back")
    print("ALL OK")
finally:
    shutil.rmtree(root)
