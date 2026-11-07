"""Inspector entry points: bin/ shim, installed launcher, python -m, and the
lazy loaders' fallback to this checkout when caiman can't be imported."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
import importlib.util, os, subprocess, sys
from pathlib import Path

insp = load_inspector()

# bin/caiman_inspector.py loaded by path re-exports the package's names
spec = importlib.util.spec_from_file_location("shim", REPO / "bin" / "caiman_inspector.py")
shim = importlib.util.module_from_spec(spec); spec.loader.exec_module(shim)
for name in ("InspectorWindow", "ComponentStore", "load_from_hdf5", "main", "_make_demo_store",
             "_hex_rgb", "PALETTE_HEX", "OVERLAY_ALPHA", "theme", "store"):
    assert getattr(shim, name) is getattr(insp, name), name
assert shim.store.ComponentStore is insp.ComponentStore
assert shim.PALETTE_HEX is insp.constants.PALETTE_HEX, "mutable tables shared, not copied"
ok("bin/caiman_inspector.py re-exports every name (same objects)")

# all three command lines reach the same argparse
env = dict(os.environ, PYTHONPATH=str(REPO))
cmds = {"shim": [sys.executable, str(REPO / "bin" / "caiman_inspector.py"), "--help"],
        "-m": [sys.executable, "-m", "utilities.inspector", "--help"],
        "launcher": [sys.executable, "-c",
                     "import sys; from utilities.inspector_launcher import main; sys.exit(main(['--help']))"]}
for k, cmd in cmds.items():
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "usage: caiman-inspector" in r.stdout, (k, r.stdout, r.stderr[-1500:])
ok("python bin/caiman_inspector.py, python -m utilities.inspector, caiman-inspector launcher")

# without an importable caiman, deps fall back to caiman/utils/*.py of this checkout
code = ("import sys; sys.modules['caiman'] = None\n"
        "from utilities.inspector import deps\n"
        "print(deps._ed().__file__); print(deps._rf().__file__)")
r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
assert r.returncode == 0, r.stderr[-1500:]
files = [Path(x).resolve() for x in r.stdout.split()]
assert files == [(REPO / "caiman/utils/event_detection.py").resolve(),
                 (REPO / "caiman/utils/roi_filter.py").resolve()], files
ok("event_detection / roi_filter load from this checkout when caiman can't import")

# store, overlay, prefs, constants, deps don't import Qt
code = ("import sys; sys.modules['caiman'] = None\n"
        "import importlib.util, types, pathlib\n"
        "pkg = types.ModuleType('utilities.inspector'); pkg.__path__ = ['utilities/inspector']\n"
        "sys.modules['utilities.inspector'] = pkg\n"
        "import utilities.inspector.store, utilities.inspector.overlay, utilities.inspector.prefs\n"
        "print(sorted(m for m in sys.modules if m.startswith(('PyQt6', 'pyqtgraph'))))")
r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
assert r.returncode == 0 and r.stdout.strip() == "[]", (r.stdout, r.stderr[-1500:])
ok("store / overlay / prefs / constants / deps are Qt-free")
print("ALL OK")
