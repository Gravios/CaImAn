"""Run every ``suite_*.py`` in this folder in its own process.

    pytest utilities/tests            # all suites
    python utilities/tests/suite_x.py # one suite, verbose

Each suite runs offscreen with a throw-away inspector preferences file and
the light ``caiman`` stand-in from ``_harness`` (no compiled CaImAn stack).
A suite that exits with ``_harness.SKIP`` (a missing optional dependency
such as PyQt6 or tifffile) is reported as skipped.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _harness import SKIP  # noqa: E402

SUITES = sorted(p.name for p in HERE.glob("suite_*.py"))


@pytest.mark.parametrize("suite", SUITES)
def test_suite(suite, tmp_path):
    env = dict(os.environ,
               QT_QPA_PLATFORM="offscreen",
               CAIMAN_INSPECTOR_PREFS=str(tmp_path / "inspector_prefs.json"),
               XDG_CONFIG_HOME=str(tmp_path / "config"),
               PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(HERE / suite)], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    if r.returncode == SKIP:
        pytest.skip(out.strip().splitlines()[-1] if out.strip() else "skipped")
    assert r.returncode == 0 and "ALL OK" in r.stdout, out[-4000:]
