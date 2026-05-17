#!/usr/bin/env python
"""Regression tests for the pipeline-spine AST-audit repair series.

Tests three behavioural fixes that affect every pipeline run (boot,
SHM lifecycle, cleanup hooks):

1. ``test_clean_stale_shm_preserves_live_kmp_registrations``
       MEDIUM — clean_stale_shm() must NOT delete
       ``__KMP_REGISTERED_LIB_<pid>`` files belonging to an alive
       process, nor ``sem.loky-<pid>`` ditto.  Files belonging to a
       dead PID must still be removed.

2. ``test_load_to_shm_is_atomic_against_midcopy_kill``
       MEDIUM — load_to_shm() must use a .tmp + rename pattern so a
       crash mid-copy leaves a .tmp file (not a corrupt shm_path), and
       the next call must detect and clean up the orphaned .tmp.

3. ``test_cupy_register_cleanup_is_idempotent``
       LOW — cupy_register_cleanup() must register at most one atexit
       handler regardless of how many times it is called.  Was
       previously documented as idempotent but wasn't.

No CaImAn runtime needed (no GPU, no movies, no cluster).
"""

import os
import re
import tempfile
import atexit
from pathlib import Path

import pytest

from caiman.utils.pipeline_setup import clean_stale_shm
from caiman.utils.shm_movie import load_to_shm
from caiman.utils import memory as memory_mod


# ── 1. clean_stale_shm: preserves live-PID transient files ───────────────────

def test_clean_stale_shm_preserves_live_kmp_registrations(tmp_path, caplog):
    """clean_stale_shm() with mixed alive/dead-PID KMP files: the alive-PID
    file is kept; the dead-PID file and any non-PID transient (psm_*) are
    removed."""
    shm  = tmp_path / "shm";  shm.mkdir()
    temp = tmp_path / "temp"; temp.mkdir()

    live_pid = os.getpid()

    # A pid almost certainly not alive on this system.  We pick a value
    # well outside any plausible PID range so the test is robust against
    # the unlikely coincidence of a recycled PID.
    dead_pid = 2_000_001

    live_kmp  = shm  / f"__KMP_REGISTERED_LIB_{live_pid}"
    dead_kmp  = shm  / f"__KMP_REGISTERED_LIB_{dead_pid}"
    live_loky = shm  / f"sem.loky-{live_pid}"
    dead_loky = shm  / f"sem.loky-{dead_pid}"
    psm       = shm  / "psm_abcd1234"   # no PID — always cleaned

    for f in (live_kmp, dead_kmp, live_loky, dead_loky, psm):
        f.write_bytes(b"x")

    import logging
    log = logging.getLogger("test_clean_stale_shm")

    clean_stale_shm(str(shm), str(temp), log)

    assert live_kmp.exists(),  "live-PID KMP file must be preserved"
    assert live_loky.exists(), "live-PID loky file must be preserved"
    assert not dead_kmp.exists(),  "dead-PID KMP file must be cleaned"
    assert not dead_loky.exists(), "dead-PID loky file must be cleaned"
    assert not psm.exists(),       "psm_* always cleaned (no PID in name)"


# ── 2. load_to_shm: atomic rename against mid-copy kill ──────────────────────

def test_load_to_shm_is_atomic_against_midcopy_kill(tmp_path):
    """If a previous run was killed mid-copy, an orphan .tmp file is left
    at <shm_path>.tmp.  load_to_shm() must remove it on entry, NOT mistake
    its size + mtime for a valid SHM copy, and re-do the copy cleanly."""
    src     = tmp_path / "src.mmap"
    shm_dir = tmp_path / "shm"; shm_dir.mkdir()

    # Source: 64 KB of deterministic content
    payload = bytes(range(256)) * 256
    src.write_bytes(payload)
    expected_size = len(payload)

    # Orphan .tmp from a previous "crash" — full size, recent mtime,
    # but contains only zeros (the failure mode the previous code
    # treated as a valid reuse target).
    shm_path = shm_dir / src.name
    shm_tmp  = shm_dir / (src.name + ".tmp")
    shm_tmp.write_bytes(b"\x00" * expected_size)

    out = load_to_shm(str(src), session="ignored", shm_dir=str(shm_dir))
    assert out == str(shm_path)

    # Final file must exist and contain the source bytes
    assert shm_path.exists(),     "SHM copy must be at final path"
    assert shm_path.read_bytes() == payload, "SHM copy must match source"

    # .tmp must NOT survive the call (either consumed by the rename or
    # cleaned at the top of load_to_shm)
    assert not shm_tmp.exists(),  "orphaned .tmp must be cleaned"


def test_load_to_shm_uses_tmp_then_rename(tmp_path, monkeypatch):
    """Verify the .tmp + rename order: _fast_copy must write to .tmp,
    NOT to the final shm_path."""
    src     = tmp_path / "src.mmap"
    shm_dir = tmp_path / "shm"; shm_dir.mkdir()
    src.write_bytes(b"data" * 1024)

    captured = {"dst": None}

    import caiman.utils.shm_movie as shm_mod
    real_copy = shm_mod._fast_copy

    def spy_copy(s, d, log):
        captured["dst"] = d
        real_copy(s, d, log)

    monkeypatch.setattr(shm_mod, "_fast_copy", spy_copy)

    out = load_to_shm(str(src), session="ignored", shm_dir=str(shm_dir))

    assert captured["dst"].endswith(".tmp"), \
        f"_fast_copy must write to a .tmp file, not directly to shm_path; got {captured['dst']}"
    assert out == str(shm_dir / src.name)


# ── 3. cupy_register_cleanup is idempotent ───────────────────────────────────

def test_cupy_register_cleanup_is_idempotent(monkeypatch):
    """Repeated calls must register at most one atexit handler.

    We monkey-patch atexit.register to count invocations rather than
    actually register cleanups (which would fire at test-process exit).
    """
    calls = []

    def fake_register(fn):
        calls.append(fn)
        return fn

    monkeypatch.setattr(atexit, "register", fake_register)
    # Reset module-level guard so we exercise the fresh-register path on
    # the first call.
    monkeypatch.setattr(memory_mod, "_cupy_cleanup_registered", False)

    memory_mod.cupy_register_cleanup()
    memory_mod.cupy_register_cleanup()
    memory_mod.cupy_register_cleanup()
    memory_mod.cupy_register_cleanup()

    assert len(calls) == 1, \
        f"cupy_register_cleanup must register at most one handler; got {len(calls)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
