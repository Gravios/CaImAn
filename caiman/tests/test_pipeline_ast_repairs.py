#!/usr/bin/env python
"""Regression tests for the CNMF-pipeline AST-audit repair series.

Each test below pins behaviour that a previous bug silently violated.
If the test fails again in the future, the bug has reappeared.

Background: the original audit and patch series fixed three real
behavioural bugs in the CNMF pipeline (one critical, two high).
These tests are minimal — no CaImAn data, no GPU, no cluster.

Tests
-----
1. ``test_paramio_gsiz_autocorrect_preserves_all_keys``
       CRITICAL — gSiz auto-correct used to wipe every other CNMF
       param by reassigning ``c`` to ``SimpleNamespace(**vars(c))``,
       which captured only ``{"_data": {...}}`` because ParamBag stores
       its dict in a private ``_data`` attribute.

2. ``test_parambag_strips_comment_keys_at_every_depth``
       REFACTOR — ParamBag.__init__ now drops ``_*`` keys at every
       depth so nested sections are safe to wildcard-unpack as kwargs.

3. ``test_cluster_nprocesses_access_tolerates_missing_keys``
       HIGH — old one-liner crashed with AttributeError when the
       ``cluster`` section existed but ``n_processes`` did not.

4. ``test_cnmf_runner_evaluate_does_not_terminate_dview``
       HIGH — static source check: CNMFRunner.evaluate() must not
       call ``.terminate()`` or ``.join()`` on the dview, per the
       class docstring's lifecycle contract.
"""

import ast
import pytest
from pathlib import Path

from caiman.utils.params_io import ParamBag, build_cnmf_opts  # noqa: F401
import caiman.utils.cnmf_runner as cnmf_runner


# ── 1. CRITICAL: gSiz auto-correct must preserve every other CNMF key ────────

def test_paramio_gsiz_autocorrect_preserves_all_keys():
    """The gSiz auto-correct path in build_cnmf_opts must keep all
    other JSON-supplied CNMF params accessible via getattr(c, key).

    The pre-fix implementation rebound ``c`` to
    ``SimpleNamespace(**vars(c))``, which (because ParamBag stores
    its data in a private ``_data`` attribute) silently dropped
    every key.  Every getattr(c, "K", 30) downstream then fell back
    to the hardcoded default, undoing the user's JSON tuning.
    """
    cnmf_dict = {
        "p": 1, "gnb": 2, "merge_thr": 0.85, "rf": 20, "stride": 10, "K": 15,
        "gSig": [6, 6], "gSiz": [17, 17],  # mismatched on purpose
        "ring_size_factor": 0.9, "min_corr": 0.4, "min_pnr": 6.0,
        "method_init": "corr_pnr",
    }
    c = ParamBag(cnmf_dict)

    # Simulate the auto-correct branch as build_cnmf_opts performs it
    # post-fix: a single in-place attribute assignment.
    c.gSiz = [25, 25]

    # gSiz is patched
    assert getattr(c, "gSiz") == (25, 25), \
        "post-patch gSiz must reflect the corrected value"

    # Every other JSON-supplied key must survive the patch
    assert getattr(c, "K", "MISSING")          == 15
    assert getattr(c, "gSig", "MISSING")       == (6, 6)
    assert getattr(c, "merge_thr", "MISSING")  == 0.85
    assert getattr(c, "rf", "MISSING")         == 20
    assert getattr(c, "stride", "MISSING")     == 10
    assert getattr(c, "method_init", "MISSING") == "corr_pnr"
    assert getattr(c, "ring_size_factor", "MISSING") == 0.9
    assert getattr(c, "min_corr", "MISSING")   == 0.4
    assert getattr(c, "min_pnr", "MISSING")    == 6.0

    # hasattr() must still return True for keys probed via the
    # **({"min_corr": c.min_corr} if hasattr(c, "min_corr") else {})
    # pattern in build_cnmf_opts — otherwise corr_pnr's sparse-2P
    # tuning silently drops out of CNMFParams.
    assert hasattr(c, "min_corr")
    assert hasattr(c, "min_pnr")


# ── 2. REFACTOR: ParamBag strips _* keys at every depth ──────────────────────

def test_parambag_strips_comment_keys_at_every_depth():
    """ParamBag must drop ``_*`` keys at every nesting level so that
    iteration via ``items()`` is safe to splat as kwargs.

    The pipeline does

        MotionCorrect(fnames, ..., **{k: v for k, v in
            _P.motion_correction.items() if not k.startswith("_")})

    and prior to the refactor that filter was load-bearing; without
    it, ``_comment`` from inside the section would have leaked into
    MotionCorrect.__init__ and crashed.  Now the contract is
    centralized: a ParamBag never contains comment keys.
    """
    raw = {
        "_comment":   "top-level comment",
        "k":          1,
        "section": {
            "_comment":     "section comment",
            "_comment_two": "another",
            "x":            42,
            "nested": {
                "_inner": "deep comment",
                "y":      "preserved",
            },
        },
    }
    b = ParamBag(raw)

    # Top level
    assert list(b.items()) == [("k", 1), ("section", b.section)]
    assert "_comment" not in b

    # First nesting
    assert list(b.section.items()) == [("x", 42), ("nested", b.section.nested)]
    assert "_comment" not in b.section
    assert "_comment_two" not in b.section

    # Deeper nesting
    assert list(b.section.nested.items()) == [("y", "preserved")]
    assert "_inner" not in b.section.nested


# ── 3. HIGH: tolerant cluster.n_processes access ─────────────────────────────

def test_cluster_nprocesses_access_tolerates_missing_keys():
    """The replacement two-step getattr pattern in template_pipeline.py
    must survive (a) missing ``cluster`` section, (b) section present
    but ``n_processes`` key absent, and (c) section + key both present.

    ParamBag.__getattr__ raises AttributeError on missing keys
    (it does NOT return None), so the old one-liner
        _json_n = getattr(_P, "cluster", None) and _P.cluster.n_processes or None
    crashed in case (b).
    """
    def access(P):
        # Exact mirror of the patched expression in template_pipeline.py
        _cluster_cfg = getattr(P, "cluster", None)
        return getattr(_cluster_cfg, "n_processes", None) if _cluster_cfg else None

    # (a) no cluster section
    P_a = ParamBag({"data": {"fr": 30}})
    assert access(P_a) is None

    # (b) cluster section present, n_processes key absent — was the crash case
    P_b = ParamBag({"cluster": {"ram_budget_frac": 0.95}})
    assert access(P_b) is None

    # (c) value present
    P_c = ParamBag({"cluster": {"n_processes": 128}})
    assert access(P_c) == 128

    # (d) explicit null in JSON → None in ParamBag (since JSON null → Python None)
    P_d = ParamBag({"cluster": {"n_processes": None}})
    assert access(P_d) is None


# ── 4. HIGH: CNMFRunner.evaluate() must NOT terminate the dview ──────────────

def test_cnmf_runner_evaluate_does_not_terminate_dview():
    """CNMFRunner's class docstring states:

        The CNMFRunner does not own the cluster lifecycle — the
        pipeline is responsible for cm.cluster.setup_cluster and
        cm.stop_server so the finally block remains in the pipeline
        where it belongs.

    Prior to the fix, evaluate() violated this contract by calling
    ``_eval_dview.terminate()`` and ``_eval_dview.join()`` after
    running evaluate_components.  The pipeline's downstream
    ``cm.stop_server(dview=dview)`` then ran on an already-terminated
    pool.  Pin the documented contract via a static source check.
    """
    src = Path(cnmf_runner.__file__).read_text()
    tree = ast.parse(src)

    evaluate_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            evaluate_func = node
            break
    assert evaluate_func is not None, "CNMFRunner.evaluate() must exist"

    # Walk evaluate()'s subtree and verify no .terminate / .join call
    # targets the dview.  Allow .join on collections (e.g., str.join);
    # the dview-killing pattern is specifically attribute access of
    # those names on the local ``_eval_dview`` or ``cnm2.dview``.
    forbidden = []
    for sub in ast.walk(evaluate_func):
        if not isinstance(sub, ast.Call):
            continue
        if not isinstance(sub.func, ast.Attribute):
            continue
        if sub.func.attr not in ("terminate", "join"):
            continue
        # Resolve the receiver's root name to filter the offending pattern
        recv = sub.func.value
        while isinstance(recv, ast.Attribute):
            recv = recv.value
        if isinstance(recv, ast.Name) and recv.id in (
            "_eval_dview", "cnm2", "dview", "self",
        ):
            forbidden.append(
                f"L{sub.lineno}: {ast.unparse(sub)}"
            )

    assert not forbidden, (
        "CNMFRunner.evaluate() must not terminate / join the dview "
        "(it is owned by the pipeline).  Found:\n  "
        + "\n  ".join(forbidden)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
