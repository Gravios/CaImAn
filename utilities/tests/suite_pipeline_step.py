"""Executes the real 'ROI filter' block from template_pipeline.py."""
from _harness import REPO, ok, require, stub_caiman  # noqa: F401
require('numpy', 'scipy', 'h5py')
stub_caiman()
import json, logging, re, sys, tempfile, types
from pathlib import Path
import numpy as np, h5py, scipy.sparse
from caiman.utils.params_io import load_pipeline_params
from caiman.utils.timing import PipelineTimer

src = (REPO / "utilities/pipelines/template_pipeline.py").read_text()
block = src[src.index("    # ── 6b. ROI filter"):src.index("    # ── 7. Oscillation analysis")]
block = "\n".join(l[4:] for l in block.splitlines())          # dedent one level

d1, d2, K, T = 40, 50, 4, 900
yy, xx = np.mgrid[:d1, :d2]
A = np.column_stack([np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8).ravel(order="F")
                     for cy, cx in ((20, 25), (2, 25), (20, 10), (30, 40))]).astype(np.float32)
A[A < 1e-3] = 0
rng = np.random.default_rng(1)
F = 0.01 * rng.standard_normal((K, T)); C = np.abs(0.001 * rng.standard_normal((K, T)))
for t0 in (100, 400, 700):
    F[0, t0:t0 + 40] += 0.5 * np.exp(-np.arange(40) / 10); C[0, t0:t0 + 40] += 0.5 * np.exp(-np.arange(40) / 10)
est = types.SimpleNamespace(A=scipy.sparse.csc_matrix(A), dims=(d1, d2), C=C, F_dff=F,
                            SNR_comp=np.array([4.0, 4.0, 4.0, 0.3]), r_values=np.array([0.9, 0.9, 0.9, 0.1]),
                            cnn_preds=None)
cparams = types.SimpleNamespace(quality={"min_SNR": 1.5, "SNR_lowest": 0.5, "rval_thr": 0.6,
                                         "rval_lowest": -1.0, "use_cnn": True, "min_cnn_thr": 0.6,
                                         "cnn_lowest": 0.1},
                                init={"gSig": (2, 2)}, data={"fr": 30.0, "decay_time": 0.4})
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def run(cfg_path, outdir, session):
    ns = dict(_P=load_pipeline_params(cfg_path), timer=PipelineTimer(logging.getLogger("t")),
              cnm2=types.SimpleNamespace(estimates=est, params=cparams), outdir=outdir,
              session=session, logger=logging.getLogger("pipe"), _extra={}, Path=Path)
    exec(block, ns)
    return ns["_extra"]

with tempfile.TemporaryDirectory() as td:
    td = Path(td); res = td / "S_results.hdf5"
    with h5py.File(res, "w") as h: h.create_dataset("estimates/C", data=C)
    extra = run(REPO / "utilities/pipelines/template_pipeline.json", td, "S")
    print("report line:", extra.get("ROI filter"))
    with h5py.File(res, "r") as h:
        g = h["roi_filter"]
        prm = json.loads(g.attrs["params"])
        assert g["passed"][()].tolist() == [True, False, True, False], g["passed"][()]
        assert g["active"][()].tolist() == [True, False, False, False]
        assert prm["min_SNR"] == 1.5 and prm["gSig"] == [2, 2] and prm["fr"] == 30.0
        assert "estimates" in h
    print("ok  template JSON: group written, quality thresholds from the run, report line")

    # config without a roi_filter section → defaults, still runs
    raw = json.loads((REPO / "utilities/pipelines/template_pipeline.json").read_text())
    raw.pop("roi_filter"); old = td / "old.json"; old.write_text(json.dumps(raw))
    with h5py.File(td / "O_results.hdf5", "w") as h: h.create_dataset("estimates/C", data=C)
    extra = run(old, td, "O")
    with h5py.File(td / "O_results.hdf5", "r") as h:
        assert "roi_filter" in h
    print("ok  old config without roi_filter section: runs with defaults")

    # disabled
    raw["roi_filter"] = {"enabled": False}; dis = td / "dis.json"; dis.write_text(json.dumps(raw))
    with h5py.File(td / "D_results.hdf5", "w") as h: h.create_dataset("estimates/C", data=C)
    extra = run(dis, td, "D")
    with h5py.File(td / "D_results.hdf5", "r") as h:
        assert "roi_filter" not in h and "ROI filter" not in extra
    print("ok  enabled=false: skipped")

    # failure → warning, no exception
    extra = run(REPO / "utilities/pipelines/template_pipeline.json", td, "MISSING")
    assert "ROI filter" not in extra
    assert not (td / "MISSING_results.hdf5").exists(), "stray results file created"
    print("ok  missing results file: warning only, no file created, pipeline continues")
print("ALL OK")
