# Installing the Gravios/CaImAn fork

## Quick start (CUDA 12.x system)

```bash
git clone https://github.com/Gravios/CaImAn.git
cd CaImAn
pip install -e .
pip install dill cupy-cuda12x
pip install nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
            nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
            nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
python -c "$(cat scripts/set_ld_library_path.py)"
```

---

## Step-by-step

### 1  Check your CUDA driver version

```bash
nvidia-smi | head -1
# Driver Version: 570.x  →  CUDA 12.x  →  use cupy-cuda12x
# Driver Version: 525.x  →  CUDA 12.x  →  use cupy-cuda12x
# Driver Version: 470.x  →  CUDA 11.x  →  use cupy-cuda11x
```

### 2  Install base dependencies

```bash
pip install -e .
```

This installs everything declared in `pyproject.toml`, including `torch`,
`keras`, `scipy`, `scikit-learn`, `opencv-python`, `h5py`, `tifffile`,
`psutil`, `peakutils`, `tqdm`, `threadpoolctl`, and all their transitive
dependencies.

> **Note:** `torch` will pull in `nvidia-cublas-13.*` and related CUDA 13
> wheels to support its own CUDA runtime. These are **not compatible** with
> CuPy — do not skip the steps below.

### 3  Install `dill` and `cupy`

These are not declared in `pyproject.toml` because `dill` has no pip-stable
version constraint that works across all platforms and `cupy` requires knowing
your CUDA version at install time.

```bash
pip install dill

# CUDA 12.x (Blackwell, Ampere, Ada — most modern systems):
pip install cupy-cuda12x

# CUDA 11.x (older Volta/Turing systems):
# pip install cupy-cuda11x
```

### 4  Install CUDA 12 runtime wheels (critical)

`pip install -e .` pulls in `torch` which installs `nvidia-cublas-13.*` — a
**CUDA 13** build. `cupy-cuda12x` needs `libcublas.so.12` and will fail
silently (falling back to CPU) if only `.so.13` is available. Install the
correct CUDA 12 variants explicitly:

```bash
pip install \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12
```

These wheels install `.so` files under `site-packages/nvidia/*/lib/` inside
your conda/venv environment. They coexist safely with the CUDA 13 wheels
that `torch` installed.

### 5  Set `LD_LIBRARY_PATH`

The `.so` files installed by the nvidia Python wheels are **not** on the
dynamic linker's default search path. Without this step, `cupy` imports
without error but silently uses CPU for all operations — causing the
38-second precompute and zero-component result seen in the logs.

**Option A — per-session (add to `~/.bashrc` or `~/.zshrc`):**

```bash
NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
export LD_LIBRARY_PATH=\
${NVIDIA_SP}/cublas/lib:\
${NVIDIA_SP}/cufft/lib:\
${NVIDIA_SP}/curand/lib:\
${NVIDIA_SP}/cusolver/lib:\
${NVIDIA_SP}/cusparse/lib:\
${NVIDIA_SP}/cuda_runtime/lib:\
${LD_LIBRARY_PATH}
```

**Option B — write a `.pth` file so it is set automatically:**

```bash
python scripts/set_ld_library_path.py
```

This script writes `nvidia_cuda12_ldpath.pth` into `site-packages/`, which
sets `LD_LIBRARY_PATH` whenever Python starts. It is idempotent and safe to
re-run after upgrades.

**Option C — add to the pipeline JSON `env` section:**

```json
"env": {
    "LD_LIBRARY_PATH": "/home/<user>/.conda/envs/caiman/lib/python3.11/site-packages/nvidia/cublas/lib:/home/<user>/.conda/envs/caiman/lib/python3.11/site-packages/nvidia/cufft/lib:/home/<user>/.conda/envs/caiman/lib/python3.11/site-packages/nvidia/curand/lib:/home/<user>/.conda/envs/caiman/lib/python3.11/site-packages/nvidia/cusolver/lib:/home/<user>/.conda/envs/caiman/lib/python3.11/site-packages/nvidia/cusparse/lib:/home/<user>/.conda/envs/caiman/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
}
```

### 6  Verify GPU is visible to CuPy

```bash
python -c "
import cupy as cp
print('CuPy version :', cp.__version__)
print('CUDA version  :', cp.cuda.runtime.runtimeGetVersion())
print('Device        :', cp.cuda.Device().use())
a = cp.zeros(1)
print('GPU alloc OK  :', a.device)
"
```

Expected output (Blackwell RTX 6000 Pro):

```
CuPy version : 13.x.x
CUDA version  : 12xxxx
Device        : <CUDA Device 0>
GPU alloc OK  : <CUDA Device 0>
```

If you see an `ImportError` about `libcublas.so.12`, repeat steps 4 and 5.

### 7  Apply the drop-in patches

The fork ships its code as an editable install, so no additional copy step
is needed after `pip install -e .` in step 2. If you are applying the tarball
to an existing upstream CaImAn installation:

```bash
cd /path/to/upstream/CaImAn
tar -xzf pipeline_refactor.tar.gz
```

---

## Diagnosing GPU issues

Run the pipeline once. In the log, look for these lines after the
`precompute_corr_pnr_filtered_fov` step:

```
precompute_corr_pnr_filtered_fov: sn  median=45.2 max=312.1 nan=0
precompute_corr_pnr_filtered_fov: PNR median=4.1  p95=11.3  >5px=18432
```

| `sn median` | `PNR p95` | Meaning |
|---|---|---|
| 10–200 ADU | > 5 | GPU working correctly |
| ≈ 0 or ≈ inf | < 1 | CuPy fell back to CPU — check steps 4 & 5 |
| 0.00 | 0.01 | `libcublas.so.12` / `libcufft.so.12` not found |

If `PNR p95 < 1`, every patch will return 0 components. The precompute step
also takes ~38 s instead of ~3 s when running on CPU.

---

## Known conflicts

| Symptom | Cause | Fix |
|---|---|---|
| `libcublas.so.12: cannot open shared object file` | `torch` installed `nvidia-cublas-13` | Step 4 above |
| `libcufft.so.12: cannot open shared object file` | `.so` not on `LD_LIBRARY_PATH` | Step 5 above |
| 0 components, 38 s precompute | Both of the above | Steps 4 + 5 |
| `NameError: name 'ssub' is not defined` | Stale install — missing patch | Re-run `tar -xzf pipeline_refactor.tar.gz` |
| `ValueError: cannot reshape array of size 4225` | Stale install — missing patch | Re-run `tar -xzf pipeline_refactor.tar.gz` |

---

## Optional dependencies

**CNN component evaluation** (`use_cnn: true` in the JSON):

```bash
pip install tensorflow  # keras backend
```

**NWB output** (pending feature):

```bash
pip install pynwb>=2.3
```

**CNMF Inspector GUI** (`bin/cnmf_inspector.py`):

```bash
pip install PyQt6 pyqtgraph
```
