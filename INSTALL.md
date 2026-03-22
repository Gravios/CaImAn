# Installing the CaImAn drop-in patch

## 1. Base dependencies

```bash
pip install -r requirements.txt
```

## 2. GPU support (required for GPU motion correction and precompute)

Check your CUDA version:
```bash
nvidia-smi | head -1
```

Then install the matching CuPy build:
```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

## 3. Deploy the drop-in

Copy the patched CaImAn files over your existing installation:

```bash
cd /path/to/CaImAn
tar -xzf pipeline_refactor.tar.gz
```

## 4. Optional dependencies

**NWB output** (pending feature):
```bash
pip install pynwb>=2.3
```

**Widefield pipeline**:
```bash
pip install ipyparallel>=8.4
```

## Notes

- `logging` is Python standard library — no install needed.
- `dill` replaces the default pickle for multiprocessing; required for
  serialising lambda functions and closures across worker processes.
- CuPy version must match your installed CUDA toolkit, not just the driver.
  If unsure, use `pip install cupy-cuda12x` for modern systems.
