#!/usr/bin/env python3
"""
set_ld_library_path.py — write a .pth file that adds CUDA 12 nvidia wheel
libraries to LD_LIBRARY_PATH whenever this Python environment starts.

Run once after installing the nvidia-*-cu12 wheels:

    python scripts/set_ld_library_path.py

This writes  nvidia_cuda12_ldpath.pth  into your site-packages directory.
The file is automatically loaded by the Python interpreter at startup, so
libcublas.so.12, libcufft.so.12, etc. are visible to CuPy and other C
extensions without requiring manual LD_LIBRARY_PATH management.

Safe to re-run — overwrites the existing .pth if already present.
"""

import os
import site
import sys
from pathlib import Path

NVIDIA_SUBDIRS = [
    "cublas/lib",
    "cufft/lib",
    "curand/lib",
    "cusolver/lib",
    "cusparse/lib",
    "cuda_runtime/lib",
    "cuda_nvrtc/lib",
]


def main() -> int:
    sp = site.getsitepackages()
    if not sp:
        print("ERROR: could not determine site-packages directory", file=sys.stderr)
        return 1

    sp_dir   = Path(sp[0])
    nv_base  = sp_dir / "nvidia"

    if not nv_base.exists():
        print(f"WARNING: {nv_base} does not exist.")
        print("         Run 'pip install nvidia-cublas-cu12 nvidia-cufft-cu12 ...' first.")
        return 1

    # Collect existing lib directories
    found = []
    missing = []
    for sub in NVIDIA_SUBDIRS:
        d = nv_base / sub
        if d.exists():
            found.append(str(d))
        else:
            missing.append(str(d))

    if not found:
        print("ERROR: no nvidia lib directories found under", nv_base)
        return 1

    if missing:
        print("WARNING: these expected directories were not found:")
        for m in missing:
            print(f"  {m}")
        print("  (install the corresponding nvidia-*-cu12 wheels)")

    # Build the .pth file content
    # A .pth file that starts with "import " is executed as Python code at startup.
    paths_str = ":".join(found)
    pth_content = (
        "import os; "
        f"os.environ.setdefault('LD_LIBRARY_PATH', ''); "
        f"os.environ['LD_LIBRARY_PATH'] = "
        f"'{paths_str}:' + os.environ['LD_LIBRARY_PATH'] "
        f"if '{found[0]}' not in os.environ.get('LD_LIBRARY_PATH', '') else "
        f"os.environ['LD_LIBRARY_PATH']\n"
    )

    pth_path = sp_dir / "nvidia_cuda12_ldpath.pth"
    pth_path.write_text(pth_content)

    print(f"Written: {pth_path}")
    print()
    print("Library paths registered:")
    for p in found:
        print(f"  {p}")
    print()
    print("LD_LIBRARY_PATH will be set automatically on next Python startup.")
    print("Verify with:")
    print('  python -c "import os; print(os.environ.get(\'LD_LIBRARY_PATH\', \'(not set)\'))"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
