#!/usr/bin/env python
"""Report the current project environment."""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_ENV_VARS = [
    "MRB_DATA_ROOT",
    "MRB_MODEL_ROOT",
    "MRB_CACHE_ROOT",
    "MRB_OUTPUT_ROOT",
    "MRB_FEATURE_ROOT",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TORCH_HOME",
]
CORE_IMPORTS = [
    ("yaml", "pyyaml"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("open_clip", "open_clip_torch"),
]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def import_status(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def print_path_status() -> None:
    print("\nResource paths:")
    for name in PATH_ENV_VARS:
        raw = os.environ.get(name)
        if not raw:
            print(f"  WARN {name}: not set")
            continue
        path = Path(raw).expanduser()
        status = "exists" if path.exists() else "missing"
        print(f"  {name}: {path} ({status})")


def print_cuda_status() -> None:
    if not import_status("torch"):
        print("\nTorch/CUDA: WARN torch is not installed")
        return

    import torch

    print("\nTorch/CUDA:")
    print(f"  torch: {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    print(f"  cuda version: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  WARN CUDA is unavailable; CPU smoke checks may still pass.")
        return
    print(f"  device count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"  gpu {idx}: {torch.cuda.get_device_name(idx)}")


def main() -> int:
    print("Merge Route Boundary environment")
    print(f"  repo: {REPO_ROOT}")
    print(f"  git commit: {git_commit()}")
    print(f"  python: {platform.python_version()} ({sys.executable})")
    print(f"  conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'not set')}")

    print("\nImports:")
    for module_name, package_name in CORE_IMPORTS:
        ok = import_status(module_name)
        label = "OK" if ok else "WARN missing"
        print(f"  {label}: {package_name}")

    print_cuda_status()
    print_path_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
