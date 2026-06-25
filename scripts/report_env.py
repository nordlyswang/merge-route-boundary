#!/usr/bin/env python
"""Emit a JSON environment report."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def cuda_info() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "torch": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_names": [],
        }

    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": gpu_names,
    }


def build_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torchvision": package_version("torchvision"),
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "huggingface_hub": package_version("huggingface-hub"),
        "mrb_data_root": os.environ.get("MRB_DATA_ROOT"),
        "mrb_model_root": os.environ.get("MRB_MODEL_ROOT"),
        "mrb_cache_root": os.environ.get("MRB_CACHE_ROOT"),
        "mrb_output_root": os.environ.get("MRB_OUTPUT_ROOT"),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
        "torch_home": os.environ.get("TORCH_HOME"),
        "git_commit": git_commit(),
    }
    report.update(cuda_info())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.dumps(build_report(), indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
