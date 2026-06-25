#!/usr/bin/env python
"""Verify configured resource paths and manifest-managed resources."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_MANIFEST = REPO_ROOT / "configs" / "resources" / "datasets.yaml"
MODELS_MANIFEST = REPO_ROOT / "configs" / "resources" / "models.yaml"
MIN_FREE_GB = 1.0
DATASET_MARKERS = {
    "CIFAR10": ["torchvision/cifar-10-batches-py"],
    "CIFAR100": ["torchvision/cifar-100-python"],
    "MNIST": ["torchvision/MNIST/raw", "torchvision/MNIST/processed"],
    "FashionMNIST": ["torchvision/FashionMNIST/raw", "torchvision/FashionMNIST/processed"],
}


def path_from_env(name: str, fallback: Path) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser()


def resource_paths() -> dict[str, Path]:
    cache_root = path_from_env("MRB_CACHE_ROOT", REPO_ROOT / "cache")
    hf_home = path_from_env("HF_HOME", cache_root / "huggingface")
    return {
        "MRB_DATA_ROOT": path_from_env("MRB_DATA_ROOT", Path("/root/rivermind-data/datasets")),
        "MRB_MODEL_ROOT": path_from_env("MRB_MODEL_ROOT", REPO_ROOT / "models"),
        "MRB_CACHE_ROOT": cache_root,
        "MRB_OUTPUT_ROOT": path_from_env("MRB_OUTPUT_ROOT", REPO_ROOT / "outputs"),
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": path_from_env("HF_HUB_CACHE", hf_home / "hub"),
        "HF_DATASETS_CACHE": path_from_env("HF_DATASETS_CACHE", hf_home / "datasets"),
        "TORCH_HOME": path_from_env("TORCH_HOME", cache_root / "torch"),
    }


def load_yaml(path: Path, key: str) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    resources = data.get(key, [])
    if not isinstance(resources, list):
        raise ValueError(f"{path} must contain a '{key}' list")
    return resources


def check_writable(path: Path) -> tuple[bool, str]:
    path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".mrb-write-test-", delete=True):
            return True, "writable"
    except OSError as exc:
        return False, str(exc)


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def dataset_name(resource: dict[str, Any]) -> str:
    return str(resource.get("name", resource.get("id", "")))


def dataset_required(resource: dict[str, Any]) -> bool:
    return bool(resource.get("default_download", resource.get("required", False)))


def dataset_markers(resource: dict[str, Any]) -> list[str]:
    root_subdir = str(resource.get("root_subdir", ""))
    markers = resource.get("storage_markers")
    if isinstance(markers, list):
        return [str(Path(root_subdir) / str(marker)) for marker in markers]
    return DATASET_MARKERS.get(dataset_name(resource), [])


def dataset_present(data_root: Path, resource: dict[str, Any]) -> bool:
    markers = dataset_markers(resource)
    return any((data_root / marker).exists() for marker in markers)


def model_metadata_present(model_root: Path, model_id: str, patterns: list[str]) -> bool:
    model_dir = model_root / "huggingface" / model_id.replace("/", "__")
    return all((model_dir / pattern).exists() for pattern in patterns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail on missing required resources.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warnings: list[str] = []
    errors: list[str] = []
    paths = resource_paths()

    print("Resource path status:")
    for name, path in paths.items():
        if not os.environ.get(name):
            warnings.append(f"{name} is not set; using fallback {path}")
        ok, message = check_writable(path)
        status = "OK" if ok else "ERROR"
        print(f"  {status} {name}: {path} ({message})")
        if not ok:
            errors.append(f"{name} is not writable: {message}")
            continue
        available = free_gb(path)
        print(f"    free: {available:.2f} GB")
        if available < MIN_FREE_GB:
            warnings.append(f"{name} has less than {MIN_FREE_GB:.1f} GB free")

    datasets = load_yaml(DATASETS_MANIFEST, "datasets")
    models = load_yaml(MODELS_MANIFEST, "models")

    print("\nRequired datasets:")
    for dataset in [item for item in datasets if dataset_required(item)]:
        dataset_id = dataset_name(dataset)
        present = dataset_present(paths["MRB_DATA_ROOT"], dataset)
        print(f"  {'OK' if present else 'WARN missing'} {dataset_id}")
        if not present:
            warnings.append(f"required dataset is not present: {dataset_id}")

    print("\nRequired model metadata:")
    for model in [item for item in models if item.get("required") is True]:
        model_id = model["id"]
        patterns = model.get("allow_patterns", [])
        present = model_metadata_present(paths["MRB_MODEL_ROOT"], model_id, patterns)
        print(f"  {'OK' if present else 'WARN missing'} {model_id}")
        if not present:
            warnings.append(f"required model metadata is not present: {model_id}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARN {warning}")

    if errors or (args.strict and warnings):
        if args.strict and warnings:
            errors.extend(warnings)
        print("\nErrors:")
        for error in errors:
            print(f"  ERROR {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
