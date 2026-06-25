#!/usr/bin/env python
"""Verify manifest-managed datasets without downloading missing files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "resources" / "datasets.yaml"
DEFAULT_DATA_ROOT = Path("/root/rivermind-data/datasets")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError(f"{path} must contain a 'datasets' list")
    return payload


def data_root_from_manifest(manifest: dict[str, Any]) -> Path:
    root_env = str(manifest.get("root_env", "MRB_DATA_ROOT"))
    fallback = manifest.get("paths", {}).get("shared_root", str(DEFAULT_DATA_ROOT))
    return Path(os.environ.get(root_env, fallback)).expanduser()


def dataset_root(data_root: Path, resource: dict[str, Any]) -> Path:
    return data_root / str(resource.get("root_subdir", ""))


def dataset_names(resource: dict[str, Any]) -> set[str]:
    values = {str(resource["name"]).lower()}
    values.update(str(alias).lower() for alias in resource.get("aliases", []))
    return values


def select_datasets(
    manifest: dict[str, Any],
    data_root: Path,
    tiers: list[int],
    names: list[str],
    all_present: bool,
) -> list[dict[str, Any]]:
    datasets = manifest["datasets"]
    requested_tiers = set(tiers)
    requested_names = {name.lower() for name in names}
    selected: list[dict[str, Any]] = []

    if all_present:
        for resource in datasets:
            if dataset_root(data_root, resource).exists():
                selected.append(resource)
    elif not requested_tiers and not requested_names:
        requested_tiers = set(manifest.get("policies", {}).get("default_tiers", [0]))

    if requested_tiers or requested_names:
        for resource in datasets:
            matches_tier = resource.get("tier") in requested_tiers
            matches_name = bool(requested_names & dataset_names(resource))
            if matches_tier or matches_name:
                selected.append(resource)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for resource in selected:
        name = str(resource["name"])
        if name not in seen:
            seen.add(name)
            deduped.append(resource)

    missing = requested_names.copy()
    for resource in deduped:
        missing -= dataset_names(resource)
    if missing:
        raise ValueError(f"Unknown dataset name(s): {', '.join(sorted(missing))}")

    return deduped


def configure_cache_env(data_root: Path) -> None:
    cache_root = data_root / "_cache"
    hf_home = cache_root / "huggingface"
    os.environ.setdefault("MRB_CACHE_ROOT", str(cache_root))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))


def torchvision_dataset(
    resource: dict[str, Any],
    root: Path,
    split: str,
    transform: object | None,
) -> object:
    try:
        from torchvision import datasets
    except Exception as exc:
        raise RuntimeError("torchvision is required to verify torchvision datasets") from exc

    dataset_cls = getattr(datasets, str(resource["torchvision_class"]))
    loader = str(resource.get("loader", "split"))

    if loader == "train_bool":
        return dataset_cls(
            root=str(root),
            train=split == "train",
            download=False,
            transform=transform,
        )
    if loader == "no_split":
        return dataset_cls(root=str(root), download=False, transform=transform)
    if loader == "split":
        return dataset_cls(root=str(root), split=split, download=False, transform=transform)

    raise ValueError(f"Unsupported torchvision loader for {resource['name']}: {loader}")


def class_count(dataset: object) -> int | None:
    for attr in ("classes", "categories"):
        values = getattr(dataset, attr, None)
        if isinstance(values, (list, tuple, dict)):
            return len(values)

    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict):
        return len(class_to_idx)

    for attr in ("targets", "labels", "_labels"):
        values = getattr(dataset, attr, None)
        if values is None:
            continue
        try:
            return len(set(int(value) for value in values))
        except Exception:
            continue
    return None


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size

    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for dirname in dirs:
            candidate = root_path / dirname
            if candidate.is_symlink():
                total += candidate.lstat().st_size
        for filename in files:
            candidate = root_path / filename
            try:
                total += candidate.lstat().st_size
            except OSError:
                continue
    return total


def verify_batch(dataset: object) -> str:
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("torch is required to verify a dataset batch") from exc

    if len(dataset) == 0:  # type: ignore[arg-type]
        raise RuntimeError("dataset split is empty")

    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, num_workers=0)  # type: ignore[arg-type]
    batch = next(iter(loader))
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise RuntimeError("dataset batch must contain image and label tensors")
    images, labels = batch[0], batch[1]
    if not isinstance(images, torch.Tensor):
        raise RuntimeError(f"image batch is not a torch.Tensor: {type(images).__name__}")
    if images.ndim < 3:
        raise RuntimeError(f"image batch has unexpected shape: {tuple(images.shape)}")
    if labels is None:
        raise RuntimeError("label batch is missing")
    return f"images={tuple(images.shape)} labels={tuple(getattr(labels, 'shape', []))}"


def verify_torchvision(resource: dict[str, Any], data_root: Path) -> dict[str, Any]:
    try:
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("torchvision transforms are required for verification") from exc

    root = dataset_root(data_root, resource)
    result: dict[str, Any] = {
        "status": "ok",
        "source": resource.get("source"),
        "root": str(root),
        "verified_at": now_iso(),
        "splits": {},
        "notes": [],
    }

    if not root.exists():
        raise RuntimeError(f"dataset root does not exist: {root}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    total_examples = 0
    detected_classes: int | None = None
    first_dataset: object | None = None

    for split in resource.get("splits", ["train"]):
        split_name = str(split)
        dataset = torchvision_dataset(resource, root, split_name, transform=transform)
        if first_dataset is None:
            first_dataset = dataset
        split_len = len(dataset)  # type: ignore[arg-type]
        result["splits"][split_name] = split_len
        total_examples += split_len
        if detected_classes is None:
            detected_classes = class_count(dataset)

    verify = resource.get("verify", {})
    min_total = verify.get("min_total_examples")
    if min_total is not None and total_examples < int(min_total):
        raise RuntimeError(
            f"total examples {total_examples} below manifest minimum {int(min_total)}"
        )

    expected_classes = verify.get("num_classes")
    min_classes = verify.get("min_num_classes")
    if expected_classes is not None and detected_classes != int(expected_classes):
        raise RuntimeError(
            f"class count {detected_classes} does not match expected {int(expected_classes)}"
        )
    if min_classes is not None and (detected_classes is None or detected_classes < int(min_classes)):
        raise RuntimeError(
            f"class count {detected_classes} below minimum {int(min_classes)}"
        )

    if first_dataset is None:
        raise RuntimeError("no dataset splits were configured")

    result["num_classes"] = detected_classes
    result["total_examples"] = total_examples
    result["batch"] = verify_batch(first_dataset)
    result["disk_gb"] = round(directory_size_bytes(root) / (1024**3), 4)
    return result


def skipped_status(resource: dict[str, Any], data_root: Path) -> dict[str, Any]:
    root = dataset_root(data_root, resource)
    source = str(resource.get("source", ""))
    if resource.get("manual") or resource.get("gated") or "manual" in source:
        status = "manual_required" if not root.exists() else "skipped_not_implemented"
    elif not root.exists():
        status = "not_downloaded"
    else:
        status = "skipped_not_implemented"
    return {
        "status": status,
        "source": source,
        "root": str(root),
        "verified_at": now_iso(),
        "notes": ["automatic verification is not implemented for this dataset source"],
    }


def verify_dataset(resource: dict[str, Any], data_root: Path) -> dict[str, Any]:
    if resource.get("source") == "torchvision":
        return verify_torchvision(resource, data_root)
    return skipped_status(resource, data_root)


def write_status(metadata_root: Path, results: dict[str, dict[str, Any]]) -> None:
    metadata_root.mkdir(parents=True, exist_ok=True)
    status_path = metadata_root / "dataset_status.json"
    existing: dict[str, Any] = {}
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(results)
    status_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_path = metadata_root / "verification_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        for name, result in results.items():
            record = {"dataset": name, **result}
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--tier", type=int, action="append", default=[], help="Dataset tier to verify.")
    parser.add_argument("--name", action="append", default=[], help="Dataset name or alias to verify.")
    parser.add_argument("--all-present", action="store_true", help="Verify datasets whose roots exist.")
    parser.add_argument("--write-status", action="store_true", help="Write dataset status metadata.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    data_root = args.data_root.expanduser() if args.data_root else data_root_from_manifest(manifest)
    metadata_root = data_root / "_metadata"
    configure_cache_env(data_root)
    selected = select_datasets(manifest, data_root, args.tier, args.name, args.all_present)

    print(f"Dataset manifest: {args.manifest}")
    print(f"Data root: {data_root}")
    print("Verification:")

    results: dict[str, dict[str, Any]] = {}
    failed = False
    for resource in selected:
        name = str(resource["name"])
        try:
            result = verify_dataset(resource, data_root)
        except Exception as exc:
            result = {
                "status": "error",
                "source": resource.get("source"),
                "root": str(dataset_root(data_root, resource)),
                "verified_at": now_iso(),
                "error": str(exc),
                "notes": [],
            }
            if resource.get("source") == "torchvision":
                failed = True
        results[name] = result
        print(f"  {name}: {result['status']}")
        if result.get("error"):
            print(f"    error: {result['error']}")
        for note in result.get("notes", []):
            print(f"    note: {note}")

    if args.write_status:
        write_status(metadata_root, results)
        print(f"Wrote status: {metadata_root / 'dataset_status.json'}")
        print(f"Appended log: {metadata_root / 'verification_log.jsonl'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
