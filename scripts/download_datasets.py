#!/usr/bin/env python
"""Download manifest-managed datasets into the shared MRB data root."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
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
    tiers: list[int],
    names: list[str],
) -> list[dict[str, Any]]:
    datasets = manifest["datasets"]
    selected: list[dict[str, Any]] = []
    requested_tiers = set(tiers)
    requested_names = {name.lower() for name in names}

    if not requested_tiers and not requested_names:
        requested_tiers = set(manifest.get("policies", {}).get("default_tiers", [0]))

    for resource in datasets:
        matches_tier = resource.get("tier") in requested_tiers
        matches_name = bool(requested_names & dataset_names(resource))
        if matches_tier or matches_name:
            selected.append(resource)

    missing = requested_names.copy()
    for resource in selected:
        missing -= dataset_names(resource)
    if missing:
        raise ValueError(f"Unknown dataset name(s): {', '.join(sorted(missing))}")

    return selected


def is_large_dataset(resource: dict[str, Any], policies: dict[str, Any]) -> bool:
    threshold = float(policies.get("large_dataset_threshold_gb", 10))
    return bool(resource.get("large")) or float(resource.get("estimated_gb", 0)) >= threshold


def skip_reason(
    resource: dict[str, Any],
    policies: dict[str, Any],
    include_large: bool,
    skip_manual: bool,
) -> str | None:
    source = str(resource.get("source", ""))
    if is_large_dataset(resource, policies) and not include_large:
        return "large dataset requires --include-large"
    if source == "torchvision":
        return None
    if resource.get("manual") or "manual" in source or resource.get("gated"):
        if skip_manual:
            return resource.get("manual_reason", "manual/gated dataset; skipping")
        return "manual/gated dataset is registered but auto-download is not implemented"
    return f"{source} dataset is registered only; auto-download is not implemented"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def check_world_size() -> None:
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"WORLD_SIZE must be an integer when set, got {raw!r}") from exc
    if world_size > 1:
        raise RuntimeError("Refusing to download datasets with WORLD_SIZE > 1")


def check_space(
    data_root: Path,
    resources: list[dict[str, Any]],
    max_gb: float | None,
    policies: dict[str, Any],
) -> None:
    estimated_gb = sum(float(resource.get("estimated_gb", 0)) for resource in resources)
    if max_gb is not None and estimated_gb > max_gb:
        raise RuntimeError(
            f"Selected datasets estimate {estimated_gb:.2f} GB, above --max-gb {max_gb:.2f}"
        )

    min_free = float(policies.get("min_free_gb_after_download", 20))
    required_gb = estimated_gb * 2.2 + min_free
    available_gb = free_gb(data_root)
    if available_gb < required_gb:
        raise RuntimeError(
            f"Insufficient free space under {data_root}: need {required_gb:.2f} GB, "
            f"available {available_gb:.2f} GB"
        )


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


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
    download: bool,
    transform: object | None = None,
) -> object:
    try:
        from torchvision import datasets
    except Exception as exc:
        raise RuntimeError("torchvision is required for torchvision dataset downloads") from exc

    class_name = str(resource["torchvision_class"])
    dataset_cls = getattr(datasets, class_name)
    extra_mirrors = [str(url) for url in resource.get("download_mirrors", [])]
    if extra_mirrors and hasattr(dataset_cls, "mirrors"):
        existing_mirrors = list(getattr(dataset_cls, "mirrors"))
        dataset_cls.mirrors = extra_mirrors + [
            url for url in existing_mirrors if url not in extra_mirrors
        ]
    loader = str(resource.get("loader", "split"))

    if loader == "train_bool":
        if split not in {"train", "test"}:
            raise ValueError(f"{resource['name']} split {split!r} is not valid for train_bool")
        return dataset_cls(
            root=str(root),
            train=split == "train",
            download=download,
            transform=transform,
        )
    if loader == "no_split":
        return dataset_cls(root=str(root), download=download, transform=transform)
    if loader == "split":
        return dataset_cls(root=str(root), split=split, download=download, transform=transform)

    raise ValueError(f"Unsupported torchvision loader for {resource['name']}: {loader}")


def dataset_is_present(resource: dict[str, Any], data_root: Path) -> bool:
    root = dataset_root(data_root, resource)
    if not root.exists():
        return False
    for split in resource.get("splits", ["train"]):
        try:
            torchvision_dataset(resource, root, str(split), download=False)
        except Exception:
            return False
    return True


def download_torchvision_dataset(resource: dict[str, Any], data_root: Path, force: bool) -> str:
    root = dataset_root(data_root, resource)
    root.mkdir(parents=True, exist_ok=True)

    if not force and dataset_is_present(resource, data_root):
        return "already_present"

    for split in resource.get("splits", ["train"]):
        torchvision_dataset(resource, root, str(split), download=True)
    return "downloaded"


def append_log(metadata_root: Path, record: dict[str, Any]) -> None:
    metadata_root.mkdir(parents=True, exist_ok=True)
    log_path = metadata_root / "download_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan only.")
    parser.add_argument("--tier", type=int, action="append", default=[], help="Dataset tier to include.")
    parser.add_argument("--name", action="append", default=[], help="Dataset name or alias to include.")
    parser.add_argument("--max-gb", type=float, default=None, help="Maximum selected estimated size.")
    parser.add_argument("--include-large", action="store_true", help="Allow large datasets.")
    parser.add_argument(
        "--skip-manual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip manual/gated datasets instead of treating them as download targets.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run dataset download calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    policies = manifest.get("policies", {})
    data_root = args.data_root.expanduser() if args.data_root else data_root_from_manifest(manifest)
    metadata_root = data_root / "_metadata"
    selected = select_datasets(manifest, args.tier, args.name)

    plan: list[tuple[dict[str, Any], str | None]] = [
        (
            resource,
            skip_reason(resource, policies, include_large=args.include_large, skip_manual=args.skip_manual),
        )
        for resource in selected
    ]
    download_targets = [resource for resource, reason in plan if reason is None]

    print(f"Dataset manifest: {args.manifest}")
    print(f"Data root: {data_root}")
    print(f"Mode: {'dry-run' if args.dry_run else 'download'}")
    print("Plan:")
    for resource, reason in plan:
        prefix = "SKIP" if reason else "GET "
        print(
            f"  {prefix} {resource['name']} tier={resource['tier']} "
            f"source={resource['source']} estimated_gb={float(resource.get('estimated_gb', 0)):.2f}"
        )
        if reason:
            print(f"       reason: {reason}")

    if args.dry_run:
        return 0

    check_world_size()
    data_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    configure_cache_env(data_root)

    if download_targets:
        check_space(data_root, download_targets, args.max_gb, policies)

    lock_path = metadata_root / "download.lock"
    with file_lock(lock_path):
        for resource, reason in plan:
            record = {
                "dataset": resource["name"],
                "source": resource.get("source"),
                "timestamp": now_iso(),
            }
            if reason:
                print(f"Skipping {resource['name']}: {reason}")
                record.update({"status": "skipped", "reason": reason})
                append_log(metadata_root, record)
                continue

            print(f"Downloading {resource['name']}...")
            try:
                status = download_torchvision_dataset(resource, data_root, force=args.force)
            except Exception as exc:
                record.update({"status": "error", "error": str(exc)})
                append_log(metadata_root, record)
                raise
            record.update({"status": status})
            append_log(metadata_root, record)
            print(f"  {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
