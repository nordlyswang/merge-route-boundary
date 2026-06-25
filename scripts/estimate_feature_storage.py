#!/usr/bin/env python
"""Estimate feature bank storage for registered datasets and a frozen backbone."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.data.datasets import get_dataset
from mrb.data.registry import DEFAULT_REGISTRY_PATH, DatasetEntry, DatasetRegistry, dataset_available, load_registry
from mrb.features.backbones import DEFAULT_BACKBONES_CONFIG, get_backbone_spec
from mrb.features.storage import format_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--backbones", type=Path, default=DEFAULT_BACKBONES_CONFIG)
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset ids to estimate.")
    parser.add_argument("--split", action="append", default=[], help="Optional split filter. Repeatable.")
    parser.add_argument("--backbone", required=True, help="Backbone id from configs/features/backbones.yaml.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--data-root", type=Path, default=None)
    return parser.parse_args()


def split_sample_count(
    registry: DatasetRegistry,
    entry: DatasetEntry,
    split: str,
    *,
    data_root: Path | None,
) -> tuple[int | None, str]:
    sample_counts = entry.config.get("sample_counts", {})
    if isinstance(sample_counts, dict) and split in sample_counts:
        return int(sample_counts[split]), "registry"

    synthetic_sizes = entry.config.get("synthetic_sizes", {})
    if isinstance(synthetic_sizes, dict) and split in synthetic_sizes:
        return int(synthetic_sizes[split]), "registry"

    env = {registry.root_env: str(data_root)} if data_root is not None else None
    if dataset_available(registry, entry, env=env):
        dataset_root = registry.dataset_root(entry, env=env)
        try:
            dataset = get_dataset(entry.dataset_id, split, registry=registry, root=dataset_root, transform=None)
            return int(len(dataset)), "local_loader"  # type: ignore[arg-type]
        except Exception:
            return None, "local_loader_failed"

    return None, "unavailable"


def estimate_bytes(num_samples: int, feature_dim: int, dtype: str) -> int:
    feature_bytes = num_samples * feature_dim * np.dtype(dtype).itemsize
    label_bytes = num_samples * np.dtype("int64").itemsize
    index_bytes = num_samples * np.dtype("int64").itemsize
    return int(feature_bytes + label_bytes + index_bytes)


def selected_splits(entry: DatasetEntry, requested: list[str]) -> list[str]:
    if not requested:
        return list(entry.splits)
    missing = sorted(set(requested) - set(entry.splits))
    if missing:
        raise ValueError(f"{entry.dataset_id} does not define split(s): {', '.join(missing)}")
    return requested


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    data_root = args.data_root or registry.data_root()
    os.environ.setdefault(registry.root_env, str(data_root))
    spec = get_backbone_spec(args.backbone, args.backbones)

    print(
        f"Backbone: {spec.backbone_id} model={spec.model_id} "
        f"D={spec.feature_dim} dtype={args.dtype}"
    )
    print(f"Data root: {data_root}")
    print("")

    total = 0
    unknown: list[str] = []
    for dataset_id in args.datasets:
        entry = registry.get(dataset_id)
        for split in selected_splits(entry, args.split):
            count, source = split_sample_count(registry, entry, split, data_root=data_root)
            if count is None:
                unknown.append(f"{entry.dataset_id}/{split}")
                print(f"WARN {entry.dataset_id:18s} {split:10s} samples=unknown source={source}")
                continue
            size = estimate_bytes(count, spec.feature_dim, args.dtype)
            total += size
            print(
                f"{entry.dataset_id:18s} {split:10s} samples={count:8d} "
                f"estimated={format_bytes(size):>12s} source={source}"
            )

    print("")
    print(f"Total estimated array storage: {format_bytes(total)}")
    if unknown:
        print("WARN missing sample counts for: " + ", ".join(unknown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
