#!/usr/bin/env python
"""Extract a frozen feature bank for one registered dataset split."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.data.datasets import get_dataset
from mrb.data.registry import DEFAULT_REGISTRY_PATH, load_registry
from mrb.features.backbones import DEFAULT_BACKBONES_CONFIG, build_backbone, get_backbone_spec
from mrb.features.extraction import extract_and_write_feature_bank
from mrb.features.storage import DEFAULT_FEATURE_ROOT, FEATURE_ROOT_ENV, feature_bank_dir, format_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--backbones", type=Path, default=DEFAULT_BACKBONES_CONFIG)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_cache_env(data_root: Path) -> None:
    cache_root = Path(os.environ.get("MRB_CACHE_ROOT", data_root / "_cache")).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", cache_root / "huggingface")).expanduser()
    os.environ.setdefault("MRB_CACHE_ROOT", str(cache_root))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    entry = registry.get(args.dataset)
    if args.split not in entry.splits:
        print(f"ERROR split {args.split!r} is not registered for {entry.dataset_id}: {entry.splits}")
        return 1

    data_root = args.data_root or registry.data_root()
    os.environ.setdefault(registry.root_env, str(data_root))
    configure_cache_env(data_root)

    feature_root = (args.feature_root or Path(os.environ.get(FEATURE_ROOT_ENV, DEFAULT_FEATURE_ROOT))).expanduser()
    spec = get_backbone_spec(args.backbone, args.backbones)
    bank_dir = feature_bank_dir(
        feature_root,
        dataset_id=entry.dataset_id,
        split=args.split,
        backbone_id=spec.backbone_id,
    )

    print(f"Dataset: {entry.dataset_id} split={args.split}")
    print(f"Backbone: {spec.backbone_id} model={spec.model_id}")
    print(f"Feature root: {feature_root}")
    print(f"Feature bank: {bank_dir}")
    print(
        f"Extraction: batch_size={args.batch_size} max_samples={args.max_samples} "
        f"dtype={args.dtype} device={args.device} overwrite={args.overwrite}"
    )

    if args.dry_run:
        exists = bank_dir.exists() or bank_dir.is_symlink()
        state = "exists" if exists else "will_create"
        print(f"Dry run: bank_state={state}; model will not be loaded and no files will be written.")
        return 0

    if (bank_dir.exists() or bank_dir.is_symlink()) and not args.overwrite:
        print(f"ERROR feature bank already exists: {bank_dir}. Pass --overwrite to replace it.")
        return 1

    try:
        backbone = build_backbone(spec.backbone_id, device=args.device, config_path=args.backbones)
    except Exception as exc:
        print(f"ERROR failed to build backbone {spec.backbone_id}: {exc}")
        return 1

    env = {registry.root_env: str(data_root)}
    dataset_root = registry.dataset_root(entry, env=env)
    try:
        dataset = get_dataset(
            entry.dataset_id,
            args.split,
            transform=backbone.dataset_transform,
            root=dataset_root,
            registry=registry,
        )
    except Exception as exc:
        print(f"ERROR failed to load dataset {entry.dataset_id}/{args.split}: {exc}")
        return 1

    try:
        written = extract_and_write_feature_bank(
            dataset,
            bank_dir,
            dataset_id=entry.dataset_id,
            split=args.split,
            backbone=backbone,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            dtype=args.dtype,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR failed to extract feature bank: {exc}")
        return 1
    features_file = written / "features.npy"
    size = features_file.stat().st_size if features_file.exists() else 0
    print(f"Wrote feature bank: {written}")
    print(f"features.npy size: {format_bytes(size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
