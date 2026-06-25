#!/usr/bin/env python
"""Verify task stream splits and run a small DataLoader smoke check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.data.datasets import default_train_split, get_dataset
from mrb.data.registry import DEFAULT_REGISTRY_PATH
from mrb.data.splits import stable_hash
from mrb.data.task_streams import TaskStream, build_task_stream
from mrb.data.transforms import build_eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--stream-config", type=Path, action="append", default=None)
    parser.add_argument("--skip-batch", action="store_true", help="Skip DataLoader batch smoke checks.")
    return parser.parse_args()


def verify_nonempty_and_reproducible(stream: TaskStream) -> list[str]:
    errors: list[str] = []
    for task in stream.tasks:
        if task.num_train <= 0:
            errors.append(f"task {task.task_id} has empty train split")
        if task.num_val <= 0:
            errors.append(f"task {task.task_id} has empty val split")
        if task.num_test <= 0:
            errors.append(f"task {task.task_id} has empty test split")
        if set(task.train_indices) & set(task.val_indices):
            errors.append(f"task {task.task_id} train and val indices overlap")
        expected_hash = stable_hash(
            {
                "task_id": task.task_id,
                "dataset_id": task.dataset_id,
                "classes": list(task.classes),
                "train_indices": list(task.train_indices),
                "val_indices": list(task.val_indices),
                "test_indices": list(task.test_indices),
                "image_size": task.image_size,
                "source_splits": dict(task.source_splits) if task.source_splits else None,
                "split_policy": task.split_policy,
            }
        )
        if task.split_hash != expected_hash:
            errors.append(f"task {task.task_id} split_hash is not stable")
    return errors


def verify_class_incremental(stream: TaskStream) -> list[str]:
    if stream.stream_type != "class_incremental":
        return []
    errors: list[str] = []
    seen_classes: set[int] = set()
    seen_train_indices: set[int] = set()
    for task in stream.tasks:
        classes = set(task.classes)
        if seen_classes & classes:
            errors.append(f"task {task.task_id} classes overlap earlier tasks")
        if seen_train_indices & set(task.train_indices):
            errors.append(f"task {task.task_id} train indices overlap earlier tasks")
        seen_classes |= classes
        seen_train_indices |= set(task.train_indices)
    return errors


def verify_rebuild(args: argparse.Namespace, stream: TaskStream) -> list[str]:
    rebuilt = build_task_stream(
        stream_id=args.stream,
        seed=args.seed,
        registry_path=args.registry,
        stream_config_paths=args.stream_config,
    )
    first = [task.split_hash for task in stream.tasks]
    second = [task.split_hash for task in rebuilt.tasks]
    if first != second:
        return ["rebuilding the stream with the same seed produced different split hashes"]
    return []


def smoke_batch(stream: TaskStream) -> list[str]:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except Exception as exc:
        return [f"torch unavailable; skipped batch smoke check: {exc}"]

    errors: list[str] = []
    transform_cache: dict[int, object] = {}
    dataset_cache: dict[tuple[str, int], object] = {}
    for task in stream.tasks:
        if task.image_size not in transform_cache:
            transform_cache[task.image_size] = build_eval_transform(task.image_size)
        cache_key = (task.dataset_id, task.image_size)
        if cache_key not in dataset_cache:
            entry = stream.registry.get(task.dataset_id)
            dataset_cache[cache_key] = get_dataset(
                task.dataset_id,
                default_train_split(entry),
                transform=transform_cache[task.image_size],
                registry=stream.registry,
            )
        indices = list(task.train_indices[: min(4, len(task.train_indices))])
        subset = Subset(dataset_cache[cache_key], indices)  # type: ignore[arg-type]
        loader = DataLoader(subset, batch_size=len(indices), shuffle=False, num_workers=0)
        try:
            batch = next(iter(loader))
            images = batch[0]
            if not isinstance(images, torch.Tensor):
                errors.append(f"task {task.task_id} image batch is not a tensor")
        except Exception as exc:
            errors.append(f"task {task.task_id} batch smoke failed: {exc}")
    return errors


def main() -> int:
    args = parse_args()
    stream = build_task_stream(
        stream_id=args.stream,
        seed=args.seed,
        registry_path=args.registry,
        stream_config_paths=args.stream_config,
    )
    errors = []
    errors.extend(verify_nonempty_and_reproducible(stream))
    errors.extend(verify_class_incremental(stream))
    errors.extend(verify_rebuild(args, stream))
    if not args.skip_batch:
        errors.extend(smoke_batch(stream))

    print(f"Stream: {stream.stream_id}")
    print(f"Seed: {stream.seed}")
    print(f"Tasks: {len(stream.tasks)}")
    for task in stream.tasks:
        print(
            f"  task={task.task_id} dataset={task.dataset_id} classes={len(task.classes)} "
            f"train={task.num_train} val={task.num_val} test={task.num_test} "
            f"hash={task.split_hash[:12]}"
        )

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  ERROR {error}")
        return 1

    print("Task stream verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
