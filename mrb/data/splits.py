"""Deterministic split manifest helpers."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TaskSplit:
    task_id: int
    dataset_id: str
    classes: tuple[int, ...]
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    image_size: int
    split_hash: str
    source_splits: Mapping[str, str] | None = None
    split_policy: str = "predefined_splits"

    @property
    def num_train(self) -> int:
        return len(self.train_indices)

    @property
    def num_val(self) -> int:
        return len(self.val_indices)

    @property
    def num_test(self) -> int:
        return len(self.test_indices)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classes"] = list(self.classes)
        payload["train_indices"] = list(self.train_indices)
        payload["val_indices"] = list(self.val_indices)
        payload["test_indices"] = list(self.test_indices)
        payload["num_train"] = self.num_train
        payload["num_val"] = self.num_val
        payload["num_test"] = self.num_test
        if self.source_splits is not None:
            payload["source_splits"] = dict(self.source_splits)
        payload["split_policy"] = self.split_policy
        return payload


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def class_order(num_classes: int, seed: int) -> list[int]:
    values = list(range(num_classes))
    random.Random(seed).shuffle(values)
    return values


def split_train_val(indices: Sequence[int], val_ratio: float, seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered = list(indices)
    random.Random(seed).shuffle(ordered)
    if not ordered:
        return (), ()
    val_count = int(round(len(ordered) * val_ratio))
    if val_ratio > 0 and len(ordered) > 1:
        val_count = max(1, val_count)
    val_count = min(val_count, max(0, len(ordered) - 1))
    val_indices = tuple(sorted(ordered[:val_count]))
    train_indices = tuple(sorted(ordered[val_count:]))
    return train_indices, val_indices


def split_train_val_test(
    indices: Sequence[int],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if train_ratio <= 0 or val_ratio < 0 or test_ratio <= 0:
        raise ValueError("holdout ratios must have train/test > 0 and val >= 0")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("holdout ratio sum must be positive")
    ordered = list(indices)
    random.Random(seed).shuffle(ordered)
    if len(ordered) < 3:
        raise ValueError("deterministic_holdout requires at least three samples")

    normalized_train = train_ratio / ratio_sum
    normalized_val = val_ratio / ratio_sum
    train_count = int(round(len(ordered) * normalized_train))
    val_count = int(round(len(ordered) * normalized_val))
    train_count = min(max(1, train_count), len(ordered) - 2)
    val_count = min(max(1 if val_ratio > 0 else 0, val_count), len(ordered) - train_count - 1)
    test_count = len(ordered) - train_count - val_count
    if test_count <= 0:
        val_count = max(0, val_count - 1)
        test_count = len(ordered) - train_count - val_count
    if test_count <= 0:
        raise ValueError("deterministic_holdout could not allocate a non-empty test split")

    train = tuple(sorted(ordered[:train_count]))
    val = tuple(sorted(ordered[train_count : train_count + val_count]))
    test = tuple(sorted(ordered[train_count + val_count : train_count + val_count + test_count]))
    return train, val, test


def indices_for_classes(targets: Sequence[int], classes: Sequence[int]) -> tuple[int, ...]:
    class_set = set(classes)
    return tuple(index for index, target in enumerate(targets) if int(target) in class_set)


def make_task_split(
    *,
    task_id: int,
    dataset_id: str,
    classes: Sequence[int],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    image_size: int,
    source_splits: Mapping[str, str] | None = None,
    split_policy: str = "predefined_splits",
) -> TaskSplit:
    hash_payload = {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "classes": list(classes),
        "train_indices": list(train_indices),
        "val_indices": list(val_indices),
        "test_indices": list(test_indices),
        "image_size": image_size,
        "source_splits": dict(source_splits) if source_splits is not None else None,
        "split_policy": split_policy,
    }
    return TaskSplit(
        task_id=task_id,
        dataset_id=dataset_id,
        classes=tuple(int(value) for value in classes),
        train_indices=tuple(int(value) for value in train_indices),
        val_indices=tuple(int(value) for value in val_indices),
        test_indices=tuple(int(value) for value in test_indices),
        image_size=image_size,
        split_hash=stable_hash(hash_payload),
        source_splits=dict(source_splits) if source_splits is not None else None,
        split_policy=split_policy,
    )


def build_class_incremental_splits(
    *,
    dataset_id: str,
    train_targets: Sequence[int],
    test_targets: Sequence[int],
    num_classes: int,
    num_tasks: int,
    classes_per_task: int,
    seed: int,
    val_ratio: float,
    image_size: int,
    class_order_seed: int | None = None,
    split_seed: int | None = None,
    train_source_split: str = "train",
    test_source_split: str = "test",
) -> list[TaskSplit]:
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive")
    if num_tasks * classes_per_task > num_classes:
        raise ValueError(
            f"num_tasks * classes_per_task exceeds num_classes: "
            f"{num_tasks} * {classes_per_task} > {num_classes}"
        )

    resolved_class_order_seed = seed if class_order_seed is None else int(class_order_seed)
    resolved_split_seed = seed if split_seed is None else int(split_seed)
    ordered_classes = class_order(num_classes, resolved_class_order_seed)
    tasks: list[TaskSplit] = []
    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        classes = tuple(ordered_classes[start : start + classes_per_task])
        train_pool = indices_for_classes(train_targets, classes)
        train_indices, val_indices = split_train_val(
            train_pool, val_ratio, resolved_split_seed + task_id + 1
        )
        test_indices = indices_for_classes(test_targets, classes)
        tasks.append(
            make_task_split(
                task_id=task_id,
                dataset_id=dataset_id,
                classes=classes,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                image_size=image_size,
                source_splits={
                    "train": str(train_source_split),
                    "val": str(train_source_split),
                    "test": str(test_source_split),
                },
                split_policy="class_incremental",
            )
        )
    return tasks


def build_dataset_incremental_splits(
    *,
    datasets: Sequence[tuple[str, Sequence[int], Sequence[int], int | None] | Mapping[str, Any]],
    seed: int,
    val_ratio: float,
    image_size: int,
) -> list[TaskSplit]:
    tasks: list[TaskSplit] = []
    for task_id, item in enumerate(datasets):
        if isinstance(item, Mapping):
            dataset_id = str(item["dataset_id"])
            train_targets = item["train_targets"]
            test_targets = item.get("test_targets", train_targets)
            num_classes = item.get("num_classes")
            split_policy = str(item.get("split_policy", "predefined_splits"))
            source_splits = dict(
                item.get(
                    "source_splits",
                    {"train": "train", "val": "train", "test": "test"},
                )
            )
            holdout = dict(item.get("holdout", {}))
        else:
            dataset_id, train_targets, test_targets, num_classes = item
            split_policy = "predefined_splits"
            source_splits = {"train": "train", "val": "train", "test": "test"}
            holdout = {}

        if split_policy == "deterministic_holdout":
            all_indices = tuple(range(len(train_targets)))
            train_indices, val_indices, test_indices = split_train_val_test(
                all_indices,
                train_ratio=float(holdout.get("train_ratio", 0.8)),
                val_ratio=float(holdout.get("val_ratio", 0.1)),
                test_ratio=float(holdout.get("test_ratio", 0.1)),
                seed=seed + task_id + 1,
            )
        else:
            train_pool = tuple(range(len(train_targets)))
            train_indices, val_indices = split_train_val(train_pool, val_ratio, seed + task_id + 1)
            test_indices = tuple(range(len(test_targets)))

        classes = tuple(range(num_classes)) if num_classes is not None else tuple(sorted(set(train_targets)))
        tasks.append(
            make_task_split(
                task_id=task_id,
                dataset_id=dataset_id,
                classes=classes,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                image_size=image_size,
                source_splits=source_splits,
                split_policy=split_policy,
            )
        )
    return tasks


def manifest_from_tasks(
    *,
    stream_id: str,
    seed: int,
    data_root: str,
    stream_config: dict[str, Any],
    registry_hash: str,
    tasks: Sequence[TaskSplit],
    created_by: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stream_id": stream_id,
        "seed": seed,
        "split_seed": seed,
        "data_root": data_root,
        "created_by": created_by,
        "stream_type": stream_config["stream_type"],
        "stream_config_hash": stable_hash(stream_config),
        "registry_hash": registry_hash,
        "tasks": [task.to_dict() for task in tasks],
    }
    if "class_order_seed" in stream_config:
        payload["class_order_seed"] = int(stream_config["class_order_seed"])
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def write_manifest(manifest: dict[str, Any], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_task_splits(tasks: Sequence[TaskSplit]) -> list[str]:
    errors: list[str] = []
    for task in tasks:
        train = set(task.train_indices)
        val = set(task.val_indices)
        if not train:
            errors.append(f"task {task.task_id} has empty train split")
        if not val:
            errors.append(f"task {task.task_id} has empty val split")
        if not task.test_indices:
            errors.append(f"task {task.task_id} has empty test split")
        if train & val:
            errors.append(f"task {task.task_id} train/val indices overlap")
    return errors
