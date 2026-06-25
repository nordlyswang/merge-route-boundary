"""Deterministic split manifest helpers."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


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
) -> TaskSplit:
    hash_payload = {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "classes": list(classes),
        "train_indices": list(train_indices),
        "val_indices": list(val_indices),
        "test_indices": list(test_indices),
        "image_size": image_size,
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

    ordered_classes = class_order(num_classes, seed)
    tasks: list[TaskSplit] = []
    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        classes = tuple(ordered_classes[start : start + classes_per_task])
        train_pool = indices_for_classes(train_targets, classes)
        train_indices, val_indices = split_train_val(train_pool, val_ratio, seed + task_id + 1)
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
            )
        )
    return tasks


def build_dataset_incremental_splits(
    *,
    datasets: Sequence[tuple[str, Sequence[int], Sequence[int], int | None]],
    seed: int,
    val_ratio: float,
    image_size: int,
) -> list[TaskSplit]:
    tasks: list[TaskSplit] = []
    for task_id, (dataset_id, train_targets, test_targets, num_classes) in enumerate(datasets):
        train_pool = tuple(range(len(train_targets)))
        train_indices, val_indices = split_train_val(train_pool, val_ratio, seed + task_id + 1)
        classes = tuple(range(num_classes)) if num_classes is not None else tuple(sorted(set(train_targets)))
        tasks.append(
            make_task_split(
                task_id=task_id,
                dataset_id=dataset_id,
                classes=classes,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=tuple(range(len(test_targets))),
                image_size=image_size,
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
        "data_root": data_root,
        "created_by": created_by,
        "stream_type": stream_config["stream_type"],
        "stream_config_hash": stable_hash(stream_config),
        "registry_hash": registry_hash,
        "tasks": [task.to_dict() for task in tasks],
    }
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
