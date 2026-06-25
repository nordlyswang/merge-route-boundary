"""Build deterministic task streams from registry and stream configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from mrb.data.datasets import (
    default_test_split,
    default_train_split,
    get_dataset,
    get_targets,
    infer_num_classes,
)
from mrb.data.registry import DEFAULT_REGISTRY_PATH, DatasetRegistry, load_registry, normalize_dataset_id
from mrb.data.splits import (
    TaskSplit,
    build_class_incremental_splits,
    build_dataset_incremental_splits,
    manifest_from_tasks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STREAM_CONFIG_DIR = REPO_ROOT / "configs" / "task_streams"


@dataclass(frozen=True)
class Task:
    task_id: int
    dataset_id: str
    classes: tuple[int, ...]
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    num_train: int
    num_val: int
    num_test: int
    image_size: int
    split_hash: str
    source_splits: dict[str, str]
    split_policy: str

    @classmethod
    def from_split(cls, task: TaskSplit) -> "Task":
        return cls(
            task_id=task.task_id,
            dataset_id=task.dataset_id,
            classes=task.classes,
            train_indices=task.train_indices,
            val_indices=task.val_indices,
            test_indices=task.test_indices,
            num_train=task.num_train,
            num_val=task.num_val,
            num_test=task.num_test,
            image_size=task.image_size,
            split_hash=task.split_hash,
            source_splits=dict(task.source_splits or {}),
            split_policy=task.split_policy,
        )


@dataclass(frozen=True)
class TaskStream:
    stream_id: str
    seed: int
    stream_type: str
    config: dict[str, Any]
    registry: DatasetRegistry
    tasks: tuple[Task, ...]
    task_splits: tuple[TaskSplit, ...]

    def to_manifest(self, *, created_by: str) -> dict[str, Any]:
        return manifest_from_tasks(
            stream_id=self.stream_id,
            seed=self.seed,
            data_root=str(self.registry.data_root()),
            stream_config=self.config,
            registry_hash=self.registry.registry_hash(),
            tasks=self.task_splits,
            created_by=created_by,
        )


def load_task_stream_configs(paths: Sequence[str | Path] | None = None) -> dict[str, dict[str, Any]]:
    config_paths = [Path(path) for path in paths] if paths else sorted(DEFAULT_STREAM_CONFIG_DIR.glob("*.yaml"))
    streams: dict[str, dict[str, Any]] = {}
    for path in config_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a mapping")
        payload_streams = payload.get("streams", payload)
        if not isinstance(payload_streams, dict):
            raise ValueError(f"{path} streams must be a mapping")
        for stream_id, config in payload_streams.items():
            if not isinstance(config, dict):
                raise ValueError(f"stream {stream_id} in {path} must be a mapping")
            if stream_id in streams:
                raise ValueError(f"duplicate stream_id: {stream_id}")
            streams[str(stream_id)] = {**config, "stream_id": str(stream_id)}
    return streams


def build_task_stream(
    stream_id: str,
    seed: int,
    registry_path: str | Path | None = None,
    stream_config_paths: Sequence[str | Path] | None = None,
) -> TaskStream:
    registry = load_registry(registry_path or DEFAULT_REGISTRY_PATH)
    configs = load_task_stream_configs(stream_config_paths)
    if stream_id not in configs:
        raise KeyError(f"Unknown stream_id: {stream_id}")
    config = dict(configs[stream_id])
    stream_type = str(config["stream_type"])

    if stream_type == "class_incremental":
        config["class_order_seed"] = int(config.get("class_order_seed", seed))
        config["split_seed"] = int(seed)
        task_splits = _build_class_incremental(config, seed, registry)
    elif stream_type == "dataset_incremental":
        config["split_seed"] = int(seed)
        task_splits = _build_dataset_incremental(config, seed, registry)
    else:
        raise ValueError(f"Unsupported stream_type {stream_type!r}")

    return TaskStream(
        stream_id=stream_id,
        seed=seed,
        stream_type=stream_type,
        config=config,
        registry=registry,
        tasks=tuple(Task.from_split(task) for task in task_splits),
        task_splits=tuple(task_splits),
    )


def build_split_manifest(
    *,
    stream_id: str,
    seed: int,
    registry_path: str | Path | None = None,
    stream_config_paths: Sequence[str | Path] | None = None,
    created_by: str,
) -> dict[str, Any]:
    stream = build_task_stream(
        stream_id=stream_id,
        seed=seed,
        registry_path=registry_path,
        stream_config_paths=stream_config_paths,
    )
    return stream.to_manifest(created_by=created_by)


def _build_class_incremental(
    config: dict[str, Any],
    seed: int,
    registry: DatasetRegistry,
) -> list[TaskSplit]:
    dataset_id = normalize_dataset_id(str(config["base_dataset"]))
    entry = registry.get(dataset_id)
    train_split = str(config.get("train_split", default_train_split(entry)))
    test_split = str(config.get("test_split", default_test_split(entry)))
    train_dataset = get_dataset(dataset_id, train_split, registry=registry)
    test_dataset = get_dataset(dataset_id, test_split, registry=registry)
    num_classes = entry.num_classes or infer_num_classes(train_dataset)
    if num_classes is None:
        raise ValueError(f"Could not infer num_classes for {dataset_id}")

    return build_class_incremental_splits(
        dataset_id=dataset_id,
        train_targets=get_targets(train_dataset),
        test_targets=get_targets(test_dataset),
        num_classes=num_classes,
        num_tasks=int(config["num_tasks"]),
        classes_per_task=int(config["classes_per_task"]),
        seed=seed,
        class_order_seed=int(config.get("class_order_seed", seed)),
        split_seed=int(config.get("split_seed", seed)),
        val_ratio=float(config.get("val_ratio", 0.1)),
        image_size=int(config.get("image_size", entry.default_image_size)),
        train_source_split=train_split,
        test_source_split=test_split,
    )


def _build_dataset_incremental(
    config: dict[str, Any],
    seed: int,
    registry: DatasetRegistry,
) -> list[TaskSplit]:
    dataset_payloads = config.get("datasets", [])
    if not isinstance(dataset_payloads, list) or not dataset_payloads:
        raise ValueError("dataset_incremental stream requires a non-empty datasets list")

    loaded: list[dict[str, Any]] = []
    for item in dataset_payloads:
        if isinstance(item, str):
            dataset_id = normalize_dataset_id(item)
            disabled = False
            train_split = None
            test_split = None
        elif isinstance(item, dict):
            dataset_id = normalize_dataset_id(str(item.get("dataset_id", item.get("id", ""))))
            disabled = bool(item.get("disabled", False))
            train_split = item.get("train_split")
            test_split = item.get("test_split")
        else:
            raise ValueError(f"Invalid dataset item: {item!r}")

        if disabled:
            continue
        entry = registry.get(dataset_id)
        train_name = str(train_split or config.get("train_split") or default_train_split(entry))
        test_name = str(test_split or config.get("test_split") or default_test_split(entry))
        loader_mode = str(entry.config.get("loader", "split"))
        if loader_mode == "no_split" and train_name == test_name:
            source_dataset = get_dataset(dataset_id, train_name, registry=registry)
            holdout = dict(
                config.get(
                    "holdout",
                    {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
                )
            )
            loaded.append(
                {
                    "dataset_id": dataset_id,
                    "train_targets": get_targets(source_dataset),
                    "test_targets": get_targets(source_dataset),
                    "num_classes": entry.num_classes or infer_num_classes(source_dataset),
                    "split_policy": str(config.get("split_policy", "deterministic_holdout")),
                    "holdout": holdout,
                    "source_splits": {
                        "train": train_name,
                        "val": train_name,
                        "test": train_name,
                    },
                }
            )
        else:
            train_dataset = get_dataset(dataset_id, train_name, registry=registry)
            test_dataset = get_dataset(dataset_id, test_name, registry=registry)
            loaded.append(
                {
                    "dataset_id": dataset_id,
                    "train_targets": get_targets(train_dataset),
                    "test_targets": get_targets(test_dataset),
                    "num_classes": entry.num_classes or infer_num_classes(train_dataset),
                    "split_policy": "predefined_splits",
                    "source_splits": {
                        "train": train_name,
                        "val": train_name,
                        "test": test_name,
                    },
                }
            )

    if not loaded:
        raise ValueError("dataset_incremental stream has no enabled datasets")

    return build_dataset_incremental_splits(
        datasets=loaded,
        seed=seed,
        val_ratio=float(config.get("val_ratio", 0.1)),
        image_size=int(config.get("image_size", 224)),
    )
