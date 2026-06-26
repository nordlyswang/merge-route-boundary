"""Feature-table construction for toy merge-vs-route baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mrb.data.splits import load_manifest
from mrb.data.task_streams import build_split_manifest
from mrb.diagnostics.feature_io import FeatureBank, load_feature_bank, select_by_indices


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "artifacts" / "manifests" / "splits"


@dataclass(frozen=True)
class TaskFeatureSplit:
    task_id: int
    dataset_id: str
    classes: tuple[int, ...]
    train_features: np.ndarray
    train_labels: np.ndarray
    train_indices: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    val_indices: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    test_indices: np.ndarray
    feature_bank_splits: dict[str, str]


@dataclass(frozen=True)
class FeatureToyDataset:
    stream_id: str
    seed: int
    backbone_id: str
    tasks: tuple[TaskFeatureSplit, ...]
    manifest: dict[str, Any]
    manifest_path: Path | None
    feature_bank_metadata: dict[str, Any]


def load_feature_toy_dataset(
    *,
    stream_id: str,
    seed: int,
    backbone_id: str,
    manifest_path: str | Path | None = None,
    feature_root: str | Path | None = None,
    max_train_per_task: int | None = None,
    max_val_per_task: int | None = None,
    max_test_per_task: int | None = None,
) -> FeatureToyDataset:
    manifest, resolved_manifest_path = _load_or_build_manifest(
        stream_id=stream_id,
        seed=seed,
        manifest_path=manifest_path,
    )
    bank_cache: dict[tuple[str, str], FeatureBank] = {}
    tasks: list[TaskFeatureSplit] = []
    metadata_records: list[dict[str, Any]] = []
    for task in _iter_tasks(manifest):
        task_id = int(task["task_id"])
        dataset_id = str(task.get("dataset_id", task.get("dataset", "")))
        classes = tuple(int(value) for value in task.get("classes", []) or [])
        train_bank, train_split = _select_task_bank(
            task,
            logical_split="train",
            dataset_id=dataset_id,
            backbone_id=backbone_id,
            feature_root=feature_root,
            bank_cache=bank_cache,
        )
        val_bank, val_split = _select_task_bank(
            task,
            logical_split="val",
            dataset_id=dataset_id,
            backbone_id=backbone_id,
            feature_root=feature_root,
            bank_cache=bank_cache,
        )
        test_bank, test_split = _select_task_bank(
            task,
            logical_split="test",
            dataset_id=dataset_id,
            backbone_id=backbone_id,
            feature_root=feature_root,
            bank_cache=bank_cache,
        )
        train_arrays = _sample_arrays(
            train_bank.features,
            train_bank.labels,
            train_bank.indices,
            max_samples=max_train_per_task,
            seed=seed + task_id + 11,
        )
        val_arrays = _sample_arrays(
            val_bank.features,
            val_bank.labels,
            val_bank.indices,
            max_samples=max_val_per_task,
            seed=seed + task_id + 17,
        )
        test_arrays = _sample_arrays(
            test_bank.features,
            test_bank.labels,
            test_bank.indices,
            max_samples=max_test_per_task,
            seed=seed + task_id + 23,
        )
        tasks.append(
            TaskFeatureSplit(
                task_id=task_id,
                dataset_id=dataset_id,
                classes=classes,
                train_features=train_arrays[0],
                train_labels=train_arrays[1],
                train_indices=train_arrays[2],
                val_features=val_arrays[0],
                val_labels=val_arrays[1],
                val_indices=val_arrays[2],
                test_features=test_arrays[0],
                test_labels=test_arrays[1],
                test_indices=test_arrays[2],
                feature_bank_splits={"train": train_split, "val": val_split, "test": test_split},
            )
        )
        metadata_records.append(
            {
                "task_id": task_id,
                "dataset_id": dataset_id,
                "classes": list(classes),
                "feature_bank_splits": {
                    "train": train_split,
                    "val": val_split,
                    "test": test_split,
                },
                "num_train": int(train_arrays[0].shape[0]),
                "num_val": int(val_arrays[0].shape[0]),
                "num_test": int(test_arrays[0].shape[0]),
                "feature_dim": int(train_arrays[0].shape[1]),
                "train_path": str(train_bank.path) if train_bank.path else None,
                "val_path": str(val_bank.path) if val_bank.path else None,
                "test_path": str(test_bank.path) if test_bank.path else None,
                "train_metadata": dict(train_bank.metadata),
                "test_metadata": dict(test_bank.metadata),
            }
        )
    if not tasks:
        raise ValueError("feature toy dataset contains no tasks")
    return FeatureToyDataset(
        stream_id=stream_id,
        seed=int(seed),
        backbone_id=backbone_id,
        tasks=tuple(tasks),
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        feature_bank_metadata={"tasks": metadata_records},
    )


def stack_task_split(
    tasks: Sequence[TaskFeatureSplit],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    task_ids: list[np.ndarray] = []
    for task in tasks:
        split_features = np.asarray(getattr(task, f"{split}_features"), dtype=np.float32)
        split_labels = np.asarray(getattr(task, f"{split}_labels"), dtype=np.int64)
        features.append(split_features)
        labels.append(split_labels)
        task_ids.append(np.full(split_labels.shape[0], task.task_id, dtype=np.int64))
    if not features:
        raise ValueError("tasks must be non-empty")
    return (
        np.vstack(features).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(task_ids).astype(np.int64),
    )


def _load_or_build_manifest(
    *,
    stream_id: str,
    seed: int,
    manifest_path: str | Path | None,
) -> tuple[dict[str, Any], Path | None]:
    if manifest_path is not None:
        path = Path(manifest_path)
        return load_manifest(path), path
    default_path = DEFAULT_MANIFEST_DIR / f"{stream_id}_seed{seed}.json"
    if default_path.exists():
        return load_manifest(default_path), default_path
    manifest = build_split_manifest(
        stream_id=stream_id,
        seed=seed,
        created_by="mrb.baselines.feature_data.load_feature_toy_dataset",
    )
    return manifest, None


def _iter_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest must contain a non-empty tasks list")
    return [dict(task) for task in tasks]


def _select_task_bank(
    task: dict[str, Any],
    *,
    logical_split: str,
    dataset_id: str,
    backbone_id: str,
    feature_root: str | Path | None,
    bank_cache: dict[tuple[str, str], FeatureBank],
) -> tuple[FeatureBank, str]:
    feature_bank_split, index_field = _task_logical_split_mapping(task, logical_split)
    cache_key = (dataset_id, feature_bank_split)
    if cache_key not in bank_cache:
        bank_cache[cache_key] = load_feature_bank(
            dataset_id=dataset_id,
            split=feature_bank_split,
            backbone_id=backbone_id,
            feature_root=feature_root,
        )
    indices = task.get(index_field)
    if indices is None:
        raise KeyError(f"task {task.get('task_id')} missing {index_field}")
    return select_by_indices(bank_cache[cache_key], indices, strict=True), feature_bank_split


def _task_logical_split_mapping(task: dict[str, Any], logical_split: str) -> tuple[str, str]:
    index_field = f"{logical_split}_indices"
    source_splits = task.get("source_splits", {})
    if isinstance(source_splits, dict) and logical_split in source_splits:
        return str(source_splits[logical_split]), index_field
    if logical_split == "val":
        return "train", "val_indices"
    return logical_split, index_field


def _sample_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_values = np.asarray(features, dtype=np.float32)
    label_values = np.asarray(labels, dtype=np.int64)
    index_values = np.asarray(indices, dtype=np.int64)
    if max_samples is None or feature_values.shape[0] <= int(max_samples):
        return feature_values, label_values, index_values
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    used: set[int] = set()
    unique_labels = sorted(int(value) for value in np.unique(label_values))
    if len(unique_labels) <= int(max_samples):
        base = int(max_samples) // max(1, len(unique_labels))
        remainder = int(max_samples) % max(1, len(unique_labels))
        for label_position, label in enumerate(unique_labels):
            rows = np.flatnonzero(label_values == label)
            shuffled = rows.copy()
            rng.shuffle(shuffled)
            take = min(rows.shape[0], base + (1 if label_position < remainder else 0))
            for row in shuffled[:take]:
                row_int = int(row)
                selected.append(row_int)
                used.add(row_int)
    if len(selected) < int(max_samples):
        remaining = np.asarray(
            [row for row in range(label_values.shape[0]) if row not in used], dtype=np.int64
        )
        rng.shuffle(remaining)
        selected.extend(int(row) for row in remaining[: int(max_samples) - len(selected)])
    row_array = np.asarray(sorted(selected[: int(max_samples)]), dtype=np.int64)
    return feature_values[row_array], label_values[row_array], index_values[row_array]
