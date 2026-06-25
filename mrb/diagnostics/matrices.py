"""Pairwise boundary diagnostic matrix construction."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mrb.data.splits import load_manifest, stable_hash
from mrb.data.task_streams import build_split_manifest
from mrb.diagnostics import METRIC_VERSION
from mrb.diagnostics.feature_io import (
    FeatureBank,
    load_feature_bank,
    select_by_classes,
    select_by_indices,
)
from mrb.diagnostics.overlap import nearest_task_centroid_confusion, prototype_margin
from mrb.diagnostics.prototypes import compute_task_prototype
from mrb.diagnostics.report import summarize_boundary_matrix
from mrb.diagnostics.separability import (
    centroid_separability,
    knn_separability,
    linear_probe_separability,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "diagnostics" / "boundary_v0"

BOUNDARY_COLUMNS = [
    "stream_id",
    "seed",
    "backbone_id",
    "logical_split",
    "feature_bank_split_i",
    "feature_bank_split_j",
    "task_i",
    "task_j",
    "dataset_i",
    "dataset_j",
    "num_samples_i",
    "num_samples_j",
    "effective_num_samples_i",
    "effective_num_samples_j",
    "num_classes_i",
    "num_classes_j",
    "centroid_l2",
    "centroid_cosine_distance",
    "separation_ratio",
    "linear_probe_auc",
    "linear_probe_auc_symmetric",
    "linear_probe_acc",
    "knn_domain_acc",
    "nearest_task_centroid_acc",
    "task_i_to_j_confusion_rate",
    "task_j_to_i_confusion_rate",
    "mean_margin_i",
    "mean_margin_j",
    "resubstitution",
    "metric_version",
    "status",
    "warning",
]


@dataclass(frozen=True)
class PreparedTask:
    task_id: int
    dataset_id: str
    classes: tuple[int, ...]
    feature_bank_split: str
    index_field: str
    selected: FeatureBank | None
    sampled: FeatureBank | None
    num_samples: int
    effective_num_samples: int
    status: str
    warning: str


def compute_pairwise_boundary_matrix(
    stream_id: str,
    seed: int,
    backbone_id: str,
    split: str = "train",
    max_samples_per_task: int | None = 500,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    strict_features: bool = True,
    min_samples_per_task: int = 50,
    feature_root: Path | None = None,
    config_path: Path | None = None,
    diagnostics_config: dict[str, Any] | None = None,
    config_hash: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    manifest, resolved_manifest_path = _load_or_build_manifest(stream_id, seed, manifest_path)
    tasks = list(_iter_tasks(manifest))
    prepared = [
        _prepare_task(
            task,
            logical_split=split,
            backbone_id=backbone_id,
            max_samples_per_task=max_samples_per_task,
            min_samples_per_task=min_samples_per_task,
            strict_features=strict_features,
            feature_root=feature_root,
            seed=seed,
            diagnostics_config=diagnostics_config,
        )
        for task in tasks
    ]
    task_by_id = {task.task_id: task for task in prepared}
    rows: list[dict[str, Any]] = []
    for task_i, task_j in itertools.combinations(prepared, 2):
        rows.append(
            _compute_pair_row(
                stream_id=stream_id,
                seed=seed,
                backbone_id=backbone_id,
                logical_split=split,
                task_i=task_i,
                task_j=task_j,
                diagnostics_config=diagnostics_config,
            )
        )

    df = pd.DataFrame(rows, columns=BOUNDARY_COLUMNS)
    feature_metadata = _feature_metadata_summary(task_by_id)
    df.attrs["manifest_path"] = str(resolved_manifest_path) if resolved_manifest_path else None
    df.attrs["feature_bank_metadata"] = feature_metadata
    df.attrs["metric_version"] = METRIC_VERSION
    df.attrs["config_hash"] = config_hash

    if output_dir is not None:
        output_path = _matrix_output_path(
            Path(output_dir),
            stream_id=stream_id,
            seed=seed,
            backbone_id=backbone_id,
            split=split,
        )
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Pass --overwrite to replace it."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        summary = summarize_boundary_matrix(
            df,
            feature_bank_metadata=feature_metadata,
            manifest_path=str(resolved_manifest_path) if resolved_manifest_path else None,
            metric_version=METRIC_VERSION,
        )
        summary.update(
            {
                "stream_id": stream_id,
                "seed": seed,
                "backbone_id": backbone_id,
                "logical_split": split,
                "config_path": str(config_path) if config_path else None,
                "config_hash": config_hash,
                "output_csv": str(output_path),
            }
        )
        summary_path = output_path.with_name(output_path.stem + "_summary.json")
        summary_path.write_text(
            json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        df.attrs["output_csv"] = str(output_path)
        df.attrs["summary_json"] = str(summary_path)
    return df


def logical_split_mapping(logical_split: str) -> tuple[str, str]:
    if logical_split == "train":
        return "train", "train_indices"
    if logical_split == "val":
        return "train", "val_indices"
    if logical_split == "test":
        return "test", "test_indices"
    return logical_split, f"{logical_split}_indices"


def task_logical_split_mapping(task: dict[str, Any], logical_split: str) -> tuple[str, str]:
    _, index_field = logical_split_mapping(logical_split)
    source_splits = task.get("source_splits", {})
    if isinstance(source_splits, dict) and logical_split in source_splits:
        return str(source_splits[logical_split]), index_field
    return logical_split_mapping(logical_split)


def _load_or_build_manifest(
    stream_id: str,
    seed: int,
    manifest_path: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    if manifest_path is not None:
        path = Path(manifest_path)
        return load_manifest(path), path

    default_path = REPO_ROOT / "artifacts" / "manifests" / "splits" / f"{stream_id}_seed{seed}.json"
    if default_path.exists():
        return load_manifest(default_path), default_path

    manifest = build_split_manifest(
        stream_id=stream_id,
        seed=seed,
        created_by="mrb.diagnostics.matrices.compute_pairwise_boundary_matrix",
    )
    return manifest, None


def _iter_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest must contain a non-empty tasks list")
    return [dict(task) for task in tasks]


def _prepare_task(
    task: dict[str, Any],
    *,
    logical_split: str,
    backbone_id: str,
    max_samples_per_task: int | None,
    min_samples_per_task: int,
    strict_features: bool,
    feature_root: Path | None,
    seed: int,
    diagnostics_config: dict[str, Any] | None,
) -> PreparedTask:
    task_id = int(task["task_id"])
    dataset_id = str(task.get("dataset_id", task.get("dataset", "")))
    classes = tuple(int(value) for value in task.get("classes", []) or [])
    feature_bank_split, index_field = task_logical_split_mapping(task, logical_split)
    warnings: list[str] = []
    try:
        bank = load_feature_bank(
            dataset_id=dataset_id,
            split=feature_bank_split,
            backbone_id=backbone_id,
            feature_root=feature_root,
        )
        indices = task.get(index_field)
        if indices is not None:
            selected = select_by_indices(bank, indices, strict=strict_features)
            missing = int(selected.metadata.get("missing_indices_count", 0))
            if missing:
                warnings.append(f"{missing} manifest indices missing from feature bank")
        elif classes:
            selected = select_by_classes(bank, classes)
            warnings.append(f"manifest missing {index_field}; selected by classes")
        else:
            selected = bank
            warnings.append(f"manifest missing {index_field}; selected all feature-bank rows")

        num_samples = int(selected.features.shape[0])
        if num_samples < min_samples_per_task:
            raise ValueError(
                f"task {task_id} has only {num_samples} usable samples; "
                f"min_samples_per_task={min_samples_per_task}"
            )
        if max_samples_per_task is not None and num_samples < max_samples_per_task:
            warnings.append(
                f"requested max_samples_per_task={max_samples_per_task}, "
                f"but only {num_samples} usable samples are available"
            )
        sampled = _sample_feature_bank(
            selected,
            max_samples=max_samples_per_task,
            seed=seed + task_id,
        )
        effective = int(sampled.features.shape[0])
        return PreparedTask(
            task_id=task_id,
            dataset_id=dataset_id,
            classes=classes,
            feature_bank_split=feature_bank_split,
            index_field=index_field,
            selected=selected,
            sampled=sampled,
            num_samples=num_samples,
            effective_num_samples=effective,
            status="ok",
            warning=_join_warnings(warnings),
        )
    except Exception as exc:
        return PreparedTask(
            task_id=task_id,
            dataset_id=dataset_id,
            classes=classes,
            feature_bank_split=feature_bank_split,
            index_field=index_field,
            selected=None,
            sampled=None,
            num_samples=0,
            effective_num_samples=0,
            status="failed",
            warning=str(exc),
        )


def _sample_feature_bank(bank: FeatureBank, *, max_samples: int | None, seed: int) -> FeatureBank:
    if max_samples is None or bank.features.shape[0] <= max_samples:
        return bank
    rng = np.random.default_rng(seed)
    selected_rows: list[int] = []
    used: set[int] = set()
    labels = np.asarray(bank.labels)
    unique_labels = sorted(int(value) for value in np.unique(labels))

    if len(unique_labels) <= max_samples:
        base = max_samples // max(1, len(unique_labels))
        remainder = max_samples % max(1, len(unique_labels))
        for label_position, label in enumerate(unique_labels):
            rows = np.flatnonzero(labels == label)
            shuffled = rows.copy()
            rng.shuffle(shuffled)
            take = min(rows.shape[0], base + (1 if label_position < remainder else 0))
            for row in shuffled[:take]:
                row_int = int(row)
                selected_rows.append(row_int)
                used.add(row_int)

    if len(selected_rows) < max_samples:
        remaining = np.asarray(
            [row for row in range(labels.shape[0]) if row not in used], dtype=np.int64
        )
        rng.shuffle(remaining)
        selected_rows.extend(int(row) for row in remaining[: max_samples - len(selected_rows)])

    row_array = np.asarray(sorted(selected_rows[:max_samples]), dtype=np.int64)
    metadata = dict(bank.metadata)
    metadata.update(
        {
            "effective_num_samples": int(row_array.shape[0]),
            "max_samples_per_task": int(max_samples),
            "sampling_seed": int(seed),
            "sampling_strategy": "deterministic_stratified_by_label",
        }
    )
    return FeatureBank(
        features=np.asarray(bank.features[row_array], dtype=np.float32),
        labels=np.asarray(bank.labels[row_array], dtype=np.int64),
        indices=np.asarray(bank.indices[row_array], dtype=np.int64),
        metadata=metadata,
        dataset_id=bank.dataset_id,
        split=bank.split,
        backbone_id=bank.backbone_id,
        path=bank.path,
    )


def _compute_pair_row(
    *,
    stream_id: str,
    seed: int,
    backbone_id: str,
    logical_split: str,
    task_i: PreparedTask,
    task_j: PreparedTask,
    diagnostics_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = _base_row(
        stream_id=stream_id,
        seed=seed,
        backbone_id=backbone_id,
        logical_split=logical_split,
        task_i=task_i,
        task_j=task_j,
    )
    if (
        task_i.status != "ok"
        or task_j.status != "ok"
        or task_i.sampled is None
        or task_j.sampled is None
    ):
        warnings = [warning for warning in (task_i.warning, task_j.warning) if warning]
        return {**base, "status": "failed", "warning": _join_warnings(warnings)}

    warnings = [warning for warning in (task_i.warning, task_j.warning) if warning]
    try:
        features_i = task_i.sampled.features
        features_j = task_j.sampled.features
        proto_i = compute_task_prototype(features_i, normalize=True)
        proto_j = compute_task_prototype(features_j, normalize=True)
        centroid = centroid_separability(features_i, features_j)
        linear_config = dict((diagnostics_config or {}).get("linear_probe", {}))
        knn_config = dict((diagnostics_config or {}).get("knn", {}))
        if bool(linear_config.get("enabled", True)):
            linear = linear_probe_separability(
                features_i,
                features_j,
                seed=seed,
                max_samples_per_task=None,
                test_size=float(linear_config.get("test_size", 0.3)),
                max_iter=int(linear_config.get("max_iter", 1000)),
                class_weight=linear_config.get("class_weight", "balanced"),
            )
        else:
            linear = {
                "linear_probe_auc": math.nan,
                "linear_probe_auc_symmetric": math.nan,
                "linear_probe_acc": math.nan,
                "reason": "linear_probe disabled by diagnostics config",
            }
        if bool(knn_config.get("enabled", True)):
            knn = knn_separability(
                features_i,
                features_j,
                k=int(knn_config.get("k", 5)),
                seed=seed,
                max_samples_per_task=None,
            )
        else:
            knn = {
                "knn_domain_acc": math.nan,
                "knn_domain_auc": math.nan,
                "reason": "knn disabled by diagnostics config",
            }
        overlap = nearest_task_centroid_confusion(features_i, features_j, proto_i, proto_j)
        margin_i = prototype_margin(features_i, proto_i, proto_j)
        margin_j = prototype_margin(features_j, proto_j, proto_i)
        for reason in (linear.get("reason"), knn.get("reason")):
            if reason:
                warnings.append(str(reason))
        return {
            **base,
            **centroid,
            "linear_probe_auc": linear["linear_probe_auc"],
            "linear_probe_auc_symmetric": linear["linear_probe_auc_symmetric"],
            "linear_probe_acc": linear["linear_probe_acc"],
            "knn_domain_acc": knn["knn_domain_acc"],
            "nearest_task_centroid_acc": overlap["nearest_task_centroid_acc"],
            "task_i_to_j_confusion_rate": overlap["task_i_to_j_confusion_rate"],
            "task_j_to_i_confusion_rate": overlap["task_j_to_i_confusion_rate"],
            "mean_margin_i": margin_i["mean_prototype_margin"],
            "mean_margin_j": margin_j["mean_prototype_margin"],
            "resubstitution": True,
            "status": "ok",
            "warning": _join_warnings(warnings),
        }
    except Exception as exc:
        warnings.append(str(exc))
        return {**base, "status": "failed", "warning": _join_warnings(warnings)}


def _base_row(
    *,
    stream_id: str,
    seed: int,
    backbone_id: str,
    logical_split: str,
    task_i: PreparedTask,
    task_j: PreparedTask,
) -> dict[str, Any]:
    return {
        "stream_id": stream_id,
        "seed": int(seed),
        "backbone_id": backbone_id,
        "logical_split": logical_split,
        "feature_bank_split_i": task_i.feature_bank_split,
        "feature_bank_split_j": task_j.feature_bank_split,
        "task_i": task_i.task_id,
        "task_j": task_j.task_id,
        "dataset_i": task_i.dataset_id,
        "dataset_j": task_j.dataset_id,
        "num_samples_i": task_i.num_samples,
        "num_samples_j": task_j.num_samples,
        "effective_num_samples_i": task_i.effective_num_samples,
        "effective_num_samples_j": task_j.effective_num_samples,
        "num_classes_i": _num_classes(task_i),
        "num_classes_j": _num_classes(task_j),
        "centroid_l2": math.nan,
        "centroid_cosine_distance": math.nan,
        "separation_ratio": math.nan,
        "linear_probe_auc": math.nan,
        "linear_probe_auc_symmetric": math.nan,
        "linear_probe_acc": math.nan,
        "knn_domain_acc": math.nan,
        "nearest_task_centroid_acc": math.nan,
        "task_i_to_j_confusion_rate": math.nan,
        "task_j_to_i_confusion_rate": math.nan,
        "mean_margin_i": math.nan,
        "mean_margin_j": math.nan,
        "resubstitution": False,
        "metric_version": METRIC_VERSION,
        "status": "failed",
        "warning": "",
    }


def _num_classes(task: PreparedTask) -> int:
    if task.classes:
        return len(task.classes)
    if task.sampled is not None:
        return int(np.unique(task.sampled.labels).shape[0])
    return 0


def _feature_metadata_summary(task_by_id: dict[int, PreparedTask]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for task_id, task in sorted(task_by_id.items()):
        if task.selected is None:
            records.append({"task_id": task_id, "status": "failed", "warning": task.warning})
            continue
        metadata = dict(task.selected.metadata)
        records.append(
            {
                "task_id": task_id,
                "status": task.status,
                "dataset_id": task.dataset_id,
                "feature_bank_split": task.feature_bank_split,
                "path": str(task.selected.path) if task.selected.path else None,
                "num_samples": task.num_samples,
                "effective_num_samples": task.effective_num_samples,
                "feature_dim": int(task.selected.features.shape[1]),
                "metadata": metadata,
            }
        )
    return {"tasks": records, "metadata_hash": stable_hash(records)}


def _matrix_output_path(
    output_dir: Path,
    *,
    stream_id: str,
    seed: int,
    backbone_id: str,
    split: str,
) -> Path:
    name = f"{stream_id}_seed{seed}_{backbone_id}_{split}.csv"
    return output_dir / name


def _join_warnings(warnings: list[str]) -> str:
    seen: set[str] = set()
    unique: list[str] = []
    for warning in warnings:
        if warning and warning not in seen:
            seen.add(warning)
            unique.append(warning)
    return "; ".join(unique)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
