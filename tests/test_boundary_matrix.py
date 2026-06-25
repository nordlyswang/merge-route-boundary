from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mrb.diagnostics.matrices import compute_pairwise_boundary_matrix
from mrb.diagnostics.report import summarize_boundary_matrix


def test_two_synthetic_tasks_generate_one_pairwise_row(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, num_tasks=2, samples_per_task=4)
    _write_toy_bank(tmp_path, num_tasks=2, samples_per_task=4)

    df = compute_pairwise_boundary_matrix(
        stream_id="toy_stream",
        seed=0,
        backbone_id="dummy",
        split="train",
        max_samples_per_task=None,
        manifest_path=manifest,
        strict_features=True,
        min_samples_per_task=2,
        feature_root=tmp_path / "features",
    )

    assert len(df) == 1
    assert df.iloc[0]["status"] == "ok"


def test_three_synthetic_tasks_generate_three_pairwise_rows(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, num_tasks=3, samples_per_task=4)
    _write_toy_bank(tmp_path, num_tasks=3, samples_per_task=4)

    df = compute_pairwise_boundary_matrix(
        stream_id="toy_stream",
        seed=0,
        backbone_id="dummy",
        split="train",
        max_samples_per_task=None,
        manifest_path=manifest,
        strict_features=True,
        min_samples_per_task=2,
        feature_root=tmp_path / "features",
    )

    assert len(df) == 3
    assert (df["status"] == "ok").all()


def test_missing_feature_bank_records_failed_pair(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path, num_tasks=2, samples_per_task=4, datasets=["toy", "missing"]
    )
    _write_toy_bank(tmp_path, num_tasks=1, samples_per_task=4, dataset_id="toy")

    df = compute_pairwise_boundary_matrix(
        stream_id="toy_stream",
        seed=0,
        backbone_id="dummy",
        split="train",
        max_samples_per_task=None,
        manifest_path=manifest,
        strict_features=True,
        min_samples_per_task=2,
        feature_root=tmp_path / "features",
    )

    assert len(df) == 1
    assert df.iloc[0]["status"] == "failed"
    assert "Feature bank not found" in df.iloc[0]["warning"]


def test_val_logical_split_maps_to_train_feature_bank(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, num_tasks=2, samples_per_task=4)
    _write_toy_bank(tmp_path, num_tasks=2, samples_per_task=4)

    df = compute_pairwise_boundary_matrix(
        stream_id="toy_stream",
        seed=0,
        backbone_id="dummy",
        split="val",
        max_samples_per_task=None,
        manifest_path=manifest,
        strict_features=True,
        min_samples_per_task=2,
        feature_root=tmp_path / "features",
    )

    assert len(df) == 1
    assert df.iloc[0]["status"] == "ok"
    assert df.iloc[0]["feature_bank_split_i"] == "train"
    assert df.iloc[0]["feature_bank_split_j"] == "train"


def test_no_split_logical_split_maps_to_all_feature_bank(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, num_tasks=2, samples_per_task=4, source_split="all")
    _write_toy_bank(tmp_path, num_tasks=2, samples_per_task=4, split="all")

    df = compute_pairwise_boundary_matrix(
        stream_id="toy_stream",
        seed=0,
        backbone_id="dummy",
        split="test",
        max_samples_per_task=None,
        manifest_path=manifest,
        strict_features=True,
        min_samples_per_task=2,
        feature_root=tmp_path / "features",
    )

    assert len(df) == 1
    assert df.iloc[0]["status"] == "ok"
    assert df.iloc[0]["feature_bank_split_i"] == "all"
    assert df.iloc[0]["feature_bank_split_j"] == "all"


def test_diagnostics_config_can_disable_probe_metrics(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, num_tasks=2, samples_per_task=4)
    _write_toy_bank(tmp_path, num_tasks=2, samples_per_task=4)

    df = compute_pairwise_boundary_matrix(
        stream_id="toy_stream",
        seed=0,
        backbone_id="dummy",
        split="train",
        max_samples_per_task=None,
        manifest_path=manifest,
        strict_features=True,
        min_samples_per_task=2,
        feature_root=tmp_path / "features",
        diagnostics_config={
            "linear_probe": {"enabled": False},
            "knn": {"enabled": False},
        },
        config_hash="abc123",
    )

    row = df.iloc[0]
    assert row["status"] == "ok"
    assert np.isnan(row["linear_probe_auc_symmetric"])
    assert np.isnan(row["knn_domain_acc"])
    assert "linear_probe disabled" in row["warning"]
    assert "knn disabled" in row["warning"]
    assert df.attrs["config_hash"] == "abc123"


def test_summary_handles_all_failed_pairs_without_metric_values() -> None:
    df = pd.DataFrame(
        [
            {
                "task_i": 0,
                "task_j": 1,
                "dataset_i": "toy",
                "dataset_j": "toy",
                "status": "failed",
                "warning": "missing bank",
            }
        ]
    )

    summary = summarize_boundary_matrix(df)

    assert summary["num_ok_pairs"] == 0
    assert summary["num_failed_pairs"] == 1
    assert summary["most_separable_pairs"] == []
    assert summary["least_separable_pairs"] == []


def _write_manifest(
    tmp_path: Path,
    *,
    num_tasks: int,
    samples_per_task: int,
    datasets: list[str] | None = None,
    source_split: str = "train",
) -> Path:
    dataset_ids = datasets or ["toy"] * num_tasks
    tasks = []
    for task_id in range(num_tasks):
        start = task_id * samples_per_task
        indices = list(range(start, start + samples_per_task))
        tasks.append(
            {
                "task_id": task_id,
                "dataset_id": dataset_ids[task_id],
                "classes": [task_id],
                "train_indices": indices,
                "val_indices": indices,
                "test_indices": indices,
                "source_splits": {
                    "train": source_split,
                    "val": source_split,
                    "test": source_split,
                },
                "split_policy": "deterministic_holdout" if source_split == "all" else "predefined_splits",
            }
        )
    path = tmp_path / "manifest.json"
    payload = {"stream_id": "toy_stream", "seed": 0, "tasks": tasks}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_toy_bank(
    tmp_path: Path,
    *,
    num_tasks: int,
    samples_per_task: int,
    dataset_id: str = "toy",
    split: str = "train",
) -> None:
    features = []
    labels = []
    indices = []
    for task_id in range(num_tasks):
        for offset in range(samples_per_task):
            features.append([float(task_id * 10 + offset), float(task_id), 1.0])
            labels.append(task_id)
            indices.append(task_id * samples_per_task + offset)
    order = np.asarray(list(reversed(range(len(indices)))), dtype=np.int64)
    features_array = np.asarray(features, dtype=np.float32)[order]
    labels_array = np.asarray(labels, dtype=np.int64)[order]
    indices_array = np.asarray(indices, dtype=np.int64)[order]
    bank_dir = tmp_path / "features" / "dummy" / dataset_id / split
    bank_dir.mkdir(parents=True)
    np.save(bank_dir / "features.npy", features_array)
    np.save(bank_dir / "labels.npy", labels_array)
    np.save(bank_dir / "indices.npy", indices_array)
    metadata = {
        "dataset_id": dataset_id,
        "split": split,
        "backbone_id": "dummy",
        "dtype": "float32",
        "feature_dim": int(features_array.shape[1]),
        "num_samples": int(features_array.shape[0]),
    }
    (bank_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
