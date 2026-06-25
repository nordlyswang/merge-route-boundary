from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mrb.diagnostics.matrices import compute_pairwise_boundary_matrix


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


def _write_manifest(
    tmp_path: Path,
    *,
    num_tasks: int,
    samples_per_task: int,
    datasets: list[str] | None = None,
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
    bank_dir = tmp_path / "features" / "dummy" / dataset_id / "train"
    bank_dir.mkdir(parents=True)
    np.save(bank_dir / "features.npy", features_array)
    np.save(bank_dir / "labels.npy", labels_array)
    np.save(bank_dir / "indices.npy", indices_array)
    metadata = {
        "dataset_id": dataset_id,
        "split": "train",
        "backbone_id": "dummy",
        "dtype": "float32",
        "feature_dim": int(features_array.shape[1]),
        "num_samples": int(features_array.shape[0]),
    }
    (bank_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
