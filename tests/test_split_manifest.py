from __future__ import annotations

from pathlib import Path

import yaml

from mrb.data.splits import stable_hash
from mrb.data.task_streams import build_split_manifest


def write_synthetic_configs(tmp_path: Path) -> tuple[Path, Path]:
    registry_path = tmp_path / "registry.yaml"
    stream_path = tmp_path / "streams.yaml"
    registry_payload = {
        "version": 1,
        "root_env": "MRB_DATA_ROOT",
        "paths": {"shared_root": str(tmp_path)},
        "datasets": {
            "toy100": {
                "tier": 0,
                "source": "synthetic",
                "root": "${MRB_DATA_ROOT}/toy100",
                "task_type": "classification",
                "num_classes": 10,
                "splits": ["train", "test"],
                "default_image_size": 32,
                "expected_available": False,
                "allow_in_smoke": False,
                "large_dataset": False,
                "manual_download": False,
                "loader_status": "supported",
                "synthetic_sizes": {"train": 1000, "test": 500},
            }
        },
    }
    stream_payload = {
        "streams": {
            "toy_split_5x2": {
                "stream_type": "class_incremental",
                "base_dataset": "toy100",
                "num_tasks": 5,
                "classes_per_task": 2,
                "val_ratio": 0.1,
                "image_size": 32,
            }
        }
    }
    registry_path.write_text(yaml.safe_dump(registry_payload), encoding="utf-8")
    stream_path.write_text(yaml.safe_dump(stream_payload), encoding="utf-8")
    return registry_path, stream_path


def test_manifest_schema_and_hash_are_stable(tmp_path: Path) -> None:
    registry_path, stream_path = write_synthetic_configs(tmp_path)

    manifest = build_split_manifest(
        stream_id="toy_split_5x2",
        seed=0,
        registry_path=registry_path,
        stream_config_paths=[stream_path],
        created_by="test",
    )
    rebuilt = build_split_manifest(
        stream_id="toy_split_5x2",
        seed=0,
        registry_path=registry_path,
        stream_config_paths=[stream_path],
        created_by="test",
    )

    assert manifest == rebuilt
    assert manifest["schema_version"] == 1
    assert manifest["stream_id"] == "toy_split_5x2"
    assert manifest["seed"] == 0
    assert len(manifest["tasks"]) == 5
    assert manifest["manifest_hash"] == rebuilt["manifest_hash"]


def test_task_split_hash_and_train_val_disjoint(tmp_path: Path) -> None:
    registry_path, stream_path = write_synthetic_configs(tmp_path)

    manifest = build_split_manifest(
        stream_id="toy_split_5x2",
        seed=0,
        registry_path=registry_path,
        stream_config_paths=[stream_path],
        created_by="test",
    )
    first = manifest["tasks"][0]
    train = set(first["train_indices"])
    val = set(first["val_indices"])

    assert train
    assert val
    assert not train & val
    assert first["split_hash"] == stable_hash(
        {
            "task_id": first["task_id"],
            "dataset_id": first["dataset_id"],
            "classes": first["classes"],
            "train_indices": first["train_indices"],
            "val_indices": first["val_indices"],
            "test_indices": first["test_indices"],
            "image_size": first["image_size"],
        }
    )
