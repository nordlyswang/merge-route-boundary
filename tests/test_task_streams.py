from __future__ import annotations

from pathlib import Path

import yaml

from mrb.data.task_streams import build_task_stream


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
                "class_order_seed": 0,
                "val_ratio": 0.1,
                "image_size": 32,
            }
        }
    }
    registry_path.write_text(yaml.safe_dump(registry_payload), encoding="utf-8")
    stream_path.write_text(yaml.safe_dump(stream_payload), encoding="utf-8")
    return registry_path, stream_path


def test_class_incremental_task_count_and_width(tmp_path: Path) -> None:
    registry_path, stream_path = write_synthetic_configs(tmp_path)

    stream = build_task_stream("toy_split_5x2", 0, registry_path, [stream_path])

    assert len(stream.tasks) == 5
    assert all(len(task.classes) == 2 for task in stream.tasks)
    assert all(task.num_train == 180 for task in stream.tasks)
    assert all(task.num_val == 20 for task in stream.tasks)
    assert all(task.num_test == 100 for task in stream.tasks)


def test_same_seed_is_reproducible(tmp_path: Path) -> None:
    registry_path, stream_path = write_synthetic_configs(tmp_path)

    left = build_task_stream("toy_split_5x2", 0, registry_path, [stream_path])
    right = build_task_stream("toy_split_5x2", 0, registry_path, [stream_path])

    assert [task.classes for task in left.tasks] == [task.classes for task in right.tasks]
    assert [task.split_hash for task in left.tasks] == [task.split_hash for task in right.tasks]


def test_different_seed_changes_splits_not_class_order(tmp_path: Path) -> None:
    registry_path, stream_path = write_synthetic_configs(tmp_path)

    left = build_task_stream("toy_split_5x2", 0, registry_path, [stream_path])
    right = build_task_stream("toy_split_5x2", 1, registry_path, [stream_path])

    assert [task.classes for task in left.tasks] == [task.classes for task in right.tasks]
    assert [task.split_hash for task in left.tasks] != [task.split_hash for task in right.tasks]


def test_different_class_order_seed_changes_class_order(tmp_path: Path) -> None:
    registry_path, stream_path = write_synthetic_configs(tmp_path)
    payload = yaml.safe_load(stream_path.read_text(encoding="utf-8"))
    payload["streams"]["toy_split_5x2"]["class_order_seed"] = 1
    stream_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    left = build_task_stream("toy_split_5x2", 0, registry_path, [stream_path])
    payload["streams"]["toy_split_5x2"]["class_order_seed"] = 2
    stream_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    right = build_task_stream("toy_split_5x2", 0, registry_path, [stream_path])

    assert [task.classes for task in left.tasks] != [task.classes for task in right.tasks]


def test_manifest_records_class_order_and_split_seed(tmp_path: Path) -> None:
    registry_path, stream_path = write_synthetic_configs(tmp_path)
    stream = build_task_stream("toy_split_5x2", 7, registry_path, [stream_path])
    manifest = stream.to_manifest(created_by="test")

    assert manifest["class_order_seed"] == 0
    assert manifest["split_seed"] == 7


def test_no_split_dataset_uses_deterministic_holdout(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    stream_path = tmp_path / "streams.yaml"
    registry_payload = {
        "version": 1,
        "root_env": "MRB_DATA_ROOT",
        "paths": {"shared_root": str(tmp_path)},
        "datasets": {
            "toy_all": {
                "tier": 1,
                "source": "synthetic",
                "root": "${MRB_DATA_ROOT}/toy_all",
                "task_type": "classification",
                "num_classes": 3,
                "splits": ["all"],
                "default_image_size": 32,
                "expected_available": False,
                "allow_in_smoke": False,
                "large_dataset": False,
                "manual_download": False,
                "loader": "no_split",
                "loader_status": "supported",
                "synthetic_sizes": {"all": 60},
            }
        },
    }
    stream_payload = {
        "streams": {
            "toy_no_split": {
                "stream_type": "dataset_incremental",
                "datasets": ["toy_all"],
                "split_policy": "deterministic_holdout",
                "holdout": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
                "image_size": 32,
            }
        }
    }
    registry_path.write_text(yaml.safe_dump(registry_payload), encoding="utf-8")
    stream_path.write_text(yaml.safe_dump(stream_payload), encoding="utf-8")

    stream = build_task_stream("toy_no_split", 0, registry_path, [stream_path])
    task = stream.tasks[0]

    assert task.split_policy == "deterministic_holdout"
    assert task.source_splits == {"train": "all", "val": "all", "test": "all"}
    assert task.num_train > 0 and task.num_val > 0 and task.num_test > 0
    assert not (set(task.train_indices) & set(task.val_indices))
    assert not (set(task.train_indices) & set(task.test_indices))
    assert not (set(task.val_indices) & set(task.test_indices))

    rebuilt = build_task_stream("toy_no_split", 0, registry_path, [stream_path])
    changed = build_task_stream("toy_no_split", 1, registry_path, [stream_path])
    assert rebuilt.tasks[0].split_hash == task.split_hash
    assert changed.tasks[0].split_hash != task.split_hash
