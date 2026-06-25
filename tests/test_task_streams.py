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


def test_different_seed_changes_class_order(tmp_path: Path) -> None:
    registry_path, stream_path = write_synthetic_configs(tmp_path)

    left = build_task_stream("toy_split_5x2", 0, registry_path, [stream_path])
    right = build_task_stream("toy_split_5x2", 1, registry_path, [stream_path])

    assert [task.classes for task in left.tasks] != [task.classes for task in right.tasks]
