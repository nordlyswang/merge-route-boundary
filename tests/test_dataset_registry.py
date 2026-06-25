from __future__ import annotations

from pathlib import Path

import yaml

from mrb.data.registry import inspect_dataset, load_registry


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs" / "datasets" / "registry.yaml"


def test_registry_config_loads() -> None:
    registry = load_registry(REGISTRY_PATH)

    assert registry.version == 1
    assert registry.root_env == "MRB_DATA_ROOT"
    assert "cifar100" in registry.datasets
    assert registry.get("cifar100").num_classes == 100


def test_env_paths_resolve(tmp_path: Path) -> None:
    registry = load_registry(REGISTRY_PATH)
    entry = registry.get("cifar10")

    resolved = registry.dataset_root(entry, env={"MRB_DATA_ROOT": str(tmp_path)})

    assert resolved == tmp_path / "torchvision"


def test_missing_dataset_status_reports_missing(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    payload = {
        "version": 1,
        "root_env": "MRB_DATA_ROOT",
        "paths": {"shared_root": str(tmp_path)},
        "datasets": {
            "missing_toy": {
                "tier": 0,
                "source": "synthetic",
                "root": "${MRB_DATA_ROOT}/missing_toy",
                "storage_markers": ["marker.file"],
                "task_type": "classification",
                "num_classes": 2,
                "splits": ["train", "test"],
                "default_image_size": 32,
                "expected_available": False,
                "allow_in_smoke": False,
                "large_dataset": False,
                "manual_download": False,
                "loader_status": "supported",
                "synthetic_sizes": {"train": 10, "test": 4},
            }
        },
    }
    registry_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    registry = load_registry(registry_path)

    status = inspect_dataset(
        registry,
        registry.get("missing_toy"),
        env={"MRB_DATA_ROOT": str(tmp_path)},
    )

    assert status["available"] is False
    assert "dataset storage markers are missing" in status["warnings"]
