from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = (".bin", ".safetensors", ".pt", ".pth", ".ckpt")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_dataset_manifest_shape() -> None:
    payload = load_yaml(REPO_ROOT / "configs" / "resources" / "datasets.yaml")
    datasets = payload["datasets"]
    assert isinstance(datasets, list)
    assert payload["version"] == 1
    assert any(item["default_download"] is True for item in datasets)

    for item in datasets:
        assert item["name"]
        assert item["tier"] in {0, 1, 2, 3}
        assert item["source"] in {
            "torchvision",
            "manual",
            "huggingface",
            "manual_or_huggingface",
            "manual_gated",
        }
        assert item["root_subdir"]
        assert isinstance(item["default_download"], bool)
        if item["source"] == "torchvision":
            assert item["torchvision_class"]
            assert item["splits"]


def test_model_manifest_shape_and_metadata_defaults() -> None:
    payload = load_yaml(REPO_ROOT / "configs" / "resources" / "models.yaml")
    models = payload["models"]
    assert isinstance(models, list)
    assert any(item["required"] is True for item in models)

    for item in models:
        assert item["id"]
        assert item["source"] == "huggingface"
        assert isinstance(item["required"], bool)
        assert item["default_mode"] == "metadata_only"
        assert item["allow_patterns"]
        for pattern in item["allow_patterns"]:
            assert not pattern.endswith(WEIGHT_SUFFIXES)


def test_no_container_config_files() -> None:
    forbidden = [
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".devcontainer",
    ]
    for name in forbidden:
        assert not (REPO_ROOT / name).exists()
