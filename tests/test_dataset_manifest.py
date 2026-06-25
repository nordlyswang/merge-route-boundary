from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "resources" / "datasets.yaml"


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def datasets_by_name() -> dict[str, dict]:
    payload = load_manifest()
    return {item["name"]: item for item in payload["datasets"]}


def test_dataset_manifest_loads() -> None:
    payload = load_manifest()
    assert payload["version"] == 1
    assert payload["root_env"] == "MRB_DATA_ROOT"
    assert payload["project_data_env"] == "MRB_PROJECT_DATA"
    assert isinstance(payload["datasets"], list)
    assert payload["datasets"]


def test_required_fields_exist() -> None:
    payload = load_manifest()
    names = [item["name"] for item in payload["datasets"]]
    assert len(names) == len(set(names))

    for item in payload["datasets"]:
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
        assert isinstance(item["estimated_gb"], (int, float))
        assert item["estimated_gb"] > 0
        assert isinstance(item["default_download"], bool)
        assert isinstance(item["verify"], dict)

        if item["source"] == "torchvision":
            assert item["torchvision_class"]
            assert item["loader"] in {"train_bool", "split", "no_split"}
            assert item["splits"]


def test_expected_datasets_by_tier() -> None:
    datasets = datasets_by_name()
    expected = {
        0: {"MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"},
        1: {
            "SVHN",
            "STL10",
            "Caltech101",
            "DTD",
            "EuroSAT",
            "OxfordIIITPet",
            "Flowers102",
            "FGVCAircraft",
        },
        2: {"Food101", "SUN397", "StanfordCars"},
        3: {
            "PACS",
            "OfficeHome",
            "ImageNetR",
            "ImageNetSketch",
            "DomainNetFull",
            "ImageNet1K",
        },
    }

    for tier, names in expected.items():
        for name in names:
            assert name in datasets
            assert datasets[name]["tier"] == tier


def test_tier0_has_default_downloads() -> None:
    for item in datasets_by_name().values():
        if item["tier"] == 0:
            assert item["default_download"] is True


def test_large_datasets_not_default_download() -> None:
    for item in datasets_by_name().values():
        if item.get("large"):
            assert item["default_download"] is False


def test_manual_datasets_not_default_download() -> None:
    for item in datasets_by_name().values():
        if item.get("manual") or item.get("gated") or "manual" in item["source"]:
            assert item["default_download"] is False
