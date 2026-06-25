from __future__ import annotations

from pathlib import Path

import yaml

from mrb.features.backbones import DEFAULT_BACKBONES_CONFIG, load_backbone_specs
from mrb.features.storage import DEFAULT_FEATURE_ROOT, FEATURE_ROOT_ENV


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_feature_backbone_manifest_loads() -> None:
    specs = load_backbone_specs(DEFAULT_BACKBONES_CONFIG)

    assert set(specs) == {"clip_vit_b32", "resnet18"}
    assert specs["clip_vit_b32"].model_id == "openai/clip-vit-base-patch32"
    assert specs["resnet18"].model_id == "torchvision/resnet18"
    assert specs["clip_vit_b32"].feature_dim == 512
    assert specs["resnet18"].feature_dim == 512


def test_backbone_manifest_is_frozen_eval_only() -> None:
    payload = yaml.safe_load(DEFAULT_BACKBONES_CONFIG.read_text(encoding="utf-8"))

    assert payload["feature_root_env"] == FEATURE_ROOT_ENV
    assert payload["paths"]["default_feature_root"] == str(DEFAULT_FEATURE_ROOT)
    for item in payload["backbones"].values():
        assert item["eval_mode"] is True
        assert item["gradients_enabled"] is False
        assert item["loader"] in {"transformers_clip", "torchvision_resnet18"}
        assert item["preprocess_id"]


def test_default_config_exposes_feature_root_env() -> None:
    payload = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))

    assert payload["paths"]["feature_root_env"] == FEATURE_ROOT_ENV
