"""Backbone manifest loading and frozen feature model construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKBONES_CONFIG = REPO_ROOT / "configs" / "features" / "backbones.yaml"


@dataclass(frozen=True)
class BackboneSpec:
    backbone_id: str
    source: str
    loader: str
    model_id: str
    feature_dim: int
    image_size: int
    preprocess_id: str
    config: Mapping[str, Any]


class FrozenBackbone:
    """Small runtime wrapper around a frozen image feature extractor."""

    def __init__(
        self,
        *,
        backbone_id: str,
        model_id: str,
        feature_dim: int,
        image_size: int,
        preprocess_id: str,
        device: str,
        dataset_transform: object | None,
        encode_fn: Callable[[Sequence[object]], object],
    ) -> None:
        self.backbone_id = backbone_id
        self.model_id = model_id
        self.feature_dim = feature_dim
        self.image_size = image_size
        self.preprocess_id = preprocess_id
        self.device = device
        self.dataset_transform = dataset_transform
        self._encode_fn = encode_fn

    def encode(self, images: Sequence[object]) -> object:
        return self._encode_fn(images)


def load_backbone_specs(path: str | Path | None = None) -> dict[str, BackboneSpec]:
    config_path = Path(path or DEFAULT_BACKBONES_CONFIG)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    backbones = payload.get("backbones")
    if not isinstance(backbones, dict) or not backbones:
        raise ValueError(f"{config_path} must contain a non-empty backbones mapping")

    specs: dict[str, BackboneSpec] = {}
    for backbone_id, item in backbones.items():
        if not isinstance(item, dict):
            raise ValueError(f"Backbone entry {backbone_id!r} must be a mapping")
        if item.get("eval_mode") is not True:
            raise ValueError(f"Backbone {backbone_id!r} must declare eval_mode: true")
        if item.get("gradients_enabled") is not False:
            raise ValueError(f"Backbone {backbone_id!r} must declare gradients_enabled: false")
        specs[str(backbone_id)] = BackboneSpec(
            backbone_id=str(backbone_id),
            source=str(item["source"]),
            loader=str(item["loader"]),
            model_id=str(item["model_id"]),
            feature_dim=int(item["feature_dim"]),
            image_size=int(item["image_size"]),
            preprocess_id=str(item["preprocess_id"]),
            config=dict(item),
        )
    return specs


def get_backbone_spec(backbone_id: str, path: str | Path | None = None) -> BackboneSpec:
    specs = load_backbone_specs(path)
    try:
        return specs[backbone_id]
    except KeyError as exc:
        raise KeyError(f"Unknown backbone_id: {backbone_id}") from exc


def resolve_device(requested: str) -> str:
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"device must be one of auto, cuda, cpu; got {requested!r}")

    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def build_backbone(
    backbone_id: str,
    *,
    device: str = "auto",
    config_path: str | Path | None = None,
) -> FrozenBackbone:
    spec = get_backbone_spec(backbone_id, config_path)
    resolved_device = resolve_device(device)
    if spec.loader == "transformers_clip":
        return _build_clip_backbone(spec, resolved_device)
    if spec.loader == "torchvision_resnet18":
        return _build_resnet18_backbone(spec, resolved_device)
    raise ValueError(f"Unsupported backbone loader for {backbone_id}: {spec.loader}")


def _freeze_model(model: object) -> object:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _build_clip_backbone(spec: BackboneSpec, device: str) -> FrozenBackbone:
    try:
        import torch
        from transformers import AutoImageProcessor, CLIPModel
    except Exception as exc:
        raise RuntimeError("transformers and torch are required for clip_vit_b32 extraction") from exc

    image_processor = AutoImageProcessor.from_pretrained(spec.model_id)
    model = CLIPModel.from_pretrained(spec.model_id)
    _freeze_model(model)
    model.to(torch.device(device))

    def encode(images: Sequence[object]) -> object:
        inputs = image_processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(torch.device(device))
        with torch.inference_mode():
            return model.get_image_features(pixel_values=pixel_values).detach().cpu()

    return FrozenBackbone(
        backbone_id=spec.backbone_id,
        model_id=spec.model_id,
        feature_dim=spec.feature_dim,
        image_size=spec.image_size,
        preprocess_id=spec.preprocess_id,
        device=device,
        dataset_transform=None,
        encode_fn=encode,
    )


def _build_resnet18_backbone(spec: BackboneSpec, device: str) -> FrozenBackbone:
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:
        raise RuntimeError("torchvision and torch are required for resnet18 extraction") from exc

    weights_name = str(spec.config.get("weights", "IMAGENET1K_V1"))
    weights = ResNet18_Weights[weights_name]
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    _freeze_model(model)
    model.to(torch.device(device))
    transform = weights.transforms()

    def encode(images: Sequence[object]) -> object:
        tensors = []
        for image in images:
            if isinstance(image, torch.Tensor):
                tensors.append(image)
            else:
                tensors.append(transform(image))
        batch = torch.stack(tensors).to(torch.device(device))
        with torch.inference_mode():
            return model(batch).detach().cpu()

    return FrozenBackbone(
        backbone_id=spec.backbone_id,
        model_id=spec.model_id,
        feature_dim=spec.feature_dim,
        image_size=spec.image_size,
        preprocess_id=spec.preprocess_id,
        device=device,
        dataset_transform=transform,
        encode_fn=encode,
    )
