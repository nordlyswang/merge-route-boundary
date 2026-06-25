#!/usr/bin/env python
"""Run quick non-training smoke checks."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def path_status() -> None:
    print("Path checks:")
    for name, fallback in {
        "MRB_DATA_ROOT": Path("/root/rivermind-data/datasets"),
        "MRB_MODEL_ROOT": REPO_ROOT / "models",
        "MRB_CACHE_ROOT": REPO_ROOT / "cache",
        "MRB_OUTPUT_ROOT": REPO_ROOT / "outputs",
        "MRB_FEATURE_ROOT": Path("/root/rivermind-data/datasets/_derived/merge-route-boundary/features"),
        "HF_HOME": REPO_ROOT / "cache" / "huggingface",
        "HF_HUB_CACHE": REPO_ROOT / "cache" / "huggingface" / "hub",
        "HF_DATASETS_CACHE": REPO_ROOT / "cache" / "huggingface" / "datasets",
        "TORCH_HOME": REPO_ROOT / "cache" / "torch",
    }.items():
        path = Path(os.environ.get(name, fallback)).expanduser()
        if not os.environ.get(name):
            print(f"  WARN {name} not set; fallback would be {path}")
        else:
            print(f"  {name}: {path}")


def import_checks() -> None:
    print("Import checks:")
    for module_name in ["yaml", "torch", "torchvision", "transformers", "datasets", "open_clip"]:
        print(f"  {'OK' if has_module(module_name) else 'WARN missing'} {module_name}")


def data_batch_check(quick: bool) -> None:
    if not has_module("torch") or not has_module("torchvision"):
        print("WARN skipping data batch check; torch/torchvision unavailable")
        return

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    data_root = Path(os.environ.get("MRB_DATA_ROOT", "/root/rivermind-data/datasets")).expanduser()
    torchvision_root = data_root / "torchvision"
    transform = transforms.ToTensor()
    try:
        dataset = datasets.CIFAR10(
            root=str(torchvision_root),
            train=True,
            download=False,
            transform=transform,
        )
        source = "CIFAR10"
    except RuntimeError:
        if not quick:
            raise
        dataset = datasets.FakeData(size=8, image_size=(3, 32, 32), num_classes=10, transform=transform)
        source = "FakeData fallback because CIFAR10 is not present"

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    print(f"Data batch: {source} images={tuple(images.shape)} labels={tuple(labels.shape)}")
    assert isinstance(images, torch.Tensor)


def model_metadata_check() -> None:
    if not has_module("transformers"):
        print("WARN skipping model metadata check; transformers unavailable")
        return

    from transformers import AutoConfig, AutoTokenizer

    model_id = "openai/clip-vit-base-patch32"
    model_root = Path(os.environ.get("MRB_MODEL_ROOT", REPO_ROOT / "models")).expanduser()
    local_dir = model_root / "huggingface" / model_id.replace("/", "__")
    source = str(local_dir) if local_dir.exists() else model_id
    local_only = not os.environ.get("MRB_SMOKE_ALLOW_DOWNLOAD")

    try:
        config = AutoConfig.from_pretrained(source, local_files_only=local_only)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
    except Exception as exc:
        print(f"WARN skipping model metadata load for {model_id}: {exc}")
        return
    print(f"Model metadata: {model_id} config={config.model_type} tokenizer={tokenizer.__class__.__name__}")


def tensor_forward_check() -> None:
    if not has_module("torch"):
        print("WARN skipping tensor check; torch unavailable")
        return

    import torch

    layer = torch.nn.Linear(8, 4)
    output = layer(torch.randn(2, 8))
    print(f"Tensor forward: output={tuple(output.shape)}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        left = torch.randn(16, 16, device=device)
        right = torch.randn(16, 16, device=device)
        product = left @ right
        torch.cuda.synchronize()
        print(f"CUDA matmul: device={product.device} shape={tuple(product.shape)}")
    else:
        print("WARN CUDA unavailable; skipped CUDA matmul")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use local-only quick checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import_checks()
    path_status()
    data_batch_check(quick=args.quick)
    model_metadata_check()
    tensor_forward_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
