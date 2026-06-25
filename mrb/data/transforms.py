"""Basic image transforms for loader smoke checks."""

from __future__ import annotations


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224) -> object:
    """Build a simple train-style transform for smoke checks."""

    try:
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("torchvision is required to build image transforms") from exc

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> object:
    """Build a deterministic eval-style transform for smoke checks."""

    try:
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("torchvision is required to build image transforms") from exc

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
