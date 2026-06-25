"""Read-only dataset loader interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from mrb.data.registry import DatasetEntry, DatasetRegistry, dataset_available, load_registry


class DatasetUnavailableError(RuntimeError):
    """Raised when a registered dataset is missing or cannot be loaded locally."""


class UnsupportedDatasetError(RuntimeError):
    """Raised when an entry is inspect-only or has no loader implementation."""


class SyntheticDataset:
    """Small deterministic dataset used by unit tests and metadata-only examples."""

    def __init__(
        self,
        *,
        size: int,
        num_classes: int,
        transform: object | None = None,
        offset: int = 0,
    ) -> None:
        self.size = size
        self.num_classes = num_classes
        self.transform = transform
        self.targets = [(index + offset) % num_classes for index in range(size)]
        self.classes = [str(index) for index in range(num_classes)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[int, int]:
        value: Any = index
        if self.transform is not None:
            value = self.transform(value)  # type: ignore[operator]
        return value, self.targets[index]


def get_dataset(
    dataset_id: str,
    split: str,
    transform: object | None = None,
    root: str | Path | None = None,
    registry_path: str | Path | None = None,
    registry: DatasetRegistry | None = None,
) -> object:
    """Return a local dataset object without downloading anything."""

    loaded_registry = registry or load_registry(registry_path)
    entry = loaded_registry.get(dataset_id)
    resolved_root = loaded_registry.dataset_root(entry, root_override=root)

    if entry.source == "synthetic":
        return _synthetic_dataset(entry, split, transform)

    if entry.loader_status == "inspect_only":
        raise UnsupportedDatasetError(
            f"{entry.dataset_id} is inspect_only; no stable read-only loader is configured"
        )
    if entry.source != "torchvision":
        raise UnsupportedDatasetError(
            f"{entry.dataset_id} source {entry.source!r} does not have a loader implementation"
        )
    if not dataset_available(loaded_registry, entry, root_override=resolved_root):
        markers = ", ".join(str(marker) for marker in entry.config.get("storage_markers", []))
        marker_text = f" required markers: {markers}" if markers else ""
        raise DatasetUnavailableError(f"{entry.dataset_id} is missing under {resolved_root}.{marker_text}")

    try:
        from torchvision import datasets
    except Exception as exc:
        raise RuntimeError("torchvision is required for torchvision dataset loaders") from exc

    class_name = entry.torchvision_class
    if not class_name:
        raise UnsupportedDatasetError(f"{entry.dataset_id} is missing torchvision_class")
    dataset_cls = getattr(datasets, class_name)
    loader = str(entry.config.get("loader", "split"))

    if loader == "train_bool":
        if split not in {"train", "test"}:
            raise ValueError(f"{entry.dataset_id} split {split!r} must be 'train' or 'test'")
        return dataset_cls(
            root=str(resolved_root),
            train=split == "train",
            download=False,
            transform=transform,
        )

    if loader == "split":
        if split not in entry.splits:
            raise ValueError(f"{entry.dataset_id} split {split!r} is not one of {entry.splits}")
        kwargs = dict(entry.config.get("loader_kwargs", {}))
        return dataset_cls(
            root=str(resolved_root),
            split=split,
            download=False,
            transform=transform,
            **kwargs,
        )

    if loader == "no_split":
        if split not in entry.splits:
            raise ValueError(f"{entry.dataset_id} split {split!r} is not one of {entry.splits}")
        return dataset_cls(root=str(resolved_root), download=False, transform=transform)

    raise UnsupportedDatasetError(f"{entry.dataset_id} has unsupported loader mode {loader!r}")


def _synthetic_dataset(entry: DatasetEntry, split: str, transform: object | None) -> SyntheticDataset:
    sizes = entry.config.get("synthetic_sizes", {})
    if split not in sizes:
        raise ValueError(f"{entry.dataset_id} split {split!r} is not configured")
    num_classes = entry.num_classes
    if num_classes is None:
        raise ValueError(f"{entry.dataset_id} synthetic entry requires num_classes")
    offset = int(entry.config.get("synthetic_offsets", {}).get(split, 0))
    return SyntheticDataset(size=int(sizes[split]), num_classes=num_classes, transform=transform, offset=offset)


def get_targets(dataset: object) -> list[int]:
    """Extract integer class targets from common dataset implementations."""

    for attr in ("targets", "labels", "_labels"):
        values = getattr(dataset, attr, None)
        if values is None:
            continue
        return _as_int_list(values)

    samples = getattr(dataset, "samples", None)
    if isinstance(samples, Sequence):
        return [int(item[1]) for item in samples]

    raise ValueError(f"Could not infer targets from dataset type {type(dataset).__name__}")


def infer_num_classes(dataset: object) -> int | None:
    for attr in ("classes", "categories"):
        values = getattr(dataset, attr, None)
        if isinstance(values, (list, tuple, dict)):
            return len(values)
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict):
        return len(class_to_idx)
    try:
        return len(set(get_targets(dataset)))
    except Exception:
        return None


def default_train_split(entry: DatasetEntry) -> str:
    if "train" in entry.splits:
        return "train"
    if "trainval" in entry.splits:
        return "trainval"
    return entry.splits[0]


def default_test_split(entry: DatasetEntry) -> str:
    if "test" in entry.splits:
        return "test"
    if "val" in entry.splits:
        return "val"
    return entry.splits[-1]


def _as_int_list(values: object) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()  # type: ignore[assignment]
    if isinstance(values, Iterable):
        return [int(value) for value in values]
    raise ValueError(f"Unsupported target container: {type(values).__name__}")
