"""Frozen feature extraction loops for local datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from mrb.features.storage import (
    build_metadata,
    sample_id_records,
    validate_feature_arrays,
    write_feature_bank,
)


class FeatureBackbone(Protocol):
    backbone_id: str
    model_id: str
    feature_dim: int
    image_size: int
    preprocess_id: str

    def encode(self, images: Sequence[object]) -> object:
        ...


@dataclass(frozen=True)
class FeatureExtractionResult:
    features: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    sample_ids: list[dict[str, Any]]


def _label_to_int(value: object) -> int:
    if hasattr(value, "item"):
        return int(value.item())  # type: ignore[union-attr]
    return int(value)


def _features_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(value)


def _dataset_item_parts(item: object) -> tuple[object, int]:
    if not isinstance(item, (tuple, list)) or len(item) < 2:
        raise ValueError(f"dataset item must be a tuple/list with image and label, got {type(item).__name__}")
    return item[0], _label_to_int(item[1])


def extract_feature_arrays(
    dataset: object,
    *,
    dataset_id: str,
    split: str,
    backbone: FeatureBackbone,
    batch_size: int,
    max_samples: int | None,
    dtype: str | np.dtype = "float16",
) -> FeatureExtractionResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset_len = len(dataset)  # type: ignore[arg-type]
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    limit = dataset_len if max_samples is None else min(int(max_samples), dataset_len)
    output_dtype = np.dtype(dtype)

    feature_batches: list[np.ndarray] = []
    labels: list[int] = []
    indices: list[int] = []

    for start in range(0, limit, batch_size):
        stop = min(start + batch_size, limit)
        batch_indices = list(range(start, stop))
        images: list[object] = []
        batch_labels: list[int] = []
        for index in batch_indices:
            image, label = _dataset_item_parts(dataset[index])  # type: ignore[index]
            images.append(image)
            batch_labels.append(label)

        encoded = _features_to_numpy(backbone.encode(images))
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        if encoded.ndim != 2:
            raise ValueError(f"backbone returned features with rank {encoded.ndim}, expected 2")
        if encoded.shape[0] != len(images):
            raise ValueError(
                f"backbone returned {encoded.shape[0]} rows for {len(images)} input images"
            )
        feature_batches.append(encoded.astype(output_dtype, copy=False))
        labels.extend(batch_labels)
        indices.extend(batch_indices)

    expected_dim = int(getattr(backbone, "feature_dim"))
    if feature_batches:
        features = np.concatenate(feature_batches, axis=0)
        if features.shape[1] != expected_dim:
            raise ValueError(
                f"backbone feature_dim={expected_dim} does not match extracted D={features.shape[1]}"
            )
    else:
        features = np.empty((0, expected_dim), dtype=output_dtype)

    labels_array = np.asarray(labels, dtype=np.int64)
    indices_array = np.asarray(indices, dtype=np.int64)
    validate_feature_arrays(features=features, labels=labels_array, indices=indices_array)
    return FeatureExtractionResult(
        features=features,
        labels=labels_array,
        indices=indices_array,
        sample_ids=sample_id_records(
            dataset_id=dataset_id,
            split=split,
            indices=indices,
            labels=labels,
        ),
    )


def extract_and_write_feature_bank(
    dataset: object,
    bank_dir: str | Path,
    *,
    dataset_id: str,
    split: str,
    backbone: FeatureBackbone,
    batch_size: int,
    max_samples: int | None,
    dtype: str | np.dtype = "float16",
    overwrite: bool = False,
) -> Path:
    result = extract_feature_arrays(
        dataset,
        dataset_id=dataset_id,
        split=split,
        backbone=backbone,
        batch_size=batch_size,
        max_samples=max_samples,
        dtype=dtype,
    )
    metadata = build_metadata(
        dataset_id=dataset_id,
        split=split,
        backbone_id=backbone.backbone_id,
        model_id=backbone.model_id,
        dtype=result.features.dtype,
        feature_dim=result.features.shape[1],
        num_samples=result.features.shape[0],
        image_size=backbone.image_size,
        preprocess_id=backbone.preprocess_id,
        extra={
            "source_num_samples": int(len(dataset)),  # type: ignore[arg-type]
            "max_samples": max_samples,
        },
    )
    return write_feature_bank(
        bank_dir,
        features=result.features,
        labels=result.labels,
        indices=result.indices,
        sample_ids=result.sample_ids,
        metadata=metadata,
        overwrite=overwrite,
    )
