"""Prototype helpers for frozen features."""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim == 1:
        norm = float(np.linalg.norm(values))
        return values / max(norm, eps)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def compute_class_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
) -> dict[int, np.ndarray]:
    feature_values = np.asarray(features, dtype=np.float32)
    label_values = np.asarray(labels, dtype=np.int64)
    prototypes: dict[int, np.ndarray] = {}
    for label in sorted(int(value) for value in np.unique(label_values)):
        proto = feature_values[label_values == label].mean(axis=0)
        prototypes[label] = (
            l2_normalize(proto) if normalize else np.asarray(proto, dtype=np.float32)
        )
    return prototypes


def compute_task_prototype(
    features: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    feature_values = np.asarray(features, dtype=np.float32)
    if feature_values.ndim != 2 or feature_values.shape[0] == 0:
        raise ValueError(
            f"features must be non-empty rank 2 [N, D], got shape {feature_values.shape}"
        )
    proto = feature_values.mean(axis=0)
    return l2_normalize(proto) if normalize else np.asarray(proto, dtype=np.float32)
