"""Distance metrics for frozen-feature diagnostics."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mrb.diagnostics.prototypes import l2_normalize


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = l2_normalize(np.asarray(a, dtype=np.float32))
    b_norm = l2_normalize(np.asarray(b, dtype=np.float32))
    similarity = float(np.dot(a_norm, b_norm))
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def pairwise_cosine_matrix(prototypes: Sequence[np.ndarray]) -> np.ndarray:
    values = [l2_normalize(np.asarray(proto, dtype=np.float32)) for proto in prototypes]
    size = len(values)
    matrix = np.zeros((size, size), dtype=np.float32)
    for i, proto_i in enumerate(values):
        for j, proto_j in enumerate(values):
            matrix[i, j] = cosine_distance(proto_i, proto_j)
    return matrix


def intra_task_variance(features: np.ndarray, prototype: np.ndarray) -> float:
    feature_values = np.asarray(features, dtype=np.float32)
    if feature_values.shape[0] == 0:
        return float("nan")
    deltas = feature_values - np.asarray(prototype, dtype=np.float32)
    return float(np.mean(np.sum(deltas * deltas, axis=1)))


def inter_task_distance(proto_i: np.ndarray, proto_j: np.ndarray) -> dict[str, float]:
    return {
        "centroid_l2": l2_distance(proto_i, proto_j),
        "centroid_cosine_distance": cosine_distance(proto_i, proto_j),
    }
