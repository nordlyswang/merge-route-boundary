"""Feature overlap diagnostics based on nearest prototypes."""

from __future__ import annotations

import numpy as np

from mrb.diagnostics.prototypes import l2_normalize


def nearest_task_centroid_confusion(
    features_i: np.ndarray,
    features_j: np.ndarray,
    proto_i: np.ndarray,
    proto_j: np.ndarray,
) -> dict[str, float]:
    norm_i = l2_normalize(np.asarray(features_i, dtype=np.float32))
    norm_j = l2_normalize(np.asarray(features_j, dtype=np.float32))
    p_i = l2_normalize(np.asarray(proto_i, dtype=np.float32))
    p_j = l2_normalize(np.asarray(proto_j, dtype=np.float32))
    i_to_i = np.linalg.norm(norm_i - p_i, axis=1)
    i_to_j = np.linalg.norm(norm_i - p_j, axis=1)
    j_to_j = np.linalg.norm(norm_j - p_j, axis=1)
    j_to_i = np.linalg.norm(norm_j - p_i, axis=1)
    i_confusion = float(np.mean(i_to_j < i_to_i)) if norm_i.shape[0] else float("nan")
    j_confusion = float(np.mean(j_to_i < j_to_j)) if norm_j.shape[0] else float("nan")
    total = norm_i.shape[0] + norm_j.shape[0]
    correct = float(np.sum(i_to_i <= i_to_j) + np.sum(j_to_j <= j_to_i))
    return {
        "nearest_task_centroid_acc": correct / total if total else float("nan"),
        "task_i_to_j_confusion_rate": i_confusion,
        "task_j_to_i_confusion_rate": j_confusion,
    }


def prototype_margin(
    features: np.ndarray,
    own_proto: np.ndarray,
    other_proto: np.ndarray,
) -> dict[str, float]:
    values = l2_normalize(np.asarray(features, dtype=np.float32))
    own = l2_normalize(np.asarray(own_proto, dtype=np.float32))
    other = l2_normalize(np.asarray(other_proto, dtype=np.float32))
    margins = np.linalg.norm(values - other, axis=1) - np.linalg.norm(values - own, axis=1)
    return {
        "mean_prototype_margin": float(np.mean(margins)) if margins.shape[0] else float("nan"),
        "min_prototype_margin": float(np.min(margins)) if margins.shape[0] else float("nan"),
    }


def nearest_class_prototype_confusion(
    features: np.ndarray,
    labels: np.ndarray,
    class_prototypes: dict[int, np.ndarray],
) -> dict[str, float]:
    if not class_prototypes:
        return {"nearest_class_prototype_acc": float("nan"), "class_confusion_rate": float("nan")}
    values = l2_normalize(np.asarray(features, dtype=np.float32))
    label_values = np.asarray(labels, dtype=np.int64)
    class_ids = sorted(class_prototypes)
    prototypes = np.vstack([l2_normalize(class_prototypes[class_id]) for class_id in class_ids])
    distances = np.sum((values[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
    nearest = np.asarray(
        [class_ids[index] for index in np.argmin(distances, axis=1)], dtype=np.int64
    )
    acc = float(np.mean(nearest == label_values)) if label_values.shape[0] else float("nan")
    return {
        "nearest_class_prototype_acc": acc,
        "class_confusion_rate": 1.0 - acc if not np.isnan(acc) else float("nan"),
    }
