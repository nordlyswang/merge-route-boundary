"""Frozen-feature separability diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mrb.diagnostics.distances import cosine_distance, intra_task_variance, l2_distance
from mrb.diagnostics.prototypes import compute_task_prototype


def centroid_separability(features_i: np.ndarray, features_j: np.ndarray) -> dict[str, float]:
    proto_i = compute_task_prototype(features_i, normalize=False)
    proto_j = compute_task_prototype(features_j, normalize=False)
    variance_i = intra_task_variance(features_i, proto_i)
    variance_j = intra_task_variance(features_j, proto_j)
    centroid_l2 = l2_distance(proto_i, proto_j)
    return {
        "centroid_l2": centroid_l2,
        "centroid_cosine_distance": cosine_distance(proto_i, proto_j),
        "separation_ratio": float(centroid_l2 / math.sqrt(variance_i + variance_j + 1e-12)),
    }


def linear_probe_separability(
    features_i: np.ndarray,
    features_j: np.ndarray,
    seed: int = 0,
    max_samples_per_task: int | None = None,
    test_size: float = 0.3,
) -> dict[str, Any]:
    left = _sample_rows(np.asarray(features_i, dtype=np.float32), max_samples_per_task, seed)
    right = _sample_rows(np.asarray(features_j, dtype=np.float32), max_samples_per_task, seed + 1)
    if left.shape[0] < 2 or right.shape[0] < 2:
        return _nan_probe_result(
            "need at least two samples per task",
            effective_i=left.shape[0],
            effective_j=right.shape[0],
        )

    x = np.vstack([left, right]).astype(np.float32, copy=False)
    y = np.asarray([0] * left.shape[0] + [1] * right.shape[0], dtype=np.int64)
    split = _stratified_train_eval_indices(y, test_size=test_size, seed=seed)
    if split is None:
        return _nan_probe_result(
            "could not create stratified train/eval split",
            effective_i=left.shape[0],
            effective_j=right.shape[0],
        )
    train_idx, eval_idx = split

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score

        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        classifier.fit(x[train_idx], y[train_idx])
        scores = classifier.predict_proba(x[eval_idx])[:, 1]
        predictions = classifier.predict(x[eval_idx])
        auc = float(roc_auc_score(y[eval_idx], scores))
        acc = float(accuracy_score(y[eval_idx], predictions))
        reason = ""
    except Exception as exc:
        auc, acc = _centroid_projection_probe(x, y, train_idx, eval_idx)
        reason = f"sklearn unavailable or failed; used centroid projection fallback: {exc}"

    return {
        "linear_probe_auc": auc,
        "linear_probe_auc_symmetric": max(auc, 1.0 - auc) if not math.isnan(auc) else float("nan"),
        "linear_probe_acc": acc,
        "linear_probe_num_train": int(train_idx.shape[0]),
        "linear_probe_num_eval": int(eval_idx.shape[0]),
        "effective_num_samples_i": int(left.shape[0]),
        "effective_num_samples_j": int(right.shape[0]),
        "reason": reason,
    }


def knn_separability(
    features_i: np.ndarray,
    features_j: np.ndarray,
    k: int = 5,
    seed: int = 0,
    max_samples_per_task: int | None = None,
) -> dict[str, Any]:
    left = _sample_rows(np.asarray(features_i, dtype=np.float32), max_samples_per_task, seed)
    right = _sample_rows(np.asarray(features_j, dtype=np.float32), max_samples_per_task, seed + 1)
    if left.shape[0] < 2 or right.shape[0] < 2:
        return {
            "knn_domain_acc": float("nan"),
            "knn_domain_auc": float("nan"),
            "reason": "need at least two samples per task",
        }

    x = np.vstack([left, right]).astype(np.float32, copy=False)
    y = np.asarray([0] * left.shape[0] + [1] * right.shape[0], dtype=np.int64)
    split = _stratified_train_eval_indices(y, test_size=0.3, seed=seed)
    if split is None:
        return {
            "knn_domain_acc": float("nan"),
            "knn_domain_auc": float("nan"),
            "reason": "could not create stratified train/eval split",
        }
    train_idx, eval_idx = split
    effective_k = max(1, min(int(k), int(train_idx.shape[0])))

    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.neighbors import KNeighborsClassifier

        classifier = KNeighborsClassifier(n_neighbors=effective_k)
        classifier.fit(x[train_idx], y[train_idx])
        scores = classifier.predict_proba(x[eval_idx])[:, 1]
        predictions = classifier.predict(x[eval_idx])
        auc = float(roc_auc_score(y[eval_idx], scores))
        acc = float(accuracy_score(y[eval_idx], predictions))
        reason = ""
    except Exception as exc:
        scores = _manual_knn_scores(x[train_idx], y[train_idx], x[eval_idx], effective_k)
        predictions = (scores >= 0.5).astype(np.int64)
        auc = _binary_auc(y[eval_idx], scores)
        acc = float(np.mean(predictions == y[eval_idx]))
        reason = f"sklearn unavailable or failed; used numpy kNN fallback: {exc}"

    return {
        "knn_domain_acc": acc,
        "knn_domain_auc": auc,
        "reason": reason,
    }


def _sample_rows(x: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or x.shape[0] <= max_samples:
        return x
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(x.shape[0], size=max_samples, replace=False))
    return x[rows]


def _stratified_train_eval_indices(
    y: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    train_rows: list[int] = []
    eval_rows: list[int] = []
    rng = np.random.default_rng(seed)
    for label in sorted(int(value) for value in np.unique(y)):
        rows = np.flatnonzero(y == label)
        if rows.shape[0] < 2:
            return None
        shuffled = rows.copy()
        rng.shuffle(shuffled)
        eval_count = int(round(rows.shape[0] * test_size))
        eval_count = min(max(1, eval_count), rows.shape[0] - 1)
        eval_rows.extend(int(value) for value in shuffled[:eval_count])
        train_rows.extend(int(value) for value in shuffled[eval_count:])
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return np.asarray(train_rows, dtype=np.int64), np.asarray(eval_rows, dtype=np.int64)


def _centroid_projection_probe(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[float, float]:
    train_x = x[train_idx]
    train_y = y[train_idx]
    proto_i = train_x[train_y == 0].mean(axis=0)
    proto_j = train_x[train_y == 1].mean(axis=0)
    direction = proto_j - proto_i
    train_scores = train_x @ direction
    threshold = float(0.5 * (train_scores[train_y == 0].mean() + train_scores[train_y == 1].mean()))
    eval_scores = x[eval_idx] @ direction
    predictions = (eval_scores >= threshold).astype(np.int64)
    return _binary_auc(y[eval_idx], eval_scores), float(np.mean(predictions == y[eval_idx]))


def _manual_knn_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    k: int,
) -> np.ndarray:
    scores = np.zeros(eval_x.shape[0], dtype=np.float32)
    for row, vector in enumerate(eval_x):
        distances = np.sum((train_x - vector) ** 2, axis=1)
        nearest = np.argpartition(distances, kth=k - 1)[:k]
        scores[row] = float(train_y[nearest].mean())
    return scores


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    if pos_count == 0 or neg_count == 0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count))


def _nan_probe_result(reason: str, *, effective_i: int, effective_j: int) -> dict[str, Any]:
    return {
        "linear_probe_auc": float("nan"),
        "linear_probe_auc_symmetric": float("nan"),
        "linear_probe_acc": float("nan"),
        "linear_probe_num_train": 0,
        "linear_probe_num_eval": 0,
        "effective_num_samples_i": int(effective_i),
        "effective_num_samples_j": int(effective_j),
        "reason": reason,
    }
