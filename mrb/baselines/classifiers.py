"""Lightweight classifiers for frozen feature baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        return array / max(norm, eps)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


@dataclass
class PrototypeClassifier:
    """Nearest class-prototype classifier with cosine similarity."""

    normalize: bool = True
    expected_classes: Sequence[int] | None = None
    classes_: np.ndarray | None = field(default=None, init=False)
    prototypes_: np.ndarray | None = field(default=None, init=False)
    warnings_: list[str] = field(default_factory=list, init=False)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "PrototypeClassifier":
        feature_values, label_values = _validate_features_labels(features, labels)
        train_features = l2_normalize(feature_values) if self.normalize else feature_values
        observed = sorted(int(value) for value in np.unique(label_values))
        expected = (
            sorted(int(value) for value in self.expected_classes)
            if self.expected_classes is not None
            else observed
        )
        missing = [label for label in expected if label not in observed]
        if missing:
            message = f"missing samples for classes: {missing}"
            self.warnings_.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        prototypes: list[np.ndarray] = []
        classes: list[int] = []
        for label in expected:
            rows = train_features[label_values == label]
            if rows.shape[0] == 0:
                continue
            prototype = rows.mean(axis=0)
            prototypes.append(l2_normalize(prototype) if self.normalize else prototype.astype(np.float32))
            classes.append(label)
        if not prototypes:
            raise ValueError("PrototypeClassifier needs at least one class with samples")

        self.classes_ = np.asarray(classes, dtype=np.int64)
        self.prototypes_ = np.vstack(prototypes).astype(np.float32)
        return self

    def decision_scores(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.prototypes_ is None:
            raise ValueError("PrototypeClassifier is not fitted")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"features must be rank 2 [N, D], got shape {values.shape}")
        query = l2_normalize(values) if self.normalize else values
        return np.asarray(query @ self.prototypes_.T, dtype=np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("PrototypeClassifier is not fitted")
        scores = self.decision_scores(features)
        return self.classes_[np.argmax(scores, axis=1)]


@dataclass
class SklearnLinearClassifier:
    """Thin sklearn LogisticRegression wrapper for optional linear baselines."""

    max_iter: int = 1000
    class_weight: str | dict[int, float] | None = "balanced"
    solver: str = "lbfgs"
    max_train_samples: int | None = None
    random_state: int = 0
    model_: LogisticRegression | None = field(default=None, init=False)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "SklearnLinearClassifier":
        feature_values, label_values = _validate_features_labels(features, labels)
        if self.max_train_samples is not None and feature_values.shape[0] > self.max_train_samples:
            feature_values, label_values = _sample_stratified(
                feature_values,
                label_values,
                max_samples=int(self.max_train_samples),
                seed=self.random_state,
            )
        if np.unique(label_values).shape[0] < 2:
            raise ValueError("SklearnLinearClassifier needs at least two classes")
        self.model_ = LogisticRegression(
            max_iter=int(self.max_iter),
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
        )
        self.model_.fit(feature_values, label_values)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("SklearnLinearClassifier is not fitted")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"features must be rank 2 [N, D], got shape {values.shape}")
        return np.asarray(self.model_.predict(values), dtype=np.int64)


def _validate_features_labels(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_values = np.asarray(features, dtype=np.float32)
    label_values = np.asarray(labels, dtype=np.int64)
    if feature_values.ndim != 2:
        raise ValueError(f"features must be rank 2 [N, D], got shape {feature_values.shape}")
    if label_values.ndim != 1:
        raise ValueError(f"labels must be rank 1 [N], got shape {label_values.shape}")
    if feature_values.shape[0] != label_values.shape[0]:
        raise ValueError(
            f"features and labels must have matching N; got {feature_values.shape[0]} "
            f"and {label_values.shape[0]}"
        )
    if feature_values.shape[0] == 0:
        raise ValueError("features must be non-empty")
    if not np.isfinite(feature_values).all():
        raise ValueError("features contains NaN or Inf")
    return feature_values, label_values


def _sample_stratified(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    used: set[int] = set()
    unique_labels = sorted(int(value) for value in np.unique(labels))
    if len(unique_labels) <= max_samples:
        base = max_samples // max(1, len(unique_labels))
        remainder = max_samples % max(1, len(unique_labels))
        for label_position, label in enumerate(unique_labels):
            rows = np.flatnonzero(labels == label)
            shuffled = rows.copy()
            rng.shuffle(shuffled)
            take = min(rows.shape[0], base + (1 if label_position < remainder else 0))
            for row in shuffled[:take]:
                row_int = int(row)
                selected.append(row_int)
                used.add(row_int)
    if len(selected) < max_samples:
        remaining = np.asarray(
            [row for row in range(labels.shape[0]) if row not in used], dtype=np.int64
        )
        rng.shuffle(remaining)
        selected.extend(int(row) for row in remaining[: max_samples - len(selected)])
    row_array = np.asarray(sorted(selected[:max_samples]), dtype=np.int64)
    return features[row_array], labels[row_array]
