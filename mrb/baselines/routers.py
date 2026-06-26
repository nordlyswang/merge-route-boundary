"""Routers for feature-level toy baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from mrb.baselines.classifiers import l2_normalize


@dataclass
class OracleRouter:
    """Router facade that returns provided true route ids."""

    def fit(self, features: np.ndarray, route_ids: np.ndarray) -> "OracleRouter":
        return self

    def predict(self, features: np.ndarray, true_route_ids: np.ndarray | None = None) -> np.ndarray:
        if true_route_ids is None:
            raise ValueError("OracleRouter.predict requires true_route_ids")
        return np.asarray(true_route_ids, dtype=np.int64)


@dataclass
class CentroidRouter:
    """Nearest centroid router using cosine similarity."""

    normalize: bool = True
    expected_route_ids: Sequence[int] | None = None
    route_ids_: np.ndarray | None = field(default=None, init=False)
    centroids_: np.ndarray | None = field(default=None, init=False)

    def fit(self, features: np.ndarray, route_ids: np.ndarray) -> "CentroidRouter":
        feature_values, route_values = _validate_features_routes(features, route_ids)
        train_features = l2_normalize(feature_values) if self.normalize else feature_values
        observed = sorted(int(value) for value in np.unique(route_values))
        expected = (
            sorted(int(value) for value in self.expected_route_ids)
            if self.expected_route_ids is not None
            else observed
        )
        centroids: list[np.ndarray] = []
        ids: list[int] = []
        for route_id in expected:
            rows = train_features[route_values == route_id]
            if rows.shape[0] == 0:
                continue
            centroid = rows.mean(axis=0)
            centroids.append(l2_normalize(centroid) if self.normalize else centroid.astype(np.float32))
            ids.append(route_id)
        if not centroids:
            raise ValueError("CentroidRouter needs at least one route with samples")
        self.route_ids_ = np.asarray(ids, dtype=np.int64)
        self.centroids_ = np.vstack(centroids).astype(np.float32)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.route_ids_ is None or self.centroids_ is None:
            raise ValueError("CentroidRouter is not fitted")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"features must be rank 2 [N, D], got shape {values.shape}")
        query = l2_normalize(values) if self.normalize else values
        scores = query @ self.centroids_.T
        return self.route_ids_[np.argmax(scores, axis=1)]


@dataclass
class SklearnLinearRouter:
    """Optional sklearn LogisticRegression router."""

    max_iter: int = 1000
    class_weight: str | dict[int, float] | None = "balanced"
    solver: str = "lbfgs"
    max_train_per_task: int | None = None
    random_state: int = 0
    model_: LogisticRegression | None = field(default=None, init=False)

    def fit(self, features: np.ndarray, route_ids: np.ndarray) -> "SklearnLinearRouter":
        feature_values, route_values = _validate_features_routes(features, route_ids)
        if self.max_train_per_task is not None:
            feature_values, route_values = _sample_per_route(
                feature_values,
                route_values,
                max_per_route=int(self.max_train_per_task),
                seed=self.random_state,
            )
        if np.unique(route_values).shape[0] < 2:
            raise ValueError("SklearnLinearRouter needs at least two route ids")
        self.model_ = LogisticRegression(
            max_iter=int(self.max_iter),
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
        )
        self.model_.fit(feature_values, route_values)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("SklearnLinearRouter is not fitted")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"features must be rank 2 [N, D], got shape {values.shape}")
        return np.asarray(self.model_.predict(values), dtype=np.int64)


@dataclass
class PrototypeEnergyRouter:
    """Route by the strongest class-prototype evidence within each task."""

    score: str = "max"
    top_k: int = 3
    normalize: bool = True
    route_ids_: np.ndarray | None = field(default=None, init=False)
    prototypes_by_route_: dict[int, np.ndarray] = field(default_factory=dict, init=False)

    def fit(
        self,
        features: np.ndarray,
        route_ids: np.ndarray,
        labels: np.ndarray,
    ) -> "PrototypeEnergyRouter":
        feature_values, route_values = _validate_features_routes(features, route_ids)
        label_values = np.asarray(labels, dtype=np.int64)
        if label_values.ndim != 1:
            raise ValueError(f"labels must be rank 1 [N], got shape {label_values.shape}")
        if label_values.shape[0] != feature_values.shape[0]:
            raise ValueError(
                f"features and labels must have matching N; got {feature_values.shape[0]} "
                f"and {label_values.shape[0]}"
            )
        train_features = l2_normalize(feature_values) if self.normalize else feature_values
        route_ids_out: list[int] = []
        prototypes_by_route: dict[int, np.ndarray] = {}
        for route_id in sorted(int(value) for value in np.unique(route_values)):
            route_mask = route_values == route_id
            route_labels = label_values[route_mask]
            route_features = train_features[route_mask]
            prototypes: list[np.ndarray] = []
            for label in sorted(int(value) for value in np.unique(route_labels)):
                rows = route_features[route_labels == label]
                if rows.shape[0] == 0:
                    continue
                prototype = rows.mean(axis=0)
                prototypes.append(
                    l2_normalize(prototype) if self.normalize else prototype.astype(np.float32)
                )
            if prototypes:
                route_ids_out.append(route_id)
                prototypes_by_route[route_id] = np.vstack(prototypes).astype(np.float32)
        if not prototypes_by_route:
            raise ValueError("PrototypeEnergyRouter needs at least one route with class prototypes")
        self.route_ids_ = np.asarray(route_ids_out, dtype=np.int64)
        self.prototypes_by_route_ = prototypes_by_route
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.route_ids_ is None or not self.prototypes_by_route_:
            raise ValueError("PrototypeEnergyRouter is not fitted")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"features must be rank 2 [N, D], got shape {values.shape}")
        query = l2_normalize(values) if self.normalize else values
        scores: list[np.ndarray] = []
        for route_id in self.route_ids_:
            similarities = query @ self.prototypes_by_route_[int(route_id)].T
            if self.score == "max":
                route_score = similarities.max(axis=1)
            elif self.score == "top_k_mean":
                k = max(1, min(int(self.top_k), similarities.shape[1]))
                route_score = np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
            else:
                raise ValueError(f"Unsupported energy router score: {self.score}")
            scores.append(np.asarray(route_score, dtype=np.float32))
        score_matrix = np.vstack(scores).T
        return self.route_ids_[np.argmax(score_matrix, axis=1)]


def _validate_features_routes(features: np.ndarray, route_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_values = np.asarray(features, dtype=np.float32)
    route_values = np.asarray(route_ids, dtype=np.int64)
    if feature_values.ndim != 2:
        raise ValueError(f"features must be rank 2 [N, D], got shape {feature_values.shape}")
    if route_values.ndim != 1:
        raise ValueError(f"route_ids must be rank 1 [N], got shape {route_values.shape}")
    if feature_values.shape[0] != route_values.shape[0]:
        raise ValueError(
            f"features and route_ids must have matching N; got {feature_values.shape[0]} "
            f"and {route_values.shape[0]}"
        )
    if feature_values.shape[0] == 0:
        raise ValueError("features must be non-empty")
    if not np.isfinite(feature_values).all():
        raise ValueError("features contains NaN or Inf")
    return feature_values, route_values


def _sample_per_route(
    features: np.ndarray,
    route_ids: np.ndarray,
    *,
    max_per_route: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_per_route <= 0:
        raise ValueError("max_per_route must be positive")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for route_id in sorted(int(value) for value in np.unique(route_ids)):
        rows = np.flatnonzero(route_ids == route_id)
        shuffled = rows.copy()
        rng.shuffle(shuffled)
        selected.extend(int(row) for row in shuffled[:max_per_route])
    row_array = np.asarray(sorted(selected), dtype=np.int64)
    return features[row_array], route_ids[row_array]
