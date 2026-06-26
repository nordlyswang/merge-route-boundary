"""Feature-level baseline helpers."""

from __future__ import annotations

from mrb.baselines.classifiers import PrototypeClassifier, SklearnLinearClassifier
from mrb.baselines.clustering import (
    ClusterAssignment,
    make_diagnostics_clusters,
    make_sequential_clusters,
    task_to_cluster_map,
)
from mrb.baselines.routers import (
    CentroidRouter,
    OracleRouter,
    PrototypeEnergyRouter,
    SklearnLinearRouter,
)


__all__ = [
    "CentroidRouter",
    "ClusterAssignment",
    "OracleRouter",
    "PrototypeEnergyRouter",
    "PrototypeClassifier",
    "SklearnLinearClassifier",
    "SklearnLinearRouter",
    "make_diagnostics_clusters",
    "make_sequential_clusters",
    "task_to_cluster_map",
]
