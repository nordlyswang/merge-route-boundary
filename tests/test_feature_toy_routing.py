from __future__ import annotations

import numpy as np

from mrb.baselines.clustering import make_sequential_clusters, task_to_cluster_map
from mrb.baselines.routers import (
    CentroidRouter,
    OracleRouter,
    PrototypeEnergyRouter,
    SklearnLinearRouter,
)


def test_centroid_router_high_accuracy_on_separable_tasks() -> None:
    rng = np.random.default_rng(0)
    task0 = rng.normal(loc=[2.0, 0.0], scale=0.05, size=(20, 2)).astype(np.float32)
    task1 = rng.normal(loc=[0.0, 2.0], scale=0.05, size=(20, 2)).astype(np.float32)
    features = np.vstack([task0, task1])
    task_ids = np.asarray([0] * 20 + [1] * 20, dtype=np.int64)

    router = CentroidRouter().fit(features, task_ids)
    predictions = router.predict(features)

    assert np.mean(predictions == task_ids) >= 0.95


def test_oracle_router_returns_true_task_ids() -> None:
    true_ids = np.asarray([3, 3, 7, 7], dtype=np.int64)
    router = OracleRouter()

    assert router.predict(np.zeros((4, 2), dtype=np.float32), true_route_ids=true_ids).tolist() == [
        3,
        3,
        7,
        7,
    ]


def test_linear_router_high_accuracy_on_linearly_separable_tasks() -> None:
    rng = np.random.default_rng(2)
    task0 = rng.normal(loc=[-2.0, 0.0], scale=0.05, size=(24, 2)).astype(np.float32)
    task1 = rng.normal(loc=[2.0, 0.0], scale=0.05, size=(24, 2)).astype(np.float32)
    features = np.vstack([task0, task1])
    task_ids = np.asarray([0] * 24 + [1] * 24, dtype=np.int64)

    router = SklearnLinearRouter(max_iter=200, random_state=0).fit(features, task_ids)
    predictions = router.predict(features)

    assert np.mean(predictions == task_ids) >= 0.95


def test_energy_router_runs_on_multimodal_tasks() -> None:
    features = np.asarray(
        [
            [2.0, 0.0],
            [2.1, 0.0],
            [-2.0, 0.0],
            [-2.1, 0.0],
            [0.0, 2.0],
            [0.0, 2.1],
            [0.0, -2.0],
            [0.0, -2.1],
        ],
        dtype=np.float32,
    )
    task_ids = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    labels = np.asarray([10, 10, 11, 11, 20, 20, 21, 21], dtype=np.int64)

    router = PrototypeEnergyRouter().fit(features, task_ids, labels)
    predictions = router.predict(features)

    assert predictions.tolist() == task_ids.tolist()


def test_cluster_map_assigns_each_task_once() -> None:
    assignment = make_sequential_clusters([0, 1, 2, 3, 4], K=2)
    mapping = task_to_cluster_map(assignment.clusters)

    assert sorted(mapping) == [0, 1, 2, 3, 4]
    assert len(set(mapping.values())) == 2
