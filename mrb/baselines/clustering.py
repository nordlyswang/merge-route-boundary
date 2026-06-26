"""Task clustering helpers for feature-level toy baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterAssignment:
    strategy: str
    K: int
    clusters: tuple[tuple[int, ...], ...]
    task_to_cluster: dict[int, int]
    warning: str = ""


def make_sequential_clusters(task_ids: Sequence[int], K: int) -> ClusterAssignment:
    ordered = sorted(int(task_id) for task_id in task_ids)
    _validate_k(ordered, K)
    split = np.array_split(np.asarray(ordered, dtype=np.int64), int(K))
    clusters = tuple(tuple(int(value) for value in part.tolist()) for part in split if part.size)
    return ClusterAssignment(
        strategy="sequential",
        K=int(K),
        clusters=clusters,
        task_to_cluster=task_to_cluster_map(clusters),
    )


def make_diagnostics_clusters(
    task_ids: Sequence[int],
    diagnostics_csv: str | Path | None,
    K: int,
    *,
    metric: str = "linear_probe_auc_symmetric",
) -> ClusterAssignment:
    ordered = sorted(int(task_id) for task_id in task_ids)
    _validate_k(ordered, K)
    if K == len(ordered):
        clusters = tuple((task_id,) for task_id in ordered)
        return ClusterAssignment(
            strategy="diagnostics_auc",
            K=int(K),
            clusters=clusters,
            task_to_cluster=task_to_cluster_map(clusters),
        )
    if diagnostics_csv is None or not Path(diagnostics_csv).exists():
        fallback = make_sequential_clusters(ordered, K)
        return ClusterAssignment(
            strategy="diagnostics_auc",
            K=int(K),
            clusters=fallback.clusters,
            task_to_cluster=fallback.task_to_cluster,
            warning="diagnostics CSV missing; fell back to sequential clusters",
        )

    df = pd.read_csv(diagnostics_csv)
    required = {"task_i", "task_j", metric}
    missing = sorted(required - set(df.columns))
    if missing:
        fallback = make_sequential_clusters(ordered, K)
        return ClusterAssignment(
            strategy="diagnostics_auc",
            K=int(K),
            clusters=fallback.clusters,
            task_to_cluster=fallback.task_to_cluster,
            warning=f"diagnostics CSV missing columns {missing}; fell back to sequential clusters",
        )

    distances: dict[tuple[int, int], float] = {}
    warnings: list[str] = []
    for row in df.to_dict(orient="records"):
        try:
            i = int(row["task_i"])
            j = int(row["task_j"])
            value = float(row[metric])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        key = tuple(sorted((i, j)))
        distances[key] = max(0.0, value - 0.5)

    clusters = [(task_id,) for task_id in ordered]
    while len(clusters) > int(K):
        best_pair: tuple[int, int] | None = None
        best_cost: float | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                cost = _cluster_distance(clusters[left], clusters[right], distances)
                if best_cost is None or cost < best_cost:
                    best_pair = (left, right)
                    best_cost = cost
        if best_pair is None:
            warnings.append("could not find diagnostics merge pair; stopped early")
            break
        left, right = best_pair
        merged = tuple(sorted(clusters[left] + clusters[right]))
        clusters = [
            cluster
            for position, cluster in enumerate(clusters)
            if position not in {left, right}
        ]
        clusters.append(merged)
        clusters = sorted(clusters, key=lambda item: item[0])

    final_clusters = tuple(tuple(int(value) for value in cluster) for cluster in clusters)
    return ClusterAssignment(
        strategy="diagnostics_auc",
        K=int(K),
        clusters=final_clusters,
        task_to_cluster=task_to_cluster_map(final_clusters),
        warning="; ".join(warnings),
    )


def task_to_cluster_map(clusters: Sequence[Sequence[int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for cluster_id, cluster in enumerate(clusters):
        if not cluster:
            raise ValueError("clusters must not contain empty groups")
        for task_id in cluster:
            task_int = int(task_id)
            if task_int in mapping:
                raise ValueError(f"task {task_int} appears in more than one cluster")
            mapping[task_int] = int(cluster_id)
    return mapping


def _validate_k(task_ids: Sequence[int], K: int) -> None:
    if not task_ids:
        raise ValueError("task_ids must be non-empty")
    if K <= 0:
        raise ValueError("K must be positive")
    if K > len(task_ids):
        raise ValueError(f"K={K} cannot exceed number of tasks={len(task_ids)}")


def _cluster_distance(
    left: Sequence[int],
    right: Sequence[int],
    distances: dict[tuple[int, int], float],
) -> float:
    pair_distances: list[float] = []
    for task_i in left:
        for task_j in right:
            pair_distances.append(float(distances.get(tuple(sorted((task_i, task_j))), 1.0)))
    return float(np.mean(pair_distances)) if pair_distances else 1.0
