"""Evaluation routines for feature-level merge-vs-route toy baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from mrb.baselines.classifiers import PrototypeClassifier, SklearnLinearClassifier
from mrb.baselines.clustering import (
    ClusterAssignment,
    make_diagnostics_clusters,
    make_sequential_clusters,
)
from mrb.baselines.feature_data import FeatureToyDataset, TaskFeatureSplit, stack_task_split
from mrb.baselines.routers import CentroidRouter, PrototypeEnergyRouter, SklearnLinearRouter


RESULT_COLUMNS = [
    "method",
    "classifier_type",
    "router_type",
    "cluster_strategy",
    "K",
    "uses_task_id",
    "uses_learned_router",
    "num_classifiers",
    "overall_acc",
    "macro_task_acc",
    "mean_task_acc",
    "last_task_acc",
    "router_acc",
    "route_vs_merge_gap",
    "oracle_gap",
    "label_mask_gain",
    "routing_error_cost",
    "classifier_specialization_gain",
    "num_test_samples",
    "status",
    "warning",
]

PER_TASK_COLUMNS = [
    "method",
    "classifier_type",
    "router_type",
    "cluster_strategy",
    "K",
    "task_id",
    "task_acc",
    "num_test_samples",
    "router_acc_task",
]

CONFUSION_COLUMNS = [
    "method",
    "classifier_type",
    "router_type",
    "cluster_strategy",
    "K",
    "true_route_id",
    "pred_route_id",
    "count",
]


class FeatureClassifier(Protocol):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "FeatureClassifier": ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class MethodEvaluation:
    result: dict[str, Any]
    per_task: list[dict[str, Any]]
    confusion: list[dict[str, Any]]


@dataclass(frozen=True)
class FeatureToyEvaluation:
    results: pd.DataFrame
    per_task: pd.DataFrame
    route_confusion: pd.DataFrame
    cluster_assignments: list[ClusterAssignment]


def evaluate_feature_toy_baselines(
    dataset: FeatureToyDataset,
    *,
    cluster_ks: Sequence[int],
    cluster_strategies: Sequence[str],
    diagnostics_csv: str | Path | None = None,
    classifier_types: Sequence[str] = ("prototype",),
    linear_config: dict[str, Any] | None = None,
    router_types: Sequence[str] | None = None,
    router_config: dict[str, dict[str, Any]] | None = None,
    controls_config: dict[str, Any] | None = None,
    centroid_router_enabled: bool | None = None,
) -> FeatureToyEvaluation:
    tasks = list(dataset.tasks)
    resolved_router_types = _resolve_router_types(router_types, centroid_router_enabled)
    controls = _resolve_controls_config(controls_config, resolved_router_types)
    evaluations: list[MethodEvaluation] = []
    assignments: list[ClusterAssignment] = []
    for classifier_type in classifier_types:
        evaluations.append(evaluate_merge_all(tasks, classifier_type, linear_config=linear_config))
        if classifier_type == "prototype" and bool(controls["oracle_task_mask"]["enabled"]):
            evaluations.append(evaluate_merge_all_oracle_task_mask(tasks))
        if classifier_type == "prototype" and bool(controls["learned_task_mask"]["enabled"]):
            for router_type in controls["learned_task_mask"]["routers"]:
                evaluations.append(
                    evaluate_merge_all_learned_task_mask(
                        tasks,
                        router_type=str(router_type),
                        router_config=router_config,
                    )
                )

        evaluations.append(
            evaluate_task_oracle_route(tasks, classifier_type, linear_config=linear_config)
        )
        for router_type in resolved_router_types:
            evaluations.append(
                evaluate_task_learned_route(
                    tasks,
                    classifier_type,
                    router_type=str(router_type),
                    linear_config=linear_config,
                    router_config=router_config,
                )
            )
        for strategy in cluster_strategies:
            for K in cluster_ks:
                assignment = _make_cluster_assignment(
                    tasks,
                    strategy=strategy,
                    K=int(K),
                    diagnostics_csv=diagnostics_csv,
                )
                assignments.append(assignment)
                evaluations.append(
                    evaluate_cluster_oracle(
                        tasks,
                        assignment,
                        classifier_type,
                        linear_config=linear_config,
                    )
                )
                if "centroid" in resolved_router_types:
                    evaluations.append(
                        evaluate_cluster_learned(
                            tasks,
                            assignment,
                            classifier_type,
                            linear_config=linear_config,
                        )
                    )

    results = pd.DataFrame([evaluation.result for evaluation in evaluations], columns=RESULT_COLUMNS)
    per_task = pd.DataFrame(
        [row for evaluation in evaluations for row in evaluation.per_task],
        columns=PER_TASK_COLUMNS,
    )
    route_confusion = pd.DataFrame(
        [row for evaluation in evaluations for row in evaluation.confusion],
        columns=CONFUSION_COLUMNS,
    )
    _fill_gap_columns(results)
    return FeatureToyEvaluation(
        results=results,
        per_task=per_task,
        route_confusion=route_confusion,
        cluster_assignments=assignments,
    )


def evaluate_merge_all(
    tasks: Sequence[TaskFeatureSplit],
    classifier_type: str = "prototype",
    *,
    linear_config: dict[str, Any] | None = None,
) -> MethodEvaluation:
    train_features, train_labels, _ = stack_task_split(tasks, "train")
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    classes = sorted(int(value) for value in np.unique(train_labels))
    classifier = _make_classifier(classifier_type, classes, linear_config)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    return _build_evaluation(
        method=f"merge_all_{classifier_type}",
        classifier_type=classifier_type,
        router_type="none",
        cluster_strategy="none",
        K=1,
        uses_task_id=False,
        uses_learned_router=False,
        num_classifiers=1,
        y_true=test_labels,
        y_pred=predictions,
        true_task_ids=test_task_ids,
        route_true=None,
        route_pred=None,
        warning=_classifier_warning(classifier),
    )


def evaluate_merge_all_oracle_task_mask(tasks: Sequence[TaskFeatureSplit]) -> MethodEvaluation:
    classifier = _fit_global_prototype(tasks)
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    predictions = predict_prototype_with_task_mask(
        classifier,
        test_features,
        route_ids=test_task_ids,
        task_classes=_task_classes(tasks),
    )
    return _build_evaluation(
        method="merge_all_prototype_oracle_task_mask",
        classifier_type="prototype",
        router_type="oracle",
        cluster_strategy="none",
        K=len(tasks),
        uses_task_id=True,
        uses_learned_router=False,
        num_classifiers=1,
        y_true=test_labels,
        y_pred=predictions,
        true_task_ids=test_task_ids,
        route_true=test_task_ids,
        route_pred=test_task_ids,
        warning=_classifier_warning(classifier),
    )


def evaluate_merge_all_learned_task_mask(
    tasks: Sequence[TaskFeatureSplit],
    *,
    router_type: str,
    router_config: dict[str, dict[str, Any]] | None = None,
) -> MethodEvaluation:
    classifier = _fit_global_prototype(tasks)
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    predicted_task_ids = _predict_task_routes(tasks, router_type, router_config)
    predictions = predict_prototype_with_task_mask(
        classifier,
        test_features,
        route_ids=predicted_task_ids,
        task_classes=_task_classes(tasks),
    )
    return _build_evaluation(
        method=f"merge_all_prototype_learned_task_mask_{router_type}",
        classifier_type="prototype",
        router_type=router_type,
        cluster_strategy="none",
        K=len(tasks),
        uses_task_id=False,
        uses_learned_router=True,
        num_classifiers=1,
        y_true=test_labels,
        y_pred=predictions,
        true_task_ids=test_task_ids,
        route_true=test_task_ids,
        route_pred=predicted_task_ids,
        warning=_classifier_warning(classifier),
    )


def evaluate_task_oracle_route(
    tasks: Sequence[TaskFeatureSplit],
    classifier_type: str = "prototype",
    *,
    linear_config: dict[str, Any] | None = None,
) -> MethodEvaluation:
    classifiers = _fit_task_classifiers(tasks, classifier_type, linear_config)
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    predictions = _predict_with_routes(test_features, test_task_ids, classifiers)
    return _build_evaluation(
        method=f"task_oracle_{classifier_type}",
        classifier_type=classifier_type,
        router_type="oracle",
        cluster_strategy="none",
        K=len(tasks),
        uses_task_id=True,
        uses_learned_router=False,
        num_classifiers=len(classifiers),
        y_true=test_labels,
        y_pred=predictions,
        true_task_ids=test_task_ids,
        route_true=test_task_ids,
        route_pred=test_task_ids,
        warning=_classifier_warnings(classifiers),
    )


def evaluate_task_learned_route(
    tasks: Sequence[TaskFeatureSplit],
    classifier_type: str = "prototype",
    *,
    router_type: str = "centroid",
    linear_config: dict[str, Any] | None = None,
    router_config: dict[str, dict[str, Any]] | None = None,
) -> MethodEvaluation:
    classifiers = _fit_task_classifiers(tasks, classifier_type, linear_config)
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    predicted_task_ids = _predict_task_routes(tasks, router_type, router_config)
    predictions = _predict_with_routes(test_features, predicted_task_ids, classifiers)
    return _build_evaluation(
        method=f"task_learned_{router_type}_router_{classifier_type}",
        classifier_type=classifier_type,
        router_type=router_type,
        cluster_strategy="none",
        K=len(tasks),
        uses_task_id=False,
        uses_learned_router=True,
        num_classifiers=len(classifiers),
        y_true=test_labels,
        y_pred=predictions,
        true_task_ids=test_task_ids,
        route_true=test_task_ids,
        route_pred=predicted_task_ids,
        warning=_classifier_warnings(classifiers),
    )


def evaluate_cluster_oracle(
    tasks: Sequence[TaskFeatureSplit],
    assignment: ClusterAssignment,
    classifier_type: str = "prototype",
    *,
    linear_config: dict[str, Any] | None = None,
) -> MethodEvaluation:
    classifiers = _fit_cluster_classifiers(tasks, assignment, classifier_type, linear_config)
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    true_cluster_ids = _map_task_ids(test_task_ids, assignment.task_to_cluster)
    predictions = _predict_with_routes(test_features, true_cluster_ids, classifiers)
    return _build_evaluation(
        method=f"cluster_oracle_{classifier_type}",
        classifier_type=classifier_type,
        router_type="oracle",
        cluster_strategy=assignment.strategy,
        K=assignment.K,
        uses_task_id=True,
        uses_learned_router=False,
        num_classifiers=len(classifiers),
        y_true=test_labels,
        y_pred=predictions,
        true_task_ids=test_task_ids,
        route_true=true_cluster_ids,
        route_pred=true_cluster_ids,
        warning=_join_warnings([assignment.warning, _classifier_warnings(classifiers)]),
    )


def evaluate_cluster_learned(
    tasks: Sequence[TaskFeatureSplit],
    assignment: ClusterAssignment,
    classifier_type: str = "prototype",
    *,
    linear_config: dict[str, Any] | None = None,
) -> MethodEvaluation:
    classifiers = _fit_cluster_classifiers(tasks, assignment, classifier_type, linear_config)
    train_features, _, train_task_ids = stack_task_split(tasks, "train")
    train_cluster_ids = _map_task_ids(train_task_ids, assignment.task_to_cluster)
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    true_cluster_ids = _map_task_ids(test_task_ids, assignment.task_to_cluster)
    router = CentroidRouter(expected_route_ids=sorted(classifiers))
    router.fit(train_features, train_cluster_ids)
    predicted_cluster_ids = router.predict(test_features)
    predictions = _predict_with_routes(test_features, predicted_cluster_ids, classifiers)
    return _build_evaluation(
        method=f"cluster_learned_{classifier_type}",
        classifier_type=classifier_type,
        router_type="centroid",
        cluster_strategy=assignment.strategy,
        K=assignment.K,
        uses_task_id=False,
        uses_learned_router=True,
        num_classifiers=len(classifiers),
        y_true=test_labels,
        y_pred=predictions,
        true_task_ids=test_task_ids,
        route_true=true_cluster_ids,
        route_pred=predicted_cluster_ids,
        warning=_join_warnings([assignment.warning, _classifier_warnings(classifiers)]),
    )


def predict_prototype_with_task_mask(
    classifier: PrototypeClassifier,
    features: np.ndarray,
    *,
    route_ids: np.ndarray,
    task_classes: dict[int, Sequence[int]],
) -> np.ndarray:
    if classifier.classes_ is None:
        raise ValueError("PrototypeClassifier is not fitted")
    route_values = np.asarray(route_ids, dtype=np.int64)
    scores = classifier.decision_scores(features)
    if scores.shape[0] != route_values.shape[0]:
        raise ValueError("features and route_ids must have matching N")
    predictions = np.full(route_values.shape[0], -1, dtype=np.int64)
    classifier_classes = np.asarray(classifier.classes_, dtype=np.int64)
    for route_id in sorted(int(value) for value in np.unique(route_values)):
        allowed = task_classes.get(route_id, ())
        allowed_mask = np.isin(classifier_classes, np.asarray(allowed, dtype=np.int64))
        if not np.any(allowed_mask):
            continue
        rows = route_values == route_id
        allowed_classes = classifier_classes[allowed_mask]
        route_scores = scores[rows][:, allowed_mask]
        predictions[rows] = allowed_classes[np.argmax(route_scores, axis=1)]
    return predictions


def _make_classifier(
    classifier_type: str,
    classes: Sequence[int],
    linear_config: dict[str, Any] | None,
) -> FeatureClassifier:
    if classifier_type == "prototype":
        return PrototypeClassifier(expected_classes=classes)
    if classifier_type == "linear":
        config = dict(linear_config or {})
        return SklearnLinearClassifier(
            max_iter=int(config.get("max_iter", 1000)),
            class_weight=config.get("class_weight", "balanced"),
            solver=str(config.get("solver", "lbfgs")),
            max_train_samples=config.get("max_train_per_task"),
            random_state=int(config.get("random_state", 0)),
        )
    raise ValueError(f"Unsupported classifier_type: {classifier_type}")


def _fit_global_prototype(tasks: Sequence[TaskFeatureSplit]) -> PrototypeClassifier:
    train_features, train_labels, _ = stack_task_split(tasks, "train")
    classes = sorted(int(value) for value in np.unique(train_labels))
    classifier = PrototypeClassifier(expected_classes=classes)
    classifier.fit(train_features, train_labels)
    return classifier


def _fit_task_classifiers(
    tasks: Sequence[TaskFeatureSplit],
    classifier_type: str,
    linear_config: dict[str, Any] | None,
) -> dict[int, FeatureClassifier]:
    classifiers: dict[int, FeatureClassifier] = {}
    for task in tasks:
        classes = task.classes or tuple(int(value) for value in np.unique(task.train_labels))
        classifier = _make_classifier(classifier_type, classes, linear_config)
        classifier.fit(task.train_features, task.train_labels)
        classifiers[task.task_id] = classifier
    return classifiers


def _fit_cluster_classifiers(
    tasks: Sequence[TaskFeatureSplit],
    assignment: ClusterAssignment,
    classifier_type: str,
    linear_config: dict[str, Any] | None,
) -> dict[int, FeatureClassifier]:
    tasks_by_id = {task.task_id: task for task in tasks}
    classifiers: dict[int, FeatureClassifier] = {}
    for cluster_id, cluster in enumerate(assignment.clusters):
        cluster_tasks = [tasks_by_id[int(task_id)] for task_id in cluster]
        features, labels, _ = stack_task_split(cluster_tasks, "train")
        classes = sorted(int(value) for value in np.unique(labels))
        classifier = _make_classifier(classifier_type, classes, linear_config)
        classifier.fit(features, labels)
        classifiers[int(cluster_id)] = classifier
    return classifiers


def _predict_with_routes(
    features: np.ndarray,
    route_ids: np.ndarray,
    classifiers: dict[int, FeatureClassifier],
) -> np.ndarray:
    predictions = np.full(route_ids.shape[0], -1, dtype=np.int64)
    for route_id in sorted(int(value) for value in np.unique(route_ids)):
        classifier = classifiers.get(route_id)
        if classifier is None:
            continue
        mask = route_ids == route_id
        predictions[mask] = classifier.predict(features[mask])
    return predictions


def _predict_task_routes(
    tasks: Sequence[TaskFeatureSplit],
    router_type: str,
    router_config: dict[str, dict[str, Any]] | None,
) -> np.ndarray:
    train_features, train_labels, train_task_ids = stack_task_split(tasks, "train")
    test_features, _, _ = stack_task_split(tasks, "test")
    router = _fit_task_router(
        router_type,
        train_features=train_features,
        train_labels=train_labels,
        train_task_ids=train_task_ids,
        task_ids=[task.task_id for task in tasks],
        router_config=router_config,
    )
    return router.predict(test_features)


def _fit_task_router(
    router_type: str,
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_task_ids: np.ndarray,
    task_ids: Sequence[int],
    router_config: dict[str, dict[str, Any]] | None,
):
    config = dict((router_config or {}).get(router_type, {}))
    if router_type == "centroid":
        router = CentroidRouter(expected_route_ids=task_ids)
        return router.fit(train_features, train_task_ids)
    if router_type == "linear":
        router = SklearnLinearRouter(
            max_iter=int(config.get("max_iter", 1000)),
            class_weight=config.get("class_weight", "balanced"),
            solver=str(config.get("solver", "lbfgs")),
            max_train_per_task=config.get("max_train_per_task"),
            random_state=int(config.get("random_state", 0)),
        )
        return router.fit(train_features, train_task_ids)
    if router_type == "energy":
        router = PrototypeEnergyRouter(
            score=str(config.get("score", "max")),
            top_k=int(config.get("top_k", 3)),
        )
        return router.fit(train_features, train_task_ids, train_labels)
    raise ValueError(f"Unsupported router_type: {router_type}")


def _build_evaluation(
    *,
    method: str,
    classifier_type: str,
    router_type: str,
    cluster_strategy: str,
    K: int,
    uses_task_id: bool,
    uses_learned_router: bool,
    num_classifiers: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_task_ids: np.ndarray,
    route_true: np.ndarray | None,
    route_pred: np.ndarray | None,
    warning: str = "",
) -> MethodEvaluation:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    true_task_ids = np.asarray(true_task_ids, dtype=np.int64)
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != true_task_ids.shape[0]:
        raise ValueError("y_true, y_pred, and true_task_ids must have matching N")
    per_task = _per_task_rows(
        method=method,
        classifier_type=classifier_type,
        router_type=router_type,
        cluster_strategy=cluster_strategy,
        K=K,
        y_true=y_true,
        y_pred=y_pred,
        true_task_ids=true_task_ids,
        route_true=route_true,
        route_pred=route_pred,
    )
    mean_task_acc = float(np.mean([row["task_acc"] for row in per_task]))
    last_task_id = max(int(row["task_id"]) for row in per_task)
    last_task_acc = next(
        float(row["task_acc"]) for row in per_task if int(row["task_id"]) == last_task_id
    )
    router_acc = math.nan
    if route_true is not None and route_pred is not None:
        router_acc = float(np.mean(np.asarray(route_true, dtype=np.int64) == route_pred))
    result = {
        "method": method,
        "classifier_type": classifier_type,
        "router_type": router_type,
        "cluster_strategy": cluster_strategy,
        "K": int(K),
        "uses_task_id": bool(uses_task_id),
        "uses_learned_router": bool(uses_learned_router),
        "num_classifiers": int(num_classifiers),
        "overall_acc": float(np.mean(y_true == y_pred)),
        "macro_task_acc": mean_task_acc,
        "mean_task_acc": mean_task_acc,
        "last_task_acc": last_task_acc,
        "router_acc": router_acc,
        "route_vs_merge_gap": math.nan,
        "oracle_gap": math.nan,
        "label_mask_gain": math.nan,
        "routing_error_cost": math.nan,
        "classifier_specialization_gain": math.nan,
        "num_test_samples": int(y_true.shape[0]),
        "status": "ok",
        "warning": warning,
    }
    return MethodEvaluation(
        result=result,
        per_task=per_task,
        confusion=_confusion_rows(
            method=method,
            classifier_type=classifier_type,
            router_type=router_type,
            cluster_strategy=cluster_strategy,
            K=K,
            route_true=route_true,
            route_pred=route_pred,
        ),
    )


def _per_task_rows(
    *,
    method: str,
    classifier_type: str,
    router_type: str,
    cluster_strategy: str,
    K: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_task_ids: np.ndarray,
    route_true: np.ndarray | None,
    route_pred: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in sorted(int(value) for value in np.unique(true_task_ids)):
        mask = true_task_ids == task_id
        router_acc_task = math.nan
        if route_true is not None and route_pred is not None:
            router_acc_task = float(np.mean(route_true[mask] == route_pred[mask]))
        rows.append(
            {
                "method": method,
                "classifier_type": classifier_type,
                "router_type": router_type,
                "cluster_strategy": cluster_strategy,
                "K": int(K),
                "task_id": int(task_id),
                "task_acc": float(np.mean(y_true[mask] == y_pred[mask])),
                "num_test_samples": int(mask.sum()),
                "router_acc_task": router_acc_task,
            }
        )
    return rows


def _confusion_rows(
    *,
    method: str,
    classifier_type: str,
    router_type: str,
    cluster_strategy: str,
    K: int,
    route_true: np.ndarray | None,
    route_pred: np.ndarray | None,
) -> list[dict[str, Any]]:
    if route_true is None or route_pred is None or router_type in {"none", "oracle"}:
        return []
    rows: list[dict[str, Any]] = []
    route_true = np.asarray(route_true, dtype=np.int64)
    route_pred = np.asarray(route_pred, dtype=np.int64)
    for true_route in sorted(int(value) for value in np.unique(route_true)):
        true_mask = route_true == true_route
        for pred_route in sorted(int(value) for value in np.unique(route_pred[true_mask])):
            rows.append(
                {
                    "method": method,
                    "classifier_type": classifier_type,
                    "router_type": router_type,
                    "cluster_strategy": cluster_strategy,
                    "K": int(K),
                    "true_route_id": int(true_route),
                    "pred_route_id": int(pred_route),
                    "count": int(np.sum(true_mask & (route_pred == pred_route))),
                }
            )
    return rows


def _fill_gap_columns(results: pd.DataFrame) -> None:
    for classifier_type in sorted(results["classifier_type"].dropna().unique()):
        classifier_mask = results["classifier_type"] == classifier_type
        merge_acc = _method_acc(results, f"merge_all_{classifier_type}")
        if merge_acc is None:
            continue
        oracle_acc = _method_acc(results, f"task_oracle_{classifier_type}")
        oracle_mask_acc = _method_acc(results, f"merge_all_{classifier_type}_oracle_task_mask")
        results.loc[classifier_mask, "route_vs_merge_gap"] = (
            results.loc[classifier_mask, "overall_acc"].astype(float) - merge_acc
        )
        if oracle_acc is not None:
            results.loc[classifier_mask, "oracle_gap"] = oracle_acc - merge_acc
        if oracle_mask_acc is not None:
            results.loc[classifier_mask, "label_mask_gain"] = oracle_mask_acc - merge_acc
            learned_mask = classifier_mask & results["method"].str.startswith(
                f"merge_all_{classifier_type}_learned_task_mask_"
            )
            results.loc[learned_mask, "routing_error_cost"] = (
                oracle_mask_acc - results.loc[learned_mask, "overall_acc"].astype(float)
            )
            if oracle_acc is not None:
                results.loc[classifier_mask, "classifier_specialization_gain"] = (
                    oracle_acc - oracle_mask_acc
                )


def _method_acc(results: pd.DataFrame, method: str) -> float | None:
    rows = results[results["method"] == method]
    if rows.empty:
        return None
    return float(rows.iloc[0]["overall_acc"])


def _make_cluster_assignment(
    tasks: Sequence[TaskFeatureSplit],
    *,
    strategy: str,
    K: int,
    diagnostics_csv: str | Path | None,
) -> ClusterAssignment:
    task_ids = [task.task_id for task in tasks]
    if strategy == "sequential":
        return make_sequential_clusters(task_ids, K)
    if strategy == "diagnostics_auc":
        return make_diagnostics_clusters(task_ids, diagnostics_csv, K)
    raise ValueError(f"Unsupported cluster strategy: {strategy}")


def _map_task_ids(task_ids: np.ndarray, task_to_cluster: dict[int, int]) -> np.ndarray:
    return np.asarray([task_to_cluster[int(task_id)] for task_id in task_ids], dtype=np.int64)


def _task_classes(tasks: Sequence[TaskFeatureSplit]) -> dict[int, tuple[int, ...]]:
    return {
        task.task_id: task.classes or tuple(int(value) for value in np.unique(task.train_labels))
        for task in tasks
    }


def _resolve_router_types(
    router_types: Sequence[str] | None,
    centroid_router_enabled: bool | None,
) -> list[str]:
    if router_types is not None:
        return [str(value) for value in router_types]
    return ["centroid"] if centroid_router_enabled is not False else []


def _resolve_controls_config(
    controls_config: dict[str, Any] | None,
    router_types: Sequence[str],
) -> dict[str, Any]:
    controls = dict(controls_config or {})
    oracle = dict(controls.get("oracle_task_mask", {}))
    learned = dict(controls.get("learned_task_mask", {}))
    oracle.setdefault("enabled", True)
    learned.setdefault("enabled", True)
    learned.setdefault("routers", list(router_types))
    return {"oracle_task_mask": oracle, "learned_task_mask": learned}


def _classifier_warning(classifier: FeatureClassifier) -> str:
    warnings_attr = getattr(classifier, "warnings_", [])
    return "; ".join(str(value) for value in warnings_attr if value)


def _classifier_warnings(classifiers: dict[int, FeatureClassifier]) -> str:
    return _join_warnings([_classifier_warning(classifier) for classifier in classifiers.values()])


def _join_warnings(warnings: Sequence[str]) -> str:
    seen: set[str] = set()
    unique: list[str] = []
    for warning in warnings:
        if warning and warning not in seen:
            seen.add(warning)
            unique.append(warning)
    return "; ".join(unique)
