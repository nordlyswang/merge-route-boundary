"""Soft, top-k, and fallback controls for frozen-feature routing baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mrb.baselines.classifiers import PrototypeClassifier
from mrb.baselines.evaluate import (
    MethodEvaluation,
    evaluate_merge_all,
    evaluate_merge_all_learned_task_mask,
    evaluate_merge_all_oracle_task_mask,
    evaluate_task_learned_route,
    evaluate_task_oracle_route,
)
from mrb.baselines.feature_data import FeatureToyDataset, TaskFeatureSplit, stack_task_split
from mrb.baselines.routers import PrototypeEnergyRouter, SklearnLinearRouter


SOFT_RESULT_COLUMNS = [
    "method",
    "control_type",
    "router_type",
    "topk",
    "tau",
    "beta",
    "selected_on",
    "val_acc",
    "overall_acc",
    "mean_task_acc",
    "last_task_acc",
    "router_acc_top1",
    "router_recall_topk",
    "fallback_rate",
    "mean_mask_size",
    "route_vs_merge_gap",
    "oracle_gap_closure",
    "num_test_samples",
    "status",
    "warning",
]

SOFT_PER_TASK_COLUMNS = [
    "method",
    "task_id",
    "task_acc",
    "num_test_samples",
    "router_acc_top1_task",
    "router_recall_topk_task",
    "fallback_rate_task",
    "mean_mask_size_task",
]

SOFT_CONFUSION_COLUMNS = [
    "method",
    "router_type",
    "true_task_id",
    "pred_task_id",
    "count",
]


@dataclass(frozen=True)
class SoftRoutingEvaluation:
    results: pd.DataFrame
    per_task: pd.DataFrame
    route_confusion: pd.DataFrame
    selected_hparams: dict[str, Any]


@dataclass(frozen=True)
class RouterBundle:
    router_type: str
    router: Any
    route_ids: np.ndarray


def evaluate_soft_routing_controls(
    dataset: FeatureToyDataset,
    *,
    router_types: Sequence[str] = ("linear", "energy"),
    router_config: dict[str, dict[str, Any]] | None = None,
    include_previous_controls: bool = True,
    topk_config: dict[str, Any] | None = None,
    fallback_config: dict[str, Any] | None = None,
    soft_prior_config: dict[str, Any] | None = None,
) -> SoftRoutingEvaluation:
    tasks = list(dataset.tasks)
    enabled_routers = [str(value) for value in router_types]
    results: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    selected_hparams: dict[str, Any] = {"fallback": {}, "soft_prior": {}}

    if include_previous_controls:
        for evaluation, control_type, topk, mean_mask_size in _previous_control_evaluations(
            tasks,
            router_types=enabled_routers,
            router_config=router_config,
        ):
            converted = _convert_previous_evaluation(
                evaluation,
                control_type=control_type,
                topk=topk,
                mean_mask_size=mean_mask_size,
            )
            results.append(converted[0])
            per_task.extend(converted[1])
            confusion.extend(converted[2])

    classifier = _fit_global_prototype(tasks)
    task_classes = _task_classes(tasks)
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    val_features, val_labels, val_task_ids = stack_task_split(tasks, "val")
    train_features, train_labels, train_task_ids = stack_task_split(tasks, "train")
    bundles = [
        _fit_router_bundle(
            router_type,
            train_features=train_features,
            train_labels=train_labels,
            train_task_ids=train_task_ids,
            router_config=router_config,
        )
        for router_type in enabled_routers
    ]

    if _config_enabled(topk_config, default=True):
        for bundle in bundles:
            if bundle.router_type not in _config_routers(topk_config, enabled_routers):
                continue
            for k in _config_ints(topk_config, "ks", default=(2, 3, 5)):
                val_predictions, _ = predict_prototype_with_topk_task_mask(
                    classifier,
                    val_features,
                    bundle.router.predict_topk(val_features, k),
                    task_classes=task_classes,
                )
                test_topk = bundle.router.predict_topk(test_features, k)
                predictions, mask_sizes = predict_prototype_with_topk_task_mask(
                    classifier,
                    test_features,
                    test_topk,
                    task_classes=task_classes,
                )
                router_top1 = test_topk[:, 0]
                rows = _rows_from_predictions(
                    method=f"merge_all_prototype_topk_task_mask_{bundle.router_type}_k{k}",
                    control_type="topk_task_mask",
                    router_type=bundle.router_type,
                    topk=k,
                    tau=math.nan,
                    beta=math.nan,
                    selected_on="none",
                    val_acc=_accuracy(val_labels, val_predictions),
                    y_true=test_labels,
                    y_pred=predictions,
                    true_task_ids=test_task_ids,
                    router_top1=router_top1,
                    router_topk=test_topk,
                    fallback_flags=None,
                    mask_sizes=mask_sizes,
                )
                results.append(rows[0])
                per_task.extend(rows[1])
                confusion.extend(_route_confusion_rows(rows[0]["method"], bundle.router_type, test_task_ids, router_top1))

    if _config_enabled(fallback_config, default=True):
        for bundle in bundles:
            if bundle.router_type not in _config_routers(fallback_config, enabled_routers):
                continue
            val_confidence = bundle.router.confidence(val_features)
            candidates = _fallback_tau_candidates(bundle.router_type, fallback_config, val_confidence)
            best_tau, best_val_acc = _select_fallback_tau(
                classifier,
                val_features,
                val_labels,
                bundle.router.predict_topk(val_features, 1),
                val_confidence,
                task_classes,
                candidates,
            )
            test_top1 = bundle.router.predict_topk(test_features, 1)
            test_confidence = bundle.router.confidence(test_features)
            predictions, fallback_flags, mask_sizes = predict_with_fallback_to_merge_all(
                classifier,
                test_features,
                test_top1,
                test_confidence,
                tau=best_tau,
                task_classes=task_classes,
            )
            router_top1 = test_top1[:, 0]
            rows = _rows_from_predictions(
                method=f"merge_all_prototype_fallback_{bundle.router_type}_val_tau",
                control_type="fallback_to_merge_all",
                router_type=bundle.router_type,
                topk=1,
                tau=best_tau,
                beta=math.nan,
                selected_on="val",
                val_acc=best_val_acc,
                y_true=test_labels,
                y_pred=predictions,
                true_task_ids=test_task_ids,
                router_top1=router_top1,
                router_topk=test_top1,
                fallback_flags=fallback_flags,
                mask_sizes=mask_sizes,
            )
            results.append(rows[0])
            per_task.extend(rows[1])
            confusion.extend(_route_confusion_rows(rows[0]["method"], bundle.router_type, test_task_ids, router_top1))
            selected_hparams["fallback"][bundle.router_type] = {
                "selected_tau": best_tau,
                "val_acc_at_selected_tau": best_val_acc,
                "fallback_rate_test": float(np.mean(fallback_flags)),
            }

    if _config_enabled(soft_prior_config, default=True):
        for bundle in bundles:
            if bundle.router_type not in _config_routers(soft_prior_config, enabled_routers):
                continue
            beta_grid = _config_floats(
                soft_prior_config,
                "beta_grid",
                default=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
            )
            val_scores = bundle.router.decision_scores(val_features)
            best_beta, best_val_acc = _select_soft_prior_beta(
                classifier,
                val_features,
                val_labels,
                val_scores,
                bundle.route_ids,
                tasks,
                beta_grid,
            )
            test_scores = bundle.router.decision_scores(test_features)
            predictions = predict_with_soft_task_prior(
                classifier,
                test_features,
                test_scores,
                route_ids=bundle.route_ids,
                tasks=tasks,
                beta=best_beta,
            )
            router_top1 = bundle.router.predict_topk(test_features, 1)[:, 0]
            mask_sizes = np.full(test_labels.shape[0], classifier.classes_.shape[0], dtype=np.float32)
            rows = _rows_from_predictions(
                method=f"merge_all_prototype_soft_prior_{bundle.router_type}_beta",
                control_type="soft_prior",
                router_type=bundle.router_type,
                topk=math.nan,
                tau=math.nan,
                beta=best_beta,
                selected_on="val",
                val_acc=best_val_acc,
                y_true=test_labels,
                y_pred=predictions,
                true_task_ids=test_task_ids,
                router_top1=router_top1,
                router_topk=router_top1.reshape(-1, 1),
                fallback_flags=None,
                mask_sizes=mask_sizes,
            )
            results.append(rows[0])
            per_task.extend(rows[1])
            confusion.extend(_route_confusion_rows(rows[0]["method"], bundle.router_type, test_task_ids, router_top1))
            selected_hparams["soft_prior"][bundle.router_type] = {
                "selected_beta": best_beta,
                "val_acc_at_selected_beta": best_val_acc,
            }

    results_df = pd.DataFrame(results, columns=SOFT_RESULT_COLUMNS)
    per_task_df = pd.DataFrame(per_task, columns=SOFT_PER_TASK_COLUMNS)
    confusion_df = pd.DataFrame(confusion, columns=SOFT_CONFUSION_COLUMNS)
    _fill_gap_columns(results_df)
    return SoftRoutingEvaluation(
        results=results_df,
        per_task=per_task_df,
        route_confusion=confusion_df,
        selected_hparams=selected_hparams,
    )


def predict_prototype_with_topk_task_mask(
    classifier: PrototypeClassifier,
    features: np.ndarray,
    topk_route_ids: np.ndarray,
    *,
    task_classes: dict[int, Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    if classifier.classes_ is None:
        raise ValueError("PrototypeClassifier is not fitted")
    route_values = np.asarray(topk_route_ids, dtype=np.int64)
    if route_values.ndim == 1:
        route_values = route_values.reshape(-1, 1)
    scores = classifier.decision_scores(features)
    if scores.shape[0] != route_values.shape[0]:
        raise ValueError("features and topk_route_ids must have matching N")
    classifier_classes = np.asarray(classifier.classes_, dtype=np.int64)
    unrestricted = classifier_classes[np.argmax(scores, axis=1)]
    predictions = unrestricted.copy()
    mask_sizes = np.full(route_values.shape[0], classifier_classes.shape[0], dtype=np.float32)
    for row_idx in range(route_values.shape[0]):
        allowed_classes = _allowed_classes_for_routes(route_values[row_idx], task_classes)
        allowed_mask = np.isin(classifier_classes, allowed_classes)
        if not np.any(allowed_mask):
            continue
        row_scores = scores[row_idx, allowed_mask]
        allowed = classifier_classes[allowed_mask]
        predictions[row_idx] = allowed[int(np.argmax(row_scores))]
        mask_sizes[row_idx] = float(allowed.shape[0])
    return predictions.astype(np.int64), mask_sizes


def compute_topk_task_recall(topk_route_ids: np.ndarray, true_task_ids: np.ndarray) -> float:
    route_values = np.asarray(topk_route_ids, dtype=np.int64)
    if route_values.ndim == 1:
        route_values = route_values.reshape(-1, 1)
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    if route_values.shape[0] != true_values.shape[0]:
        raise ValueError("topk_route_ids and true_task_ids must have matching N")
    return float(np.mean(np.any(route_values == true_values[:, None], axis=1)))


def predict_with_fallback_to_merge_all(
    classifier: PrototypeClassifier,
    features: np.ndarray,
    top1_route_ids: np.ndarray,
    confidence: np.ndarray,
    *,
    tau: float,
    task_classes: dict[int, Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if classifier.classes_ is None:
        raise ValueError("PrototypeClassifier is not fitted")
    confidence_values = np.asarray(confidence, dtype=np.float32)
    if confidence_values.ndim != 1:
        raise ValueError("confidence must be rank 1 [N]")
    merge_predictions = classifier.predict(features)
    routed_predictions, routed_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        features,
        top1_route_ids,
        task_classes=task_classes,
    )
    if confidence_values.shape[0] != merge_predictions.shape[0]:
        raise ValueError("confidence and features must have matching N")
    route_mask = confidence_values >= float(tau)
    predictions = merge_predictions.copy()
    predictions[route_mask] = routed_predictions[route_mask]
    mask_sizes = np.full(predictions.shape[0], classifier.classes_.shape[0], dtype=np.float32)
    mask_sizes[route_mask] = routed_mask_sizes[route_mask]
    fallback_flags = ~route_mask
    return predictions.astype(np.int64), fallback_flags, mask_sizes


def predict_with_soft_task_prior(
    classifier: PrototypeClassifier,
    features: np.ndarray,
    router_scores: np.ndarray,
    *,
    route_ids: np.ndarray,
    tasks: Sequence[TaskFeatureSplit],
    beta: float,
) -> np.ndarray:
    if classifier.classes_ is None:
        raise ValueError("PrototypeClassifier is not fitted")
    prototype_scores = classifier.decision_scores(features)
    score_values = np.asarray(router_scores, dtype=np.float32)
    if prototype_scores.shape[0] != score_values.shape[0]:
        raise ValueError("features and router_scores must have matching N")
    class_task_ids = _class_task_ids(classifier.classes_, tasks)
    route_to_col = {int(route_id): col for col, route_id in enumerate(route_ids)}
    log_prior = _router_log_prior(score_values)
    prior_by_class = np.zeros_like(prototype_scores, dtype=np.float32)
    for class_col, task_id in enumerate(class_task_ids):
        route_col = route_to_col.get(int(task_id))
        if route_col is not None:
            prior_by_class[:, class_col] = log_prior[:, route_col]
    adjusted = prototype_scores + float(beta) * prior_by_class
    return np.asarray(classifier.classes_, dtype=np.int64)[np.argmax(adjusted, axis=1)]


def _previous_control_evaluations(
    tasks: Sequence[TaskFeatureSplit],
    *,
    router_types: Sequence[str],
    router_config: dict[str, dict[str, Any]] | None,
) -> list[tuple[MethodEvaluation, str, float, float]]:
    evaluations: list[tuple[MethodEvaluation, str, float, float]] = [
        (evaluate_merge_all(tasks, "prototype"), "merge_all", math.nan, float(_num_classes(tasks))),
        (
            evaluate_merge_all_oracle_task_mask(tasks),
            "oracle_task_mask",
            1.0,
            float(_mean_task_class_count(tasks)),
        ),
    ]
    for router_type in router_types:
        evaluations.append(
            (
                evaluate_merge_all_learned_task_mask(
                    tasks,
                    router_type=str(router_type),
                    router_config=router_config,
                ),
                "hard_task_mask",
                1.0,
                float(_mean_task_class_count(tasks)),
            )
        )
    evaluations.append(
        (
            evaluate_task_oracle_route(tasks, "prototype"),
            "oracle_route",
            1.0,
            float(_mean_task_class_count(tasks)),
        )
    )
    for router_type in router_types:
        evaluations.append(
            (
                evaluate_task_learned_route(
                    tasks,
                    "prototype",
                    router_type=str(router_type),
                    router_config=router_config,
                ),
                "hard_route",
                1.0,
                float(_mean_task_class_count(tasks)),
            )
        )
    return evaluations


def _convert_previous_evaluation(
    evaluation: MethodEvaluation,
    *,
    control_type: str,
    topk: float,
    mean_mask_size: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result = evaluation.result
    router_acc = _float_or_nan(result.get("router_acc"))
    row = {
        "method": result["method"],
        "control_type": control_type,
        "router_type": result["router_type"],
        "topk": topk,
        "tau": math.nan,
        "beta": math.nan,
        "selected_on": "none",
        "val_acc": math.nan,
        "overall_acc": float(result["overall_acc"]),
        "mean_task_acc": float(result["mean_task_acc"]),
        "last_task_acc": float(result["last_task_acc"]),
        "router_acc_top1": router_acc,
        "router_recall_topk": router_acc,
        "fallback_rate": 0.0,
        "mean_mask_size": mean_mask_size,
        "route_vs_merge_gap": math.nan,
        "oracle_gap_closure": math.nan,
        "num_test_samples": int(result["num_test_samples"]),
        "status": result["status"],
        "warning": result.get("warning", ""),
    }
    if control_type in {"oracle_task_mask", "oracle_route"}:
        row["router_acc_top1"] = 1.0
        row["router_recall_topk"] = 1.0
    if control_type == "merge_all":
        row["router_acc_top1"] = math.nan
        row["router_recall_topk"] = math.nan

    per_task = []
    for task_row in evaluation.per_task:
        task_router_acc = _float_or_nan(task_row.get("router_acc_task"))
        if control_type in {"oracle_task_mask", "oracle_route"}:
            task_router_acc = 1.0
        if control_type == "merge_all":
            task_router_acc = math.nan
        per_task.append(
            {
                "method": task_row["method"],
                "task_id": int(task_row["task_id"]),
                "task_acc": float(task_row["task_acc"]),
                "num_test_samples": int(task_row["num_test_samples"]),
                "router_acc_top1_task": task_router_acc,
                "router_recall_topk_task": task_router_acc,
                "fallback_rate_task": 0.0,
                "mean_mask_size_task": mean_mask_size,
            }
        )
    confusion = [
        {
            "method": row["method"],
            "router_type": result["router_type"],
            "true_task_id": int(confusion_row["true_route_id"]),
            "pred_task_id": int(confusion_row["pred_route_id"]),
            "count": int(confusion_row["count"]),
        }
        for confusion_row in evaluation.confusion
    ]
    return row, per_task, confusion


def _rows_from_predictions(
    *,
    method: str,
    control_type: str,
    router_type: str,
    topk: float,
    tau: float,
    beta: float,
    selected_on: str,
    val_acc: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_task_ids: np.ndarray,
    router_top1: np.ndarray | None,
    router_topk: np.ndarray | None,
    fallback_flags: np.ndarray | None,
    mask_sizes: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_true_values = np.asarray(y_true, dtype=np.int64)
    y_pred_values = np.asarray(y_pred, dtype=np.int64)
    task_values = np.asarray(true_task_ids, dtype=np.int64)
    mask_values = np.asarray(mask_sizes, dtype=np.float32)
    if y_true_values.shape[0] != y_pred_values.shape[0] or y_true_values.shape[0] != task_values.shape[0]:
        raise ValueError("y_true, y_pred, and true_task_ids must have matching N")
    per_task = _per_task_rows(
        method=method,
        y_true=y_true_values,
        y_pred=y_pred_values,
        true_task_ids=task_values,
        router_top1=router_top1,
        router_topk=router_topk,
        fallback_flags=fallback_flags,
        mask_sizes=mask_values,
    )
    last_task_id = max(int(row["task_id"]) for row in per_task)
    last_task_acc = next(
        float(row["task_acc"]) for row in per_task if int(row["task_id"]) == last_task_id
    )
    fallback_rate = 0.0 if fallback_flags is None else float(np.mean(fallback_flags))
    row = {
        "method": method,
        "control_type": control_type,
        "router_type": router_type,
        "topk": topk,
        "tau": tau,
        "beta": beta,
        "selected_on": selected_on,
        "val_acc": val_acc,
        "overall_acc": _accuracy(y_true_values, y_pred_values),
        "mean_task_acc": float(np.mean([row["task_acc"] for row in per_task])),
        "last_task_acc": last_task_acc,
        "router_acc_top1": _router_top1_acc(router_top1, task_values),
        "router_recall_topk": _router_topk_recall(router_topk, task_values),
        "fallback_rate": fallback_rate,
        "mean_mask_size": float(np.mean(mask_values)),
        "route_vs_merge_gap": math.nan,
        "oracle_gap_closure": math.nan,
        "num_test_samples": int(y_true_values.shape[0]),
        "status": "ok",
        "warning": "",
    }
    return row, per_task


def _per_task_rows(
    *,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_task_ids: np.ndarray,
    router_top1: np.ndarray | None,
    router_topk: np.ndarray | None,
    fallback_flags: np.ndarray | None,
    mask_sizes: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in sorted(int(value) for value in np.unique(true_task_ids)):
        mask = true_task_ids == task_id
        fallback_rate = 0.0 if fallback_flags is None else float(np.mean(fallback_flags[mask]))
        rows.append(
            {
                "method": method,
                "task_id": int(task_id),
                "task_acc": _accuracy(y_true[mask], y_pred[mask]),
                "num_test_samples": int(mask.sum()),
                "router_acc_top1_task": _router_top1_acc(
                    None if router_top1 is None else router_top1[mask],
                    true_task_ids[mask],
                ),
                "router_recall_topk_task": _router_topk_recall(
                    None if router_topk is None else router_topk[mask],
                    true_task_ids[mask],
                ),
                "fallback_rate_task": fallback_rate,
                "mean_mask_size_task": float(np.mean(mask_sizes[mask])),
            }
        )
    return rows


def _route_confusion_rows(
    method: str,
    router_type: str,
    true_task_ids: np.ndarray,
    predicted_task_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    pred_values = np.asarray(predicted_task_ids, dtype=np.int64)
    for true_task_id in sorted(int(value) for value in np.unique(true_values)):
        true_mask = true_values == true_task_id
        for pred_task_id in sorted(int(value) for value in np.unique(pred_values[true_mask])):
            rows.append(
                {
                    "method": method,
                    "router_type": router_type,
                    "true_task_id": int(true_task_id),
                    "pred_task_id": int(pred_task_id),
                    "count": int(np.sum(true_mask & (pred_values == pred_task_id))),
                }
            )
    return rows


def _fit_global_prototype(tasks: Sequence[TaskFeatureSplit]) -> PrototypeClassifier:
    train_features, train_labels, _ = stack_task_split(tasks, "train")
    classes = sorted(int(value) for value in np.unique(train_labels))
    classifier = PrototypeClassifier(expected_classes=classes)
    return classifier.fit(train_features, train_labels)


def _fit_router_bundle(
    router_type: str,
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_task_ids: np.ndarray,
    router_config: dict[str, dict[str, Any]] | None,
) -> RouterBundle:
    config = dict((router_config or {}).get(router_type, {}))
    if router_type == "linear":
        router = SklearnLinearRouter(
            max_iter=int(config.get("max_iter", 1000)),
            class_weight=config.get("class_weight", "balanced"),
            solver=str(config.get("solver", "lbfgs")),
            max_train_per_task=config.get("max_train_per_task"),
            random_state=int(config.get("random_state", 0)),
        ).fit(train_features, train_task_ids)
        return RouterBundle(
            router_type=router_type,
            router=router,
            route_ids=np.asarray(router.model_.classes_, dtype=np.int64),
        )
    if router_type == "energy":
        router = PrototypeEnergyRouter(
            score=str(config.get("score", "max")),
            top_k=int(config.get("top_k", 3)),
        ).fit(train_features, train_task_ids, train_labels)
        return RouterBundle(
            router_type=router_type,
            router=router,
            route_ids=np.asarray(router.route_ids_, dtype=np.int64),
        )
    raise ValueError(f"Unsupported soft routing router_type: {router_type}")


def _select_fallback_tau(
    classifier: PrototypeClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    top1_route_ids: np.ndarray,
    confidence: np.ndarray,
    task_classes: dict[int, Sequence[int]],
    candidates: Sequence[float],
) -> tuple[float, float]:
    best_tau = float(candidates[0])
    best_acc = -1.0
    best_fallback_rate = math.inf
    for tau in candidates:
        predictions, fallback_flags, _ = predict_with_fallback_to_merge_all(
            classifier,
            features,
            top1_route_ids,
            confidence,
            tau=float(tau),
            task_classes=task_classes,
        )
        acc = _accuracy(labels, predictions)
        fallback_rate = float(np.mean(fallback_flags))
        if acc > best_acc + 1e-12 or (
            abs(acc - best_acc) <= 1e-12 and fallback_rate < best_fallback_rate
        ):
            best_tau = float(tau)
            best_acc = acc
            best_fallback_rate = fallback_rate
    return best_tau, best_acc


def _select_soft_prior_beta(
    classifier: PrototypeClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    router_scores: np.ndarray,
    route_ids: np.ndarray,
    tasks: Sequence[TaskFeatureSplit],
    beta_grid: Sequence[float],
) -> tuple[float, float]:
    best_beta = float(beta_grid[0])
    best_acc = -1.0
    for beta in beta_grid:
        predictions = predict_with_soft_task_prior(
            classifier,
            features,
            router_scores,
            route_ids=route_ids,
            tasks=tasks,
            beta=float(beta),
        )
        acc = _accuracy(labels, predictions)
        if acc > best_acc + 1e-12:
            best_beta = float(beta)
            best_acc = acc
    return best_beta, best_acc


def _fallback_tau_candidates(
    router_type: str,
    config: dict[str, Any] | None,
    confidence: np.ndarray,
) -> list[float]:
    values = dict(config or {})
    if router_type == "linear":
        candidates = values.get(
            "linear_tau_grid",
            values.get("tau_grid", [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]),
        )
        return _unique_floats(candidates)
    quantiles = values.get("energy_tau_quantiles", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
    confidence_values = np.asarray(confidence, dtype=np.float32)
    candidates = [float(np.quantile(confidence_values, float(q))) for q in quantiles]
    return _unique_floats(candidates)


def _fill_gap_columns(results: pd.DataFrame) -> None:
    merge_acc = _method_acc(results, "merge_all_prototype")
    oracle_mask_acc = _method_acc(results, "merge_all_prototype_oracle_task_mask")
    if merge_acc is None:
        return
    results.loc[:, "route_vs_merge_gap"] = results["overall_acc"].astype(float) - merge_acc
    if oracle_mask_acc is None or abs(oracle_mask_acc - merge_acc) <= 1e-12:
        return
    denominator = oracle_mask_acc - merge_acc
    results.loc[:, "oracle_gap_closure"] = (
        results["overall_acc"].astype(float) - merge_acc
    ) / denominator


def _method_acc(results: pd.DataFrame, method: str) -> float | None:
    rows = results[results["method"] == method]
    if rows.empty:
        return None
    return float(rows.iloc[0]["overall_acc"])


def _task_classes(tasks: Sequence[TaskFeatureSplit]) -> dict[int, tuple[int, ...]]:
    return {
        task.task_id: task.classes or tuple(int(value) for value in np.unique(task.train_labels))
        for task in tasks
    }


def _allowed_classes_for_routes(
    route_ids: np.ndarray,
    task_classes: dict[int, Sequence[int]],
) -> np.ndarray:
    allowed: set[int] = set()
    for route_id in np.asarray(route_ids, dtype=np.int64).tolist():
        allowed.update(int(value) for value in task_classes.get(int(route_id), ()))
    return np.asarray(sorted(allowed), dtype=np.int64)


def _class_task_ids(classes: np.ndarray, tasks: Sequence[TaskFeatureSplit]) -> np.ndarray:
    class_to_task: dict[int, int] = {}
    for task in tasks:
        for class_id in task.classes or tuple(int(value) for value in np.unique(task.train_labels)):
            class_to_task[int(class_id)] = int(task.task_id)
    return np.asarray([class_to_task.get(int(class_id), -1) for class_id in classes], dtype=np.int64)


def _router_log_prior(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    score_values = np.asarray(scores, dtype=np.float32)
    row_sums = np.sum(score_values, axis=1)
    if np.all(score_values >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-4):
        return np.asarray(np.log(np.clip(score_values, eps, 1.0)), dtype=np.float32)
    shifted = score_values - np.max(score_values, axis=1, keepdims=True)
    log_norm = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return np.asarray(shifted - log_norm, dtype=np.float32)


def _config_enabled(config: dict[str, Any] | None, *, default: bool) -> bool:
    if config is None:
        return default
    return bool(dict(config).get("enabled", default))


def _config_routers(config: dict[str, Any] | None, default: Sequence[str]) -> list[str]:
    if config is None or "routers" not in config:
        return [str(value) for value in default]
    return [str(value) for value in config.get("routers", [])]


def _config_ints(
    config: dict[str, Any] | None,
    key: str,
    *,
    default: Sequence[int],
) -> list[int]:
    values = default if config is None else config.get(key, default)
    return [int(value) for value in values]


def _config_floats(
    config: dict[str, Any] | None,
    key: str,
    *,
    default: Sequence[float],
) -> list[float]:
    values = default if config is None else config.get(key, default)
    return [float(value) for value in values]


def _unique_floats(values: Sequence[float]) -> list[float]:
    unique: list[float] = []
    for value in values:
        number = float(value)
        if not any(abs(number - existing) <= 1e-12 for existing in unique):
            unique.append(number)
    if not unique:
        raise ValueError("at least one candidate value is required")
    return unique


def _num_classes(tasks: Sequence[TaskFeatureSplit]) -> int:
    labels: set[int] = set()
    for task in tasks:
        labels.update(int(value) for value in np.unique(task.train_labels))
    return len(labels)


def _mean_task_class_count(tasks: Sequence[TaskFeatureSplit]) -> float:
    return float(np.mean([len(_task_classes([task])[task.task_id]) for task in tasks]))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values = np.asarray(y_true, dtype=np.int64)
    pred_values = np.asarray(y_pred, dtype=np.int64)
    if true_values.shape[0] == 0:
        return math.nan
    return float(np.mean(true_values == pred_values))


def _router_top1_acc(router_top1: np.ndarray | None, true_task_ids: np.ndarray) -> float:
    if router_top1 is None:
        return math.nan
    return _accuracy(true_task_ids, router_top1)


def _router_topk_recall(router_topk: np.ndarray | None, true_task_ids: np.ndarray) -> float:
    if router_topk is None:
        return math.nan
    return compute_topk_task_recall(router_topk, true_task_ids)


def _float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number
