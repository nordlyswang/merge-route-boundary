"""Router calibration and mask-size frontier experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mrb.baselines.calibration import (
    CalibrationMetrics,
    compute_calibration_metrics,
    extract_linear_router_scores,
    log_probabilities,
    predict_topk_from_probabilities,
    select_temperature,
    temperature_scaled_probabilities,
)
from mrb.baselines.classifiers import PrototypeClassifier
from mrb.baselines.feature_data import FeatureToyDataset, TaskFeatureSplit, stack_task_split
from mrb.baselines.routers import SklearnLinearRouter
from mrb.baselines.routing_controls import (
    predict_prototype_with_topk_task_mask,
    predict_with_fallback_to_merge_all,
)
from mrb.data.splits import stable_hash


ROUTER_CALIBRATION_RESULT_COLUMNS = [
    "method",
    "control_type",
    "router_type",
    "calibrated",
    "temperature",
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
    "router_ece",
    "router_nll",
    "router_brier",
    "fallback_rate",
    "mean_mask_size",
    "route_vs_merge_gap",
    "oracle_gap_closure",
    "mask_size_cost",
    "oracle_topk_acc_std",
    "num_test_samples",
    "status",
    "warning",
]

ROUTER_CALIBRATION_PER_TASK_COLUMNS = [
    "method",
    "task_id",
    "task_acc",
    "num_test_samples",
    "router_acc_top1_task",
    "router_recall_topk_task",
    "fallback_rate_task",
    "mean_mask_size_task",
]

ROUTER_CALIBRATION_CONFUSION_COLUMNS = [
    "method",
    "router_type",
    "true_task_id",
    "pred_task_id",
    "count",
]

DEFAULT_PREVIOUS_BEST_FALLBACK_ACC = 0.6732


@dataclass(frozen=True)
class RouterCalibrationEvaluation:
    results: pd.DataFrame
    per_task: pd.DataFrame
    route_confusion: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class RouterCalibrationArtifacts:
    classifier: PrototypeClassifier
    router: SklearnLinearRouter
    task_classes: dict[int, tuple[int, ...]]
    train_features: np.ndarray
    train_labels: np.ndarray
    train_task_ids: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    val_task_ids: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    test_task_ids: np.ndarray
    route_ids: np.ndarray
    score_source: str
    val_scores: np.ndarray
    test_scores: np.ndarray
    selected_temperature: float
    uncalibrated_val_metrics: CalibrationMetrics
    uncalibrated_test_metrics: CalibrationMetrics
    calibrated_val_metrics: CalibrationMetrics
    calibrated_test_metrics: CalibrationMetrics
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    uncalibrated_val_probabilities: np.ndarray
    uncalibrated_test_probabilities: np.ndarray


def evaluate_router_calibration_frontier(
    dataset: FeatureToyDataset,
    *,
    router_config: dict[str, Any] | None = None,
    calibration_config: dict[str, Any] | None = None,
    fallback_config: dict[str, Any] | None = None,
    soft_prior_config: dict[str, Any] | None = None,
    topk_config: dict[str, Any] | None = None,
    oracle_topk_config: dict[str, Any] | None = None,
    previous_best_fallback_acc: float = DEFAULT_PREVIOUS_BEST_FALLBACK_ACC,
    config: dict[str, Any] | None = None,
) -> RouterCalibrationEvaluation:
    tasks = list(dataset.tasks)
    artifacts = _fit_and_calibrate(
        tasks,
        router_config=router_config,
        calibration_config=calibration_config,
    )
    rows: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []

    for row, task_rows in _base_rows(artifacts):
        rows.append(row)
        per_task.extend(task_rows)

    previous_row, previous_task_rows, previous_confusion = _evaluate_fallback(
        artifacts,
        method="merge_all_prototype_fallback_linear_val_tau",
        control_type="previous_uncalibrated_fallback",
        router_type="linear_uncalibrated",
        calibrated=False,
        probabilities=artifacts.uncalibrated_test_probabilities,
        val_probabilities=artifacts.uncalibrated_val_probabilities,
        temperature=1.0,
        fallback_config=fallback_config,
    )
    rows.append(previous_row)
    per_task.extend(previous_task_rows)
    confusion.extend(previous_confusion)

    if _config_enabled(fallback_config, default=True):
        row, task_rows, conf_rows = _evaluate_fallback(
            artifacts,
            method="merge_all_prototype_fallback_linear_calibrated_tau",
            control_type="calibrated_fallback",
            router_type="linear_calibrated",
            calibrated=True,
            probabilities=artifacts.test_probabilities,
            val_probabilities=artifacts.val_probabilities,
            temperature=artifacts.selected_temperature,
            fallback_config=fallback_config,
        )
        rows.append(row)
        per_task.extend(task_rows)
        confusion.extend(conf_rows)

    if _config_enabled(soft_prior_config, default=True):
        row, task_rows, conf_rows = _evaluate_soft_prior(
            artifacts,
            beta_grid=_config_floats(
                soft_prior_config,
                "beta_grid",
                default=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
            ),
        )
        rows.append(row)
        per_task.extend(task_rows)
        confusion.extend(conf_rows)

    topk_rows: list[dict[str, Any]] = []
    if _config_enabled(topk_config, default=True):
        for k in _config_ints(topk_config, "ks", default=(1, 2, 3, 4, 5, 7, 10)):
            row, task_rows, conf_rows = _evaluate_topk_frontier(artifacts, k)
            rows.append(row)
            topk_rows.append(row)
            per_task.extend(task_rows)
            confusion.extend(conf_rows)

    oracle_rows: list[dict[str, Any]] = []
    if _config_enabled(oracle_topk_config, default=True):
        for k in _config_ints(oracle_topk_config, "ks", default=(1, 2, 3, 4, 5, 7, 10)):
            row, task_rows = _evaluate_oracle_topk(
                artifacts,
                k,
                repeats=int(dict(oracle_topk_config or {}).get("random_extra_repeats", 5)),
                seed=int(dict(oracle_topk_config or {}).get("random_seed", 0)),
            )
            rows.append(row)
            oracle_rows.append(row)
            per_task.extend(task_rows)

    results = pd.DataFrame(rows, columns=ROUTER_CALIBRATION_RESULT_COLUMNS)
    per_task_df = pd.DataFrame(per_task, columns=ROUTER_CALIBRATION_PER_TASK_COLUMNS)
    confusion_df = pd.DataFrame(confusion, columns=ROUTER_CALIBRATION_CONFUSION_COLUMNS)
    _fill_gap_columns(results)
    filled_topk_rows = results[results["control_type"] == "topk_task_mask"].to_dict(
        orient="records"
    )
    filled_oracle_rows = results[
        results["control_type"] == "oracle_topk_decomposition"
    ].to_dict(orient="records")
    summary = summarize_router_calibration_results(
        results,
        artifacts=artifacts,
        topk_rows=filled_topk_rows,
        oracle_rows=filled_oracle_rows,
        previous_best_fallback_acc=previous_best_fallback_acc,
        config=config or {},
        feature_bank_metadata=dataset.feature_bank_metadata,
        manifest_path=dataset.manifest_path,
    )
    return RouterCalibrationEvaluation(
        results=results,
        per_task=per_task_df,
        route_confusion=confusion_df,
        summary=summary,
    )


def write_router_calibration_outputs(
    evaluation: RouterCalibrationEvaluation,
    *,
    output_dir: str | Path,
    stream_id: str,
    seed: int,
    backbone_id: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{stream_id}_seed{seed}_{backbone_id}"
    paths = {
        "results_csv": output_root / f"{prefix}_results.csv",
        "per_task_csv": output_root / f"{prefix}_per_task.csv",
        "route_confusion_csv": output_root / f"{prefix}_route_confusion.csv",
        "summary_json": output_root / f"{prefix}_summary.json",
    }
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Output already exists: {joined}. Pass --overwrite to replace.")
    evaluation.results.to_csv(paths["results_csv"], index=False)
    evaluation.per_task.to_csv(paths["per_task_csv"], index=False)
    evaluation.route_confusion.to_csv(paths["route_confusion_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(_json_safe(evaluation.summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def predict_with_calibrated_soft_prior(
    classifier: PrototypeClassifier,
    features: np.ndarray,
    log_prior: np.ndarray,
    *,
    route_ids: np.ndarray,
    tasks: Sequence[TaskFeatureSplit],
    beta: float,
) -> np.ndarray:
    if classifier.classes_ is None:
        raise ValueError("PrototypeClassifier is not fitted")
    prototype_scores = classifier.decision_scores(features)
    prior_values = np.asarray(log_prior, dtype=np.float32)
    if prior_values.shape[0] != prototype_scores.shape[0]:
        raise ValueError("features and log_prior must have matching N")
    class_task_ids = _class_task_ids(classifier.classes_, tasks)
    route_to_col = {int(route_id): col for col, route_id in enumerate(route_ids)}
    prior_by_class = np.zeros_like(prototype_scores, dtype=np.float32)
    for class_col, task_id in enumerate(class_task_ids):
        route_col = route_to_col.get(int(task_id))
        if route_col is not None:
            prior_by_class[:, class_col] = prior_values[:, route_col]
    adjusted = prototype_scores + float(beta) * prior_by_class
    return np.asarray(classifier.classes_, dtype=np.int64)[np.argmax(adjusted, axis=1)]


def oracle_topk_route_sets(
    true_task_ids: np.ndarray,
    task_ids: Sequence[int],
    *,
    k: int,
    seed: int,
) -> np.ndarray:
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    available = np.asarray(sorted(int(task_id) for task_id in task_ids), dtype=np.int64)
    if k <= 0:
        raise ValueError("k must be positive")
    limit = min(int(k), available.shape[0])
    if limit == available.shape[0]:
        return np.tile(available.reshape(1, -1), (true_values.shape[0], 1))
    rng = np.random.default_rng(seed)
    route_sets = np.empty((true_values.shape[0], limit), dtype=np.int64)
    for row_idx, true_task_id in enumerate(true_values):
        candidates = available[available != int(true_task_id)]
        extra = rng.choice(candidates, size=limit - 1, replace=False) if limit > 1 else []
        route_sets[row_idx] = np.asarray([int(true_task_id), *extra], dtype=np.int64)
    return route_sets


def summarize_router_calibration_results(
    results: pd.DataFrame,
    *,
    artifacts: RouterCalibrationArtifacts,
    topk_rows: Sequence[dict[str, Any]],
    oracle_rows: Sequence[dict[str, Any]],
    previous_best_fallback_acc: float,
    config: dict[str, Any],
    feature_bank_metadata: dict[str, Any],
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    merge_acc = _method_value(results, "merge_all_prototype", "overall_acc")
    oracle_acc = _method_value(results, "merge_all_prototype_oracle_task_mask", "overall_acc")
    calibrated_fallback = _method_brief(
        results,
        "merge_all_prototype_fallback_linear_calibrated_tau",
    )
    calibrated_soft = _method_brief(
        results,
        "merge_all_prototype_soft_prior_linear_calibrated_beta",
    )
    best_method = _best_method(results)
    best_learned_control = _best_learned_control(results)
    conclusion = _make_conclusion(
        best_learned_control=best_learned_control,
        calibrated_soft=calibrated_soft,
        previous_best_fallback_acc=previous_best_fallback_acc,
        merge_acc=merge_acc,
    )
    return {
        "merge_all_acc": merge_acc,
        "oracle_task_mask_acc": oracle_acc,
        "previous_best_fallback_acc": float(previous_best_fallback_acc),
        "best_calibrated_fallback": calibrated_fallback,
        "best_calibrated_soft_prior": calibrated_soft,
        "best_topk_by_val": _best_row_by(topk_rows, "val_acc"),
        "best_topk_by_test": _best_row_by(topk_rows, "overall_acc"),
        "topk_frontier": [_result_brief(row) for row in topk_rows],
        "oracle_topk_decomposition": [_result_brief(row) for row in oracle_rows],
        "selected_temperature": artifacts.selected_temperature,
        "score_source": artifacts.score_source,
        "router_calibration_metrics": {
            "uncalibrated_val": asdict(artifacts.uncalibrated_val_metrics),
            "calibrated_val": asdict(artifacts.calibrated_val_metrics),
            "uncalibrated_test": asdict(artifacts.uncalibrated_test_metrics),
            "calibrated_test": asdict(artifacts.calibrated_test_metrics),
        },
        "best_method_by_overall_acc": best_method,
        "best_learned_control_by_overall_acc": best_learned_control,
        "num_methods": int(results.shape[0]),
        "num_failed_methods": int((results["status"] != "ok").sum()),
        "failed_methods": results.loc[results["status"] != "ok", "method"].tolist(),
        "has_nan_overall_acc": bool(results["overall_acc"].isna().any()),
        "config_hash": stable_hash(config),
        "feature_bank_metadata": feature_bank_metadata,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "conclusion": conclusion,
    }


def _fit_and_calibrate(
    tasks: Sequence[TaskFeatureSplit],
    *,
    router_config: dict[str, Any] | None,
    calibration_config: dict[str, Any] | None,
) -> RouterCalibrationArtifacts:
    train_features, train_labels, train_task_ids = stack_task_split(tasks, "train")
    val_features, val_labels, val_task_ids = stack_task_split(tasks, "val")
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    classifier = _fit_global_prototype(tasks)
    router = _fit_linear_router(
        train_features,
        train_task_ids,
        router_config=router_config,
    )
    val_payload = extract_linear_router_scores(router, val_features)
    test_payload = extract_linear_router_scores(router, test_features)
    temperatures = _config_floats(
        calibration_config,
        "temperatures",
        default=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0),
    )
    n_bins = int(dict(calibration_config or {}).get("ece_bins", 15))
    selection = select_temperature(
        val_payload.scores,
        val_task_ids,
        val_payload.route_ids,
        temperatures,
        n_bins=n_bins,
    )
    uncalibrated_val = temperature_scaled_probabilities(val_payload.scores, 1.0)
    uncalibrated_test = temperature_scaled_probabilities(test_payload.scores, 1.0)
    calibrated_val = temperature_scaled_probabilities(
        val_payload.scores,
        selection.temperature,
    )
    calibrated_test = temperature_scaled_probabilities(
        test_payload.scores,
        selection.temperature,
    )
    return RouterCalibrationArtifacts(
        classifier=classifier,
        router=router,
        task_classes=_task_classes(tasks),
        train_features=train_features,
        train_labels=train_labels,
        train_task_ids=train_task_ids,
        val_features=val_features,
        val_labels=val_labels,
        val_task_ids=val_task_ids,
        test_features=test_features,
        test_labels=test_labels,
        test_task_ids=test_task_ids,
        route_ids=val_payload.route_ids,
        score_source=val_payload.score_source,
        val_scores=val_payload.scores,
        test_scores=test_payload.scores,
        selected_temperature=selection.temperature,
        uncalibrated_val_metrics=compute_calibration_metrics(
            uncalibrated_val,
            val_task_ids,
            val_payload.route_ids,
            n_bins=n_bins,
        ),
        uncalibrated_test_metrics=compute_calibration_metrics(
            uncalibrated_test,
            test_task_ids,
            test_payload.route_ids,
            n_bins=n_bins,
        ),
        calibrated_val_metrics=selection.val_metrics,
        calibrated_test_metrics=compute_calibration_metrics(
            calibrated_test,
            test_task_ids,
            test_payload.route_ids,
            n_bins=n_bins,
        ),
        val_probabilities=calibrated_val,
        test_probabilities=calibrated_test,
        uncalibrated_val_probabilities=uncalibrated_val,
        uncalibrated_test_probabilities=uncalibrated_test,
    )


def _base_rows(
    artifacts: RouterCalibrationArtifacts,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    merge_predictions = artifacts.classifier.predict(artifacts.test_features)
    merge_mask_sizes = np.full(
        artifacts.test_labels.shape[0],
        _num_classes(artifacts.classifier),
        dtype=np.float32,
    )
    oracle_predictions, oracle_mask_sizes = predict_prototype_with_topk_task_mask(
        artifacts.classifier,
        artifacts.test_features,
        artifacts.test_task_ids.reshape(-1, 1),
        task_classes=artifacts.task_classes,
    )
    return [
        _rows_from_predictions(
            method="merge_all_prototype",
            control_type="merge_all",
            router_type="none",
            calibrated=False,
            temperature=math.nan,
            topk=math.nan,
            tau=math.nan,
            beta=math.nan,
            selected_on="none",
            val_acc=math.nan,
            y_true=artifacts.test_labels,
            y_pred=merge_predictions,
            true_task_ids=artifacts.test_task_ids,
            router_top1=None,
            router_topk=None,
            router_metrics=None,
            fallback_flags=None,
            mask_sizes=merge_mask_sizes,
            oracle_topk_acc_std=math.nan,
        ),
        _rows_from_predictions(
            method="merge_all_prototype_oracle_task_mask",
            control_type="oracle_task_mask",
            router_type="oracle",
            calibrated=False,
            temperature=math.nan,
            topk=1,
            tau=math.nan,
            beta=math.nan,
            selected_on="none",
            val_acc=math.nan,
            y_true=artifacts.test_labels,
            y_pred=oracle_predictions,
            true_task_ids=artifacts.test_task_ids,
            router_top1=artifacts.test_task_ids,
            router_topk=artifacts.test_task_ids.reshape(-1, 1),
            router_metrics=None,
            fallback_flags=None,
            mask_sizes=oracle_mask_sizes,
            oracle_topk_acc_std=math.nan,
        ),
    ]


def _evaluate_fallback(
    artifacts: RouterCalibrationArtifacts,
    *,
    method: str,
    control_type: str,
    router_type: str,
    calibrated: bool,
    probabilities: np.ndarray,
    val_probabilities: np.ndarray,
    temperature: float,
    fallback_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tau_grid = _config_floats(
        fallback_config,
        "tau_grid",
        default=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    )
    val_top1 = predict_topk_from_probabilities(val_probabilities, artifacts.route_ids, 1)
    val_confidence = val_probabilities.max(axis=1)
    best_tau, best_val_acc = _select_fallback_tau(
        artifacts,
        val_top1,
        val_confidence,
        tau_grid,
    )
    test_top1 = predict_topk_from_probabilities(probabilities, artifacts.route_ids, 1)
    predictions, fallback_flags, mask_sizes = predict_with_fallback_to_merge_all(
        artifacts.classifier,
        artifacts.test_features,
        test_top1,
        probabilities.max(axis=1),
        tau=best_tau,
        task_classes=artifacts.task_classes,
    )
    router_metrics = artifacts.calibrated_test_metrics if calibrated else artifacts.uncalibrated_test_metrics
    row, per_task = _rows_from_predictions(
        method=method,
        control_type=control_type,
        router_type=router_type,
        calibrated=calibrated,
        temperature=temperature,
        topk=1,
        tau=best_tau,
        beta=math.nan,
        selected_on="val",
        val_acc=best_val_acc,
        y_true=artifacts.test_labels,
        y_pred=predictions,
        true_task_ids=artifacts.test_task_ids,
        router_top1=test_top1[:, 0],
        router_topk=test_top1,
        router_metrics=router_metrics,
        fallback_flags=fallback_flags,
        mask_sizes=mask_sizes,
        oracle_topk_acc_std=math.nan,
    )
    return (
        row,
        per_task,
        _route_confusion_rows(method, router_type, artifacts.test_task_ids, test_top1[:, 0]),
    )


def _evaluate_soft_prior(
    artifacts: RouterCalibrationArtifacts,
    *,
    beta_grid: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    val_log_prior = log_probabilities(artifacts.val_probabilities)
    test_log_prior = log_probabilities(artifacts.test_probabilities)
    best_beta = float(beta_grid[0])
    best_val_acc = -1.0
    for beta in beta_grid:
        val_predictions = predict_with_calibrated_soft_prior(
            artifacts.classifier,
            artifacts.val_features,
            val_log_prior,
            route_ids=artifacts.route_ids,
            tasks=_tasks_from_classes(artifacts.task_classes),
            beta=float(beta),
        )
        acc = _accuracy(artifacts.val_labels, val_predictions)
        if acc > best_val_acc + 1e-12:
            best_beta = float(beta)
            best_val_acc = acc
    predictions = predict_with_calibrated_soft_prior(
        artifacts.classifier,
        artifacts.test_features,
        test_log_prior,
        route_ids=artifacts.route_ids,
        tasks=_tasks_from_classes(artifacts.task_classes),
        beta=best_beta,
    )
    top1 = predict_topk_from_probabilities(artifacts.test_probabilities, artifacts.route_ids, 1)
    mask_sizes = np.full(
        artifacts.test_labels.shape[0],
        _num_classes(artifacts.classifier),
        dtype=np.float32,
    )
    row, per_task = _rows_from_predictions(
        method="merge_all_prototype_soft_prior_linear_calibrated_beta",
        control_type="calibrated_soft_prior",
        router_type="linear_calibrated",
        calibrated=True,
        temperature=artifacts.selected_temperature,
        topk=math.nan,
        tau=math.nan,
        beta=best_beta,
        selected_on="val",
        val_acc=best_val_acc,
        y_true=artifacts.test_labels,
        y_pred=predictions,
        true_task_ids=artifacts.test_task_ids,
        router_top1=top1[:, 0],
        router_topk=top1,
        router_metrics=artifacts.calibrated_test_metrics,
        fallback_flags=None,
        mask_sizes=mask_sizes,
        oracle_topk_acc_std=math.nan,
    )
    return (
        row,
        per_task,
        _route_confusion_rows(
            row["method"],
            "linear_calibrated",
            artifacts.test_task_ids,
            top1[:, 0],
        ),
    )


def _evaluate_topk_frontier(
    artifacts: RouterCalibrationArtifacts,
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    val_topk = predict_topk_from_probabilities(artifacts.val_probabilities, artifacts.route_ids, k)
    val_predictions, _ = predict_prototype_with_topk_task_mask(
        artifacts.classifier,
        artifacts.val_features,
        val_topk,
        task_classes=artifacts.task_classes,
    )
    test_topk = predict_topk_from_probabilities(
        artifacts.test_probabilities,
        artifacts.route_ids,
        k,
    )
    predictions, mask_sizes = predict_prototype_with_topk_task_mask(
        artifacts.classifier,
        artifacts.test_features,
        test_topk,
        task_classes=artifacts.task_classes,
    )
    row, per_task = _rows_from_predictions(
        method=f"merge_all_prototype_topk_task_mask_linear_k{k}",
        control_type="topk_task_mask",
        router_type="linear_calibrated",
        calibrated=True,
        temperature=artifacts.selected_temperature,
        topk=k,
        tau=math.nan,
        beta=math.nan,
        selected_on="val",
        val_acc=_accuracy(artifacts.val_labels, val_predictions),
        y_true=artifacts.test_labels,
        y_pred=predictions,
        true_task_ids=artifacts.test_task_ids,
        router_top1=test_topk[:, 0],
        router_topk=test_topk,
        router_metrics=artifacts.calibrated_test_metrics,
        fallback_flags=None,
        mask_sizes=mask_sizes,
        oracle_topk_acc_std=math.nan,
    )
    return (
        row,
        per_task,
        _route_confusion_rows(row["method"], "linear_calibrated", artifacts.test_task_ids, test_topk[:, 0]),
    )


def _evaluate_oracle_topk(
    artifacts: RouterCalibrationArtifacts,
    k: int,
    *,
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_ids = sorted(artifacts.task_classes)
    repeat_accs: list[float] = []
    repeat_predictions: list[np.ndarray] = []
    repeat_masks: list[np.ndarray] = []
    for repeat in range(max(1, int(repeats))):
        route_sets = oracle_topk_route_sets(
            artifacts.test_task_ids,
            task_ids,
            k=k,
            seed=seed + repeat * 1009 + int(k) * 37,
        )
        predictions, mask_sizes = predict_prototype_with_topk_task_mask(
            artifacts.classifier,
            artifacts.test_features,
            route_sets,
            task_classes=artifacts.task_classes,
        )
        repeat_predictions.append(predictions)
        repeat_masks.append(mask_sizes)
        repeat_accs.append(_accuracy(artifacts.test_labels, predictions))
    mean_predictions = repeat_predictions[0]
    mean_mask_sizes = repeat_masks[0]
    row, per_task = _rows_from_predictions(
        method=f"oracle_topk_mask_random_extra_k{k}",
        control_type="oracle_topk_decomposition",
        router_type="oracle_random_extra",
        calibrated=False,
        temperature=math.nan,
        topk=k,
        tau=math.nan,
        beta=math.nan,
        selected_on="none",
        val_acc=math.nan,
        y_true=artifacts.test_labels,
        y_pred=mean_predictions,
        true_task_ids=artifacts.test_task_ids,
        router_top1=artifacts.test_task_ids,
        router_topk=np.repeat(artifacts.test_task_ids.reshape(-1, 1), min(k, len(task_ids)), axis=1),
        router_metrics=None,
        fallback_flags=None,
        mask_sizes=mean_mask_sizes,
        oracle_topk_acc_std=float(np.std(repeat_accs)),
    )
    row["overall_acc"] = float(np.mean(repeat_accs))
    row["mean_task_acc"] = float(np.mean([task_row["task_acc"] for task_row in per_task]))
    return row, per_task


def _select_fallback_tau(
    artifacts: RouterCalibrationArtifacts,
    val_top1: np.ndarray,
    val_confidence: np.ndarray,
    tau_grid: Sequence[float],
) -> tuple[float, float]:
    best_tau = float(tau_grid[0])
    best_acc = -1.0
    best_fallback_rate = math.inf
    for tau in tau_grid:
        predictions, fallback_flags, _ = predict_with_fallback_to_merge_all(
            artifacts.classifier,
            artifacts.val_features,
            val_top1,
            val_confidence,
            tau=float(tau),
            task_classes=artifacts.task_classes,
        )
        acc = _accuracy(artifacts.val_labels, predictions)
        fallback_rate = float(np.mean(fallback_flags))
        if acc > best_acc + 1e-12 or (
            abs(acc - best_acc) <= 1e-12 and fallback_rate < best_fallback_rate
        ):
            best_tau = float(tau)
            best_acc = acc
            best_fallback_rate = fallback_rate
    return best_tau, best_acc


def _rows_from_predictions(
    *,
    method: str,
    control_type: str,
    router_type: str,
    calibrated: bool,
    temperature: float,
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
    router_metrics: CalibrationMetrics | None,
    fallback_flags: np.ndarray | None,
    mask_sizes: np.ndarray,
    oracle_topk_acc_std: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_true_values = np.asarray(y_true, dtype=np.int64)
    y_pred_values = np.asarray(y_pred, dtype=np.int64)
    task_values = np.asarray(true_task_ids, dtype=np.int64)
    mask_values = np.asarray(mask_sizes, dtype=np.float32)
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
    row = {
        "method": method,
        "control_type": control_type,
        "router_type": router_type,
        "calibrated": bool(calibrated),
        "temperature": temperature,
        "topk": topk,
        "tau": tau,
        "beta": beta,
        "selected_on": selected_on,
        "val_acc": val_acc,
        "overall_acc": _accuracy(y_true_values, y_pred_values),
        "mean_task_acc": float(np.mean([row["task_acc"] for row in per_task])),
        "last_task_acc": last_task_acc,
        "router_acc_top1": _nan_if_none(router_metrics, "top1_acc", router_top1, task_values),
        "router_recall_topk": _router_topk_recall(router_topk, task_values),
        "router_ece": math.nan if router_metrics is None else router_metrics.ece,
        "router_nll": math.nan if router_metrics is None else router_metrics.nll,
        "router_brier": math.nan if router_metrics is None else router_metrics.brier,
        "fallback_rate": 0.0 if fallback_flags is None else float(np.mean(fallback_flags)),
        "mean_mask_size": float(np.mean(mask_values)),
        "route_vs_merge_gap": math.nan,
        "oracle_gap_closure": math.nan,
        "mask_size_cost": math.nan,
        "oracle_topk_acc_std": oracle_topk_acc_std,
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
                "fallback_rate_task": 0.0
                if fallback_flags is None
                else float(np.mean(fallback_flags[mask])),
                "mean_mask_size_task": float(np.mean(mask_sizes[mask])),
            }
        )
    return rows


def _fill_gap_columns(results: pd.DataFrame) -> None:
    merge_acc = _method_value(results, "merge_all_prototype", "overall_acc")
    oracle_acc = _method_value(results, "merge_all_prototype_oracle_task_mask", "overall_acc")
    if merge_acc is None:
        return
    results.loc[:, "route_vs_merge_gap"] = results["overall_acc"].astype(float) - merge_acc
    if oracle_acc is not None and abs(oracle_acc - merge_acc) > 1e-12:
        results.loc[:, "oracle_gap_closure"] = (
            results["overall_acc"].astype(float) - merge_acc
        ) / (oracle_acc - merge_acc)
        oracle_mask = results["control_type"] == "oracle_topk_decomposition"
        results.loc[oracle_mask, "mask_size_cost"] = (
            oracle_acc - results.loc[oracle_mask, "overall_acc"].astype(float)
        )


def _fit_global_prototype(tasks: Sequence[TaskFeatureSplit]) -> PrototypeClassifier:
    train_features, train_labels, _ = stack_task_split(tasks, "train")
    classes = sorted(int(value) for value in np.unique(train_labels))
    return PrototypeClassifier(expected_classes=classes).fit(train_features, train_labels)


def _fit_linear_router(
    train_features: np.ndarray,
    train_task_ids: np.ndarray,
    *,
    router_config: dict[str, Any] | None,
) -> SklearnLinearRouter:
    config = dict(router_config or {})
    return SklearnLinearRouter(
        max_iter=int(config.get("max_iter", 1000)),
        class_weight=config.get("class_weight", "balanced"),
        solver=str(config.get("solver", "lbfgs")),
        max_train_per_task=config.get("max_train_per_task"),
        random_state=int(config.get("random_state", 0)),
    ).fit(train_features, train_task_ids)


def _task_classes(tasks: Sequence[TaskFeatureSplit]) -> dict[int, tuple[int, ...]]:
    return {
        task.task_id: task.classes or tuple(int(value) for value in np.unique(task.train_labels))
        for task in tasks
    }


def _tasks_from_classes(task_classes: dict[int, tuple[int, ...]]) -> list[TaskFeatureSplit]:
    tasks: list[TaskFeatureSplit] = []
    empty = np.zeros((1, 1), dtype=np.float32)
    empty_labels = np.asarray([0], dtype=np.int64)
    for task_id, classes in sorted(task_classes.items()):
        tasks.append(
            TaskFeatureSplit(
                task_id=int(task_id),
                dataset_id="synthetic_task_classes",
                classes=tuple(int(value) for value in classes),
                train_features=empty,
                train_labels=empty_labels,
                train_indices=empty_labels,
                val_features=empty,
                val_labels=empty_labels,
                val_indices=empty_labels,
                test_features=empty,
                test_labels=empty_labels,
                test_indices=empty_labels,
                feature_bank_splits={},
            )
        )
    return tasks


def _class_task_ids(classes: np.ndarray, tasks: Sequence[TaskFeatureSplit]) -> np.ndarray:
    class_to_task: dict[int, int] = {}
    for task in tasks:
        for class_id in task.classes:
            class_to_task[int(class_id)] = int(task.task_id)
    return np.asarray([class_to_task.get(int(class_id), -1) for class_id in classes], dtype=np.int64)


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


def _best_method(results: pd.DataFrame) -> dict[str, Any] | None:
    ok = results[results["status"] == "ok"] if "status" in results else results
    if ok.empty:
        return None
    return _result_brief(ok.sort_values("overall_acc", ascending=False).iloc[0].to_dict())


def _best_learned_control(results: pd.DataFrame) -> dict[str, Any] | None:
    controls = results[
        results["control_type"].isin(
            ["calibrated_fallback", "calibrated_soft_prior", "topk_task_mask"]
        )
    ]
    if controls.empty:
        return None
    return _result_brief(controls.sort_values("overall_acc", ascending=False).iloc[0].to_dict())


def _method_brief(results: pd.DataFrame, method: str) -> dict[str, Any] | None:
    rows = results[results["method"] == method]
    if rows.empty:
        return None
    return _result_brief(rows.iloc[0].to_dict())


def _result_brief(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "method",
        "control_type",
        "router_type",
        "calibrated",
        "temperature",
        "topk",
        "tau",
        "beta",
        "val_acc",
        "overall_acc",
        "router_acc_top1",
        "router_recall_topk",
        "router_ece",
        "router_nll",
        "router_brier",
        "fallback_rate",
        "mean_mask_size",
        "route_vs_merge_gap",
        "oracle_gap_closure",
        "mask_size_cost",
        "oracle_topk_acc_std",
    ]
    return {key: _json_safe(row.get(key)) for key in keys if key in row}


def _best_row_by(rows: Sequence[dict[str, Any]], column: str) -> dict[str, Any] | None:
    if not rows:
        return None
    valid = [row for row in rows if not math.isnan(float(row.get(column, math.nan)))]
    if not valid:
        return None
    return _result_brief(max(valid, key=lambda row: float(row[column])))


def _make_conclusion(
    *,
    best_learned_control: dict[str, Any] | None,
    calibrated_soft: dict[str, Any] | None,
    previous_best_fallback_acc: float,
    merge_acc: float | None,
) -> str:
    pieces: list[str] = []
    if best_learned_control is not None:
        best_acc = float(best_learned_control["overall_acc"])
        if best_acc > previous_best_fallback_acc:
            pieces.append("best learned control exceeds previous best fallback")
        else:
            pieces.append("best learned control does not exceed previous best fallback")
        if merge_acc is not None and best_acc > merge_acc:
            pieces.append("best learned control exceeds merge_all")
    if calibrated_soft is not None and float(calibrated_soft.get("beta") or 0.0) == 0.0:
        pieces.append("calibrated router prior still does not improve prototype score")
    return "; ".join(pieces)


def _method_value(results: pd.DataFrame, method: str, column: str) -> float | None:
    rows = results[results["method"] == method]
    if rows.empty:
        return None
    value = rows.iloc[0][column]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _config_enabled(config: dict[str, Any] | None, *, default: bool) -> bool:
    if config is None:
        return default
    return bool(dict(config).get("enabled", default))


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


def _num_classes(classifier: PrototypeClassifier) -> int:
    if classifier.classes_ is None:
        raise ValueError("classifier is not fitted")
    return int(classifier.classes_.shape[0])


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values = np.asarray(y_true, dtype=np.int64)
    pred_values = np.asarray(y_pred, dtype=np.int64)
    return float(np.mean(true_values == pred_values))


def _router_top1_acc(router_top1: np.ndarray | None, true_task_ids: np.ndarray) -> float:
    if router_top1 is None:
        return math.nan
    return _accuracy(true_task_ids, router_top1)


def _router_topk_recall(router_topk: np.ndarray | None, true_task_ids: np.ndarray) -> float:
    if router_topk is None:
        return math.nan
    values = np.asarray(router_topk, dtype=np.int64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    return float(np.mean(np.any(values == true_values[:, None], axis=1)))


def _nan_if_none(
    router_metrics: CalibrationMetrics | None,
    attr: str,
    router_top1: np.ndarray | None,
    true_task_ids: np.ndarray,
) -> float:
    if router_metrics is not None:
        return float(getattr(router_metrics, attr))
    return _router_top1_acc(router_top1, true_task_ids)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
