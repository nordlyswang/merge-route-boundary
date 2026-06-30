"""Pairwise-aware routing policy for frozen-feature baselines."""

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
    predict_topk_from_probabilities,
    select_temperature,
    temperature_scaled_probabilities,
)
from mrb.baselines.classifiers import PrototypeClassifier
from mrb.baselines.feature_data import FeatureToyDataset, TaskFeatureSplit, stack_task_split
from mrb.baselines.routers import SklearnLinearRouter
from mrb.baselines.routing_controls import predict_prototype_with_topk_task_mask
from mrb.data.splits import stable_hash


ACTION_TOP1_MASK = "top1_mask"
ACTION_TOP2_MASK = "top2_mask"
ACTION_FALLBACK = "fallback_merge_all"

PAIRWISE_RESULT_COLUMNS = [
    "method",
    "policy_type",
    "router_type",
    "selected_on",
    "val_acc",
    "overall_acc",
    "mean_task_acc",
    "last_task_acc",
    "route_vs_merge_gap",
    "oracle_gap_closure",
    "tau_fallback",
    "tau_margin",
    "tau_pair_auc",
    "tau_pair_confusion",
    "action_top1_rate",
    "action_top2_rate",
    "action_fallback_rate",
    "mean_mask_size",
    "router_top1_acc",
    "router_top2_recall",
    "confidence_auc_for_correct_route",
    "num_test_samples",
    "status",
    "warning",
]

PAIRWISE_PER_TASK_COLUMNS = [
    "method",
    "task_id",
    "task_acc",
    "num_test_samples",
    "action_top1_rate_task",
    "action_top2_rate_task",
    "action_fallback_rate_task",
    "mean_mask_size_task",
    "router_top1_acc_task",
    "router_top2_recall_task",
    "gain_vs_merge_task",
    "gain_vs_hard_top1_task",
]

PAIRWISE_PAIR_COLUMNS = [
    "top1_task",
    "top2_task",
    "num_samples",
    "pair_auc",
    "pair_separation_ratio",
    "pair_confusion_prior",
    "mean_margin",
    "mean_confidence",
    "action_top1_rate",
    "action_top2_rate",
    "action_fallback_rate",
    "accuracy_when_pair_appears",
]

THRESHOLD_GRID_COLUMNS = [
    "method",
    "tau_fallback",
    "tau_margin",
    "tau_pair_auc",
    "tau_pair_confusion",
    "val_acc",
    "mean_mask_size",
    "action_top1_rate",
    "action_top2_rate",
    "action_fallback_rate",
    "status",
    "warning",
]

DEFAULT_PREVIOUS_BEST_FALLBACK_ACC = 0.6732


@dataclass(frozen=True)
class PairwisePolicyEvaluation:
    results: pd.DataFrame
    per_task: pd.DataFrame
    pairwise_pairs: pd.DataFrame
    threshold_grid: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class PolicyThresholds:
    tau_fallback: float
    tau_margin: float
    tau_pair_auc: float
    tau_pair_confusion: float


@dataclass(frozen=True)
class PairwiseSignals:
    top2_routes: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray
    pair_auc: np.ndarray
    pair_separation_ratio: np.ndarray
    pair_confusion_prior: np.ndarray
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairwiseDiagnosticsValue:
    pair_auc: float
    pair_separation_ratio: float
    warning: str = ""


@dataclass(frozen=True)
class PairwiseDiagnosticsLookup:
    values: dict[tuple[int, int], PairwiseDiagnosticsValue]
    default_pair_auc: float = 1.0
    default_separation_ratio: float = math.nan
    source_missing: bool = False

    @classmethod
    def from_dataframe(
        cls,
        diagnostics: pd.DataFrame | None,
        *,
        default_pair_auc: float = 1.0,
        default_separation_ratio: float = math.nan,
    ) -> "PairwiseDiagnosticsLookup":
        if diagnostics is None or diagnostics.empty:
            return cls(
                values={},
                default_pair_auc=default_pair_auc,
                default_separation_ratio=default_separation_ratio,
                source_missing=True,
            )
        if not {"task_i", "task_j"}.issubset(diagnostics.columns):
            return cls(
                values={},
                default_pair_auc=default_pair_auc,
                default_separation_ratio=default_separation_ratio,
                source_missing=True,
            )
        values: dict[tuple[int, int], PairwiseDiagnosticsValue] = {}
        for row in diagnostics.to_dict(orient="records"):
            try:
                task_i = int(row["task_i"])
                task_j = int(row["task_j"])
            except (TypeError, ValueError):
                continue
            pair_auc = _first_finite(
                row,
                (
                    "diagnostics_linear_probe_auc_symmetric",
                    "linear_probe_auc_symmetric",
                    "linear_probe_auc",
                ),
                default_pair_auc,
            )
            separation = _first_finite(
                row,
                ("diagnostics_separation_ratio", "separation_ratio"),
                default_separation_ratio,
            )
            values[_pair_key(task_i, task_j)] = PairwiseDiagnosticsValue(
                pair_auc=pair_auc,
                pair_separation_ratio=separation,
            )
        return cls(
            values=values,
            default_pair_auc=default_pair_auc,
            default_separation_ratio=default_separation_ratio,
            source_missing=False,
        )

    def lookup(self, task_i: int, task_j: int) -> PairwiseDiagnosticsValue:
        key = _pair_key(task_i, task_j)
        value = self.values.get(key)
        if value is not None:
            return value
        warning = f"missing diagnostics pair ({key[0]}, {key[1]}); used safe default"
        if self.source_missing:
            warning = "diagnostics table missing; used safe defaults"
        return PairwiseDiagnosticsValue(
            pair_auc=self.default_pair_auc,
            pair_separation_ratio=self.default_separation_ratio,
            warning=warning,
        )


@dataclass(frozen=True)
class PairwisePolicyArtifacts:
    classifier: PrototypeClassifier
    router: SklearnLinearRouter
    task_classes: dict[int, tuple[int, ...]]
    val_features: np.ndarray
    val_labels: np.ndarray
    val_task_ids: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    test_task_ids: np.ndarray
    route_ids: np.ndarray
    score_source: str
    selected_temperature: float
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    calibration_metrics_val: CalibrationMetrics
    calibration_metrics_test: CalibrationMetrics


@dataclass(frozen=True)
class SplitPredictions:
    merge_predictions: np.ndarray
    top1_predictions: np.ndarray
    top1_mask_sizes: np.ndarray
    top2_predictions: np.ndarray
    top2_mask_sizes: np.ndarray
    full_mask_sizes: np.ndarray


def evaluate_pairwise_policy(
    dataset: FeatureToyDataset,
    *,
    router_config: dict[str, Any] | None = None,
    calibration_config: dict[str, Any] | None = None,
    policy_config: dict[str, Any] | None = None,
    diagnostics_csv: str | Path | None = None,
    previous_best_fallback_acc: float = DEFAULT_PREVIOUS_BEST_FALLBACK_ACC,
    config: dict[str, Any] | None = None,
) -> PairwisePolicyEvaluation:
    """Run Pairwise-aware Routing Policy v0 on an already-loaded feature dataset."""

    policy_values = dict(policy_config or {})
    artifacts = _fit_artifacts(
        list(dataset.tasks),
        router_config=router_config,
        calibration_config=calibration_config,
    )
    diagnostics = PairwiseDiagnosticsLookup.from_dataframe(
        _load_diagnostics(diagnostics_csv),
        default_pair_auc=float(policy_values.get("default_pair_auc", 1.0)),
    )
    val_top2 = predict_topk_from_probabilities(
        artifacts.val_probabilities,
        artifacts.route_ids,
        2,
    )
    test_top2 = predict_topk_from_probabilities(
        artifacts.test_probabilities,
        artifacts.route_ids,
        2,
    )
    prior_lookup = build_pair_confusion_prior(artifacts.val_task_ids, val_top2[:, 0])
    val_signals = build_pairwise_signals(
        val_top2,
        artifacts.val_probabilities,
        diagnostics,
        prior_lookup,
    )
    test_signals = build_pairwise_signals(
        test_top2,
        artifacts.test_probabilities,
        diagnostics,
        prior_lookup,
    )
    val_predictions = _split_predictions(
        artifacts.classifier,
        artifacts.val_features,
        val_top2,
        task_classes=artifacts.task_classes,
    )
    test_predictions = _split_predictions(
        artifacts.classifier,
        artifacts.test_features,
        test_top2,
        task_classes=artifacts.task_classes,
    )
    threshold_grid = evaluate_threshold_grid(
        signals=val_signals,
        predictions=val_predictions,
        labels=artifacts.val_labels,
        thresholds=_threshold_candidates(policy_values),
    )
    selected_thresholds = select_best_thresholds(threshold_grid)
    selected_val_row = _selected_grid_row(threshold_grid, selected_thresholds)

    rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    confidence_auc = compute_confidence_auc_for_correct_route(
        artifacts.test_probabilities.max(axis=1),
        test_top2[:, 0] == artifacts.test_task_ids,
    )
    val_confidence_auc = compute_confidence_auc_for_correct_route(
        artifacts.val_probabilities.max(axis=1),
        val_top2[:, 0] == artifacts.val_task_ids,
    )

    def append_method(
        *,
        method: str,
        policy_type: str,
        selected_on: str,
        val_acc: float,
        y_pred: np.ndarray,
        router_topk: np.ndarray | None,
        actions: np.ndarray | None,
        mask_sizes: np.ndarray,
        thresholds: PolicyThresholds | None = None,
        warning: str = "",
    ) -> None:
        row, task_rows = rows_from_predictions(
            method=method,
            policy_type=policy_type,
            router_type="linear_calibrated" if router_topk is not None else "none",
            selected_on=selected_on,
            val_acc=val_acc,
            y_true=artifacts.test_labels,
            y_pred=y_pred,
            true_task_ids=artifacts.test_task_ids,
            router_topk=router_topk,
            actions=actions,
            mask_sizes=mask_sizes,
            confidence_auc_for_correct_route=confidence_auc,
            thresholds=thresholds,
            warning=warning,
        )
        rows.append(row)
        per_task_rows.extend(task_rows)

    merge_val = artifacts.classifier.predict(artifacts.val_features)
    merge_test = test_predictions.merge_predictions
    append_method(
        method="merge_all_prototype",
        policy_type="merge_all",
        selected_on="none",
        val_acc=_accuracy(artifacts.val_labels, merge_val),
        y_pred=merge_test,
        router_topk=None,
        actions=np.full(artifacts.test_labels.shape[0], ACTION_FALLBACK, dtype=object),
        mask_sizes=test_predictions.full_mask_sizes,
    )

    val_oracle, _ = predict_prototype_with_topk_task_mask(
        artifacts.classifier,
        artifacts.val_features,
        artifacts.val_task_ids.reshape(-1, 1),
        task_classes=artifacts.task_classes,
    )
    test_oracle, test_oracle_sizes = predict_prototype_with_topk_task_mask(
        artifacts.classifier,
        artifacts.test_features,
        artifacts.test_task_ids.reshape(-1, 1),
        task_classes=artifacts.task_classes,
    )
    append_method(
        method="oracle_task_mask",
        policy_type="oracle_task_mask",
        selected_on="none",
        val_acc=_accuracy(artifacts.val_labels, val_oracle),
        y_pred=test_oracle,
        router_topk=artifacts.test_task_ids.reshape(-1, 1),
        actions=np.full(artifacts.test_labels.shape[0], ACTION_TOP1_MASK, dtype=object),
        mask_sizes=test_oracle_sizes,
    )

    append_method(
        method="hard_top1_linear",
        policy_type="hard_top1",
        selected_on="none",
        val_acc=_accuracy(artifacts.val_labels, val_predictions.top1_predictions),
        y_pred=test_predictions.top1_predictions,
        router_topk=test_top2[:, :1],
        actions=np.full(artifacts.test_labels.shape[0], ACTION_TOP1_MASK, dtype=object),
        mask_sizes=test_predictions.top1_mask_sizes,
    )

    for k in _topk_values(policy_values):
        val_topk = predict_topk_from_probabilities(artifacts.val_probabilities, artifacts.route_ids, k)
        test_topk = predict_topk_from_probabilities(
            artifacts.test_probabilities,
            artifacts.route_ids,
            k,
        )
        val_pred, _ = predict_prototype_with_topk_task_mask(
            artifacts.classifier,
            artifacts.val_features,
            val_topk,
            task_classes=artifacts.task_classes,
        )
        test_pred, test_sizes = predict_prototype_with_topk_task_mask(
            artifacts.classifier,
            artifacts.test_features,
            test_topk,
            task_classes=artifacts.task_classes,
        )
        append_method(
            method=f"topk_linear_k{k}",
            policy_type="topk",
            selected_on="val",
            val_acc=_accuracy(artifacts.val_labels, val_pred),
            y_pred=test_pred,
            router_topk=test_topk,
            actions=np.full(artifacts.test_labels.shape[0], ACTION_TOP2_MASK, dtype=object),
            mask_sizes=test_sizes,
        )

    fallback_threshold, fallback_val_acc = _select_fallback_tau(
        artifacts.val_labels,
        val_signals.confidence,
        val_predictions,
        _fallback_tau_grid(policy_values),
    )
    fallback_actions = np.where(
        test_signals.confidence < fallback_threshold,
        ACTION_FALLBACK,
        ACTION_TOP1_MASK,
    ).astype(object)
    fallback_test_pred, fallback_test_sizes = apply_actions(
        fallback_actions,
        test_predictions,
    )
    append_method(
        method="fallback_linear_tau",
        policy_type="fallback",
        selected_on="val",
        val_acc=fallback_val_acc,
        y_pred=fallback_test_pred,
        router_topk=test_top2[:, :1],
        actions=fallback_actions,
        mask_sizes=fallback_test_sizes,
        thresholds=PolicyThresholds(fallback_threshold, math.nan, math.nan, math.nan),
    )

    pairwise_actions = choose_pairwise_actions(test_signals, selected_thresholds)
    pairwise_pred, pairwise_sizes = apply_actions(pairwise_actions, test_predictions)
    pairwise_warning = "; ".join(sorted(set((*val_signals.warnings, *test_signals.warnings))))
    append_method(
        method="pairwise_policy_linear_combined",
        policy_type="pairwise_policy",
        selected_on="val",
        val_acc=float(selected_val_row["val_acc"]),
        y_pred=pairwise_pred,
        router_topk=test_top2,
        actions=pairwise_actions,
        mask_sizes=pairwise_sizes,
        thresholds=selected_thresholds,
        warning=pairwise_warning,
    )

    results = pd.DataFrame(rows, columns=PAIRWISE_RESULT_COLUMNS)
    _fill_gap_columns(results)
    previous_row = _previous_best_control_row(results)
    if previous_row is not None:
        results = pd.concat(
            [results, pd.DataFrame([previous_row], columns=PAIRWISE_RESULT_COLUMNS)],
            ignore_index=True,
        )
    per_task = pd.DataFrame(per_task_rows, columns=PAIRWISE_PER_TASK_COLUMNS)
    _fill_per_task_gains(per_task)
    pairwise_pairs = build_pairwise_policy_pairs(
        top2_routes=test_top2,
        signals=test_signals,
        actions=pairwise_actions,
        y_true=artifacts.test_labels,
        y_pred=pairwise_pred,
    )
    summary = summarize_pairwise_policy_results(
        results,
        per_task,
        pairwise_pairs,
        selected_thresholds=selected_thresholds,
        threshold_grid=threshold_grid,
        artifacts=artifacts,
        confidence_auc_val=val_confidence_auc,
        previous_best_fallback_acc=previous_best_fallback_acc,
        config=config or {},
        feature_bank_metadata=dataset.feature_bank_metadata,
        manifest_path=dataset.manifest_path,
    )
    return PairwisePolicyEvaluation(
        results=results,
        per_task=per_task,
        pairwise_pairs=pairwise_pairs,
        threshold_grid=threshold_grid,
        summary=summary,
    )


def build_pairwise_signals(
    top2_routes: np.ndarray,
    probabilities: np.ndarray,
    diagnostics: PairwiseDiagnosticsLookup,
    pair_confusion_prior: dict[tuple[int, int], float] | None = None,
) -> PairwiseSignals:
    routes = np.asarray(top2_routes, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float32)
    if routes.ndim != 2 or routes.shape[1] < 2:
        raise ValueError("top2_routes must have shape [N, >=2]")
    if probs.ndim != 2 or probs.shape[0] != routes.shape[0]:
        raise ValueError("probabilities and top2_routes must have matching N")
    confidence = np.max(probs, axis=1).astype(np.float32)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).astype(np.float32)
    prior_lookup = pair_confusion_prior or {}
    unique_pairs = {
        (int(row[0]), int(row[1])): _pair_key(int(row[0]), int(row[1]))
        for row in routes[:, :2]
    }
    diagnostic_by_ordered: dict[tuple[int, int], PairwiseDiagnosticsValue] = {}
    prior_by_ordered: dict[tuple[int, int], float] = {}
    warnings: set[str] = set()
    for ordered_pair, pair_key in unique_pairs.items():
        value = diagnostics.lookup(*ordered_pair)
        if value.warning:
            warnings.add(value.warning)
        diagnostic_by_ordered[ordered_pair] = value
        prior_by_ordered[ordered_pair] = float(prior_lookup.get(pair_key, 0.0))

    pair_auc = np.empty(routes.shape[0], dtype=np.float32)
    pair_separation = np.empty(routes.shape[0], dtype=np.float32)
    pair_prior = np.empty(routes.shape[0], dtype=np.float32)
    for idx, row in enumerate(routes[:, :2]):
        key = (int(row[0]), int(row[1]))
        value = diagnostic_by_ordered[key]
        pair_auc[idx] = float(value.pair_auc)
        pair_separation[idx] = float(value.pair_separation_ratio)
        pair_prior[idx] = float(prior_by_ordered[key])

    return PairwiseSignals(
        top2_routes=routes[:, :2],
        confidence=confidence,
        margin=margin,
        pair_auc=pair_auc,
        pair_separation_ratio=pair_separation,
        pair_confusion_prior=pair_prior,
        warnings=tuple(sorted(warnings)),
    )


def build_pair_confusion_prior(
    true_task_ids: np.ndarray,
    top1_task_ids: np.ndarray,
) -> dict[tuple[int, int], float]:
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    pred_values = np.asarray(top1_task_ids, dtype=np.int64)
    if true_values.ndim != 1 or pred_values.ndim != 1:
        raise ValueError("true_task_ids and top1_task_ids must be rank 1")
    if true_values.shape[0] != pred_values.shape[0]:
        raise ValueError("true_task_ids and top1_task_ids must have matching N")
    task_ids = sorted(int(value) for value in np.unique(np.concatenate([true_values, pred_values])))
    row_totals = {task_id: int(np.sum(true_values == task_id)) for task_id in task_ids}
    directed: dict[tuple[int, int], float] = {}
    for true_task in task_ids:
        denom = row_totals[true_task]
        for pred_task in task_ids:
            if true_task == pred_task:
                continue
            count = int(np.sum((true_values == true_task) & (pred_values == pred_task)))
            directed[(true_task, pred_task)] = 0.0 if denom == 0 else count / denom
    symmetric: dict[tuple[int, int], float] = {}
    for idx, task_i in enumerate(task_ids):
        for task_j in task_ids[idx + 1 :]:
            symmetric[(task_i, task_j)] = float(
                directed.get((task_i, task_j), 0.0) + directed.get((task_j, task_i), 0.0)
            )
    return symmetric


def choose_pairwise_actions(
    signals: PairwiseSignals,
    thresholds: PolicyThresholds,
) -> np.ndarray:
    confidence = np.asarray(signals.confidence, dtype=np.float32)
    margin = np.asarray(signals.margin, dtype=np.float32)
    pair_auc = np.asarray(signals.pair_auc, dtype=np.float32)
    prior = np.asarray(signals.pair_confusion_prior, dtype=np.float32)
    _validate_same_length(confidence, margin, pair_auc, prior)

    actions = np.full(confidence.shape[0], ACTION_TOP1_MASK, dtype=object)
    fallback_mask = confidence < float(thresholds.tau_fallback)
    risky_pair = (pair_auc < float(thresholds.tau_pair_auc)) | (
        prior > float(thresholds.tau_pair_confusion)
    )
    top2_mask = (~fallback_mask) & (margin < float(thresholds.tau_margin)) & risky_pair
    actions[top2_mask] = ACTION_TOP2_MASK
    actions[fallback_mask] = ACTION_FALLBACK
    return actions


def apply_actions(
    actions: np.ndarray,
    predictions: SplitPredictions,
) -> tuple[np.ndarray, np.ndarray]:
    action_values = np.asarray(actions, dtype=object)
    out = predictions.top1_predictions.copy()
    mask_sizes = predictions.top1_mask_sizes.copy()
    top2_mask = action_values == ACTION_TOP2_MASK
    fallback_mask = action_values == ACTION_FALLBACK
    out[top2_mask] = predictions.top2_predictions[top2_mask]
    out[fallback_mask] = predictions.merge_predictions[fallback_mask]
    mask_sizes[top2_mask] = predictions.top2_mask_sizes[top2_mask]
    mask_sizes[fallback_mask] = predictions.full_mask_sizes[fallback_mask]
    return out.astype(np.int64), mask_sizes.astype(np.float32)


def evaluate_threshold_grid(
    *,
    signals: PairwiseSignals,
    predictions: SplitPredictions,
    labels: np.ndarray,
    thresholds: Sequence[PolicyThresholds],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_values = np.asarray(labels, dtype=np.int64)
    for candidate in thresholds:
        actions = choose_pairwise_actions(signals, candidate)
        y_pred, mask_sizes = apply_actions(actions, predictions)
        action_summary = compute_action_summary(actions, mask_sizes)
        rows.append(
            {
                "method": "pairwise_policy_linear_combined",
                "tau_fallback": candidate.tau_fallback,
                "tau_margin": candidate.tau_margin,
                "tau_pair_auc": candidate.tau_pair_auc,
                "tau_pair_confusion": candidate.tau_pair_confusion,
                "val_acc": _accuracy(label_values, y_pred),
                "mean_mask_size": action_summary["mean_mask_size"],
                "action_top1_rate": action_summary["action_top1_rate"],
                "action_top2_rate": action_summary["action_top2_rate"],
                "action_fallback_rate": action_summary["action_fallback_rate"],
                "status": "ok",
                "warning": "",
            }
        )
    return pd.DataFrame(rows, columns=THRESHOLD_GRID_COLUMNS)


def select_best_thresholds(grid: pd.DataFrame) -> PolicyThresholds:
    required = {
        "val_acc",
        "mean_mask_size",
        "action_fallback_rate",
        "tau_fallback",
        "tau_margin",
        "tau_pair_auc",
        "tau_pair_confusion",
    }
    missing = sorted(required.difference(grid.columns))
    if missing:
        raise ValueError(f"threshold grid missing required columns: {missing}")
    candidates = grid.copy()
    if "status" in candidates:
        candidates = candidates[candidates["status"] == "ok"]
    if candidates.empty:
        raise ValueError("threshold grid has no ok candidates")
    best = candidates.sort_values(
        [
            "val_acc",
            "mean_mask_size",
            "action_fallback_rate",
            "tau_fallback",
            "tau_margin",
            "tau_pair_auc",
            "tau_pair_confusion",
        ],
        ascending=[False, True, True, True, True, True, True],
        kind="mergesort",
    ).iloc[0]
    return PolicyThresholds(
        tau_fallback=float(best["tau_fallback"]),
        tau_margin=float(best["tau_margin"]),
        tau_pair_auc=float(best["tau_pair_auc"]),
        tau_pair_confusion=float(best["tau_pair_confusion"]),
    )


def compute_action_summary(actions: np.ndarray, mask_sizes: np.ndarray) -> dict[str, float]:
    action_values = np.asarray(actions, dtype=object)
    mask_values = np.asarray(mask_sizes, dtype=np.float32)
    if action_values.ndim != 1 or mask_values.ndim != 1:
        raise ValueError("actions and mask_sizes must be rank 1")
    if action_values.shape[0] != mask_values.shape[0]:
        raise ValueError("actions and mask_sizes must have matching N")
    return {
        "action_top1_rate": _action_rate(action_values, ACTION_TOP1_MASK),
        "action_top2_rate": _action_rate(action_values, ACTION_TOP2_MASK),
        "action_fallback_rate": _action_rate(action_values, ACTION_FALLBACK),
        "mean_mask_size": float(np.mean(mask_values)) if mask_values.shape[0] else math.nan,
    }


def rows_from_predictions(
    *,
    method: str,
    policy_type: str,
    router_type: str,
    selected_on: str,
    val_acc: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_task_ids: np.ndarray,
    router_topk: np.ndarray | None,
    actions: np.ndarray | None,
    mask_sizes: np.ndarray,
    confidence_auc_for_correct_route: float,
    thresholds: PolicyThresholds | None = None,
    warning: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_true_values = np.asarray(y_true, dtype=np.int64)
    y_pred_values = np.asarray(y_pred, dtype=np.int64)
    task_values = np.asarray(true_task_ids, dtype=np.int64)
    mask_values = np.asarray(mask_sizes, dtype=np.float32)
    if actions is None:
        action_values = np.full(y_true_values.shape[0], "", dtype=object)
    else:
        action_values = np.asarray(actions, dtype=object)
    _validate_same_length(y_true_values, y_pred_values, task_values, mask_values, action_values)
    task_rows = _per_task_rows(
        method=method,
        y_true=y_true_values,
        y_pred=y_pred_values,
        true_task_ids=task_values,
        router_topk=router_topk,
        actions=action_values,
        mask_sizes=mask_values,
    )
    action_summary = compute_action_summary(action_values, mask_values)
    last_task_id = max(int(row["task_id"]) for row in task_rows)
    last_task_acc = next(
        float(row["task_acc"]) for row in task_rows if int(row["task_id"]) == last_task_id
    )
    top1 = None if router_topk is None else _topk_values_array(router_topk)[:, 0]
    threshold_values = thresholds or PolicyThresholds(math.nan, math.nan, math.nan, math.nan)
    row = {
        "method": method,
        "policy_type": policy_type,
        "router_type": router_type,
        "selected_on": selected_on,
        "val_acc": val_acc,
        "overall_acc": _accuracy(y_true_values, y_pred_values),
        "mean_task_acc": float(np.mean([task_row["task_acc"] for task_row in task_rows])),
        "last_task_acc": last_task_acc,
        "route_vs_merge_gap": math.nan,
        "oracle_gap_closure": math.nan,
        "tau_fallback": threshold_values.tau_fallback,
        "tau_margin": threshold_values.tau_margin,
        "tau_pair_auc": threshold_values.tau_pair_auc,
        "tau_pair_confusion": threshold_values.tau_pair_confusion,
        "action_top1_rate": action_summary["action_top1_rate"],
        "action_top2_rate": action_summary["action_top2_rate"],
        "action_fallback_rate": action_summary["action_fallback_rate"],
        "mean_mask_size": action_summary["mean_mask_size"],
        "router_top1_acc": math.nan if top1 is None else _accuracy(task_values, top1),
        "router_top2_recall": _router_topk_recall(router_topk, task_values),
        "confidence_auc_for_correct_route": confidence_auc_for_correct_route,
        "num_test_samples": int(y_true_values.shape[0]),
        "status": "ok",
        "warning": warning,
    }
    return row, task_rows


def build_pairwise_policy_pairs(
    *,
    top2_routes: np.ndarray,
    signals: PairwiseSignals,
    actions: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    routes = np.asarray(top2_routes, dtype=np.int64)
    action_values = np.asarray(actions, dtype=object)
    y_true_values = np.asarray(y_true, dtype=np.int64)
    y_pred_values = np.asarray(y_pred, dtype=np.int64)
    _validate_same_length(routes[:, 0], action_values, y_true_values, y_pred_values)
    rows: list[dict[str, Any]] = []
    for top1_task, top2_task in sorted({(int(row[0]), int(row[1])) for row in routes[:, :2]}):
        mask = (routes[:, 0] == top1_task) & (routes[:, 1] == top2_task)
        rows.append(
            {
                "top1_task": top1_task,
                "top2_task": top2_task,
                "num_samples": int(np.sum(mask)),
                "pair_auc": _masked_mean(signals.pair_auc, mask),
                "pair_separation_ratio": _masked_mean(signals.pair_separation_ratio, mask),
                "pair_confusion_prior": _masked_mean(signals.pair_confusion_prior, mask),
                "mean_margin": _masked_mean(signals.margin, mask),
                "mean_confidence": _masked_mean(signals.confidence, mask),
                "action_top1_rate": _action_rate(action_values[mask], ACTION_TOP1_MASK),
                "action_top2_rate": _action_rate(action_values[mask], ACTION_TOP2_MASK),
                "action_fallback_rate": _action_rate(action_values[mask], ACTION_FALLBACK),
                "accuracy_when_pair_appears": _accuracy(y_true_values[mask], y_pred_values[mask]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=PAIRWISE_PAIR_COLUMNS)
    return pd.DataFrame(rows, columns=PAIRWISE_PAIR_COLUMNS).sort_values(
        ["num_samples", "action_top2_rate", "action_fallback_rate"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def write_pairwise_policy_outputs(
    evaluation: PairwisePolicyEvaluation,
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
        "pairwise_policy_pairs_csv": output_root / f"{prefix}_pairwise_policy_pairs.csv",
        "summary_json": output_root / f"{prefix}_summary.json",
        "threshold_grid_csv": output_root / f"{prefix}_threshold_grid.csv",
    }
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Output already exists: {joined}. Pass --overwrite to replace.")

    evaluation.results.to_csv(paths["results_csv"], index=False)
    evaluation.per_task.to_csv(paths["per_task_csv"], index=False)
    evaluation.pairwise_pairs.to_csv(paths["pairwise_policy_pairs_csv"], index=False)
    evaluation.threshold_grid.to_csv(paths["threshold_grid_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(_json_safe(evaluation.summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def compute_oracle_gap_closure(overall_acc: float, merge_acc: float, oracle_acc: float) -> float:
    denom = float(oracle_acc) - float(merge_acc)
    if abs(denom) <= 1e-12:
        return math.nan
    return (float(overall_acc) - float(merge_acc)) / denom


def compute_confidence_auc_for_correct_route(
    confidence: np.ndarray,
    route_correct: np.ndarray,
) -> float:
    scores = np.asarray(confidence, dtype=np.float64)
    labels = np.asarray(route_correct, dtype=bool)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError("confidence and route_correct must be rank 1 with matching N")
    positives = int(np.sum(labels))
    negatives = int(labels.shape[0] - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = _rank_average(scores)
    positive_rank_sum = float(np.sum(ranks[labels]))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return float(auc)


def summarize_pairwise_policy_results(
    results: pd.DataFrame,
    per_task: pd.DataFrame,
    pairwise_pairs: pd.DataFrame,
    *,
    selected_thresholds: PolicyThresholds,
    threshold_grid: pd.DataFrame,
    artifacts: PairwisePolicyArtifacts,
    confidence_auc_val: float,
    previous_best_fallback_acc: float,
    config: dict[str, Any],
    feature_bank_metadata: dict[str, Any],
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    pairwise = _method_brief(results, "pairwise_policy_linear_combined")
    fallback = _method_brief(results, "fallback_linear_tau")
    topk_k2 = _method_brief(results, "topk_linear_k2")
    merge = _method_brief(results, "merge_all_prototype")
    oracle = _method_brief(results, "oracle_task_mask")
    best_learned = _best_learned_control(results)
    best_task_gains = _task_gain_briefs(per_task, "pairwise_policy_linear_combined", ascending=False)
    worst_task_gains = _task_gain_briefs(per_task, "pairwise_policy_linear_combined", ascending=True)
    top_pairs = [
        _pair_brief(row)
        for row in pairwise_pairs.sort_values(
            ["action_top2_rate", "action_fallback_rate", "num_samples"],
            ascending=[False, False, False],
        )
        .head(10)
        .to_dict(orient="records")
    ]
    pairwise_acc = math.nan if pairwise is None else float(pairwise["overall_acc"])
    fallback_acc = math.nan if fallback is None else float(fallback["overall_acc"])
    fallback_mask = math.nan if fallback is None else float(fallback["mean_mask_size"])
    pairwise_mask = math.nan if pairwise is None else float(pairwise["mean_mask_size"])
    return {
        "selected_thresholds": asdict(selected_thresholds),
        "selected_temperature": artifacts.selected_temperature,
        "score_source": artifacts.score_source,
        "router_calibration_metrics": {
            "val": asdict(artifacts.calibration_metrics_val),
            "test": asdict(artifacts.calibration_metrics_test),
            "confidence_auc_for_correct_route_val": confidence_auc_val,
        },
        "merge_all": merge,
        "oracle_task_mask": oracle,
        "fallback_linear_tau": fallback,
        "topk_linear_k2": topk_k2,
        "pairwise_policy_linear_combined": pairwise,
        "best_learned_control": best_learned,
        "previous_best_fallback_acc": float(previous_best_fallback_acc),
        "pairwise_exceeds_previous_best_fallback": bool(
            _is_finite(pairwise_acc) and pairwise_acc > previous_best_fallback_acc
        ),
        "pairwise_exceeds_current_fallback": bool(
            _is_finite(pairwise_acc) and _is_finite(fallback_acc) and pairwise_acc > fallback_acc
        ),
        "pairwise_lowers_mean_mask_vs_fallback": bool(
            _is_finite(pairwise_mask) and _is_finite(fallback_mask) and pairwise_mask < fallback_mask
        ),
        "best_task_gains_vs_merge": best_task_gains,
        "worst_task_gains_vs_merge": worst_task_gains,
        "top_pairwise_policy_pairs": top_pairs,
        "num_threshold_candidates": int(threshold_grid.shape[0]),
        "num_methods": int(results.shape[0]),
        "num_failed_methods": int((results["status"] != "ok").sum()),
        "failed_methods": results.loc[results["status"] != "ok", "method"].tolist(),
        "has_nan_overall_acc": bool(results["overall_acc"].isna().any()),
        "config_hash": stable_hash(config),
        "feature_bank_metadata": feature_bank_metadata,
        "manifest_path": str(manifest_path) if manifest_path else None,
    }


def _fit_artifacts(
    tasks: Sequence[TaskFeatureSplit],
    *,
    router_config: dict[str, Any] | None,
    calibration_config: dict[str, Any] | None,
) -> PairwisePolicyArtifacts:
    router_values = dict(router_config or {})
    if str(router_values.get("type", "linear")) != "linear":
        raise ValueError("pairwise_policy_v0 currently supports router.type=linear only")
    train_features, train_labels, train_task_ids = stack_task_split(tasks, "train")
    val_features, val_labels, val_task_ids = stack_task_split(tasks, "val")
    test_features, test_labels, test_task_ids = stack_task_split(tasks, "test")
    classifier = PrototypeClassifier(
        expected_classes=sorted(int(value) for value in np.unique(train_labels))
    ).fit(train_features, train_labels)
    router = SklearnLinearRouter(
        max_iter=int(router_values.get("max_iter", 1000)),
        class_weight=router_values.get("class_weight", "balanced"),
        solver=str(router_values.get("solver", "lbfgs")),
        max_train_per_task=router_values.get("max_train_per_task"),
        random_state=int(router_values.get("random_state", 0)),
    ).fit(train_features, train_task_ids)
    val_payload = extract_linear_router_scores(router, val_features)
    test_payload = extract_linear_router_scores(router, test_features)
    temperature = _selected_temperature(
        router_values=router_values,
        calibration_config=calibration_config,
        val_scores=val_payload.scores,
        val_task_ids=val_task_ids,
        route_ids=val_payload.route_ids,
    )
    ece_bins = int(dict(calibration_config or {}).get("ece_bins", 15))
    val_probabilities = temperature_scaled_probabilities(val_payload.scores, temperature)
    test_probabilities = temperature_scaled_probabilities(test_payload.scores, temperature)
    return PairwisePolicyArtifacts(
        classifier=classifier,
        router=router,
        task_classes=_task_classes(tasks),
        val_features=val_features,
        val_labels=val_labels,
        val_task_ids=val_task_ids,
        test_features=test_features,
        test_labels=test_labels,
        test_task_ids=test_task_ids,
        route_ids=val_payload.route_ids,
        score_source=val_payload.score_source,
        selected_temperature=temperature,
        val_probabilities=val_probabilities,
        test_probabilities=test_probabilities,
        calibration_metrics_val=compute_calibration_metrics(
            val_probabilities,
            val_task_ids,
            val_payload.route_ids,
            n_bins=ece_bins,
        ),
        calibration_metrics_test=compute_calibration_metrics(
            test_probabilities,
            test_task_ids,
            test_payload.route_ids,
            n_bins=ece_bins,
        ),
    )


def _split_predictions(
    classifier: PrototypeClassifier,
    features: np.ndarray,
    top2_routes: np.ndarray,
    *,
    task_classes: dict[int, tuple[int, ...]],
) -> SplitPredictions:
    merge_predictions = classifier.predict(features)
    top1_predictions, top1_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        features,
        top2_routes[:, :1],
        task_classes=task_classes,
    )
    top2_predictions, top2_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        features,
        top2_routes[:, :2],
        task_classes=task_classes,
    )
    full_mask_sizes = np.full(
        merge_predictions.shape[0],
        classifier.classes_.shape[0],
        dtype=np.float32,
    )
    return SplitPredictions(
        merge_predictions=merge_predictions,
        top1_predictions=top1_predictions,
        top1_mask_sizes=top1_mask_sizes,
        top2_predictions=top2_predictions,
        top2_mask_sizes=top2_mask_sizes,
        full_mask_sizes=full_mask_sizes,
    )


def _threshold_candidates(policy_config: dict[str, Any]) -> list[PolicyThresholds]:
    fallback = _config_floats(
        policy_config,
        "tau_fallback_grid",
        default=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8),
    )
    margin = _config_floats(
        policy_config,
        "tau_margin_grid",
        default=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3),
    )
    pair_auc = _config_floats(
        policy_config,
        "tau_pair_auc_grid",
        default=(0.90, 0.93, 0.95, 0.97, 0.99),
    )
    pair_confusion = _config_floats(
        policy_config,
        "tau_pair_confusion_grid",
        default=(0.02, 0.04, 0.06, 0.08),
    )
    return [
        PolicyThresholds(
            tau_fallback=tau_fallback,
            tau_margin=tau_margin,
            tau_pair_auc=tau_pair_auc,
            tau_pair_confusion=tau_pair_confusion,
        )
        for tau_fallback in fallback
        for tau_margin in margin
        for tau_pair_auc in pair_auc
        for tau_pair_confusion in pair_confusion
    ]


def _selected_grid_row(grid: pd.DataFrame, thresholds: PolicyThresholds) -> pd.Series:
    mask = (
        np.isclose(grid["tau_fallback"].astype(float), thresholds.tau_fallback)
        & np.isclose(grid["tau_margin"].astype(float), thresholds.tau_margin)
        & np.isclose(grid["tau_pair_auc"].astype(float), thresholds.tau_pair_auc)
        & np.isclose(grid["tau_pair_confusion"].astype(float), thresholds.tau_pair_confusion)
    )
    if not bool(mask.any()):
        raise ValueError("selected thresholds not found in grid")
    return grid.loc[mask].iloc[0]


def _select_fallback_tau(
    labels: np.ndarray,
    confidence: np.ndarray,
    predictions: SplitPredictions,
    candidates: Sequence[float],
) -> tuple[float, float]:
    label_values = np.asarray(labels, dtype=np.int64)
    best_tau = float(candidates[0])
    best_acc = -1.0
    best_fallback_rate = math.inf
    for tau in candidates:
        actions = np.where(confidence < float(tau), ACTION_FALLBACK, ACTION_TOP1_MASK).astype(
            object
        )
        y_pred, _ = apply_actions(actions, predictions)
        acc = _accuracy(label_values, y_pred)
        fallback_rate = _action_rate(actions, ACTION_FALLBACK)
        if acc > best_acc + 1e-12 or (
            abs(acc - best_acc) <= 1e-12 and fallback_rate < best_fallback_rate
        ):
            best_tau = float(tau)
            best_acc = acc
            best_fallback_rate = fallback_rate
    return best_tau, best_acc


def _per_task_rows(
    *,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_task_ids: np.ndarray,
    router_topk: np.ndarray | None,
    actions: np.ndarray,
    mask_sizes: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    router_values = None if router_topk is None else _topk_values_array(router_topk)
    for task_id in sorted(int(value) for value in np.unique(true_task_ids)):
        mask = true_task_ids == task_id
        task_actions = actions[mask]
        rows.append(
            {
                "method": method,
                "task_id": task_id,
                "task_acc": _accuracy(y_true[mask], y_pred[mask]),
                "num_test_samples": int(np.sum(mask)),
                "action_top1_rate_task": _action_rate(task_actions, ACTION_TOP1_MASK),
                "action_top2_rate_task": _action_rate(task_actions, ACTION_TOP2_MASK),
                "action_fallback_rate_task": _action_rate(task_actions, ACTION_FALLBACK),
                "mean_mask_size_task": float(np.mean(mask_sizes[mask])),
                "router_top1_acc_task": math.nan
                if router_values is None
                else _accuracy(true_task_ids[mask], router_values[mask, 0]),
                "router_top2_recall_task": math.nan
                if router_values is None
                else _router_topk_recall(router_values[mask], true_task_ids[mask]),
                "gain_vs_merge_task": math.nan,
                "gain_vs_hard_top1_task": math.nan,
            }
        )
    return rows


def _fill_gap_columns(results: pd.DataFrame) -> None:
    merge_acc = _method_value(results, "merge_all_prototype", "overall_acc")
    oracle_acc = _method_value(results, "oracle_task_mask", "overall_acc")
    if merge_acc is None:
        return
    results.loc[:, "route_vs_merge_gap"] = results["overall_acc"].astype(float) - merge_acc
    if oracle_acc is not None:
        results.loc[:, "oracle_gap_closure"] = [
            compute_oracle_gap_closure(value, merge_acc, oracle_acc)
            for value in results["overall_acc"].astype(float)
        ]


def _fill_per_task_gains(per_task: pd.DataFrame) -> None:
    merge = _task_acc_lookup(per_task, "merge_all_prototype")
    hard = _task_acc_lookup(per_task, "hard_top1_linear")
    for idx, row in per_task.iterrows():
        task_id = int(row["task_id"])
        task_acc = float(row["task_acc"])
        if task_id in merge:
            per_task.at[idx, "gain_vs_merge_task"] = task_acc - merge[task_id]
        if task_id in hard:
            per_task.at[idx, "gain_vs_hard_top1_task"] = task_acc - hard[task_id]


def _previous_best_control_row(results: pd.DataFrame) -> dict[str, Any] | None:
    candidates = results[
        results["method"].str.startswith("topk_linear_")
        | results["method"].isin(["hard_top1_linear", "fallback_linear_tau"])
    ]
    if candidates.empty:
        return None
    best = candidates.sort_values(
        ["overall_acc", "mean_mask_size", "action_fallback_rate"],
        ascending=[False, True, True],
    ).iloc[0]
    row = best.to_dict()
    row["method"] = "previous_best_learned_control"
    row["policy_type"] = "previous_best_learned_control"
    row["warning"] = f"copied_from={best['method']}"
    return row


def _task_acc_lookup(per_task: pd.DataFrame, method: str) -> dict[int, float]:
    rows = per_task[per_task["method"] == method]
    return {int(row["task_id"]): float(row["task_acc"]) for row in rows.to_dict(orient="records")}


def _load_diagnostics(path: str | Path | None) -> pd.DataFrame | None:
    if path in (None, ""):
        return None
    value = Path(path)
    if not value.exists():
        return None
    return pd.read_csv(value)


def _selected_temperature(
    *,
    router_values: dict[str, Any],
    calibration_config: dict[str, Any] | None,
    val_scores: np.ndarray,
    val_task_ids: np.ndarray,
    route_ids: np.ndarray,
) -> float:
    if not bool(router_values.get("use_temperature", True)):
        return 1.0
    requested = router_values.get("selected_temperature", "auto")
    if requested not in (None, "", "auto"):
        return float(requested)
    calibration = dict(calibration_config or {})
    if not bool(calibration.get("enabled", True)):
        return 1.0
    temperatures = calibration.get(
        "temperatures",
        [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    )
    ece_bins = int(calibration.get("ece_bins", 15))
    return float(
        select_temperature(
            val_scores,
            val_task_ids,
            route_ids,
            temperatures,
            n_bins=ece_bins,
        ).temperature
    )


def _task_classes(tasks: Sequence[TaskFeatureSplit]) -> dict[int, tuple[int, ...]]:
    return {
        task.task_id: task.classes or tuple(int(value) for value in np.unique(task.train_labels))
        for task in tasks
    }


def _topk_values(policy_config: dict[str, Any]) -> list[int]:
    values = policy_config.get("topk_controls", [2, 3, 4, 5, 7, 10])
    out: list[int] = []
    for value in values:
        number = int(value)
        if number > 0 and number not in out:
            out.append(number)
    if 2 not in out:
        out.insert(0, 2)
    return out


def _fallback_tau_grid(policy_config: dict[str, Any]) -> list[float]:
    return _config_floats(
        policy_config,
        "tau_fallback_grid",
        default=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8),
    )


def _config_floats(
    config: dict[str, Any] | None,
    key: str,
    *,
    default: Sequence[float],
) -> list[float]:
    values = default if config is None else config.get(key, default)
    out: list[float] = []
    for value in values:
        number = float(value)
        if not any(abs(number - existing) <= 1e-12 for existing in out):
            out.append(number)
    if not out:
        raise ValueError(f"{key} must contain at least one value")
    return out


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


def _method_brief(results: pd.DataFrame, method: str) -> dict[str, Any] | None:
    rows = results[results["method"] == method]
    if rows.empty:
        return None
    return _result_brief(rows.iloc[0].to_dict())


def _best_learned_control(results: pd.DataFrame) -> dict[str, Any] | None:
    controls = results[
        results["method"].str.startswith("topk_linear_")
        | results["method"].isin(["hard_top1_linear", "fallback_linear_tau"])
    ]
    if controls.empty:
        return None
    return _result_brief(
        controls.sort_values("overall_acc", ascending=False).iloc[0].to_dict()
    )


def _result_brief(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "method",
        "policy_type",
        "router_type",
        "selected_on",
        "val_acc",
        "overall_acc",
        "mean_task_acc",
        "last_task_acc",
        "route_vs_merge_gap",
        "oracle_gap_closure",
        "tau_fallback",
        "tau_margin",
        "tau_pair_auc",
        "tau_pair_confusion",
        "action_top1_rate",
        "action_top2_rate",
        "action_fallback_rate",
        "mean_mask_size",
        "router_top1_acc",
        "router_top2_recall",
        "confidence_auc_for_correct_route",
        "warning",
    ]
    return {key: _json_safe(row.get(key)) for key in keys if key in row}


def _task_gain_briefs(per_task: pd.DataFrame, method: str, *, ascending: bool) -> list[dict[str, Any]]:
    rows = per_task[per_task["method"] == method]
    if rows.empty:
        return []
    ordered = rows.sort_values("gain_vs_merge_task", ascending=ascending).head(5)
    return [
        {
            "task_id": int(row["task_id"]),
            "task_acc": float(row["task_acc"]),
            "gain_vs_merge_task": float(row["gain_vs_merge_task"]),
            "gain_vs_hard_top1_task": float(row["gain_vs_hard_top1_task"]),
        }
        for row in ordered.to_dict(orient="records")
    ]


def _pair_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "top1_task": int(row["top1_task"]),
        "top2_task": int(row["top2_task"]),
        "num_samples": int(row["num_samples"]),
        "pair_auc": _json_safe(row["pair_auc"]),
        "pair_confusion_prior": _json_safe(row["pair_confusion_prior"]),
        "mean_margin": _json_safe(row["mean_margin"]),
        "mean_confidence": _json_safe(row["mean_confidence"]),
        "action_top1_rate": _json_safe(row["action_top1_rate"]),
        "action_top2_rate": _json_safe(row["action_top2_rate"]),
        "action_fallback_rate": _json_safe(row["action_fallback_rate"]),
        "accuracy_when_pair_appears": _json_safe(row["accuracy_when_pair_appears"]),
    }


def _topk_values_array(router_topk: np.ndarray) -> np.ndarray:
    values = np.asarray(router_topk, dtype=np.int64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def _router_topk_recall(router_topk: np.ndarray | None, true_task_ids: np.ndarray) -> float:
    if router_topk is None:
        return math.nan
    values = _topk_values_array(router_topk)
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    return float(np.mean(np.any(values == true_values[:, None], axis=1)))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values = np.asarray(y_true, dtype=np.int64)
    pred_values = np.asarray(y_pred, dtype=np.int64)
    if true_values.shape[0] == 0:
        return math.nan
    return float(np.mean(true_values == pred_values))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    value_array = np.asarray(values, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    if not np.any(mask_array):
        return math.nan
    finite_values = value_array[mask_array]
    if finite_values.shape[0] == 0:
        return math.nan
    if np.isnan(finite_values).all():
        return math.nan
    return float(np.nanmean(finite_values))


def _action_rate(actions: np.ndarray, action: str) -> float:
    action_values = np.asarray(actions, dtype=object)
    if action_values.shape[0] == 0:
        return math.nan
    return float(np.mean(action_values == action))


def _first_finite(row: dict[str, Any], columns: Sequence[str], default: float) -> float:
    for column in columns:
        if column not in row:
            continue
        try:
            number = float(row[column])
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return float(default)


def _pair_key(task_i: int, task_j: int) -> tuple[int, int]:
    left = int(task_i)
    right = int(task_j)
    return (left, right) if left <= right else (right, left)


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _validate_same_length(*arrays: np.ndarray) -> None:
    lengths = [np.asarray(array).shape[0] for array in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"arrays must have matching first dimension, got {lengths}")


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
