"""Router error anatomy analyses for frozen-feature routing baselines."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mrb.baselines.calibration import (
    compute_calibration_metrics,
    extract_linear_router_scores,
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


BUCKET_ORDER = (
    "top1_correct",
    "in_top2_not_top1",
    "in_top3_not_top2",
    "in_top5_not_top3",
    "not_in_top5",
)

REQUIRED_SUMMARY_KEYS = (
    "router_type",
    "top1_acc",
    "top2_recall",
    "top3_recall",
    "top5_recall",
    "mean_confidence_correct",
    "mean_confidence_wrong",
    "confidence_auc_for_correct_route",
    "best_task_by_router_acc",
    "worst_task_by_router_acc",
    "most_confused_pair",
    "merge_all_acc",
    "oracle_task_mask_acc",
    "hard_route_acc",
    "topk_k2_acc",
    "fallback_acc",
    "oracle_gap",
    "hard_route_gap",
    "topk_k2_gap",
    "fallback_gap",
    "diagnostics_confusion_correlations",
    "interpretation",
)


@dataclass(frozen=True)
class RouterErrorAnatomyEvaluation:
    confusion_counts: pd.DataFrame
    confusion_rates: pd.DataFrame
    symmetric_confusion: pd.DataFrame
    top_confusions: pd.DataFrame
    per_task_reliability: pd.DataFrame
    confidence_bins: pd.DataFrame
    confidence_summary: dict[str, Any]
    topk_error_buckets: pd.DataFrame
    diagnostics_confusion_pairs: pd.DataFrame
    diagnostics_confusion_summary: dict[str, Any]
    task_gap_decomposition: pd.DataFrame
    sample_level: pd.DataFrame | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class MethodOutputs:
    predictions: np.ndarray
    mask_sizes: np.ndarray
    fallback_flags: np.ndarray | None = None


def evaluate_router_error_anatomy(
    dataset: FeatureToyDataset,
    *,
    router_config: dict[str, Any] | None = None,
    calibration_config: dict[str, Any] | None = None,
    fallback_config: dict[str, Any] | None = None,
    analysis_config: dict[str, Any] | None = None,
    diagnostics_csv: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> RouterErrorAnatomyEvaluation:
    """Run Router Error Anatomy v0 on an already-loaded frozen feature dataset."""

    analysis = dict(analysis_config or {})
    router_values = dict(router_config or {})
    if str(router_values.get("type", "linear")) != "linear":
        raise ValueError("router_error_anatomy_v0 currently supports router.type=linear only")

    tasks = list(dataset.tasks)
    task_classes = _task_classes(tasks)
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
    selected_temperature = _selected_temperature(
        router_values=router_values,
        calibration_config=calibration_config,
        val_scores=val_payload.scores,
        val_task_ids=val_task_ids,
        route_ids=val_payload.route_ids,
    )
    val_probabilities = temperature_scaled_probabilities(
        val_payload.scores,
        selected_temperature,
    )
    test_probabilities = temperature_scaled_probabilities(
        test_payload.scores,
        selected_temperature,
    )
    route_ids = np.asarray(test_payload.route_ids, dtype=np.int64)
    topk_values = _analysis_topk_values(analysis)
    topk_routes = {
        k: predict_topk_from_probabilities(test_probabilities, route_ids, k)
        for k in topk_values
    }
    val_top1 = predict_topk_from_probabilities(val_probabilities, route_ids, 1)
    top1_routes = topk_routes[1]
    confidence = np.asarray(test_probabilities.max(axis=1), dtype=np.float32)
    route_correct = top1_routes[:, 0] == test_task_ids

    outputs = _method_outputs(
        classifier=classifier,
        task_classes=task_classes,
        test_features=test_features,
        test_labels=test_labels,
        test_task_ids=test_task_ids,
        val_features=val_features,
        val_labels=val_labels,
        val_top1=val_top1,
        val_confidence=np.asarray(val_probabilities.max(axis=1), dtype=np.float32),
        test_confidence=confidence,
        topk_routes=topk_routes,
        fallback_config=fallback_config,
    )
    per_task_accs = {
        name: _per_task_accuracy(test_labels, method.predictions, test_task_ids)
        for name, method in outputs.items()
    }
    confusion_counts = build_confusion_count_matrix(test_task_ids, top1_routes[:, 0], route_ids)
    confusion_rates = build_confusion_rate_matrix(confusion_counts)
    symmetric_confusion = build_symmetric_confusion_matrix(confusion_rates)
    diagnostics = _load_diagnostics(diagnostics_csv)
    diagnostics_pairs, diagnostics_summary = build_diagnostics_confusion_pairs(
        confusion_rates,
        diagnostics,
        route_ids,
    )
    diagnostics_lookup = _diagnostics_lookup(diagnostics_pairs)
    top_confusions = build_top_confusions_table(
        confusion_counts,
        confusion_rates,
        per_task_accs,
        diagnostics_lookup,
    )
    confidence_summary = build_confidence_summary(
        test_probabilities,
        test_task_ids,
        route_ids,
        confidence,
        route_correct,
        n_bins=int(analysis.get("confidence_bins", 10)),
    )
    confidence_bins = build_confidence_bin_table(
        confidence,
        route_correct,
        test_labels == outputs["hard_linear_route"].predictions,
        outputs["fallback"].fallback_flags,
        test_labels == outputs["fallback"].predictions,
        test_labels == outputs["merge_all"].predictions,
        n_bins=int(analysis.get("confidence_bins", 10)),
    )
    topk_error_buckets = build_topk_error_bucket_table(
        assign_topk_error_buckets(topk_routes[5], test_task_ids),
        test_labels,
        outputs,
        confidence,
        total_samples=test_labels.shape[0],
    )
    per_task_reliability = build_per_task_reliability_table(
        test_task_ids,
        test_labels,
        route_correct,
        topk_routes,
        confidence,
        outputs,
    )
    task_gap_decomposition = build_task_gap_decomposition_table(per_task_accs)
    sample_level = (
        build_sample_level_table(
            test_labels,
            test_task_ids,
            topk_routes,
            confidence,
            outputs,
        )
        if bool(analysis.get("save_sample_level", False))
        else None
    )
    summary = build_router_error_anatomy_summary(
        router_type="linear",
        topk_routes=topk_routes,
        true_task_ids=test_task_ids,
        confidence_summary=confidence_summary,
        per_task_reliability=per_task_reliability,
        top_confusions=top_confusions,
        per_task_accs=per_task_accs,
        diagnostics_summary=diagnostics_summary,
        config=config or {},
        selected_temperature=selected_temperature,
        score_source=test_payload.score_source,
        feature_bank_metadata=dataset.feature_bank_metadata,
        manifest_path=dataset.manifest_path,
    )
    return RouterErrorAnatomyEvaluation(
        confusion_counts=confusion_counts,
        confusion_rates=confusion_rates,
        symmetric_confusion=symmetric_confusion,
        top_confusions=top_confusions,
        per_task_reliability=per_task_reliability,
        confidence_bins=confidence_bins,
        confidence_summary=confidence_summary,
        topk_error_buckets=topk_error_buckets,
        diagnostics_confusion_pairs=diagnostics_pairs,
        diagnostics_confusion_summary=diagnostics_summary,
        task_gap_decomposition=task_gap_decomposition,
        sample_level=sample_level,
        summary=summary,
    )


def write_router_error_anatomy_outputs(
    evaluation: RouterErrorAnatomyEvaluation,
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
        "confusion_counts_csv": output_root / f"{prefix}_linear_confusion_counts.csv",
        "confusion_rates_csv": output_root / f"{prefix}_linear_confusion_rates.csv",
        "symmetric_confusion_csv": output_root / f"{prefix}_linear_confusion_symmetric.csv",
        "top_confusions_csv": output_root / f"{prefix}_top_confusions.csv",
        "per_task_reliability_csv": output_root / f"{prefix}_per_task_reliability.csv",
        "confidence_bins_csv": output_root / f"{prefix}_confidence_bins.csv",
        "confidence_summary_json": output_root / f"{prefix}_confidence_summary.json",
        "topk_error_buckets_csv": output_root / f"{prefix}_topk_error_buckets.csv",
        "diagnostics_confusion_pairs_csv": output_root
        / f"{prefix}_diagnostics_confusion_pairs.csv",
        "diagnostics_confusion_summary_json": output_root
        / f"{prefix}_diagnostics_confusion_summary.json",
        "task_gap_decomposition_csv": output_root / f"{prefix}_task_gap_decomposition.csv",
        "summary_json": output_root / f"{prefix}_router_error_anatomy_summary.json",
    }
    if evaluation.sample_level is not None:
        paths["sample_level_csv"] = output_root / f"{prefix}_sample_level_router_diagnostics.csv"
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Output already exists: {joined}. Pass --overwrite to replace.")

    evaluation.confusion_counts.to_csv(paths["confusion_counts_csv"], index=False)
    evaluation.confusion_rates.to_csv(paths["confusion_rates_csv"], index=False)
    evaluation.symmetric_confusion.to_csv(paths["symmetric_confusion_csv"], index=False)
    evaluation.top_confusions.to_csv(paths["top_confusions_csv"], index=False)
    evaluation.per_task_reliability.to_csv(paths["per_task_reliability_csv"], index=False)
    evaluation.confidence_bins.to_csv(paths["confidence_bins_csv"], index=False)
    paths["confidence_summary_json"].write_text(
        json.dumps(_json_safe(evaluation.confidence_summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation.topk_error_buckets.to_csv(paths["topk_error_buckets_csv"], index=False)
    evaluation.diagnostics_confusion_pairs.to_csv(
        paths["diagnostics_confusion_pairs_csv"],
        index=False,
    )
    paths["diagnostics_confusion_summary_json"].write_text(
        json.dumps(
            _json_safe(evaluation.diagnostics_confusion_summary),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation.task_gap_decomposition.to_csv(paths["task_gap_decomposition_csv"], index=False)
    if evaluation.sample_level is not None:
        evaluation.sample_level.to_csv(paths["sample_level_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(_json_safe(evaluation.summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def build_confusion_count_matrix(
    true_task_ids: np.ndarray,
    predicted_task_ids: np.ndarray,
    task_ids: Sequence[int] | np.ndarray | None = None,
) -> pd.DataFrame:
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    pred_values = np.asarray(predicted_task_ids, dtype=np.int64)
    if true_values.ndim != 1 or pred_values.ndim != 1:
        raise ValueError("true_task_ids and predicted_task_ids must be rank 1")
    if true_values.shape[0] != pred_values.shape[0]:
        raise ValueError("true_task_ids and predicted_task_ids must have matching N")
    routes = _sorted_task_ids(true_values, pred_values, task_ids)
    rows: list[dict[str, Any]] = []
    for true_task in routes:
        mask = true_values == true_task
        row = {"true_task": int(true_task)}
        for pred_task in routes:
            row[f"pred_task_{int(pred_task)}"] = int(np.sum(mask & (pred_values == pred_task)))
        rows.append(row)
    return pd.DataFrame(rows, columns=["true_task", *[f"pred_task_{int(t)}" for t in routes]])


def build_confusion_rate_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    pred_columns = _pred_columns(counts)
    rows: list[dict[str, Any]] = []
    for row in counts.to_dict(orient="records"):
        total = sum(int(row[column]) for column in pred_columns)
        out = {"true_task": int(row["true_task"])}
        for column in pred_columns:
            out[column] = math.nan if total == 0 else float(row[column]) / float(total)
        rows.append(out)
    return pd.DataFrame(rows, columns=["true_task", *pred_columns])


def build_symmetric_confusion_matrix(rates: pd.DataFrame) -> pd.DataFrame:
    task_ids = [int(value) for value in rates["true_task"].tolist()]
    rate_lookup = _rate_lookup(rates)
    rows: list[dict[str, Any]] = []
    for task_i in task_ids:
        row = {"task": task_i}
        for task_j in task_ids:
            if task_i == task_j:
                value = 0.0
            else:
                value = rate_lookup.get((task_i, task_j), 0.0) + rate_lookup.get(
                    (task_j, task_i),
                    0.0,
                )
            row[f"task_{task_j}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows, columns=["task", *[f"task_{task_id}" for task_id in task_ids]])


def build_top_confusions_table(
    counts: pd.DataFrame,
    rates: pd.DataFrame,
    per_task_accs: dict[str, dict[int, float]],
    diagnostics_lookup: dict[tuple[int, int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    diagnostics_values = diagnostics_lookup or {}
    rate_lookup = _rate_lookup(rates)
    rows: list[dict[str, Any]] = []
    for count_row in counts.to_dict(orient="records"):
        true_task = int(count_row["true_task"])
        for column in _pred_columns(counts):
            pred_task = _task_id_from_pred_column(column)
            if pred_task == true_task:
                continue
            count = int(count_row[column])
            if count <= 0:
                continue
            pair_key = _pair_key(true_task, pred_task)
            diag = diagnostics_values.get(pair_key, {})
            rows.append(
                {
                    "true_task": true_task,
                    "predicted_task": pred_task,
                    "count": count,
                    "rate_given_true_task": rate_lookup.get((true_task, pred_task), math.nan),
                    "true_task_acc_under_merge": per_task_accs["merge_all"].get(
                        true_task,
                        math.nan,
                    ),
                    "true_task_acc_under_hard_route": per_task_accs[
                        "hard_linear_route"
                    ].get(true_task, math.nan),
                    "true_task_acc_under_topk_k2": per_task_accs["topk_k2"].get(
                        true_task,
                        math.nan,
                    ),
                    "true_task_acc_under_fallback": per_task_accs["fallback"].get(
                        true_task,
                        math.nan,
                    ),
                    "diagnostics_auc_sym": diag.get(
                        "diagnostics_linear_probe_auc_symmetric",
                        math.nan,
                    ),
                    "separation_ratio": diag.get("diagnostics_separation_ratio", math.nan),
                }
            )
    columns = [
        "true_task",
        "predicted_task",
        "count",
        "rate_given_true_task",
        "true_task_acc_under_merge",
        "true_task_acc_under_hard_route",
        "true_task_acc_under_topk_k2",
        "true_task_acc_under_fallback",
        "diagnostics_auc_sym",
        "separation_ratio",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["count", "rate_given_true_task", "true_task", "predicted_task"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )


def assign_topk_error_buckets(topk_route_ids: np.ndarray, true_task_ids: np.ndarray) -> np.ndarray:
    route_values = np.asarray(topk_route_ids, dtype=np.int64)
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    if route_values.ndim == 1:
        route_values = route_values.reshape(-1, 1)
    if true_values.ndim != 1 or true_values.shape[0] != route_values.shape[0]:
        raise ValueError("topk_route_ids and true_task_ids must have matching N")
    buckets: list[str] = []
    for routes, true_task in zip(route_values, true_values, strict=True):
        if routes.shape[0] >= 1 and int(routes[0]) == int(true_task):
            buckets.append("top1_correct")
        elif _contains(routes[:2], int(true_task)):
            buckets.append("in_top2_not_top1")
        elif _contains(routes[:3], int(true_task)):
            buckets.append("in_top3_not_top2")
        elif _contains(routes[:5], int(true_task)):
            buckets.append("in_top5_not_top3")
        else:
            buckets.append("not_in_top5")
    return np.asarray(buckets, dtype=object)


def build_confidence_summary(
    probabilities: np.ndarray,
    true_task_ids: np.ndarray,
    route_ids: np.ndarray,
    confidence: np.ndarray,
    route_correct: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, Any]:
    confidence_values = np.asarray(confidence, dtype=np.float32)
    correct_values = np.asarray(route_correct, dtype=bool)
    metrics = compute_calibration_metrics(
        probabilities,
        true_task_ids,
        route_ids,
        n_bins=n_bins,
    )
    return {
        "mean_confidence_correct": _masked_mean(confidence_values, correct_values),
        "mean_confidence_wrong": _masked_mean(confidence_values, ~correct_values),
        "median_confidence_correct": _masked_median(confidence_values, correct_values),
        "median_confidence_wrong": _masked_median(confidence_values, ~correct_values),
        "confidence_auc_for_correct_route": compute_confidence_auc_for_correct_route(
            confidence_values,
            correct_values,
        ),
        "ece": metrics.ece,
        "n_bins": int(n_bins),
    }


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


def build_confidence_bin_table(
    confidence: np.ndarray,
    route_correct: np.ndarray,
    hard_route_correct: np.ndarray,
    fallback_flags: np.ndarray | None,
    fallback_correct: np.ndarray,
    merge_all_correct: np.ndarray,
    *,
    n_bins: int,
) -> pd.DataFrame:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    confidence_values = np.asarray(confidence, dtype=np.float32)
    route_correct_values = np.asarray(route_correct, dtype=bool)
    hard_values = np.asarray(hard_route_correct, dtype=bool)
    fallback_values = np.zeros_like(route_correct_values) if fallback_flags is None else np.asarray(
        fallback_flags,
        dtype=bool,
    )
    fallback_correct_values = np.asarray(fallback_correct, dtype=bool)
    merge_values = np.asarray(merge_all_correct, dtype=bool)
    _validate_same_length(
        confidence_values,
        route_correct_values,
        hard_values,
        fallback_values,
        fallback_correct_values,
        merge_values,
    )
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for idx in range(int(n_bins)):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        if idx == 0:
            mask = (confidence_values >= low) & (confidence_values <= high)
        else:
            mask = (confidence_values > low) & (confidence_values <= high)
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "num_samples": int(np.sum(mask)),
                "mean_confidence": _masked_mean(confidence_values, mask),
                "route_accuracy": _masked_mean(route_correct_values.astype(np.float32), mask),
                "hard_route_acc": _masked_mean(hard_values.astype(np.float32), mask),
                "fallback_rate": _masked_mean(fallback_values.astype(np.float32), mask),
                "fallback_acc": _masked_mean(fallback_correct_values.astype(np.float32), mask),
                "merge_all_acc": _masked_mean(merge_values.astype(np.float32), mask),
            }
        )
    return pd.DataFrame(rows)


def build_topk_error_bucket_table(
    buckets: np.ndarray,
    y_true: np.ndarray,
    outputs: dict[str, MethodOutputs],
    confidence: np.ndarray,
    *,
    total_samples: int,
) -> pd.DataFrame:
    bucket_values = np.asarray(buckets, dtype=object)
    y_true_values = np.asarray(y_true, dtype=np.int64)
    confidence_values = np.asarray(confidence, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        mask = bucket_values == bucket
        rows.append(
            {
                "bucket": bucket,
                "num_samples": int(np.sum(mask)),
                "fraction": 0.0 if total_samples == 0 else float(np.sum(mask)) / total_samples,
                "merge_all_acc": _method_acc_for_mask(outputs["merge_all"], y_true_values, mask),
                "hard_route_acc": _method_acc_for_mask(
                    outputs["hard_linear_route"],
                    y_true_values,
                    mask,
                ),
                "topk_k2_acc": _method_acc_for_mask(outputs["topk_k2"], y_true_values, mask),
                "topk_k3_acc": _method_acc_for_mask(outputs["topk_k3"], y_true_values, mask),
                "topk_k5_acc": _method_acc_for_mask(outputs["topk_k5"], y_true_values, mask),
                "fallback_acc": _method_acc_for_mask(outputs["fallback"], y_true_values, mask),
                "oracle_task_mask_acc": _method_acc_for_mask(
                    outputs["oracle_task_mask"],
                    y_true_values,
                    mask,
                ),
                "mean_confidence": _masked_mean(confidence_values, mask),
                "mean_mask_size": _masked_mean(outputs["topk_k5"].mask_sizes, mask),
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics_confusion_pairs(
    confusion_rates: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    task_ids: Sequence[int] | np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rate_lookup = _rate_lookup(confusion_rates)
    diag_lookup = _raw_diagnostics_lookup(diagnostics)
    rows: list[dict[str, Any]] = []
    routes = sorted(int(value) for value in task_ids)
    for idx, task_i in enumerate(routes):
        for task_j in routes[idx + 1 :]:
            diag = diag_lookup.get(_pair_key(task_i, task_j), {})
            confusion_i_to_j = float(rate_lookup.get((task_i, task_j), 0.0))
            confusion_j_to_i = float(rate_lookup.get((task_j, task_i), 0.0))
            rows.append(
                {
                    "task_i": task_i,
                    "task_j": task_j,
                    "diagnostics_linear_probe_auc_symmetric": diag.get(
                        "linear_probe_auc_symmetric",
                        math.nan,
                    ),
                    "diagnostics_knn_acc": diag.get("knn_domain_acc", math.nan),
                    "diagnostics_separation_ratio": diag.get("separation_ratio", math.nan),
                    "confusion_i_to_j_rate": confusion_i_to_j,
                    "confusion_j_to_i_rate": confusion_j_to_i,
                    "symmetric_confusion_rate": confusion_i_to_j + confusion_j_to_i,
                    "oracle_topk_mask_cost_pair": math.nan,
                }
            )
    columns = [
        "task_i",
        "task_j",
        "diagnostics_linear_probe_auc_symmetric",
        "diagnostics_knn_acc",
        "diagnostics_separation_ratio",
        "confusion_i_to_j_rate",
        "confusion_j_to_i_rate",
        "symmetric_confusion_rate",
        "oracle_topk_mask_cost_pair",
    ]
    pairs = pd.DataFrame(rows, columns=columns)
    summary = _diagnostics_confusion_summary(pairs, diagnostics is None)
    return pairs, summary


def build_per_task_reliability_table(
    true_task_ids: np.ndarray,
    y_true: np.ndarray,
    route_correct: np.ndarray,
    topk_routes: dict[int, np.ndarray],
    confidence: np.ndarray,
    outputs: dict[str, MethodOutputs],
) -> pd.DataFrame:
    task_values = np.asarray(true_task_ids, dtype=np.int64)
    y_true_values = np.asarray(y_true, dtype=np.int64)
    confidence_values = np.asarray(confidence, dtype=np.float32)
    route_correct_values = np.asarray(route_correct, dtype=bool)
    rows: list[dict[str, Any]] = []
    for task_id in sorted(int(value) for value in np.unique(task_values)):
        mask = task_values == task_id
        method_accs = {
            name: _method_acc_for_mask(method, y_true_values, mask)
            for name, method in outputs.items()
        }
        learned = {
            "hard_linear_route": method_accs["hard_linear_route"],
            "topk_k2": method_accs["topk_k2"],
            "topk_k3": method_accs["topk_k3"],
            "fallback": method_accs["fallback"],
        }
        best_control, best_acc = _best_named_value(learned)
        oracle_gap = method_accs["oracle_task_mask"] - method_accs["merge_all"]
        rows.append(
            {
                "task_id": task_id,
                "num_test_samples": int(np.sum(mask)),
                "router_top1_acc": _masked_mean(route_correct_values.astype(np.float32), mask),
                "router_top2_recall": _topk_recall_for_mask(topk_routes[2], task_values, mask),
                "router_top3_recall": _topk_recall_for_mask(topk_routes[3], task_values, mask),
                "router_top5_recall": _topk_recall_for_mask(topk_routes[5], task_values, mask),
                "mean_confidence": _masked_mean(confidence_values, mask),
                "mean_confidence_correct": _masked_mean(
                    confidence_values,
                    mask & route_correct_values,
                ),
                "mean_confidence_wrong": _masked_mean(
                    confidence_values,
                    mask & ~route_correct_values,
                ),
                "merge_all_acc": method_accs["merge_all"],
                "oracle_task_mask_acc": method_accs["oracle_task_mask"],
                "hard_linear_route_acc": method_accs["hard_linear_route"],
                "topk_k2_acc": method_accs["topk_k2"],
                "topk_k3_acc": method_accs["topk_k3"],
                "fallback_acc": method_accs["fallback"],
                "fallback_rate": _masked_mean(
                    outputs["fallback"].fallback_flags.astype(np.float32),
                    mask,
                )
                if outputs["fallback"].fallback_flags is not None
                else 0.0,
                "fallback_gain_vs_hard_route": method_accs["fallback"]
                - method_accs["hard_linear_route"],
                "best_learned_control": best_control,
                "best_learned_control_acc": best_acc,
                "oracle_gap_closure_task": math.nan
                if abs(oracle_gap) <= 1e-12
                else (best_acc - method_accs["merge_all"]) / oracle_gap,
            }
        )
    return pd.DataFrame(rows)


def build_task_gap_decomposition_table(
    per_task_accs: dict[str, dict[int, float]],
) -> pd.DataFrame:
    tasks = sorted(int(task_id) for task_id in per_task_accs["merge_all"])
    rows: list[dict[str, Any]] = []
    for task_id in tasks:
        merge_acc = per_task_accs["merge_all"].get(task_id, math.nan)
        oracle_acc = per_task_accs["oracle_task_mask"].get(task_id, math.nan)
        hard_acc = per_task_accs["hard_linear_route"].get(task_id, math.nan)
        topk_acc = per_task_accs["topk_k2"].get(task_id, math.nan)
        fallback_acc = per_task_accs["fallback"].get(task_id, math.nan)
        rows.append(
            {
                "task_id": task_id,
                "merge_all_acc": merge_acc,
                "oracle_task_mask_acc": oracle_acc,
                "hard_linear_route_acc": hard_acc,
                "topk_k2_acc": topk_acc,
                "fallback_acc": fallback_acc,
                "oracle_gap": oracle_acc - merge_acc,
                "hard_route_gap": hard_acc - merge_acc,
                "topk_k2_gap": topk_acc - merge_acc,
                "fallback_gap": fallback_acc - merge_acc,
                "hard_route_error_cost": oracle_acc - hard_acc,
                "topk_error_cost": oracle_acc - topk_acc,
                "fallback_error_cost": oracle_acc - fallback_acc,
            }
        )
    return pd.DataFrame(rows)


def build_sample_level_table(
    y_true: np.ndarray,
    true_task_ids: np.ndarray,
    topk_routes: dict[int, np.ndarray],
    confidence: np.ndarray,
    outputs: dict[str, MethodOutputs],
) -> pd.DataFrame:
    y_true_values = np.asarray(y_true, dtype=np.int64)
    task_values = np.asarray(true_task_ids, dtype=np.int64)
    top1 = np.asarray(topk_routes[1], dtype=np.int64)[:, 0]
    fallback_flags = outputs["fallback"].fallback_flags
    return pd.DataFrame(
        {
            "sample_index": np.arange(y_true_values.shape[0], dtype=np.int64),
            "true_label": y_true_values,
            "true_task": task_values,
            "predicted_task": top1,
            "route_correct": top1 == task_values,
            "confidence": np.asarray(confidence, dtype=np.float32),
            "top2_contains_true": _topk_contains(topk_routes[2], task_values),
            "top3_contains_true": _topk_contains(topk_routes[3], task_values),
            "top5_contains_true": _topk_contains(topk_routes[5], task_values),
            "merge_all_pred": outputs["merge_all"].predictions,
            "hard_route_pred": outputs["hard_linear_route"].predictions,
            "fallback_pred": outputs["fallback"].predictions,
            "fallback_used": np.zeros_like(task_values, dtype=bool)
            if fallback_flags is None
            else fallback_flags,
        }
    )


def build_router_error_anatomy_summary(
    *,
    router_type: str,
    topk_routes: dict[int, np.ndarray],
    true_task_ids: np.ndarray,
    confidence_summary: dict[str, Any],
    per_task_reliability: pd.DataFrame,
    top_confusions: pd.DataFrame,
    per_task_accs: dict[str, dict[int, float]],
    diagnostics_summary: dict[str, Any],
    config: dict[str, Any],
    selected_temperature: float,
    score_source: str,
    feature_bank_metadata: dict[str, Any],
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    merge_acc = _mean_task_weighted_acc(per_task_accs["merge_all"], true_values)
    oracle_acc = _mean_task_weighted_acc(per_task_accs["oracle_task_mask"], true_values)
    hard_acc = _mean_task_weighted_acc(per_task_accs["hard_linear_route"], true_values)
    topk_k2_acc = _mean_task_weighted_acc(per_task_accs["topk_k2"], true_values)
    topk_k3_acc = _mean_task_weighted_acc(per_task_accs["topk_k3"], true_values)
    fallback_acc = _mean_task_weighted_acc(per_task_accs["fallback"], true_values)
    router_rows = per_task_reliability.sort_values("router_top1_acc")
    worst_task = _task_router_brief(router_rows.iloc[0]) if not router_rows.empty else None
    best_task = _task_router_brief(router_rows.iloc[-1]) if not router_rows.empty else None
    most_confused = None if top_confusions.empty else _confusion_brief(
        top_confusions.iloc[0].to_dict()
    )
    top_confused_pairs = [
        _confusion_brief(row)
        for row in top_confusions.head(5).to_dict(orient="records")
    ]
    fallback_task_gains = _top_task_gains(
        per_task_reliability,
        "fallback_gain_vs_hard_route",
    )
    topk_task_gains = _topk_task_gains(per_task_reliability)
    summary = {
        "router_type": router_type,
        "selected_temperature": selected_temperature,
        "score_source": score_source,
        "top1_acc": _topk_recall(topk_routes[1], true_values),
        "top2_recall": _topk_recall(topk_routes[2], true_values),
        "top3_recall": _topk_recall(topk_routes[3], true_values),
        "top5_recall": _topk_recall(topk_routes[5], true_values),
        "mean_confidence_correct": confidence_summary["mean_confidence_correct"],
        "mean_confidence_wrong": confidence_summary["mean_confidence_wrong"],
        "confidence_auc_for_correct_route": confidence_summary[
            "confidence_auc_for_correct_route"
        ],
        "best_task_by_router_acc": best_task,
        "worst_task_by_router_acc": worst_task,
        "most_confused_pair": most_confused,
        "top_confused_pairs": top_confused_pairs,
        "merge_all_acc": merge_acc,
        "oracle_task_mask_acc": oracle_acc,
        "hard_route_acc": hard_acc,
        "topk_k2_acc": topk_k2_acc,
        "topk_k3_acc": topk_k3_acc,
        "fallback_acc": fallback_acc,
        "oracle_gap": oracle_acc - merge_acc,
        "hard_route_gap": hard_acc - merge_acc,
        "topk_k2_gap": topk_k2_acc - merge_acc,
        "fallback_gap": fallback_acc - merge_acc,
        "diagnostics_confusion_correlations": {
            "pearson_auc_vs_confusion": diagnostics_summary.get(
                "pearson_auc_vs_confusion",
                math.nan,
            ),
            "spearman_auc_vs_confusion": diagnostics_summary.get(
                "spearman_auc_vs_confusion",
                math.nan,
            ),
            "pearson_separation_vs_confusion": diagnostics_summary.get(
                "pearson_separation_vs_confusion",
                math.nan,
            ),
            "spearman_separation_vs_confusion": diagnostics_summary.get(
                "spearman_separation_vs_confusion",
                math.nan,
            ),
        },
        "fallback_top_task_gains": fallback_task_gains,
        "topk_k2_top_task_gains": topk_task_gains,
        "config_hash": stable_hash(config),
        "feature_bank_metadata": feature_bank_metadata,
        "manifest_path": str(manifest_path) if manifest_path else None,
    }
    summary["interpretation"] = _interpretation(
        summary,
        diagnostics_summary=diagnostics_summary,
        fallback_task_gains=fallback_task_gains,
        topk_task_gains=topk_task_gains,
    )
    for key in REQUIRED_SUMMARY_KEYS:
        summary.setdefault(key, None)
    return summary


def _selected_temperature(
    *,
    router_values: dict[str, Any],
    calibration_config: dict[str, Any] | None,
    val_scores: np.ndarray,
    val_task_ids: np.ndarray,
    route_ids: np.ndarray,
) -> float:
    if not bool(router_values.get("use_calibrated_temperature", True)):
        return 1.0
    requested = router_values.get("selected_temperature", "auto")
    if requested not in (None, "", "auto"):
        return float(requested)
    calibration = dict(calibration_config or {})
    temperatures = calibration.get(
        "temperatures",
        [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    )
    ece_bins = int(calibration.get("ece_bins", 15))
    selection = select_temperature(
        val_scores,
        val_task_ids,
        route_ids,
        temperatures,
        n_bins=ece_bins,
    )
    return float(selection.temperature)


def _analysis_topk_values(analysis: dict[str, Any]) -> list[int]:
    values = {int(value) for value in analysis.get("topk_values", [1, 2, 3, 5])}
    values.update({1, 2, 3, 5})
    return sorted(value for value in values if value > 0)


def _method_outputs(
    *,
    classifier: PrototypeClassifier,
    task_classes: dict[int, tuple[int, ...]],
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_task_ids: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_top1: np.ndarray,
    val_confidence: np.ndarray,
    test_confidence: np.ndarray,
    topk_routes: dict[int, np.ndarray],
    fallback_config: dict[str, Any] | None,
) -> dict[str, MethodOutputs]:
    full_mask = np.full(
        test_labels.shape[0],
        classifier.classes_.shape[0],
        dtype=np.float32,
    )
    merge_predictions = classifier.predict(test_features)
    oracle_predictions, oracle_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        test_features,
        test_task_ids.reshape(-1, 1),
        task_classes=task_classes,
    )
    outputs = {
        "merge_all": MethodOutputs(merge_predictions, full_mask),
        "oracle_task_mask": MethodOutputs(oracle_predictions, oracle_mask_sizes),
    }
    for k in (1, 2, 3, 5):
        predictions, mask_sizes = predict_prototype_with_topk_task_mask(
            classifier,
            test_features,
            topk_routes[k],
            task_classes=task_classes,
        )
        name = "hard_linear_route" if k == 1 else f"topk_k{k}"
        outputs[name] = MethodOutputs(predictions, mask_sizes)
    tau_grid = _fallback_tau_grid(fallback_config)
    tau, _ = _select_fallback_tau(
        classifier,
        val_features,
        val_labels,
        val_top1,
        val_confidence,
        task_classes,
        tau_grid,
    )
    return _add_fallback_output(
        outputs,
        classifier=classifier,
        test_features=test_features,
        top1_routes=topk_routes[1],
        test_confidence=test_confidence,
        task_classes=task_classes,
        tau=tau,
    )


def _add_fallback_output(
    outputs: dict[str, MethodOutputs],
    *,
    classifier: PrototypeClassifier,
    test_features: np.ndarray,
    top1_routes: np.ndarray,
    test_confidence: np.ndarray | None,
    task_classes: dict[int, tuple[int, ...]],
    tau: float,
) -> dict[str, MethodOutputs]:
    predictions, fallback_flags, mask_sizes = predict_with_fallback_to_merge_all(
        classifier,
        test_features,
        top1_routes,
        test_confidence,
        tau=tau,
        task_classes=task_classes,
    )
    outputs["fallback"] = MethodOutputs(predictions, mask_sizes, fallback_flags)
    return outputs


def _select_fallback_tau(
    classifier: PrototypeClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    top1_route_ids: np.ndarray,
    confidence: np.ndarray,
    task_classes: dict[int, tuple[int, ...]],
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


def _fallback_tau_grid(config: dict[str, Any] | None) -> list[float]:
    values = dict(config or {})
    candidates = values.get(
        "tau_grid",
        values.get("linear_tau_grid", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
    )
    unique: list[float] = []
    for value in candidates:
        number = float(value)
        if not any(abs(number - existing) <= 1e-12 for existing in unique):
            unique.append(number)
    if not unique:
        raise ValueError("fallback tau grid must contain at least one value")
    return unique


def _task_classes(tasks: Sequence[TaskFeatureSplit]) -> dict[int, tuple[int, ...]]:
    return {
        task.task_id: task.classes or tuple(int(value) for value in np.unique(task.train_labels))
        for task in tasks
    }


def _per_task_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_task_ids: np.ndarray,
) -> dict[int, float]:
    true_values = np.asarray(y_true, dtype=np.int64)
    pred_values = np.asarray(y_pred, dtype=np.int64)
    task_values = np.asarray(true_task_ids, dtype=np.int64)
    return {
        int(task_id): _accuracy(true_values[task_values == task_id], pred_values[task_values == task_id])
        for task_id in sorted(int(value) for value in np.unique(task_values))
    }


def _load_diagnostics(path: str | Path | None) -> pd.DataFrame | None:
    if path in (None, ""):
        return None
    value = Path(path)
    if not value.exists():
        return None
    return pd.read_csv(value)


def _raw_diagnostics_lookup(diagnostics: pd.DataFrame | None) -> dict[tuple[int, int], dict[str, Any]]:
    if diagnostics is None or diagnostics.empty:
        return {}
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    if not {"task_i", "task_j"}.issubset(diagnostics.columns):
        return lookup
    for row in diagnostics.to_dict(orient="records"):
        try:
            key = _pair_key(int(row["task_i"]), int(row["task_j"]))
        except (TypeError, ValueError):
            continue
        lookup[key] = row
    return lookup


def _diagnostics_lookup(pairs: pd.DataFrame) -> dict[tuple[int, int], dict[str, float]]:
    lookup: dict[tuple[int, int], dict[str, float]] = {}
    for row in pairs.to_dict(orient="records"):
        lookup[_pair_key(int(row["task_i"]), int(row["task_j"]))] = row
    return lookup


def _diagnostics_confusion_summary(
    pairs: pd.DataFrame,
    diagnostics_missing: bool,
) -> dict[str, Any]:
    auc = pairs["diagnostics_linear_probe_auc_symmetric"].to_numpy(dtype=np.float64)
    separation = pairs["diagnostics_separation_ratio"].to_numpy(dtype=np.float64)
    confusion = pairs["symmetric_confusion_rate"].to_numpy(dtype=np.float64)
    high_high = _pair_briefs(
        pairs.sort_values(
            ["diagnostics_linear_probe_auc_symmetric", "symmetric_confusion_rate"],
            ascending=[False, False],
        ).head(5)
    )
    low_low = _pair_briefs(
        pairs.sort_values(
            ["diagnostics_linear_probe_auc_symmetric", "symmetric_confusion_rate"],
            ascending=[True, True],
        ).head(5)
    )
    return {
        "diagnostics_missing": diagnostics_missing,
        "num_pairs": int(pairs.shape[0]),
        "pearson_auc_vs_confusion": _pearson(auc, confusion),
        "spearman_auc_vs_confusion": _spearman(auc, confusion),
        "pearson_separation_vs_confusion": _pearson(separation, confusion),
        "spearman_separation_vs_confusion": _spearman(separation, confusion),
        "top_pairs_high_diagnostics_but_high_confusion": high_high,
        "top_pairs_low_diagnostics_but_low_confusion": low_low,
    }


def _pair_briefs(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows.to_dict(orient="records"):
        out.append(
            {
                "task_i": int(row["task_i"]),
                "task_j": int(row["task_j"]),
                "diagnostics_linear_probe_auc_symmetric": row[
                    "diagnostics_linear_probe_auc_symmetric"
                ],
                "diagnostics_separation_ratio": row["diagnostics_separation_ratio"],
                "symmetric_confusion_rate": row["symmetric_confusion_rate"],
            }
        )
    return out


def _interpretation(
    summary: dict[str, Any],
    *,
    diagnostics_summary: dict[str, Any],
    fallback_task_gains: list[dict[str, Any]],
    topk_task_gains: list[dict[str, Any]],
) -> list[str]:
    notes: list[str] = []
    top1 = float(summary["top1_acc"])
    hard_gap = float(summary["hard_route_gap"])
    oracle_gap = float(summary["oracle_gap"])
    if top1 < 0.8 or hard_gap <= 0.0:
        notes.append(
            "Router top-1 is a primary bottleneck: hard routing does not reliably close "
            "the oracle-mask gap."
        )
    else:
        notes.append("Router top-1 is not the only bottleneck; mask-size cost remains visible.")
    auc = summary["confidence_auc_for_correct_route"]
    if _is_finite(auc) and float(auc) >= 0.7:
        notes.append("Confidence separates correct and wrong routes well enough for fallback.")
    elif _is_finite(auc):
        notes.append("Confidence only weakly separates correct and wrong routes.")
    if summary.get("most_confused_pair"):
        pair = summary["most_confused_pair"]
        notes.append(
            f"Dominant confusion is true task {pair['true_task']} routed to "
            f"{pair['predicted_task']}."
        )
    corr = diagnostics_summary.get("pearson_auc_vs_confusion")
    if _is_finite(corr) and float(corr) < -0.3:
        notes.append("Diagnostics AUC is negatively correlated with real routing confusion.")
    elif _is_finite(corr):
        notes.append("Diagnostics AUC is not a strong standalone predictor of confusion.")
    if topk_task_gains:
        tasks = ", ".join(str(item["task_id"]) for item in topk_task_gains[:3])
        notes.append(f"Top-k gains are concentrated on task(s): {tasks}.")
    if fallback_task_gains:
        tasks = ", ".join(str(item["task_id"]) for item in fallback_task_gains[:3])
        notes.append(f"Fallback gains are concentrated on task(s): {tasks}.")
    if oracle_gap > 0.0:
        cost = float(summary["oracle_task_mask_acc"]) - float(summary["fallback_acc"])
        notes.append(
            f"Remaining fallback error cost is {cost:.6f}, so the route-vs-merge gap is "
            "still mostly constrained by routing error and mask-size tradeoffs."
        )
    return notes


def _top_task_gains(df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if df.empty or column not in df:
        return []
    rows = df.sort_values(column, ascending=False).head(5)
    return [
        {"task_id": int(row["task_id"]), "gain": float(row[column])}
        for row in rows.to_dict(orient="records")
        if _is_finite(row[column]) and float(row[column]) > 0.0
    ]


def _topk_task_gains(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    work["topk_k2_gain_vs_hard_route"] = (
        work["topk_k2_acc"].astype(float) - work["hard_linear_route_acc"].astype(float)
    )
    return _top_task_gains(work, "topk_k2_gain_vs_hard_route")


def _task_router_brief(row: pd.Series) -> dict[str, Any]:
    return {
        "task_id": int(row["task_id"]),
        "router_top1_acc": float(row["router_top1_acc"]),
        "router_top2_recall": float(row["router_top2_recall"]),
        "hard_linear_route_acc": float(row["hard_linear_route_acc"]),
    }


def _confusion_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "true_task": int(row["true_task"]),
        "predicted_task": int(row["predicted_task"]),
        "count": int(row["count"]),
        "rate_given_true_task": float(row["rate_given_true_task"]),
        "true_task_acc_under_merge": float(row["true_task_acc_under_merge"]),
        "true_task_acc_under_hard_route": float(row["true_task_acc_under_hard_route"]),
        "true_task_acc_under_topk_k2": float(row["true_task_acc_under_topk_k2"]),
        "true_task_acc_under_fallback": float(row["true_task_acc_under_fallback"]),
        "diagnostics_auc_sym": row.get("diagnostics_auc_sym"),
        "separation_ratio": row.get("separation_ratio"),
    }


def _method_acc_for_mask(method: MethodOutputs, y_true: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    return _accuracy(y_true[mask], method.predictions[mask])


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values = np.asarray(y_true, dtype=np.int64)
    pred_values = np.asarray(y_pred, dtype=np.int64)
    if true_values.shape[0] == 0:
        return math.nan
    return float(np.mean(true_values == pred_values))


def _topk_recall_for_mask(topk: np.ndarray, true_task_ids: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    return _topk_recall(topk[mask], true_task_ids[mask])


def _topk_recall(topk: np.ndarray, true_task_ids: np.ndarray) -> float:
    route_values = np.asarray(topk, dtype=np.int64)
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    if route_values.ndim == 1:
        route_values = route_values.reshape(-1, 1)
    if true_values.shape[0] == 0:
        return math.nan
    return float(np.mean(np.any(route_values == true_values[:, None], axis=1)))


def _topk_contains(topk: np.ndarray, true_task_ids: np.ndarray) -> np.ndarray:
    route_values = np.asarray(topk, dtype=np.int64)
    true_values = np.asarray(true_task_ids, dtype=np.int64)
    if route_values.ndim == 1:
        route_values = route_values.reshape(-1, 1)
    return np.any(route_values == true_values[:, None], axis=1)


def _mean_task_weighted_acc(per_task: dict[int, float], true_task_ids: np.ndarray) -> float:
    task_values = np.asarray(true_task_ids, dtype=np.int64)
    numerator = 0.0
    denominator = 0
    for task_id, acc in per_task.items():
        count = int(np.sum(task_values == int(task_id)))
        numerator += float(acc) * count
        denominator += count
    return math.nan if denominator == 0 else numerator / denominator


def _best_named_value(values: dict[str, float]) -> tuple[str, float]:
    filtered = {key: value for key, value in values.items() if _is_finite(value)}
    if not filtered:
        return "", math.nan
    key = max(filtered, key=lambda item: filtered[item])
    return key, float(filtered[key])


def _sorted_task_ids(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    task_ids: Sequence[int] | np.ndarray | None,
) -> list[int]:
    if task_ids is None:
        values = np.concatenate([true_values, pred_values])
    else:
        values = np.asarray(task_ids, dtype=np.int64)
    return sorted(int(value) for value in np.unique(values))


def _pred_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if str(column).startswith("pred_task_")]


def _task_id_from_pred_column(column: str) -> int:
    return int(str(column).removeprefix("pred_task_"))


def _rate_lookup(rates: pd.DataFrame) -> dict[tuple[int, int], float]:
    lookup: dict[tuple[int, int], float] = {}
    for row in rates.to_dict(orient="records"):
        true_task = int(row["true_task"])
        for column in _pred_columns(rates):
            lookup[(true_task, _task_id_from_pred_column(column))] = float(row[column])
    return lookup


def _pair_key(task_i: int, task_j: int) -> tuple[int, int]:
    return tuple(sorted((int(task_i), int(task_j))))


def _contains(values: np.ndarray, target: int) -> bool:
    return bool(np.any(np.asarray(values, dtype=np.int64) == int(target)))


def _validate_same_length(*arrays: np.ndarray) -> None:
    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise ValueError("all arrays must have matching N")


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    mask_values = np.asarray(mask, dtype=bool)
    if not np.any(mask_values):
        return math.nan
    return float(np.mean(np.asarray(values)[mask_values]))


def _masked_median(values: np.ndarray, mask: np.ndarray) -> float:
    mask_values = np.asarray(mask, dtype=bool)
    if not np.any(mask_values):
        return math.nan
    return float(np.median(np.asarray(values)[mask_values]))


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _pearson(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return math.nan
    x = x[mask]
    y = y[mask]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator <= 0.0:
        return math.nan
    return float(np.sum(x_centered * y_centered) / denominator)


def _spearman(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return math.nan
    return _pearson(_rank_average(x[mask]), _rank_average(y[mask]))


def _is_finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Path):
        return str(value)
    return value
