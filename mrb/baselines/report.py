"""Output helpers for feature toy baseline runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mrb.baselines.evaluate import FeatureToyEvaluation
from mrb.baselines.routing_controls import SoftRoutingEvaluation
from mrb.data.splits import stable_hash


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "baselines" / "feature_toy_v0"
DEFAULT_SOFT_ROUTING_OUTPUT_DIR = REPO_ROOT / "artifacts" / "baselines" / "soft_routing_v0"


def write_feature_toy_outputs(
    evaluation: FeatureToyEvaluation,
    *,
    output_dir: str | Path,
    stream_id: str,
    seed: int,
    backbone_id: str,
    config: dict[str, Any],
    feature_bank_metadata: dict[str, Any],
    manifest_path: str | Path | None,
    diagnostics_csv: str | Path | None,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{stream_id}_seed{seed}_{backbone_id}"
    paths = {
        "results_csv": output_root / f"{prefix}_results.csv",
        "per_task_csv": output_root / f"{prefix}_per_task.csv",
        "route_confusion_csv": output_root / f"{prefix}_route_confusion.csv",
        "cluster_assignments_json": output_root / f"{prefix}_cluster_assignments.json",
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
    cluster_payload = _cluster_payload(evaluation)
    paths["cluster_assignments_json"].write_text(
        json.dumps(_json_safe(cluster_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = summarize_feature_toy_results(
        evaluation.results,
        config=config,
        feature_bank_metadata=feature_bank_metadata,
        manifest_path=manifest_path,
        diagnostics_csv=diagnostics_csv,
        cluster_assignments=cluster_payload,
    )
    paths["summary_json"].write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def summarize_feature_toy_results(
    results: pd.DataFrame,
    *,
    config: dict[str, Any],
    feature_bank_metadata: dict[str, Any],
    manifest_path: str | Path | None,
    diagnostics_csv: str | Path | None,
    cluster_assignments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ok_results = results[results["status"] == "ok"] if "status" in results else results
    best = None
    if not ok_results.empty:
        best_row = ok_results.sort_values("overall_acc", ascending=False).iloc[0]
        best = {
            "method": best_row["method"],
            "classifier_type": best_row["classifier_type"],
            "cluster_strategy": best_row["cluster_strategy"],
            "K": int(best_row["K"]),
            "overall_acc": float(best_row["overall_acc"]),
        }
    merge_acc = _method_value(results, "merge_all_prototype", "overall_acc")
    oracle_acc = _method_value(results, "task_oracle_prototype", "overall_acc")
    learned_acc = _method_value(results, "task_learned_centroid_router_prototype", "overall_acc")
    learned_router_acc = _method_value(
        results, "task_learned_centroid_router_prototype", "router_acc"
    )
    oracle_mask_acc = _method_value(
        results, "merge_all_prototype_oracle_task_mask", "overall_acc"
    )
    failed = results[results["status"] != "ok"] if "status" in results else results.iloc[0:0]
    return {
        "best_method_by_overall_acc": best,
        "merge_all_prototype_acc": merge_acc,
        "merge_all_prototype_oracle_task_mask_acc": oracle_mask_acc,
        "task_oracle_prototype_acc": oracle_acc,
        "learned_route_centroid_prototype_acc": learned_acc,
        "learned_route_centroid_router_acc": learned_router_acc,
        "oracle_route_gap": None if merge_acc is None or oracle_acc is None else oracle_acc - merge_acc,
        "learned_route_gap": None if merge_acc is None or learned_acc is None else learned_acc - merge_acc,
        "label_mask_gain": None
        if merge_acc is None or oracle_mask_acc is None
        else oracle_mask_acc - merge_acc,
        "classifier_specialization_gain": None
        if oracle_acc is None or oracle_mask_acc is None
        else oracle_acc - oracle_mask_acc,
        "routing_error_cost": _routing_error_cost(results),
        "learned_router_metrics": _learned_router_metrics(results),
        "cluster_frontier": _cluster_frontier(results),
        "num_methods": int(results.shape[0]),
        "num_failed_methods": int(failed.shape[0]),
        "failed_methods": failed["method"].tolist() if not failed.empty else [],
        "has_nan_overall_acc": bool(results["overall_acc"].isna().any()),
        "config_hash": stable_hash(config),
        "feature_bank_metadata": feature_bank_metadata,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "diagnostics_csv": str(diagnostics_csv) if diagnostics_csv else None,
        "cluster_assignments": cluster_assignments or [],
    }


def write_soft_routing_outputs(
    evaluation: SoftRoutingEvaluation,
    *,
    output_dir: str | Path,
    stream_id: str,
    seed: int,
    backbone_id: str,
    config: dict[str, Any],
    feature_bank_metadata: dict[str, Any],
    manifest_path: str | Path | None,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{stream_id}_seed{seed}_{backbone_id}"
    paths = {
        "results_csv": output_root / f"{prefix}_results.csv",
        "per_task_csv": output_root / f"{prefix}_per_task.csv",
        "route_confusion_csv": output_root / f"{prefix}_route_confusion.csv",
        "selected_hparams_json": output_root / f"{prefix}_selected_hparams.json",
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
    paths["selected_hparams_json"].write_text(
        json.dumps(_json_safe(evaluation.selected_hparams), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = summarize_soft_routing_results(
        evaluation.results,
        config=config,
        feature_bank_metadata=feature_bank_metadata,
        manifest_path=manifest_path,
        selected_hparams=evaluation.selected_hparams,
    )
    paths["summary_json"].write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def summarize_soft_routing_results(
    results: pd.DataFrame,
    *,
    config: dict[str, Any],
    feature_bank_metadata: dict[str, Any],
    manifest_path: str | Path | None,
    selected_hparams: dict[str, Any],
) -> dict[str, Any]:
    ok_results = results[results["status"] == "ok"] if "status" in results else results
    best = None
    if not ok_results.empty:
        best_row = ok_results.sort_values("overall_acc", ascending=False).iloc[0]
        best = _soft_result_brief(best_row)
    merge_acc = _method_value(results, "merge_all_prototype", "overall_acc")
    oracle_mask_acc = _method_value(
        results, "merge_all_prototype_oracle_task_mask", "overall_acc"
    )
    failed = results[results["status"] != "ok"] if "status" in results else results.iloc[0:0]
    return {
        "best_method_by_overall_acc": best,
        "merge_all_prototype_acc": merge_acc,
        "merge_all_prototype_oracle_task_mask_acc": oracle_mask_acc,
        "task_oracle_prototype_acc": _method_value(
            results,
            "task_oracle_prototype",
            "overall_acc",
        ),
        "hard_learned_linear_acc": _method_value(
            results,
            "merge_all_prototype_learned_task_mask_linear",
            "overall_acc",
        ),
        "hard_learned_energy_acc": _method_value(
            results,
            "merge_all_prototype_learned_task_mask_energy",
            "overall_acc",
        ),
        "task_learned_linear_router_prototype_acc": _method_value(
            results,
            "task_learned_linear_router_prototype",
            "overall_acc",
        ),
        "task_learned_energy_router_prototype_acc": _method_value(
            results,
            "task_learned_energy_router_prototype",
            "overall_acc",
        ),
        "topk_best": _best_by_control(results, "topk_task_mask"),
        "fallback_best": _best_by_control(results, "fallback_to_merge_all"),
        "soft_prior_best": _best_by_control(results, "soft_prior"),
        "selected_hparams": selected_hparams,
        "num_methods": int(results.shape[0]),
        "num_failed_methods": int(failed.shape[0]),
        "failed_methods": failed["method"].tolist() if not failed.empty else [],
        "has_nan_overall_acc": bool(results["overall_acc"].isna().any()),
        "exceeds_merge_all": None
        if best is None or merge_acc is None
        else bool(float(best["overall_acc"]) > merge_acc),
        "config_hash": stable_hash(config),
        "feature_bank_metadata": feature_bank_metadata,
        "manifest_path": str(manifest_path) if manifest_path else None,
    }


def _cluster_payload(evaluation: FeatureToyEvaluation) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, tuple[tuple[int, ...], ...]]] = set()
    payload: list[dict[str, Any]] = []
    for assignment in evaluation.cluster_assignments:
        key = (assignment.strategy, assignment.K, assignment.clusters)
        if key in seen:
            continue
        seen.add(key)
        payload.append(
            {
                "strategy": assignment.strategy,
                "K": int(assignment.K),
                "clusters": [list(cluster) for cluster in assignment.clusters],
                "task_to_cluster": {
                    str(task_id): int(cluster_id)
                    for task_id, cluster_id in sorted(assignment.task_to_cluster.items())
                },
                "warning": assignment.warning,
            }
        )
    return payload


def _best_by_control(results: pd.DataFrame, control_type: str) -> dict[str, Any]:
    rows = results[results["control_type"] == control_type] if "control_type" in results else results.iloc[0:0]
    best: dict[str, Any] = {}
    for router_type in sorted(str(value) for value in rows["router_type"].dropna().unique()):
        router_rows = rows[rows["router_type"] == router_type]
        if router_rows.empty:
            continue
        best_row = router_rows.sort_values("overall_acc", ascending=False).iloc[0]
        best[router_type] = _soft_result_brief(best_row)
    return best


def _soft_result_brief(row: pd.Series) -> dict[str, Any]:
    return {
        "method": row["method"],
        "control_type": row["control_type"],
        "router_type": row["router_type"],
        "topk": _float_or_none(row.get("topk")),
        "tau": _float_or_none(row.get("tau")),
        "beta": _float_or_none(row.get("beta")),
        "val_acc": _float_or_none(row.get("val_acc")),
        "overall_acc": _float_or_none(row.get("overall_acc")),
        "route_vs_merge_gap": _float_or_none(row.get("route_vs_merge_gap")),
        "oracle_gap_closure": _float_or_none(row.get("oracle_gap_closure")),
        "fallback_rate": _float_or_none(row.get("fallback_rate")),
        "mean_mask_size": _float_or_none(row.get("mean_mask_size")),
    }


def _cluster_frontier(results: pd.DataFrame) -> list[dict[str, Any]]:
    if results.empty:
        return []
    cluster = results[results["method"].isin(["cluster_oracle_prototype", "cluster_learned_prototype"])]
    rows: list[dict[str, Any]] = []
    for row in cluster.sort_values(["cluster_strategy", "method", "K"]).to_dict(orient="records"):
        rows.append(
            {
                "method": row["method"],
                "cluster_strategy": row["cluster_strategy"],
                "K": int(row["K"]),
                "overall_acc": _float_or_none(row["overall_acc"]),
                "router_acc": _float_or_none(row["router_acc"]),
                "route_vs_merge_gap": _float_or_none(row["route_vs_merge_gap"]),
                "status": row["status"],
                "warning": row.get("warning", ""),
            }
        )
    return rows


def _routing_error_cost(results: pd.DataFrame) -> dict[str, float | None]:
    rows = results[results["method"].str.startswith("merge_all_prototype_learned_task_mask_")]
    return {
        str(row["router_type"]): _float_or_none(row.get("routing_error_cost"))
        for row in rows.to_dict(orient="records")
    }


def _learned_router_metrics(results: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    rows = results[results["method"].str.startswith("task_learned_")]
    metrics: dict[str, dict[str, float | None]] = {}
    for row in rows.to_dict(orient="records"):
        router_type = str(row["router_type"])
        metrics[router_type] = {
            "overall_acc": _float_or_none(row.get("overall_acc")),
            "router_acc": _float_or_none(row.get("router_acc")),
            "route_vs_merge_gap": _float_or_none(row.get("route_vs_merge_gap")),
        }
    return metrics


def _method_value(results: pd.DataFrame, method: str, column: str) -> float | None:
    rows = results[results["method"] == method]
    if rows.empty:
        return None
    return _float_or_none(rows.iloc[0][column])


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


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
