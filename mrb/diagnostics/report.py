"""Summary helpers for boundary diagnostic matrices."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mrb.diagnostics import METRIC_VERSION


def summarize_boundary_matrix(
    df: pd.DataFrame,
    *,
    feature_bank_metadata: dict[str, Any] | None = None,
    manifest_path: str | None = None,
    metric_version: str = METRIC_VERSION,
) -> dict[str, Any]:
    ok = df[df["status"] == "ok"] if "status" in df else df.iloc[0:0]
    tasks = sorted(set(df.get("task_i", [])) | set(df.get("task_j", [])))
    metric = "linear_probe_auc_symmetric"
    if not ok.empty and metric in ok:
        sorted_pairs = ok.sort_values(metric, ascending=False)
    else:
        sorted_pairs = ok
    least_pairs = (
        sorted_pairs.tail(5).sort_values(metric)
        if not sorted_pairs.empty and metric in sorted_pairs
        else sorted_pairs.iloc[0:0]
    )
    return {
        "num_tasks": len(tasks),
        "num_pairs": int(len(df)),
        "num_ok_pairs": int(len(ok)),
        "num_failed_pairs": int(len(df) - len(ok)),
        "mean_linear_probe_auc_symmetric": _mean_or_none(ok, "linear_probe_auc_symmetric"),
        "mean_knn_domain_acc": _mean_or_none(ok, "knn_domain_acc"),
        "mean_centroid_l2": _mean_or_none(ok, "centroid_l2"),
        "mean_separation_ratio": _mean_or_none(ok, "separation_ratio"),
        "most_separable_pairs": _pair_records(sorted_pairs.head(5), metric),
        "least_separable_pairs": _pair_records(least_pairs, metric),
        "feature_bank_metadata": feature_bank_metadata or {},
        "manifest_path": manifest_path,
        "metric_version": metric_version,
    }


def _mean_or_none(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df:
        return None
    value = df[column].mean(skipna=True)
    return None if pd.isna(value) else float(value)


def _pair_records(df: pd.DataFrame, metric: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        value = row.get(metric)
        records.append(
            {
                "task_i": int(row["task_i"]),
                "task_j": int(row["task_j"]),
                metric: None if pd.isna(value) else float(value),
                "dataset_i": row.get("dataset_i"),
                "dataset_j": row.get("dataset_j"),
            }
        )
    return records
