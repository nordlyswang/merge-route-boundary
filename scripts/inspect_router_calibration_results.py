#!/usr/bin/env python
"""Inspect router calibration frontier result CSVs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    summary = _load_summary(args.input)
    print(f"file={args.input}")
    if summary:
        print(f"selected_temperature={summary.get('selected_temperature')}")
        print(f"score_source={summary.get('score_source')}")
        _print_calibration(summary.get("router_calibration_metrics", {}))
    _print_method(df, "merge_all_prototype")
    _print_method(df, "merge_all_prototype_oracle_task_mask")
    _print_method(df, "merge_all_prototype_fallback_linear_val_tau")
    _print_method(df, "merge_all_prototype_fallback_linear_calibrated_tau")
    _print_method(df, "merge_all_prototype_soft_prior_linear_calibrated_beta")
    _print_frontier(df, "topk_task_mask", "top-k learned frontier")
    _print_frontier(df, "oracle_topk_decomposition", "oracle top-k decomposition")
    failed = df[df["status"] != "ok"] if "status" in df else df.iloc[0:0]
    print(f"failed_methods={failed.shape[0]}")
    print(f"has_nan_overall_acc={bool(df['overall_acc'].isna().any())}")
    if not failed.empty:
        for row in failed.to_dict(orient="records"):
            print(f"  {row['method']} warning={row.get('warning', '')}")
    if summary:
        print(f"conclusion={summary.get('conclusion')}")
    return 0


def _load_summary(results_path: Path) -> dict:
    summary_path = results_path.with_name(results_path.name.replace("_results.csv", "_summary.json"))
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _print_calibration(metrics: dict) -> None:
    for label in ("uncalibrated_val", "calibrated_val", "uncalibrated_test", "calibrated_test"):
        values = metrics.get(label, {})
        if not values:
            continue
        print(
            f"{label}: ece={_fmt(values.get('ece'))} nll={_fmt(values.get('nll'))} "
            f"brier={_fmt(values.get('brier'))} top1={_fmt(values.get('top1_acc'))} "
            f"top2={_fmt(values.get('top2_recall'))} top3={_fmt(values.get('top3_recall'))}"
        )


def _print_method(df: pd.DataFrame, method: str) -> None:
    rows = df[df["method"] == method]
    if rows.empty:
        print(f"{method}: missing")
        return
    print(_format_row(rows.iloc[0]))


def _print_frontier(df: pd.DataFrame, control_type: str, label: str) -> None:
    rows = df[df["control_type"] == control_type]
    if rows.empty:
        print(f"{label}: missing")
        return
    print(f"{label}:")
    for row in rows.sort_values("topk").to_dict(orient="records"):
        print(f"  {_format_row(pd.Series(row))}")


def _format_row(row: pd.Series) -> str:
    return (
        f"{row['method']}: overall_acc={_fmt(row.get('overall_acc'))} "
        f"val_acc={_fmt(row.get('val_acc'))} topk={_fmt(row.get('topk'))} "
        f"tau={_fmt(row.get('tau'))} beta={_fmt(row.get('beta'))} "
        f"router_acc_top1={_fmt(row.get('router_acc_top1'))} "
        f"router_recall_topk={_fmt(row.get('router_recall_topk'))} "
        f"router_ece={_fmt(row.get('router_ece'))} "
        f"fallback_rate={_fmt(row.get('fallback_rate'))} "
        f"mean_mask_size={_fmt(row.get('mean_mask_size'))} "
        f"route_vs_merge_gap={_fmt(row.get('route_vs_merge_gap'))} "
        f"oracle_gap_closure={_fmt(row.get('oracle_gap_closure'))} "
        f"mask_size_cost={_fmt(row.get('mask_size_cost'))}"
    )


def _fmt(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(number):
        return "nan"
    return f"{number:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
