#!/usr/bin/env python
"""Inspect Pairwise-aware Routing Policy v0 result CSVs."""

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
    results = pd.read_csv(args.input)
    summary = _load_summary(args.input)
    per_task = _load_optional_csv(args.input, "_per_task.csv")
    pairs = _load_optional_csv(args.input, "_pairwise_policy_pairs.csv")

    print(f"file={args.input}")
    if summary:
        print(f"selected_temperature={summary.get('selected_temperature')}")
        thresholds = summary.get("selected_thresholds", {})
        print(
            "selected thresholds: "
            f"tau_fallback={_fmt(thresholds.get('tau_fallback'))} "
            f"tau_margin={_fmt(thresholds.get('tau_margin'))} "
            f"tau_pair_auc={_fmt(thresholds.get('tau_pair_auc'))} "
            f"tau_pair_confusion={_fmt(thresholds.get('tau_pair_confusion'))}"
        )

    _print_method(results, "merge_all_prototype")
    _print_method(results, "oracle_task_mask")
    _print_method(results, "hard_top1_linear")
    _print_method(results, "topk_linear_k2")
    _print_method(results, "fallback_linear_tau")
    _print_method(results, "previous_best_learned_control")
    _print_method(results, "pairwise_policy_linear_combined")
    _print_comparison(results)
    _print_task_gains(per_task)
    _print_top_pairs(pairs)

    failed = results[results["status"] != "ok"] if "status" in results else results.iloc[0:0]
    print(f"failed_methods={failed.shape[0]}")
    print(f"has_nan_overall_acc={bool(results['overall_acc'].isna().any())}")
    if not failed.empty:
        for row in failed.to_dict(orient="records"):
            print(f"  {row['method']} warning={row.get('warning', '')}")
    return 0


def _load_summary(results_path: Path) -> dict:
    summary_path = results_path.with_name(results_path.name.replace("_results.csv", "_summary.json"))
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_optional_csv(results_path: Path, suffix: str) -> pd.DataFrame:
    path = results_path.with_name(results_path.name.replace("_results.csv", suffix))
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _print_method(df: pd.DataFrame, method: str) -> None:
    rows = df[df["method"] == method]
    if rows.empty:
        print(f"{method}: missing")
        return
    row = rows.iloc[0]
    print(
        f"{method}: overall_acc={_fmt(row.get('overall_acc'))} "
        f"val_acc={_fmt(row.get('val_acc'))} "
        f"route_vs_merge_gap={_fmt(row.get('route_vs_merge_gap'))} "
        f"oracle_gap_closure={_fmt(row.get('oracle_gap_closure'))} "
        f"top1={_fmt(row.get('action_top1_rate'))} "
        f"top2={_fmt(row.get('action_top2_rate'))} "
        f"fallback={_fmt(row.get('action_fallback_rate'))} "
        f"mean_mask_size={_fmt(row.get('mean_mask_size'))}"
    )


def _print_comparison(df: pd.DataFrame) -> None:
    pairwise = _method_row(df, "pairwise_policy_linear_combined")
    fallback = _method_row(df, "fallback_linear_tau")
    topk_k2 = _method_row(df, "topk_linear_k2")
    if pairwise is None:
        return
    print("comparison against fallback and top-k k=2:")
    if fallback is not None:
        print(
            f"  vs fallback: acc_delta={_fmt(pairwise['overall_acc'] - fallback['overall_acc'])} "
            f"mask_delta={_fmt(pairwise['mean_mask_size'] - fallback['mean_mask_size'])}"
        )
    if topk_k2 is not None:
        print(
            f"  vs topk_k2: acc_delta={_fmt(pairwise['overall_acc'] - topk_k2['overall_acc'])} "
            f"mask_delta={_fmt(pairwise['mean_mask_size'] - topk_k2['mean_mask_size'])}"
        )
    print(f"  exceeds fallback 0.6732={bool(float(pairwise['overall_acc']) > 0.6732)}")


def _print_task_gains(per_task: pd.DataFrame) -> None:
    if per_task.empty:
        print("best/worst task gains: missing per_task.csv")
        return
    rows = per_task[per_task["method"] == "pairwise_policy_linear_combined"]
    if rows.empty:
        print("best/worst task gains: missing pairwise rows")
        return
    print("best task gains vs merge:")
    for row in rows.sort_values("gain_vs_merge_task", ascending=False).head(5).to_dict(
        orient="records"
    ):
        print(
            f"  task={int(row['task_id'])} gain={_fmt(row['gain_vs_merge_task'])} "
            f"acc={_fmt(row['task_acc'])}"
        )
    print("worst task gains vs merge:")
    for row in rows.sort_values("gain_vs_merge_task", ascending=True).head(5).to_dict(
        orient="records"
    ):
        print(
            f"  task={int(row['task_id'])} gain={_fmt(row['gain_vs_merge_task'])} "
            f"acc={_fmt(row['task_acc'])}"
        )


def _print_top_pairs(pairs: pd.DataFrame) -> None:
    if pairs.empty:
        print("top pairwise policy pairs: missing")
        return
    print("top pairwise policy pairs:")
    ordered = pairs.sort_values(
        ["action_top2_rate", "action_fallback_rate", "num_samples"],
        ascending=[False, False, False],
    ).head(10)
    for row in ordered.to_dict(orient="records"):
        print(
            f"  top1={int(row['top1_task'])} top2={int(row['top2_task'])} "
            f"n={int(row['num_samples'])} pair_auc={_fmt(row['pair_auc'])} "
            f"prior={_fmt(row['pair_confusion_prior'])} "
            f"top2_rate={_fmt(row['action_top2_rate'])} "
            f"fallback_rate={_fmt(row['action_fallback_rate'])} "
            f"acc={_fmt(row['accuracy_when_pair_appears'])}"
        )


def _method_row(df: pd.DataFrame, method: str) -> pd.Series | None:
    rows = df[df["method"] == method]
    if rows.empty:
        return None
    return rows.iloc[0]


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
