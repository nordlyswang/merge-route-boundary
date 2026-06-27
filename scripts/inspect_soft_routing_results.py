#!/usr/bin/env python
"""Inspect soft routing control result CSVs."""

from __future__ import annotations

import argparse
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
    print(f"file={args.input}")
    _print_method(df, "merge_all_prototype")
    _print_method(df, "merge_all_prototype_oracle_task_mask")
    _print_method(df, "merge_all_prototype_learned_task_mask_linear")
    _print_method(df, "merge_all_prototype_learned_task_mask_energy")
    _print_method(df, "task_oracle_prototype")
    _print_method(df, "task_learned_linear_router_prototype")
    _print_method(df, "task_learned_energy_router_prototype")
    _print_best(df)
    _print_control_best(df, "topk_task_mask")
    _print_control_best(df, "fallback_to_merge_all")
    _print_control_best(df, "soft_prior")
    failed = df[df["status"] != "ok"] if "status" in df else df.iloc[0:0]
    print(f"failed_methods={failed.shape[0]}")
    print(f"has_nan_overall_acc={bool(df['overall_acc'].isna().any())}")
    if not failed.empty:
        for row in failed.to_dict(orient="records"):
            print(f"  {row['method']} warning={row.get('warning', '')}")
    return 0


def _print_method(df: pd.DataFrame, method: str) -> None:
    rows = df[df["method"] == method]
    if rows.empty:
        print(f"{method}: missing")
        return
    row = rows.iloc[0]
    print(_format_row(method, row))


def _print_best(df: pd.DataFrame) -> None:
    ok = df[df["status"] == "ok"] if "status" in df else df
    if ok.empty:
        print("best_method_by_overall_acc: none")
        return
    row = ok.sort_values("overall_acc", ascending=False).iloc[0]
    print(f"best_method_by_overall_acc: {_format_row(str(row['method']), row)}")


def _print_control_best(df: pd.DataFrame, control_type: str) -> None:
    rows = df[df["control_type"] == control_type]
    if rows.empty:
        print(f"{control_type}: missing")
        return
    print(f"{control_type}:")
    for router_type in sorted(str(value) for value in rows["router_type"].dropna().unique()):
        router_rows = rows[rows["router_type"] == router_type]
        row = router_rows.sort_values("overall_acc", ascending=False).iloc[0]
        print(f"  {router_type}: {_format_row(str(row['method']), row)}")


def _format_row(method: str, row: pd.Series) -> str:
    return (
        f"{method}: overall_acc={float(row['overall_acc']):.6f} "
        f"mean_task_acc={float(row['mean_task_acc']):.6f} "
        f"router_acc_top1={_fmt(row.get('router_acc_top1'))} "
        f"router_recall_topk={_fmt(row.get('router_recall_topk'))} "
        f"fallback_rate={_fmt(row.get('fallback_rate'))} "
        f"mean_mask_size={_fmt(row.get('mean_mask_size'))} "
        f"route_vs_merge_gap={_fmt(row.get('route_vs_merge_gap'))} "
        f"oracle_gap_closure={_fmt(row.get('oracle_gap_closure'))} "
        f"topk={_fmt(row.get('topk'))} tau={_fmt(row.get('tau'))} beta={_fmt(row.get('beta'))}"
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
