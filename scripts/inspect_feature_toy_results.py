#!/usr/bin/env python
"""Inspect feature toy baseline result CSVs."""

from __future__ import annotations

import argparse
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
    _print_method(df, "merge_all_prototype_learned_task_mask_centroid")
    _print_method(df, "merge_all_prototype_learned_task_mask_linear")
    _print_method(df, "merge_all_prototype_learned_task_mask_energy")
    _print_method(df, "task_oracle_prototype")
    _print_method(df, "task_learned_centroid_router_prototype")
    _print_method(df, "task_learned_linear_router_prototype")
    _print_method(df, "task_learned_energy_router_prototype")
    _print_route_gap(df)
    _print_decomposition(df)
    _print_frontier(df)
    failed = df[df["status"] != "ok"] if "status" in df else df.iloc[0:0]
    print(f"failed_methods={failed.shape[0]}")
    print(f"has_nan_overall_acc={bool(df['overall_acc'].isna().any())}")
    if not failed.empty:
        for row in failed.to_dict(orient="records"):
            print(f"  {row['method']} K={row['K']} warning={row.get('warning', '')}")
    return 0


def _print_method(df: pd.DataFrame, method: str) -> None:
    rows = df[df["method"] == method]
    if rows.empty:
        print(f"{method}: missing")
        return
    row = rows.iloc[0]
    print(
        f"{method}: overall_acc={float(row['overall_acc']):.6f} "
        f"mean_task_acc={float(row['mean_task_acc']):.6f} "
        f"router_acc={row['router_acc']} "
        f"route_vs_merge_gap={float(row['route_vs_merge_gap']):.6f}"
    )


def _print_route_gap(df: pd.DataFrame) -> None:
    merge = _method_acc(df, "merge_all_prototype")
    oracle = _method_acc(df, "task_oracle_prototype")
    learned = _method_acc(df, "task_learned_centroid_router_prototype")
    if merge is None:
        return
    if oracle is not None:
        print(f"oracle_route_gap={oracle - merge:.6f}")
    if learned is not None:
        print(f"learned_route_gap={learned - merge:.6f}")
        if oracle is not None:
            print(f"learned_vs_oracle_gap={learned - oracle:.6f}")


def _print_decomposition(df: pd.DataFrame) -> None:
    merge = _method_acc(df, "merge_all_prototype")
    oracle_mask = _method_acc(df, "merge_all_prototype_oracle_task_mask")
    oracle = _method_acc(df, "task_oracle_prototype")
    if merge is None or oracle_mask is None:
        return
    print(f"label_mask_gain={oracle_mask - merge:.6f}")
    if oracle is not None:
        print(f"classifier_specialization_gain={oracle - oracle_mask:.6f}")
    for method in (
        "merge_all_prototype_learned_task_mask_centroid",
        "merge_all_prototype_learned_task_mask_linear",
        "merge_all_prototype_learned_task_mask_energy",
    ):
        rows = df[df["method"] == method]
        if rows.empty:
            continue
        row = rows.iloc[0]
        print(
            f"routing_error_cost[{row['router_type']}]="
            f"{oracle_mask - float(row['overall_acc']):.6f}"
        )


def _print_frontier(df: pd.DataFrame) -> None:
    cluster = df[df["method"].isin(["cluster_oracle_prototype", "cluster_learned_prototype"])]
    if cluster.empty:
        print("cluster_frontier: none")
        return
    print("cluster_frontier:")
    for row in cluster.sort_values(["cluster_strategy", "method", "K"]).to_dict(orient="records"):
        print(
            f"  strategy={row['cluster_strategy']} method={row['method']} K={int(row['K'])} "
            f"overall_acc={float(row['overall_acc']):.6f} router_acc={row['router_acc']} "
            f"gap={float(row['route_vs_merge_gap']):.6f}"
        )


def _method_acc(df: pd.DataFrame, method: str) -> float | None:
    rows = df[df["method"] == method]
    if rows.empty:
        return None
    return float(rows.iloc[0]["overall_acc"])


if __name__ == "__main__":
    raise SystemExit(main())
