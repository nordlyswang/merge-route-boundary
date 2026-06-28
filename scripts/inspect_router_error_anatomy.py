#!/usr/bin/env python
"""Inspect Router Error Anatomy v0 summary output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    print(f"summary={args.summary}")
    print(
        "router_recall: "
        f"top1={_fmt(summary.get('top1_acc'))} "
        f"top2={_fmt(summary.get('top2_recall'))} "
        f"top3={_fmt(summary.get('top3_recall'))} "
        f"top5={_fmt(summary.get('top5_recall'))}"
    )
    print(f"worst_per_task_router_acc={summary.get('worst_task_by_router_acc')}")
    print("most_confused_pairs:")
    for row in summary.get("top_confused_pairs", []):
        print(
            "  "
            f"true={row.get('true_task')} predicted={row.get('predicted_task')} "
            f"count={row.get('count')} rate={_fmt(row.get('rate_given_true_task'))}"
        )
    print(
        "confidence: "
        f"mean_correct={_fmt(summary.get('mean_confidence_correct'))} "
        f"mean_wrong={_fmt(summary.get('mean_confidence_wrong'))} "
        f"auc={_fmt(summary.get('confidence_auc_for_correct_route'))}"
    )
    correlations = summary.get("diagnostics_confusion_correlations", {})
    print(
        "diagnostics_confusion_correlation: "
        f"pearson_auc={_fmt(correlations.get('pearson_auc_vs_confusion'))} "
        f"spearman_auc={_fmt(correlations.get('spearman_auc_vs_confusion'))} "
        f"pearson_separation={_fmt(correlations.get('pearson_separation_vs_confusion'))} "
        f"spearman_separation={_fmt(correlations.get('spearman_separation_vs_confusion'))}"
    )
    print(
        "method_gaps: "
        f"oracle_gap={_fmt(summary.get('oracle_gap'))} "
        f"hard_route_gap={_fmt(summary.get('hard_route_gap'))} "
        f"topk_k2_gap={_fmt(summary.get('topk_k2_gap'))} "
        f"fallback_gap={_fmt(summary.get('fallback_gap'))}"
    )
    print(f"topk_k2_top_task_gains={summary.get('topk_k2_top_task_gains')}")
    print(f"fallback_top_task_gains={summary.get('fallback_top_task_gains')}")
    print("final_recommendation:")
    for item in summary.get("interpretation", []):
        print(f"  {item}")
    return 0


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
