#!/usr/bin/env python
"""Inspect generated boundary diagnostics matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.diagnostics.matrices import DEFAULT_OUTPUT_DIR  # noqa: E402
from mrb.diagnostics.report import summarize_boundary_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [args.input] if args.input else [
        path for path in sorted(args.output_dir.glob("*.csv")) if not path.stem.endswith("_matrix")
    ]
    if not paths:
        print(f"No diagnostics CSV files found under {args.output_dir}")
        return 0
    for path in paths:
        if path is None:
            continue
        df = pd.read_csv(path)
        summary = summarize_boundary_matrix(df)
        print(f"file={path}")
        print(
            f"pairs={summary['num_pairs']} ok={summary['num_ok_pairs']} "
            f"failed={summary['num_failed_pairs']}"
        )
        print(
            "means "
            f"auc_sym={summary['mean_linear_probe_auc_symmetric']} "
            f"knn_acc={summary['mean_knn_domain_acc']} "
            f"separation_ratio={summary['mean_separation_ratio']}"
        )
        _print_pairs("most", summary["most_separable_pairs"], "linear_probe_auc_symmetric")
        _print_pairs("least", summary["least_separable_pairs"], "linear_probe_auc_symmetric")
        failed = df[df["status"] == "failed"] if "status" in df else df.iloc[0:0]
        if not failed.empty:
            print("warnings:")
            for warning, count in failed["warning"].fillna("").value_counts().head(5).items():
                print(f"  count={count} warning={warning}")
    return 0


def _print_pairs(label: str, pairs: list[dict], metric: str) -> None:
    print(f"{label}:")
    if not pairs:
        print("  none")
        return
    for item in pairs:
        print(f"  task {item['task_i']} vs {item['task_j']} {metric}={item.get(metric)}")


if __name__ == "__main__":
    raise SystemExit(main())
