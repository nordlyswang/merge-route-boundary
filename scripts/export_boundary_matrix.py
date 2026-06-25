#!/usr/bin/env python
"""Export a pairwise diagnostics CSV as a task x task metric matrix."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--diagonal", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    if args.metric not in df.columns:
        print(f"ERROR metric {args.metric!r} is not in {args.input}")
        return 1
    tasks = sorted(set(df["task_i"].astype(int)) | set(df["task_j"].astype(int)))
    index = {task_id: offset for offset, task_id in enumerate(tasks)}
    diagonal = args.diagonal
    if diagonal is None:
        diagonal = 1.0 if ("acc" in args.metric or "auc" in args.metric) else 0.0
    matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, diagonal)
    for _, row in df.iterrows():
        value = row[args.metric]
        if "status" in row and row["status"] != "ok":
            value = math.nan
        i = index[int(row["task_i"])]
        j = index[int(row["task_j"])]
        matrix[i, j] = value
        matrix[j, i] = value
    output = args.output or args.input.with_name(f"{args.input.stem}_{args.metric}_matrix.csv")
    matrix_df = pd.DataFrame(matrix, index=tasks, columns=tasks)
    matrix_df.index.name = "task_id"
    matrix_df.to_csv(output)
    print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
