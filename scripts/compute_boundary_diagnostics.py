#!/usr/bin/env python
"""Compute pairwise frozen-feature boundary diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.diagnostics.matrices import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    compute_pairwise_boundary_matrix,
)
from mrb.data.splits import stable_hash  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "diagnostics" / "boundary_v0.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--max-samples-per-task", type=int, default=None)
    parser.add_argument("--min-samples-per-task", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    strict = parser.add_mutually_exclusive_group()
    strict.add_argument(
        "--strict-features", dest="strict_features", action="store_true", default=None
    )
    strict.add_argument("--allow-partial-features", dest="strict_features", action="store_false")
    parser.add_argument("--fail-on-any-failed-pair", action="store_true")
    parser.add_argument("--min-ok-ratio", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    diagnostics = config.get("diagnostics", {})
    outputs = config.get("outputs", {})
    split = args.split or str(diagnostics.get("default_split", "train"))
    max_samples = (
        args.max_samples_per_task
        if args.max_samples_per_task is not None
        else diagnostics.get("max_samples_per_task", 500)
    )
    min_samples = (
        args.min_samples_per_task
        if args.min_samples_per_task is not None
        else int(diagnostics.get("min_samples_per_task", 50))
    )
    strict_features = (
        bool(args.strict_features)
        if args.strict_features is not None
        else bool(diagnostics.get("strict_features", True))
    )
    output_dir = args.output_dir or Path(outputs.get("root", DEFAULT_OUTPUT_DIR))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    df = compute_pairwise_boundary_matrix(
        stream_id=args.stream,
        seed=args.seed,
        backbone_id=args.backbone,
        split=split,
        max_samples_per_task=None if max_samples is None else int(max_samples),
        output_dir=output_dir,
        manifest_path=args.manifest,
        strict_features=strict_features,
        min_samples_per_task=min_samples,
        feature_root=args.feature_root,
        config_path=args.config,
        diagnostics_config=diagnostics,
        config_hash=stable_hash(config) if config else None,
        overwrite=args.overwrite,
    )
    ok_count = int((df["status"] == "ok").sum())
    failed_count = int((df["status"] == "failed").sum())
    ok_ratio = float(ok_count / len(df)) if len(df) else 1.0
    print(
        f"rows={len(df)} ok={ok_count} failed={failed_count} ok_ratio={ok_ratio:.3f}"
    )
    if "output_csv" in df.attrs:
        print(f"csv={df.attrs['output_csv']}")
    if "summary_json" in df.attrs:
        print(f"summary={df.attrs['summary_json']}")
    min_ok_ratio = (
        float(args.min_ok_ratio)
        if args.min_ok_ratio is not None
        else (1.0 if strict_features else 0.8)
    )
    gate = evaluate_quality_gate(
        failed_pairs=failed_count,
        ok_ratio=ok_ratio,
        min_ok_ratio=min_ok_ratio,
        fail_on_any_failed_pair=args.fail_on_any_failed_pair or strict_features,
    )
    if not gate["ok"]:
        print("ERROR boundary diagnostics quality gate failed:")
        print(f"  failed_pairs={failed_count}")
        print(f"  min_ok_ratio={min_ok_ratio}")
        print(f"  ok_ratio={ok_ratio:.3f}")
        warnings = Counter(
            str(value)
            for value in df.get("warning", [])
            if value is not None and str(value).strip()
        )
        if warnings:
            print("  top warnings:")
            for warning, count in warnings.most_common(5):
                print(f"  - count={count} {warning}")
        return 1
    return 0


def evaluate_quality_gate(
    *,
    failed_pairs: int,
    ok_ratio: float,
    min_ok_ratio: float,
    fail_on_any_failed_pair: bool,
) -> dict[str, object]:
    errors: list[str] = []
    if fail_on_any_failed_pair and failed_pairs > 0:
        errors.append("failed pairs are not allowed")
    if ok_ratio < min_ok_ratio:
        errors.append("ok ratio below threshold")
    return {"ok": not errors, "errors": errors}


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
