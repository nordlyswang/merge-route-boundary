#!/usr/bin/env python
"""Run Pairwise-aware Routing Policy v0 on frozen features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.baselines.feature_data import load_feature_toy_dataset  # noqa: E402
from mrb.baselines.pairwise_policy import (  # noqa: E402
    evaluate_pairwise_policy,
    write_pairwise_policy_outputs,
)
from mrb.data.splits import stable_hash  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "baselines" / "pairwise_policy_v0.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "baselines" / "pairwise_policy_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stream", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--diagnostics-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    effective = _effective_config(config, args)
    experiment = effective["experiment"]
    data_config = effective["data"]
    inputs = effective["inputs"]
    output_dir = _resolve_path(effective["outputs"].get("root", DEFAULT_OUTPUT_DIR))
    if args.output_dir is not None:
        output_dir = _resolve_path(args.output_dir)

    stream_id = str(experiment["stream"])
    seed = int(experiment["seed"])
    backbone_id = str(experiment["backbone"])
    manifest_path = _optional_resolve_path(experiment.get("split_manifest"))
    diagnostics_csv = _optional_resolve_path(inputs.get("diagnostics_csv"))

    dataset = load_feature_toy_dataset(
        stream_id=stream_id,
        seed=seed,
        backbone_id=backbone_id,
        manifest_path=manifest_path,
        feature_root=args.feature_root,
        max_train_per_task=data_config.get("max_train_per_task"),
        max_val_per_task=data_config.get("max_val_per_task"),
        max_test_per_task=data_config.get("max_test_per_task"),
    )
    evaluation = evaluate_pairwise_policy(
        dataset,
        router_config=dict(effective.get("router", {})),
        calibration_config=dict(effective.get("calibration", {})),
        policy_config=dict(effective.get("policy", {})),
        diagnostics_csv=diagnostics_csv,
        config=effective,
    )
    paths = write_pairwise_policy_outputs(
        evaluation,
        output_dir=output_dir,
        stream_id=stream_id,
        seed=seed,
        backbone_id=backbone_id,
        overwrite=args.overwrite,
    )

    print(f"config_hash={stable_hash(effective)}")
    for label, path in paths.items():
        print(f"{label}={path}")
    _print_key_metrics(evaluation.summary)
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _effective_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    effective = {
        "experiment": dict(config.get("experiment", {})),
        "inputs": dict(config.get("inputs", {})),
        "data": dict(config.get("data", {})),
        "router": dict(config.get("router", {})),
        "calibration": dict(config.get("calibration", {})),
        "policy": dict(config.get("policy", {})),
        "outputs": dict(config.get("outputs", {})),
    }
    if args.stream is not None:
        effective["experiment"]["stream"] = args.stream
    if args.seed is not None:
        effective["experiment"]["seed"] = int(args.seed)
    if args.backbone is not None:
        effective["experiment"]["backbone"] = args.backbone
    if args.manifest is not None:
        effective["experiment"]["split_manifest"] = str(args.manifest)
    if args.diagnostics_csv is not None:
        effective["inputs"]["diagnostics_csv"] = str(args.diagnostics_csv)
    if args.quick:
        effective["data"]["max_train_per_task"] = 1000
        effective["data"]["max_val_per_task"] = 500
        effective["data"]["max_test_per_task"] = 500
        effective["router"]["max_train_per_task"] = 1000
        effective["policy"]["tau_fallback_grid"] = [0.0, 0.5, 0.6, 0.7]
        effective["policy"]["tau_margin_grid"] = [0.05, 0.1, 0.2]
        effective["policy"]["tau_pair_auc_grid"] = [0.93, 0.97, 0.99]
        effective["policy"]["tau_pair_confusion_grid"] = [0.04, 0.08]
    _validate_effective_config(effective)
    return effective


def _validate_effective_config(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    for key in ("stream", "seed", "backbone"):
        if key not in experiment or experiment[key] in (None, ""):
            raise ValueError(f"experiment.{key} is required")
    router_type = config.get("router", {}).get("type", "linear")
    if router_type != "linear":
        raise ValueError("pairwise_policy_v0 currently supports router.type=linear only")


def _resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _optional_resolve_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(value)


def _print_key_metrics(summary: dict[str, Any]) -> None:
    thresholds = summary["selected_thresholds"]
    pairwise = summary.get("pairwise_policy_linear_combined") or {}
    print(f"selected_temperature={summary['selected_temperature']}")
    print(
        "selected_thresholds="
        f"tau_fallback={thresholds['tau_fallback']} "
        f"tau_margin={thresholds['tau_margin']} "
        f"tau_pair_auc={thresholds['tau_pair_auc']} "
        f"tau_pair_confusion={thresholds['tau_pair_confusion']}"
    )
    print(
        "pairwise_policy_linear_combined: "
        f"overall_acc={pairwise.get('overall_acc')} "
        f"val_acc={pairwise.get('val_acc')} "
        f"route_vs_merge_gap={pairwise.get('route_vs_merge_gap')} "
        f"oracle_gap_closure={pairwise.get('oracle_gap_closure')} "
        f"mean_mask_size={pairwise.get('mean_mask_size')}"
    )
    print(
        "actions: "
        f"top1={pairwise.get('action_top1_rate')} "
        f"top2={pairwise.get('action_top2_rate')} "
        f"fallback={pairwise.get('action_fallback_rate')}"
    )
    print(
        "comparison: "
        f"exceeds_previous_best_fallback={summary['pairwise_exceeds_previous_best_fallback']} "
        f"lowers_mean_mask_vs_fallback={summary['pairwise_lowers_mean_mask_vs_fallback']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
