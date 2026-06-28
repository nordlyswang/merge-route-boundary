#!/usr/bin/env python
"""Run Router Error Anatomy v0 on frozen feature banks."""

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
from mrb.baselines.router_error_anatomy import (  # noqa: E402
    evaluate_router_error_anatomy,
    write_router_error_anatomy_outputs,
)
from mrb.data.splits import stable_hash  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "baselines" / "router_error_anatomy_v0.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "baselines" / "router_error_anatomy_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stream", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
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
    output_dir = _resolve_path(effective["outputs"].get("root", DEFAULT_OUTPUT_DIR))
    if args.output_dir is not None:
        output_dir = _resolve_path(args.output_dir)

    stream_id = str(experiment["stream"])
    seed = int(experiment["seed"])
    backbone_id = str(experiment["backbone"])
    manifest_path = _optional_resolve_path(experiment.get("split_manifest"))
    diagnostics_csv = _optional_resolve_path(effective["inputs"].get("diagnostics_csv"))

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
    evaluation = evaluate_router_error_anatomy(
        dataset,
        router_config=dict(effective.get("router", {})),
        calibration_config=dict(effective.get("calibration", {})),
        fallback_config=dict(effective.get("fallback", {})),
        analysis_config=dict(effective.get("analysis", {})),
        diagnostics_csv=diagnostics_csv,
        config=effective,
    )
    paths = write_router_error_anatomy_outputs(
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
        "data": dict(config.get("data", {})),
        "inputs": dict(config.get("inputs", {})),
        "router": dict(config.get("router", {})),
        "calibration": dict(config.get("calibration", {})),
        "fallback": dict(config.get("fallback", {})),
        "analysis": dict(config.get("analysis", {})),
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
    if args.quick:
        effective["data"]["max_train_per_task"] = 1000
        effective["data"]["max_val_per_task"] = 500
        effective["data"]["max_test_per_task"] = 500
        effective["router"]["max_train_per_task"] = 1000
    _validate_effective_config(effective)
    return effective


def _validate_effective_config(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    for key in ("stream", "seed", "backbone"):
        if key not in experiment or experiment[key] in (None, ""):
            raise ValueError(f"experiment.{key} is required")
    if str(config.get("router", {}).get("type", "linear")) != "linear":
        raise ValueError("router_error_anatomy_v0 currently supports router.type=linear only")


def _resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _optional_resolve_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(value)


def _print_key_metrics(summary: dict[str, Any]) -> None:
    print(f"router_type={summary['router_type']}")
    print(f"selected_temperature={summary.get('selected_temperature')}")
    print(f"top1_acc={summary['top1_acc']:.6f}")
    print(f"top2_recall={summary['top2_recall']:.6f}")
    print(f"top3_recall={summary['top3_recall']:.6f}")
    print(f"top5_recall={summary['top5_recall']:.6f}")
    print(f"merge_all_acc={summary['merge_all_acc']:.6f}")
    print(f"oracle_task_mask_acc={summary['oracle_task_mask_acc']:.6f}")
    print(f"hard_route_acc={summary['hard_route_acc']:.6f}")
    print(f"topk_k2_acc={summary['topk_k2_acc']:.6f}")
    print(f"fallback_acc={summary['fallback_acc']:.6f}")
    print(
        "confidence_correct_vs_wrong="
        f"{summary['mean_confidence_correct']:.6f}/"
        f"{summary['mean_confidence_wrong']:.6f}"
    )
    print(
        "confidence_auc_for_correct_route="
        f"{summary['confidence_auc_for_correct_route']:.6f}"
    )
    print(f"worst_task_by_router_acc={summary.get('worst_task_by_router_acc')}")
    print(f"most_confused_pair={summary.get('most_confused_pair')}")
    print(f"interpretation={summary.get('interpretation')}")


if __name__ == "__main__":
    raise SystemExit(main())
