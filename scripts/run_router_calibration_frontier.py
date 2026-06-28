#!/usr/bin/env python
"""Run router calibration and mask-size frontier experiments."""

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
from mrb.baselines.router_calibration_frontier import (  # noqa: E402
    evaluate_router_calibration_frontier,
    write_router_calibration_outputs,
)
from mrb.data.splits import stable_hash  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "baselines" / "router_calibration_v0.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "baselines" / "router_calibration_v0"


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
    evaluation = evaluate_router_calibration_frontier(
        dataset,
        router_config=dict(effective.get("router", {})),
        calibration_config=dict(effective.get("calibration", {})),
        fallback_config=dict(effective.get("fallback", {})),
        soft_prior_config=dict(effective.get("soft_prior", {})),
        topk_config=dict(effective.get("topk_frontier", {})),
        oracle_topk_config=dict(effective.get("oracle_topk_decomposition", {})),
        config=effective,
    )
    paths = write_router_calibration_outputs(
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
        "router": dict(config.get("router", {})),
        "calibration": dict(config.get("calibration", {})),
        "fallback": dict(config.get("fallback", {})),
        "soft_prior": dict(config.get("soft_prior", {})),
        "topk_frontier": dict(config.get("topk_frontier", {})),
        "oracle_topk_decomposition": dict(config.get("oracle_topk_decomposition", {})),
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
        effective["oracle_topk_decomposition"]["random_extra_repeats"] = 2
    _validate_effective_config(effective)
    return effective


def _validate_effective_config(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    for key in ("stream", "seed", "backbone"):
        if key not in experiment or experiment[key] in (None, ""):
            raise ValueError(f"experiment.{key} is required")
    router_type = config.get("router", {}).get("type", "linear")
    if router_type != "linear":
        raise ValueError("router_calibration_v0 currently supports router.type=linear only")


def _resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _optional_resolve_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(value)


def _print_key_metrics(summary: dict[str, Any]) -> None:
    print(f"selected_temperature={summary['selected_temperature']}")
    print(f"merge_all_acc={summary['merge_all_acc']}")
    print(f"oracle_task_mask_acc={summary['oracle_task_mask_acc']}")
    print(f"previous_best_fallback_acc={summary['previous_best_fallback_acc']}")
    _print_brief("best_calibrated_fallback", summary.get("best_calibrated_fallback"))
    _print_brief("best_calibrated_soft_prior", summary.get("best_calibrated_soft_prior"))
    _print_brief("best_topk_by_val", summary.get("best_topk_by_val"))
    _print_brief("best_topk_by_test", summary.get("best_topk_by_test"))
    _print_brief("best_learned_control", summary.get("best_learned_control_by_overall_acc"))
    print(f"conclusion={summary['conclusion']}")


def _print_brief(label: str, row: dict[str, Any] | None) -> None:
    if row is None:
        print(f"{label}=none")
        return
    print(
        f"{label}: method={row['method']} overall_acc={row['overall_acc']} "
        f"val_acc={row.get('val_acc')} gap={row.get('route_vs_merge_gap')} "
        f"oracle_gap_closure={row.get('oracle_gap_closure')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
