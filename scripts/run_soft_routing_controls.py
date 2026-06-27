#!/usr/bin/env python
"""Run soft, top-k, and fallback routing controls on frozen features."""

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
from mrb.baselines.report import (  # noqa: E402
    DEFAULT_SOFT_ROUTING_OUTPUT_DIR,
    write_soft_routing_outputs,
)
from mrb.baselines.routing_controls import evaluate_soft_routing_controls  # noqa: E402
from mrb.data.splits import stable_hash  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "baselines" / "soft_routing_v0.yaml"


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
    output_dir = _resolve_path(effective["outputs"].get("root", DEFAULT_SOFT_ROUTING_OUTPUT_DIR))
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
    evaluation = evaluate_soft_routing_controls(
        dataset,
        router_types=_enabled_router_types(effective),
        router_config=dict(effective.get("routers", {})),
        include_previous_controls=bool(
            effective.get("base_methods", {}).get("include_previous_controls", True)
        ),
        topk_config=dict(effective.get("topk_controls", {})),
        fallback_config=dict(effective.get("fallback_controls", {})),
        soft_prior_config=dict(effective.get("soft_prior_controls", {})),
    )
    paths = write_soft_routing_outputs(
        evaluation,
        output_dir=output_dir,
        stream_id=stream_id,
        seed=seed,
        backbone_id=backbone_id,
        config=effective,
        feature_bank_metadata=dataset.feature_bank_metadata,
        manifest_path=dataset.manifest_path,
        overwrite=args.overwrite,
    )

    print(f"config_hash={stable_hash(effective)}")
    for label, path in paths.items():
        print(f"{label}={path}")
    _print_key_metrics(evaluation.results)
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
        "base_methods": dict(config.get("base_methods", {})),
        "routers": {
            key: dict(value)
            for key, value in dict(config.get("routers", {})).items()
            if isinstance(value, dict)
        },
        "topk_controls": dict(config.get("topk_controls", {})),
        "fallback_controls": dict(config.get("fallback_controls", {})),
        "soft_prior_controls": dict(config.get("soft_prior_controls", {})),
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
        effective["routers"].setdefault("linear", {})["max_train_per_task"] = 1000
    _validate_effective_config(effective)
    return effective


def _validate_effective_config(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    for key in ("stream", "seed", "backbone"):
        if key not in experiment or experiment[key] in (None, ""):
            raise ValueError(f"experiment.{key} is required")
    if not _enabled_router_types(config):
        raise ValueError("At least one router must be enabled")


def _enabled_router_types(config: dict[str, Any]) -> list[str]:
    routers = config.get("routers", {})
    enabled = [
        router_type
        for router_type in ("linear", "energy")
        if bool(routers.get(router_type, {}).get("enabled", False))
    ]
    return enabled


def _resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _optional_resolve_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(value)


def _print_key_metrics(results) -> None:
    for method in (
        "merge_all_prototype",
        "merge_all_prototype_oracle_task_mask",
        "merge_all_prototype_learned_task_mask_linear",
        "merge_all_prototype_learned_task_mask_energy",
        "task_oracle_prototype",
        "task_learned_linear_router_prototype",
        "task_learned_energy_router_prototype",
    ):
        _print_method(results, method)
    for control_type in ("topk_task_mask", "fallback_to_merge_all", "soft_prior"):
        rows = results[results["control_type"] == control_type]
        if rows.empty:
            continue
        best = rows.sort_values("overall_acc", ascending=False).iloc[0]
        print(
            f"best_{control_type}: method={best['method']} "
            f"overall_acc={float(best['overall_acc']):.6f} "
            f"gap={float(best['route_vs_merge_gap']):.6f} "
            f"oracle_gap_closure={float(best['oracle_gap_closure']):.6f}"
        )


def _print_method(results, method: str) -> None:
    rows = results[results["method"] == method]
    if rows.empty:
        print(f"{method}: missing")
        return
    row = rows.iloc[0]
    print(
        f"{method}: overall_acc={float(row['overall_acc']):.6f} "
        f"router_acc_top1={row['router_acc_top1']} "
        f"gap={float(row['route_vs_merge_gap']):.6f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
