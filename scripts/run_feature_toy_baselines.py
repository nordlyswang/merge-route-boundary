#!/usr/bin/env python
"""Run feature-level merge-vs-route toy baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.baselines.evaluate import evaluate_feature_toy_baselines  # noqa: E402
from mrb.baselines.feature_data import load_feature_toy_dataset  # noqa: E402
from mrb.baselines.report import DEFAULT_OUTPUT_DIR, write_feature_toy_outputs  # noqa: E402
from mrb.data.splits import stable_hash  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "baselines" / "feature_toy_v0.yaml"


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
    parser.add_argument("--cluster-ks", default=None, help="Comma-separated K values, e.g. 1,2,4,5,10")
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
    diagnostics_csv = _optional_resolve_path(experiment.get("diagnostics_csv"))

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
    classifier_types = _enabled_classifier_types(effective)
    evaluation = evaluate_feature_toy_baselines(
        dataset,
        cluster_ks=[int(value) for value in effective["clusters"]["ks"]],
        cluster_strategies=[str(value) for value in effective["clusters"]["strategies"]],
        diagnostics_csv=diagnostics_csv,
        classifier_types=classifier_types,
        linear_config=dict(effective.get("classifiers", {}).get("linear", {})),
        router_types=_enabled_router_types(effective),
        router_config=dict(effective.get("routers", {})),
        controls_config=dict(effective.get("controls", {})),
    )
    paths = write_feature_toy_outputs(
        evaluation,
        output_dir=output_dir,
        stream_id=stream_id,
        seed=seed,
        backbone_id=backbone_id,
        config=effective,
        feature_bank_metadata=dataset.feature_bank_metadata,
        manifest_path=dataset.manifest_path,
        diagnostics_csv=diagnostics_csv,
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
        "classifiers": {
            key: dict(value)
            for key, value in dict(config.get("classifiers", {})).items()
            if isinstance(value, dict)
        },
        "routers": {
            key: dict(value)
            for key, value in dict(config.get("routers", {})).items()
            if isinstance(value, dict)
        },
        "controls": dict(config.get("controls", {})),
        "clusters": dict(config.get("clusters", {})),
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
        effective["experiment"]["diagnostics_csv"] = str(args.diagnostics_csv)
    if args.cluster_ks is not None:
        effective["clusters"]["ks"] = [
            int(value.strip()) for value in args.cluster_ks.split(",") if value.strip()
        ]
    if args.quick:
        effective["data"]["max_train_per_task"] = 1000
        effective["data"]["max_test_per_task"] = 500
        effective["classifiers"].setdefault("linear", {})["enabled"] = False
        effective["routers"].setdefault("linear", {})["enabled"] = True
        effective["routers"]["linear"]["max_train_per_task"] = 1000
        effective["routers"].setdefault("centroid", {})["enabled"] = True
        effective["routers"].setdefault("energy", {})["enabled"] = True
    _validate_effective_config(effective)
    return effective


def _validate_effective_config(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    for key in ("stream", "seed", "backbone"):
        if key not in experiment or experiment[key] in (None, ""):
            raise ValueError(f"experiment.{key} is required")
    clusters = config["clusters"]
    if not clusters.get("ks"):
        raise ValueError("clusters.ks must be non-empty")
    if not clusters.get("strategies"):
        raise ValueError("clusters.strategies must be non-empty")


def _enabled_classifier_types(config: dict[str, Any]) -> list[str]:
    classifiers = config.get("classifiers", {})
    enabled: list[str] = []
    if bool(classifiers.get("prototype", {}).get("enabled", True)):
        enabled.append("prototype")
    if bool(classifiers.get("linear", {}).get("enabled", False)):
        enabled.append("linear")
    if not enabled:
        raise ValueError("At least one classifier must be enabled")
    return enabled


def _enabled_router_types(config: dict[str, Any]) -> list[str]:
    routers = config.get("routers", {})
    enabled = [
        router_type
        for router_type in ("centroid", "linear", "energy")
        if bool(routers.get(router_type, {}).get("enabled", False))
    ]
    if not enabled:
        raise ValueError("At least one task router must be enabled")
    return enabled


def _resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _optional_resolve_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(value)


def _print_key_metrics(results) -> None:
    methods = (
        "merge_all_prototype",
        "merge_all_prototype_oracle_task_mask",
        "merge_all_prototype_learned_task_mask_centroid",
        "merge_all_prototype_learned_task_mask_linear",
        "merge_all_prototype_learned_task_mask_energy",
        "task_oracle_prototype",
        "task_learned_centroid_router_prototype",
        "task_learned_linear_router_prototype",
        "task_learned_energy_router_prototype",
    )
    for method in methods:
        rows = results[results["method"] == method]
        if rows.empty:
            continue
        row = rows.iloc[0]
        print(
            f"{method}: overall_acc={row['overall_acc']:.6f} "
            f"router_acc={row['router_acc']} gap={row['route_vs_merge_gap']:.6f}"
        )
    first = results.iloc[0]
    if "label_mask_gain" in results:
        print(
            "decomposition: "
            f"label_mask_gain={first['label_mask_gain']} "
            f"classifier_specialization_gain={first['classifier_specialization_gain']}"
        )
    cluster = results[
        (results["method"] == "cluster_learned_prototype")
        & (results["cluster_strategy"] == "sequential")
    ]
    if not cluster.empty:
        frontier = ", ".join(
            f"K={int(row.K)} acc={float(row.overall_acc):.6f}"
            for row in cluster.sort_values("K").itertuples()
        )
        print(f"sequential_cluster_learned_frontier: {frontier}")


if __name__ == "__main__":
    raise SystemExit(main())
