#!/usr/bin/env python
"""Inspect registered datasets without downloading or modifying data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.data.fingerprint import registry_fingerprints
from mrb.data.registry import (
    DEFAULT_REGISTRY_PATH,
    check_project_data_symlink,
    inspect_registry,
    load_registry,
)


DEFAULT_STATUS_JSON = REPO_ROOT / "artifacts" / "dataset_status.json"


def format_gb(size_bytes: int) -> float:
    return size_bytes / (1024**3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--project-data", type=Path, default=None)
    parser.add_argument(
        "--status-json",
        type=Path,
        default=DEFAULT_STATUS_JSON,
        help="Small local audit JSON output. Use --no-status-json to disable.",
    )
    parser.add_argument("--no-status-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    data_root = args.data_root or registry.data_root()
    project_data = args.project_data or registry.project_data()
    os.environ.setdefault(registry.root_env, str(data_root))

    print(f"Registry: {registry.path}")
    print(f"MRB_DATA_ROOT: {data_root}")
    print(f"Project data: {project_data}")
    if not data_root.exists():
        print(f"ERROR data root does not exist: {data_root}")
        return 1

    link_status = check_project_data_symlink(project_data, data_root)
    print(f"Project data mode: {link_status['mode']}")
    if link_status["target"]:
        print(f"Project data target: {link_status['target']}")
    for warning in link_status["warnings"]:
        print(f"WARN {warning}")

    statuses = inspect_registry(registry, data_root=data_root, load_smoke_metadata=True)
    print("\nDatasets:")
    missing_expected = []
    for status in statuses:
        state = "available" if status["available"] else "missing"
        if status["expected_available"] and not status["available"]:
            missing_expected.append(status["dataset_id"])
        split_sizes = status.get("split_sizes", {})
        split_text = ""
        if split_sizes:
            split_text = " " + ", ".join(f"{name}={size}" for name, size in split_sizes.items())
        print(
            f"  {state:9s} tier={status['tier']} {status['dataset_id']:18s} "
            f"size={format_gb(status['disk_usage_bytes']):.4f} GB "
            f"loader={status['loader_status']}{split_text}"
        )
        for warning in status.get("warnings", []):
            print(f"    WARN {warning}")

    if not args.no_status_json and args.status_json:
        report = {
            "schema_version": 1,
            "registry": str(registry.path),
            "data_root": str(data_root),
            "project_data_status": link_status,
            "datasets": statuses,
            "fingerprints": registry_fingerprints(registry, max_paths=100),
        }
        args.status_json.parent.mkdir(parents=True, exist_ok=True)
        args.status_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote status JSON: {args.status_json}")

    if missing_expected:
        print("\nWARN expected datasets missing: " + ", ".join(missing_expected))
    if link_status["mode"] != "full_symlink":
        print("WARN project data path is not the full symlink requested for this phase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
