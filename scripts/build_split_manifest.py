#!/usr/bin/env python
"""Build a deterministic task stream split manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.data.registry import DEFAULT_REGISTRY_PATH
from mrb.data.splits import write_manifest
from mrb.data.task_streams import build_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, help="Stream id from configs/task_streams/*.yaml.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--stream-config",
        type=Path,
        action="append",
        default=None,
        help="Optional stream config YAML. Defaults to configs/task_streams/*.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_split_manifest(
        stream_id=args.stream,
        seed=args.seed,
        registry_path=args.registry,
        stream_config_paths=args.stream_config,
        created_by="scripts/build_split_manifest.py",
    )
    write_manifest(manifest, args.output)
    print(f"Wrote split manifest: {args.output}")
    print(f"stream_id={manifest['stream_id']} seed={manifest['seed']} tasks={len(manifest['tasks'])}")
    print(f"manifest_hash={manifest['manifest_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
