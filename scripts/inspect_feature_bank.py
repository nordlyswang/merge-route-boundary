#!/usr/bin/env python
"""Inspect available frozen feature banks under MRB_FEATURE_ROOT."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.features.storage import DEFAULT_FEATURE_ROOT, FEATURE_ROOT_ENV, format_bytes, list_feature_banks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_root = (args.feature_root or Path(os.environ.get(FEATURE_ROOT_ENV, DEFAULT_FEATURE_ROOT))).expanduser()
    print(f"MRB_FEATURE_ROOT: {feature_root}")
    print("canonical_layout: MRB_FEATURE_ROOT/<dataset_id>/<split>/<backbone_id>/")
    banks = list_feature_banks(feature_root)
    if not banks:
        print("No feature banks found.")
        return 0

    for bank in banks:
        if not bank.get("valid_metadata"):
            print(f"WARN {bank['path']} metadata_error={bank.get('error')}")
            continue
        metadata = bank["metadata"]
        print(
            f"{metadata.get('dataset_id'):18s} {metadata.get('split'):10s} "
            f"{metadata.get('backbone_id'):14s} "
            f"N={metadata.get('num_samples')} D={metadata.get('feature_dim')} "
            f"dtype={metadata.get('dtype')} size={format_bytes(int(bank['size_bytes']))} "
            f"path={bank['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
