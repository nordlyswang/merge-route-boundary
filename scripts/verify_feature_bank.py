#!/usr/bin/env python
"""Verify a stored frozen feature bank."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.features.storage import DEFAULT_FEATURE_ROOT, FEATURE_ROOT_ENV, feature_bank_dir
from mrb.features.verify import verify_feature_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--backbone", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_root = (args.feature_root or Path(os.environ.get(FEATURE_ROOT_ENV, DEFAULT_FEATURE_ROOT))).expanduser()
    bank_dir = feature_bank_dir(
        feature_root,
        dataset_id=args.dataset,
        split=args.split,
        backbone_id=args.backbone,
    )
    result = verify_feature_bank(bank_dir)
    print(f"Feature bank: {result.path}")
    if result.ok:
        metadata = result.metadata
        print(
            "OK "
            f"N={metadata.get('num_samples')} D={metadata.get('feature_dim')} "
            f"dtype={metadata.get('dtype')}"
        )
        for warning in result.warnings:
            print(f"WARN {warning}")
        return 0

    for error in result.errors:
        print(f"ERROR {error}")
    for warning in result.warnings:
        print(f"WARN {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
