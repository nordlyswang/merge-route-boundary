#!/usr/bin/env python
"""Create the project features symlink pointing at MRB_FEATURE_ROOT."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.features.storage import DEFAULT_FEATURE_ROOT, FEATURE_ROOT_ENV, project_feature_link


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path(os.environ.get(FEATURE_ROOT_ENV, DEFAULT_FEATURE_ROOT)),
        help="Shared feature cache root. Defaults to MRB_FEATURE_ROOT or the project recommendation.",
    )
    parser.add_argument(
        "--project-features",
        type=Path,
        default=project_feature_link(),
        help="Project-local symlink path.",
    )
    return parser.parse_args()


def create_feature_symlink(project_features: Path, feature_root: Path) -> None:
    feature_root = feature_root.expanduser()
    project_features = project_features.expanduser()
    feature_root.mkdir(parents=True, exist_ok=True)
    project_features.parent.mkdir(parents=True, exist_ok=True)

    if project_features.is_symlink():
        if project_features.resolve() == feature_root.resolve():
            print(f"Project features symlink already exists: {project_features} -> {feature_root}")
            return
        print(
            f"ERROR: {project_features} points to {project_features.resolve()}, "
            f"expected {feature_root.resolve()}."
        )
        raise SystemExit(1)

    if project_features.exists():
        if project_features.is_dir() and not any(project_features.iterdir()):
            project_features.rmdir()
        else:
            print(f"ERROR: {project_features} exists and is not an empty directory. Refusing to overwrite.")
            raise SystemExit(1)

    project_features.symlink_to(feature_root, target_is_directory=True)
    print(f"Project features symlink: {project_features} -> {feature_root}")
    print(f"{FEATURE_ROOT_ENV}: {feature_root}")


def main() -> int:
    args = parse_args()
    create_feature_symlink(args.project_features, args.feature_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
