#!/usr/bin/env python
"""Report shared dataset storage without deleting or moving files."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrb.data.registry import DEFAULT_DATA_ROOT, check_project_data_symlink, directory_size_bytes


def gb(size_bytes: int) -> float:
    return size_bytes / (1024**3)


def direct_children(data_root: Path) -> list[tuple[Path, int]]:
    if not data_root.exists():
        return []
    return [(entry, directory_size_bytes(entry)) for entry in sorted(data_root.iterdir(), key=lambda item: item.name)]


def duplicate_name_report(data_root: Path) -> list[tuple[str, list[str]]]:
    names: dict[str, list[str]] = defaultdict(list)
    if not data_root.exists():
        return []
    for root, dirs, _files in os.walk(data_root, followlinks=False):
        dirs.sort()
        for dirname in dirs:
            path = Path(root) / dirname
            if dirname.startswith(".") or dirname.startswith("_"):
                continue
            names[dirname.lower()].append(str(path.relative_to(data_root)))
    return sorted((name, paths) for name, paths in names.items() if len(paths) > 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("MRB_DATA_ROOT", DEFAULT_DATA_ROOT)))
    parser.add_argument("--project-data", type=Path, default=Path(os.environ.get("MRB_PROJECT_DATA", REPO_ROOT / "data")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser()
    project_data = args.project_data.expanduser()
    usage_base = data_root if data_root.exists() else data_root.parent
    usage = shutil.disk_usage(usage_base)

    print(f"MRB_DATA_ROOT: {data_root}")
    print(f"Project data: {project_data}")
    print(f"Filesystem: total={gb(usage.total):.2f} GB used={gb(usage.used):.2f} GB free={gb(usage.free):.2f} GB")
    print(f"Data root total usage: {gb(directory_size_bytes(data_root)):.4f} GB")

    link_status = check_project_data_symlink(project_data, data_root)
    print(f"Project data mode: {link_status['mode']}")
    for warning in link_status["warnings"]:
        print(f"WARN {warning}")

    print("\nTop-level data root entries:")
    for entry, size in direct_children(data_root):
        kind = "symlink" if entry.is_symlink() else "dir" if entry.is_dir() else "file"
        print(f"  {entry.name:18s} {gb(size):10.4f} GB {kind}")

    duplicates = duplicate_name_report(data_root)
    print("\nPotential duplicate directory names:")
    if not duplicates:
        print("  none detected")
    for name, paths in duplicates[:30]:
        print(f"  {name}: {', '.join(paths[:8])}")
    if len(duplicates) > 30:
        print(f"  ... {len(duplicates) - 30} more duplicate-name groups omitted")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
