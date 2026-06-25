#!/usr/bin/env python
"""Report storage usage for the shared MRB dataset root."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/root/rivermind-data/datasets")
DEFAULT_PROJECT_DATA = REPO_ROOT / "data"


def gb(size_bytes: int) -> float:
    return size_bytes / (1024**3)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size

    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for dirname in dirs:
            candidate = root_path / dirname
            if candidate.is_symlink():
                total += candidate.lstat().st_size
        for filename in files:
            candidate = root_path / filename
            try:
                total += candidate.lstat().st_size
            except OSError:
                continue
    return total


def top_level_sizes(data_root: Path) -> list[dict[str, Any]]:
    if not data_root.exists():
        return []
    entries: list[dict[str, Any]] = []
    for entry in sorted(data_root.iterdir(), key=lambda item: item.name):
        entries.append(
            {
                "path": entry.name,
                "type": "symlink" if entry.is_symlink() else "directory" if entry.is_dir() else "file",
                "gb": round(gb(size_bytes(entry)), 4),
            }
        )
    return entries


def largest_entries(data_root: Path, limit: int) -> list[dict[str, Any]]:
    if not data_root.exists():
        return []
    entries: list[dict[str, Any]] = []

    for root, dirs, files in os.walk(data_root, followlinks=False):
        root_path = Path(root)
        for dirname in dirs:
            candidate = root_path / dirname
            try:
                entries.append(
                    {
                        "path": str(candidate.relative_to(data_root)),
                        "type": "symlink" if candidate.is_symlink() else "directory",
                        "gb": round(gb(size_bytes(candidate)), 4),
                    }
                )
            except OSError:
                continue
        for filename in files:
            candidate = root_path / filename
            try:
                entries.append(
                    {
                        "path": str(candidate.relative_to(data_root)),
                        "type": "file",
                        "gb": round(gb(candidate.lstat().st_size), 4),
                    }
                )
            except OSError:
                continue

    entries.sort(key=lambda item: item["gb"], reverse=True)
    return entries[:limit]


def symlink_status(project_data: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not project_data.exists():
        return entries
    for entry in sorted(project_data.iterdir(), key=lambda item: item.name):
        if not entry.is_symlink():
            continue
        entries.append({"path": str(entry), "target": os.readlink(entry)})
    return entries


def build_report(data_root: Path, project_data: Path, top: int) -> dict[str, Any]:
    usage = shutil.disk_usage(data_root if data_root.exists() else data_root.parent)
    return {
        "generated_at": now_iso(),
        "data_root": str(data_root),
        "project_data": str(project_data),
        "filesystem": {
            "total_gb": round(gb(usage.total), 2),
            "used_gb": round(gb(usage.used), 2),
            "free_gb": round(gb(usage.free), 2),
        },
        "project_data_links": symlink_status(project_data),
        "top_level": top_level_sizes(data_root),
        "largest_entries": largest_entries(data_root, top),
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"MRB_DATA_ROOT: {report['data_root']}")
    print(f"Project data dir: {report['project_data']}")
    fs = report["filesystem"]
    print(
        f"Filesystem: total={fs['total_gb']:.2f} GB "
        f"used={fs['used_gb']:.2f} GB free={fs['free_gb']:.2f} GB"
    )

    print("Project data links:")
    if not report["project_data_links"]:
        print("  WARN no project data symlinks found")
    for item in report["project_data_links"]:
        print(f"  {item['path']} -> {item['target']}")

    print("Top-level dataset root usage:")
    if not report["top_level"]:
        print("  WARN data root is missing or empty")
    for item in report["top_level"]:
        print(f"  {item['path']}: {item['gb']:.4f} GB ({item['type']})")

    print("Largest files/directories:")
    if not report["largest_entries"]:
        print("  WARN no entries found")
    for item in report["largest_entries"]:
        print(f"  {item['path']}: {item['gb']:.4f} GB ({item['type']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("MRB_DATA_ROOT", DEFAULT_DATA_ROOT)),
    )
    parser.add_argument(
        "--project-data",
        type=Path,
        default=Path(os.environ.get("MRB_PROJECT_DATA", DEFAULT_PROJECT_DATA)),
    )
    parser.add_argument("--top", type=int, default=20, help="Number of largest entries to show.")
    parser.add_argument("--json", type=Path, default=None, help="Optional report JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser()
    project_data = args.project_data.expanduser()
    report = build_report(data_root, project_data, args.top)
    print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
