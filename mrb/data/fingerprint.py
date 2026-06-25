"""Lightweight dataset fingerprints for local audit reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mrb.data.registry import (
    DatasetEntry,
    DatasetRegistry,
    dataset_available,
    dataset_size_bytes,
    dataset_storage_paths,
    stable_hash,
)


def dataset_fingerprint(
    registry: DatasetRegistry,
    entry: DatasetEntry,
    *,
    max_paths: int = 200,
) -> dict[str, Any]:
    """Fingerprint metadata without hashing image contents."""

    root = registry.dataset_root(entry)
    storage_paths = dataset_storage_paths(registry, entry)
    paths: list[str] = []
    file_count = 0
    latest_mtime = 0.0
    earliest_mtime: float | None = None

    for storage_path in storage_paths:
        if not storage_path.exists():
            continue
        if storage_path.is_file():
            try:
                stat = storage_path.lstat()
            except OSError:
                continue
            file_count += 1
            latest_mtime = max(latest_mtime, stat.st_mtime)
            earliest_mtime = stat.st_mtime if earliest_mtime is None else min(earliest_mtime, stat.st_mtime)
            if len(paths) < max_paths:
                paths.append(str(storage_path.relative_to(root)))
            continue
        for current_root, dirs, files in os.walk(storage_path, followlinks=False):
            dirs.sort()
            files.sort()
            root_path = Path(current_root)
            for filename in files:
                candidate = root_path / filename
                try:
                    stat = candidate.lstat()
                except OSError:
                    continue
                file_count += 1
                latest_mtime = max(latest_mtime, stat.st_mtime)
                earliest_mtime = stat.st_mtime if earliest_mtime is None else min(earliest_mtime, stat.st_mtime)
                if len(paths) < max_paths:
                    paths.append(str(candidate.relative_to(root)))

    return {
        "dataset_id": entry.dataset_id,
        "root": str(root),
        "storage_paths": [str(path) for path in storage_paths],
        "available": dataset_available(registry, entry),
        "file_count": file_count,
        "total_disk_size_bytes": dataset_size_bytes(registry, entry),
        "first_paths_hash": stable_hash(paths),
        "first_paths_sampled": len(paths),
        "mtime_summary": {
            "earliest": earliest_mtime,
            "latest": latest_mtime or None,
        },
        "num_classes": entry.num_classes,
        "splits": list(entry.splits),
        "registry_hash": registry.registry_hash(),
    }


def registry_fingerprints(registry: DatasetRegistry, *, max_paths: int = 200) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "registry_path": str(registry.path),
        "registry_hash": registry.registry_hash(),
        "datasets": [
            dataset_fingerprint(registry, entry, max_paths=max_paths)
            for entry in registry.datasets.values()
        ],
    }
