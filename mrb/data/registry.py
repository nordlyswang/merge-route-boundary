"""Dataset registry loading, path resolution, and status inspection."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "datasets" / "registry.yaml"
DEFAULT_DATA_ROOT = Path("/root/rivermind-data/datasets")
DEFAULT_PROJECT_DATA = REPO_ROOT / "data"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class RegistryError(ValueError):
    """Raised when a dataset registry is malformed."""


@dataclass(frozen=True)
class DatasetEntry:
    """One dataset entry from the registry."""

    dataset_id: str
    tier: int
    source: str
    root: str
    task_type: str
    num_classes: int | None
    splits: tuple[str, ...]
    default_image_size: int
    expected_available: bool
    allow_in_smoke: bool
    large_dataset: bool
    manual_download: bool
    loader_status: str
    config: Mapping[str, Any]

    @property
    def torchvision_class(self) -> str | None:
        value = self.config.get("torchvision_class")
        return str(value) if value else None


class DatasetRegistry:
    """Loaded dataset registry with environment-aware path helpers."""

    def __init__(
        self,
        *,
        path: Path,
        version: int,
        root_env: str,
        project_data_env: str,
        default_data_root: Path,
        default_project_data: Path,
        datasets: Mapping[str, DatasetEntry],
        raw: Mapping[str, Any],
    ) -> None:
        self.path = path
        self.version = version
        self.root_env = root_env
        self.project_data_env = project_data_env
        self.default_data_root = default_data_root
        self.default_project_data = default_project_data
        self.datasets = dict(datasets)
        self.raw = dict(raw)

    def get(self, dataset_id: str) -> DatasetEntry:
        key = normalize_dataset_id(dataset_id)
        try:
            return self.datasets[key]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset_id: {dataset_id}") from exc

    def data_root(self, env: Mapping[str, str] | None = None) -> Path:
        env_map = env or os.environ
        raw = env_map.get(self.root_env, str(self.default_data_root))
        return Path(raw).expanduser()

    def project_data(self, env: Mapping[str, str] | None = None) -> Path:
        env_map = env or os.environ
        raw = env_map.get(self.project_data_env, str(self.default_project_data))
        return Path(raw).expanduser()

    def dataset_root(
        self,
        entry_or_id: DatasetEntry | str,
        *,
        root_override: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Path:
        if isinstance(entry_or_id, DatasetEntry):
            entry = entry_or_id
        else:
            entry = self.get(entry_or_id)
        if root_override is not None:
            return Path(root_override).expanduser()
        return resolve_path(entry.root, env=env, defaults={self.root_env: str(self.default_data_root)})

    def registry_hash(self) -> str:
        return stable_hash(self.raw)


def normalize_dataset_id(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_env_string(
    value: str,
    *,
    env: Mapping[str, str] | None = None,
    defaults: Mapping[str, str] | None = None,
) -> str:
    env_map = env or os.environ
    default_map = defaults or {}

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        fallback = match.group(2)
        if name in env_map:
            return env_map[name]
        if name in default_map:
            return default_map[name]
        if fallback is not None:
            return fallback
        return ""

    return _ENV_PATTERN.sub(replace, value)


def resolve_path(
    value: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    defaults: Mapping[str, str] | None = None,
) -> Path:
    raw = str(value)
    return Path(resolve_env_string(raw, env=env, defaults=defaults)).expanduser()


def load_registry(path: str | Path | None = None) -> DatasetRegistry:
    registry_path = Path(path or DEFAULT_REGISTRY_PATH)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RegistryError(f"{registry_path} must contain a YAML mapping")

    datasets_raw = payload.get("datasets")
    if not isinstance(datasets_raw, dict) or not datasets_raw:
        raise RegistryError(f"{registry_path} must contain a non-empty 'datasets' mapping")

    root_env = str(payload.get("root_env", "MRB_DATA_ROOT"))
    project_data_env = str(payload.get("project_data_env", "MRB_PROJECT_DATA"))
    defaults = payload.get("paths", {})
    default_data_root = Path(defaults.get("shared_root", DEFAULT_DATA_ROOT)).expanduser()
    default_project_data = Path(defaults.get("project_data_dir", DEFAULT_PROJECT_DATA)).expanduser()

    datasets: dict[str, DatasetEntry] = {}
    for key, item in datasets_raw.items():
        if not isinstance(item, dict):
            raise RegistryError(f"Dataset entry {key!r} must be a mapping")
        dataset_id = normalize_dataset_id(str(item.get("dataset_id", key)))
        splits = item.get("splits", [])
        if not isinstance(splits, list) or not splits:
            raise RegistryError(f"Dataset entry {dataset_id!r} must define non-empty splits")
        if dataset_id in datasets:
            raise RegistryError(f"Duplicate dataset_id: {dataset_id}")
        num_classes = item.get("num_classes")
        datasets[dataset_id] = DatasetEntry(
            dataset_id=dataset_id,
            tier=int(item["tier"]),
            source=str(item["source"]),
            root=str(item["root"]),
            task_type=str(item.get("task_type", "classification")),
            num_classes=int(num_classes) if num_classes is not None else None,
            splits=tuple(str(split) for split in splits),
            default_image_size=int(item.get("default_image_size", 224)),
            expected_available=bool(item.get("expected_available", False)),
            allow_in_smoke=bool(item.get("allow_in_smoke", False)),
            large_dataset=bool(item.get("large_dataset", False)),
            manual_download=bool(item.get("manual_download", False)),
            loader_status=str(item.get("loader_status", "supported")),
            config={**item, "dataset_id": dataset_id},
        )

    return DatasetRegistry(
        path=registry_path,
        version=int(payload.get("version", 1)),
        root_env=root_env,
        project_data_env=project_data_env,
        default_data_root=default_data_root,
        default_project_data=default_project_data,
        datasets=datasets,
        raw=payload,
    )


def directory_size_bytes(path: Path) -> int:
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


def dataset_available(
    registry: DatasetRegistry,
    entry: DatasetEntry,
    *,
    root_override: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    root = registry.dataset_root(entry, root_override=root_override, env=env)
    markers = entry.config.get("storage_markers", [])
    if markers:
        return all((root / str(marker)).exists() for marker in markers)
    return root.exists()


def dataset_storage_paths(
    registry: DatasetRegistry,
    entry: DatasetEntry,
    *,
    root_override: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> list[Path]:
    root = registry.dataset_root(entry, root_override=root_override, env=env)
    markers = entry.config.get("storage_markers", [])
    if markers:
        return [root / str(marker) for marker in markers]
    return [root]


def dataset_size_bytes(
    registry: DatasetRegistry,
    entry: DatasetEntry,
    *,
    root_override: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    return sum(
        directory_size_bytes(path)
        for path in dataset_storage_paths(registry, entry, root_override=root_override, env=env)
    )


def check_project_data_symlink(project_data: Path, data_root: Path) -> dict[str, Any]:
    """Return a non-mutating status report for the project data path."""

    status: dict[str, Any] = {
        "path": str(project_data),
        "expected_target": str(data_root),
        "exists": project_data.exists() or project_data.is_symlink(),
        "is_symlink": project_data.is_symlink(),
        "target": None,
        "resolves_to_data_root": False,
        "mode": "missing",
        "warnings": [],
    }
    if project_data.is_symlink():
        target = os.readlink(project_data)
        status["target"] = target
        try:
            status["resolves_to_data_root"] = project_data.resolve() == data_root.resolve()
        except OSError:
            status["resolves_to_data_root"] = False
        status["mode"] = "full_symlink" if status["resolves_to_data_root"] else "other_symlink"
        if not status["resolves_to_data_root"]:
            status["warnings"].append("project data symlink does not resolve to data root")
        return status

    if project_data.is_dir():
        child_links = []
        wrong_children = []
        for child in sorted(project_data.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                child_links.append(child.name)
                try:
                    expected = data_root / child.name
                    if child.resolve() != expected.resolve():
                        wrong_children.append(child.name)
                except OSError:
                    wrong_children.append(child.name)
        status["mode"] = "child_symlinks" if child_links else "directory"
        status["child_symlinks"] = child_links
        if child_links:
            status["warnings"].append(
                "project data is a directory with child symlinks; this task expects one full symlink"
            )
        if wrong_children:
            status["warnings"].append(
                "some child symlinks do not resolve under the expected data root: "
                + ", ".join(wrong_children)
            )
        return status

    if status["exists"]:
        status["mode"] = "file"
        status["warnings"].append("project data path exists but is neither directory nor symlink")
    else:
        status["warnings"].append("project data path is missing")
    return status


def inspect_dataset(
    registry: DatasetRegistry,
    entry: DatasetEntry,
    *,
    root_override: str | Path | None = None,
    data_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    load_metadata: bool | None = None,
    load_smoke_metadata: bool | None = None,
) -> dict[str, Any]:
    if data_root is not None and env is None:
        env = {registry.root_env: str(data_root)}
    should_load_metadata = bool(load_smoke_metadata if load_smoke_metadata is not None else load_metadata)
    root = registry.dataset_root(entry, root_override=root_override, env=env)
    available = dataset_available(registry, entry, root_override=root_override, env=env)
    result: dict[str, Any] = {
        "dataset_id": entry.dataset_id,
        "tier": entry.tier,
        "source": entry.source,
        "root": str(root),
        "task_type": entry.task_type,
        "num_classes": entry.num_classes,
        "splits": list(entry.splits),
        "available": available,
        "expected_available": entry.expected_available,
        "allow_in_smoke": entry.allow_in_smoke,
        "large_dataset": entry.large_dataset,
        "manual_download": entry.manual_download,
        "loader_status": entry.loader_status,
        "custom_layout": bool(entry.config.get("custom_layout", False)),
        "estimated_gb": float(entry.config.get("estimated_gb", 0.0)),
        "disk_usage_bytes": dataset_size_bytes(registry, entry, root_override=root_override, env=env),
        "storage_paths": [
            str(path)
            for path in dataset_storage_paths(registry, entry, root_override=root_override, env=env)
        ],
        "split_sizes": {},
        "warnings": [],
    }
    if not available:
        result["warnings"].append("dataset storage markers are missing")
        return result
    if entry.loader_status == "inspect_only":
        result["warnings"].append("dataset is registered for inspection only")
        return result
    if not should_load_metadata:
        return result

    try:
        from mrb.data.datasets import get_dataset, infer_num_classes
    except Exception as exc:
        result["warnings"].append(f"dataset loader unavailable: {exc}")
        return result

    for split in entry.splits:
        try:
            dataset = get_dataset(
                entry.dataset_id,
                split,
                registry=registry,
                root=root_override,
                transform=None,
            )
            result["split_sizes"][split] = len(dataset)  # type: ignore[arg-type]
            if result["num_classes"] is None:
                result["num_classes"] = infer_num_classes(dataset)
        except Exception as exc:
            result["warnings"].append(f"{split} metadata load failed: {exc}")
    return result


def inspect_registry(
    registry: DatasetRegistry,
    *,
    data_root: str | Path | None = None,
    load_smoke_metadata: bool = True,
) -> list[dict[str, Any]]:
    env = None
    if data_root is not None:
        env = {registry.root_env: str(Path(data_root).expanduser())}
    statuses = []
    for entry in registry.datasets.values():
        statuses.append(
            inspect_dataset(
                registry,
                entry,
                env=env,
                load_metadata=load_smoke_metadata and entry.allow_in_smoke,
            )
        )
    return statuses
