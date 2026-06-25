"""Feature bank path and persistence helpers."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mrb.data.registry import directory_size_bytes, normalize_dataset_id


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT_ENV = "MRB_FEATURE_ROOT"
DEFAULT_FEATURE_ROOT = Path("/root/rivermind-data/datasets/_derived/merge-route-boundary/features")
FEATURE_FILE_NAMES = (
    "features.npy",
    "labels.npy",
    "indices.npy",
    "sample_ids.jsonl",
    "metadata.json",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_component(value: str) -> str:
    normalized = normalize_dataset_id(value)
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in normalized)


def feature_root(env: Mapping[str, str] | None = None) -> Path:
    env_map = env or os.environ
    return Path(env_map.get(FEATURE_ROOT_ENV, str(DEFAULT_FEATURE_ROOT))).expanduser()


def project_feature_link() -> Path:
    return REPO_ROOT / "features"


def feature_bank_dir(
    root: str | Path,
    *,
    dataset_id: str,
    split: str,
    backbone_id: str,
) -> Path:
    return (
        Path(root).expanduser()
        / normalize_component(dataset_id)
        / normalize_component(split)
        / normalize_component(backbone_id)
    )


def sample_id_records(
    *,
    dataset_id: str,
    split: str,
    indices: Sequence[int],
    labels: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": normalize_dataset_id(dataset_id),
            "split": str(split),
            "index": int(index),
            "label": int(label),
        }
        for index, label in zip(indices, labels, strict=True)
    ]


def build_metadata(
    *,
    dataset_id: str,
    split: str,
    backbone_id: str,
    model_id: str,
    dtype: str | np.dtype,
    feature_dim: int,
    num_samples: int,
    image_size: int,
    preprocess_id: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset_id": normalize_dataset_id(dataset_id),
        "split": str(split),
        "backbone_id": normalize_component(backbone_id),
        "model_id": str(model_id),
        "dtype": np.dtype(dtype).name,
        "feature_dim": int(feature_dim),
        "num_samples": int(num_samples),
        "image_size": int(image_size),
        "preprocess_id": str(preprocess_id),
        "created_at": now_iso(),
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def validate_feature_arrays(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> None:
    if features.ndim != 2:
        raise ValueError(f"features must be rank 2 [N, D], got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be rank 1 [N], got shape {labels.shape}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be rank 1 [N], got shape {indices.shape}")
    if features.shape[0] != labels.shape[0] or labels.shape[0] != indices.shape[0]:
        raise ValueError(
            "features, labels, and indices must have matching N; "
            f"got {features.shape[0]}, {labels.shape[0]}, {indices.shape[0]}"
        )


def write_feature_bank(
    bank_dir: str | Path,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    sample_ids: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    target = Path(bank_dir).expanduser()
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Feature bank already exists: {target}")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    features_array = np.asarray(features)
    labels_array = np.asarray(labels, dtype=np.int64)
    indices_array = np.asarray(indices, dtype=np.int64)
    validate_feature_arrays(features=features_array, labels=labels_array, indices=indices_array)

    target.mkdir(parents=True, exist_ok=False)
    np.save(target / "features.npy", features_array)
    np.save(target / "labels.npy", labels_array)
    np.save(target / "indices.npy", indices_array)

    with (target / "sample_ids.jsonl").open("w", encoding="utf-8") as handle:
        for record in sample_ids:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")

    (target / "metadata.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_feature_metadata(bank_dir: str | Path) -> dict[str, Any]:
    path = Path(bank_dir).expanduser() / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def feature_bank_size_bytes(bank_dir: str | Path) -> int:
    return directory_size_bytes(Path(bank_dir).expanduser())


def format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def list_feature_banks(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return []

    banks: list[dict[str, Any]] = []
    for metadata_path in sorted(root_path.rglob("metadata.json")):
        bank_dir = metadata_path.parent
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            banks.append(
                {
                    "path": str(bank_dir),
                    "valid_metadata": False,
                    "error": f"metadata decode failed: {exc}",
                    "size_bytes": feature_bank_size_bytes(bank_dir),
                }
            )
            continue
        banks.append(
            {
                "path": str(bank_dir),
                "valid_metadata": True,
                "metadata": metadata,
                "size_bytes": feature_bank_size_bytes(bank_dir),
            }
        )
    return banks
