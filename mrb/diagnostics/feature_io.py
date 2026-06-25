"""Feature bank loading and selection helpers for diagnostics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mrb.features.storage import FEATURE_ROOT_ENV, normalize_component


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FeatureBank:
    features: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    metadata: dict[str, Any]
    dataset_id: str
    split: str
    backbone_id: str
    path: Path | None = None


def resolve_feature_root(feature_root: str | Path | None = None) -> Path:
    if feature_root is not None:
        return Path(feature_root).expanduser()
    env_value = os.environ.get(FEATURE_ROOT_ENV)
    if env_value:
        return Path(env_value).expanduser()
    project_link = REPO_ROOT / "features"
    if project_link.exists() or project_link.is_symlink():
        return project_link
    raise EnvironmentError(
        "MRB_FEATURE_ROOT is not set. Please export it or create the project feature symlink."
    )


def candidate_feature_bank_dirs(
    *,
    dataset_id: str,
    split: str,
    backbone_id: str,
    feature_root: str | Path | None = None,
) -> list[Path]:
    root = resolve_feature_root(feature_root)
    dataset = normalize_component(dataset_id)
    source_split = normalize_component(split)
    backbone = normalize_component(backbone_id)
    return [
        root / backbone / dataset / source_split,
        root / dataset / source_split / backbone,
    ]


def load_feature_bank(
    dataset_id: str,
    split: str,
    backbone_id: str,
    feature_root: str | Path | None = None,
) -> FeatureBank:
    candidates = candidate_feature_bank_dirs(
        dataset_id=dataset_id,
        split=split,
        backbone_id=backbone_id,
        feature_root=feature_root,
    )
    bank_dir = next((path for path in candidates if path.exists()), None)
    if bank_dir is None:
        locations = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Feature bank not found. Tried: {locations}")

    missing_files = [
        name
        for name in ("features.npy", "labels.npy", "indices.npy", "metadata.json")
        if not (bank_dir / name).exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Feature bank {bank_dir} missing: {', '.join(missing_files)}")

    metadata = json.loads((bank_dir / "metadata.json").read_text(encoding="utf-8"))
    bank = FeatureBank(
        features=np.asarray(np.load(bank_dir / "features.npy"), dtype=np.float32),
        labels=np.asarray(np.load(bank_dir / "labels.npy"), dtype=np.int64),
        indices=np.asarray(np.load(bank_dir / "indices.npy"), dtype=np.int64),
        metadata=dict(metadata),
        dataset_id=str(metadata.get("dataset_id", dataset_id)),
        split=str(metadata.get("split", split)),
        backbone_id=str(metadata.get("backbone_id", backbone_id)),
        path=bank_dir,
    )
    validate_feature_bank(bank)
    return bank


def validate_feature_bank(bank: FeatureBank) -> None:
    features = np.asarray(bank.features)
    labels = np.asarray(bank.labels)
    indices = np.asarray(bank.indices)
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
    if not np.isfinite(features).all():
        raise ValueError("features contains NaN or Inf")
    if "feature_dim" in bank.metadata and int(bank.metadata["feature_dim"]) != int(
        features.shape[1]
    ):
        raise ValueError(
            f"metadata.feature_dim={bank.metadata['feature_dim']} does not match "
            f"features.shape[1]={features.shape[1]}"
        )
    if np.unique(indices).shape[0] != indices.shape[0]:
        raise ValueError("indices contains duplicate dataset indices")


def select_by_indices(
    bank: FeatureBank,
    indices: Sequence[int],
    *,
    strict: bool = True,
) -> FeatureBank:
    requested = [int(index) for index in indices]
    index_to_row = {int(index): row for row, index in enumerate(bank.indices.tolist())}
    rows: list[int] = []
    missing: list[int] = []
    for index in requested:
        row = index_to_row.get(index)
        if row is None:
            missing.append(index)
        else:
            rows.append(row)

    if missing and strict:
        preview = ", ".join(str(value) for value in missing[:10])
        suffix = "" if len(missing) <= 10 else ", ..."
        raise KeyError(
            f"{len(missing)} requested indices are missing from feature bank "
            f"{bank.path or ''}: {preview}{suffix}"
        )

    row_array = np.asarray(rows, dtype=np.int64)
    metadata = dict(bank.metadata)
    metadata.update(
        {
            "requested_num_indices": len(requested),
            "missing_indices_count": len(missing),
            "missing_indices_preview": missing[:10],
        }
    )
    return replace(
        bank,
        features=np.asarray(bank.features[row_array], dtype=np.float32),
        labels=np.asarray(bank.labels[row_array], dtype=np.int64),
        indices=np.asarray(bank.indices[row_array], dtype=np.int64),
        metadata=metadata,
    )


def select_by_classes(
    bank: FeatureBank,
    classes: Sequence[int],
) -> FeatureBank:
    class_values = np.asarray([int(value) for value in classes], dtype=np.int64)
    mask = np.isin(bank.labels, class_values)
    metadata = dict(bank.metadata)
    metadata.update({"selected_classes": class_values.tolist()})
    return replace(
        bank,
        features=np.asarray(bank.features[mask], dtype=np.float32),
        labels=np.asarray(bank.labels[mask], dtype=np.int64),
        indices=np.asarray(bank.indices[mask], dtype=np.int64),
        metadata=metadata,
    )
