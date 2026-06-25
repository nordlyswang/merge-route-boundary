from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mrb.diagnostics.distances import cosine_distance
from mrb.diagnostics.feature_io import load_feature_bank, select_by_indices
from mrb.diagnostics.overlap import nearest_task_centroid_confusion
from mrb.diagnostics.prototypes import (
    compute_class_prototypes,
    compute_task_prototype,
    l2_normalize,
)
from mrb.diagnostics.separability import linear_probe_separability


def test_l2_normalize_outputs_unit_norm() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normalized = l2_normalize(values)

    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_compute_class_prototypes_outputs_one_per_class() -> None:
    features = np.asarray([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    prototypes = compute_class_prototypes(features, labels)

    assert sorted(prototypes) == [0, 1]
    assert prototypes[0].shape == (2,)


def test_cosine_distance_range_is_reasonable() -> None:
    assert np.isclose(cosine_distance(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])), 0.0)
    assert np.isclose(cosine_distance(np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0])), 2.0)


def test_linear_probe_separability_high_for_synthetic_clusters() -> None:
    rng = np.random.default_rng(0)
    features_i = rng.normal(loc=-2.0, scale=0.1, size=(40, 8)).astype(np.float32)
    features_j = rng.normal(loc=2.0, scale=0.1, size=(40, 8)).astype(np.float32)

    result = linear_probe_separability(features_i, features_j, seed=0)

    assert result["linear_probe_auc_symmetric"] >= 0.95
    assert result["linear_probe_acc"] >= 0.9


def test_nearest_task_centroid_confusion_high_for_separable_data() -> None:
    rng = np.random.default_rng(1)
    features_i = rng.normal(loc=-3.0, scale=0.1, size=(20, 4)).astype(np.float32)
    features_j = rng.normal(loc=3.0, scale=0.1, size=(20, 4)).astype(np.float32)
    proto_i = compute_task_prototype(features_i)
    proto_j = compute_task_prototype(features_j)

    result = nearest_task_centroid_confusion(features_i, features_j, proto_i, proto_j)

    assert result["nearest_task_centroid_acc"] >= 0.95


def test_select_by_indices_uses_original_indices_not_row_positions(tmp_path: Path) -> None:
    bank_dir = tmp_path / "dummy" / "toy" / "train"
    _write_bank(
        bank_dir,
        dataset_id="toy",
        split="train",
        backbone_id="dummy",
        features=np.asarray([[30.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32),
        labels=np.asarray([3, 1, 2], dtype=np.int64),
        indices=np.asarray([30, 10, 20], dtype=np.int64),
    )

    bank = load_feature_bank("toy", "train", "dummy", feature_root=tmp_path)
    selected = select_by_indices(bank, [10, 30], strict=True)

    assert selected.indices.tolist() == [10, 30]
    assert selected.features[:, 0].tolist() == [10.0, 30.0]


def _write_bank(
    bank_dir: Path,
    *,
    dataset_id: str,
    split: str,
    backbone_id: str,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> None:
    bank_dir.mkdir(parents=True)
    np.save(bank_dir / "features.npy", features)
    np.save(bank_dir / "labels.npy", labels)
    np.save(bank_dir / "indices.npy", indices)
    metadata = {
        "dataset_id": dataset_id,
        "split": split,
        "backbone_id": backbone_id,
        "dtype": str(features.dtype),
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
    }
    (bank_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
