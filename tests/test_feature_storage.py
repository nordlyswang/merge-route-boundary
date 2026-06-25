from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mrb.features.storage import (
    build_metadata,
    feature_bank_dir,
    list_feature_banks,
    read_feature_metadata,
    sample_id_records,
    write_feature_bank,
)
from mrb.features.verify import verify_feature_bank


def write_tiny_bank(root: Path, *, features: np.ndarray | None = None) -> Path:
    bank_dir = feature_bank_dir(root, dataset_id="toy", split="train", backbone_id="dummy")
    features_array = (
        features
        if features is not None
        else np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float16)
    )
    labels = np.asarray([0, 1, 0], dtype=np.int64)
    indices = np.asarray([0, 1, 2], dtype=np.int64)
    metadata = build_metadata(
        dataset_id="toy",
        split="train",
        backbone_id="dummy",
        model_id="dummy/model",
        dtype=features_array.dtype,
        feature_dim=2,
        num_samples=3,
        image_size=32,
        preprocess_id="identity",
    )
    return write_feature_bank(
        bank_dir,
        features=features_array,
        labels=labels,
        indices=indices,
        sample_ids=sample_id_records(dataset_id="toy", split="train", indices=indices, labels=labels),
        metadata=metadata,
    )


def test_write_and_verify_feature_bank(tmp_path: Path) -> None:
    bank_dir = write_tiny_bank(tmp_path)

    assert (bank_dir / "features.npy").exists()
    assert (bank_dir / "labels.npy").exists()
    assert (bank_dir / "indices.npy").exists()
    assert (bank_dir / "sample_ids.jsonl").exists()
    assert (bank_dir / "metadata.json").exists()

    metadata = read_feature_metadata(bank_dir)
    assert metadata["dataset_id"] == "toy"
    assert metadata["split"] == "train"
    assert metadata["backbone_id"] == "dummy"
    assert metadata["dtype"] == "float16"
    assert metadata["feature_dim"] == 2
    assert metadata["num_samples"] == 3

    result = verify_feature_bank(bank_dir)
    assert result.ok, result.errors


def test_feature_bank_dir_uses_canonical_layout(tmp_path: Path) -> None:
    assert feature_bank_dir(
        tmp_path,
        dataset_id="cifar10",
        split="train",
        backbone_id="clip_vit_b32",
    ) == tmp_path / "cifar10" / "train" / "clip_vit_b32"


def test_write_feature_bank_refuses_existing_without_overwrite(tmp_path: Path) -> None:
    bank_dir = write_tiny_bank(tmp_path)

    with pytest.raises(FileExistsError):
        write_tiny_bank(tmp_path)

    banks = list_feature_banks(tmp_path)
    assert len(banks) == 1
    assert banks[0]["path"] == str(bank_dir)


def test_verify_feature_bank_rejects_nan(tmp_path: Path) -> None:
    features = np.asarray([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]], dtype=np.float16)
    bank_dir = write_tiny_bank(tmp_path, features=features)

    result = verify_feature_bank(bank_dir)

    assert not result.ok
    assert any("NaN or Inf" in error for error in result.errors)
