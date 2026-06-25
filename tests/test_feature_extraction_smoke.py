from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from mrb.features.extraction import extract_and_write_feature_bank, extract_feature_arrays
from mrb.features.verify import verify_feature_bank


class TinyImageDataset:
    targets = [0, 1, 0, 1, 0]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return f"image-{index}", self.targets[index]


class DummyBackbone:
    backbone_id = "dummy"
    model_id = "dummy/model"
    feature_dim = 3
    image_size = 32
    preprocess_id = "identity"

    def encode(self, images: Sequence[object]) -> np.ndarray:
        rows = []
        for image in images:
            index = int(str(image).split("-")[-1])
            rows.append([float(index), float(index + 1), 1.0])
        return np.asarray(rows, dtype=np.float32)


def test_extract_feature_arrays_honors_max_samples() -> None:
    result = extract_feature_arrays(
        TinyImageDataset(),
        dataset_id="toy",
        split="train",
        backbone=DummyBackbone(),
        batch_size=2,
        max_samples=3,
        dtype="float16",
    )

    assert result.features.shape == (3, 3)
    assert result.features.dtype == np.float16
    assert result.labels.tolist() == [0, 1, 0]
    assert result.indices.tolist() == [0, 1, 2]
    assert result.sample_ids[2] == {"dataset_id": "toy", "split": "train", "index": 2, "label": 0}


def test_extract_and_write_feature_bank_smoke(tmp_path: Path) -> None:
    bank_dir = tmp_path / "features" / "toy" / "train" / "dummy"

    written = extract_and_write_feature_bank(
        TinyImageDataset(),
        bank_dir,
        dataset_id="toy",
        split="train",
        backbone=DummyBackbone(),
        batch_size=2,
        max_samples=4,
        dtype="float16",
    )

    assert written == bank_dir
    result = verify_feature_bank(bank_dir)
    assert result.ok, result.errors
    assert result.metadata["num_samples"] == 4
    assert result.metadata["feature_dim"] == 3
    assert result.metadata["dtype"] == "float16"
