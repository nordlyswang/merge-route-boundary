"""Validation for stored frozen feature banks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mrb.features.storage import FEATURE_FILE_NAMES


@dataclass(frozen=True)
class FeatureBankVerification:
    path: Path
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any]


def verify_feature_bank(bank_dir: str | Path) -> FeatureBankVerification:
    path = Path(bank_dir).expanduser()
    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}

    for filename in FEATURE_FILE_NAMES:
        if not (path / filename).exists():
            errors.append(f"missing {filename}")
    if errors:
        return FeatureBankVerification(path, False, tuple(errors), tuple(warnings), metadata)

    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"metadata.json is not valid JSON: {exc}")
        return FeatureBankVerification(path, False, tuple(errors), tuple(warnings), metadata)

    try:
        features = np.load(path / "features.npy", mmap_mode="r")
        labels = np.load(path / "labels.npy", mmap_mode="r")
        indices = np.load(path / "indices.npy", mmap_mode="r")
    except Exception as exc:
        errors.append(f"failed to load numpy arrays: {exc}")
        return FeatureBankVerification(path, False, tuple(errors), tuple(warnings), metadata)

    if features.ndim != 2:
        errors.append(f"features.npy must have rank 2 [N, D], got shape {features.shape}")
    if labels.ndim != 1:
        errors.append(f"labels.npy must have rank 1 [N], got shape {labels.shape}")
    if indices.ndim != 1:
        errors.append(f"indices.npy must have rank 1 [N], got shape {indices.shape}")

    if features.ndim == 2:
        num_samples = int(features.shape[0])
        feature_dim = int(features.shape[1])
        if int(metadata.get("num_samples", -1)) != num_samples:
            errors.append(
                f"metadata num_samples={metadata.get('num_samples')} does not match features N={num_samples}"
            )
        if int(metadata.get("feature_dim", -1)) != feature_dim:
            errors.append(
                f"metadata feature_dim={metadata.get('feature_dim')} does not match features D={feature_dim}"
            )
    else:
        num_samples = -1

    if labels.ndim == 1 and num_samples >= 0 and labels.shape[0] != num_samples:
        errors.append(f"labels length {labels.shape[0]} does not match features N={num_samples}")
    if indices.ndim == 1 and num_samples >= 0 and indices.shape[0] != num_samples:
        errors.append(f"indices length {indices.shape[0]} does not match features N={num_samples}")
    if labels.ndim == 1 and indices.ndim == 1 and labels.shape[0] != indices.shape[0]:
        errors.append(f"labels length {labels.shape[0]} does not match indices length {indices.shape[0]}")

    expected_dtype = metadata.get("dtype")
    if expected_dtype is None:
        errors.append("metadata missing dtype")
    else:
        try:
            expected_dtype_name = np.dtype(str(expected_dtype)).name
            if features.dtype.name != expected_dtype_name:
                errors.append(
                    f"metadata dtype={expected_dtype_name} does not match features dtype={features.dtype.name}"
                )
        except TypeError:
            errors.append(f"metadata dtype is invalid: {expected_dtype!r}")

    try:
        if not bool(np.isfinite(features).all()):
            errors.append("features.npy contains NaN or Inf")
    except TypeError:
        errors.append(f"features.npy dtype must be numeric, got {features.dtype}")

    sample_id_lines = 0
    with (path / "sample_ids.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                warnings.append(f"sample_ids.jsonl line {line_number} is blank")
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"sample_ids.jsonl line {line_number} is invalid JSON: {exc}")
                continue
            missing = {"dataset_id", "split", "index", "label"} - set(record)
            if missing:
                errors.append(
                    f"sample_ids.jsonl line {line_number} missing fields: {', '.join(sorted(missing))}"
                )
            sample_id_lines += 1
    if num_samples >= 0 and sample_id_lines != num_samples:
        errors.append(f"sample_ids.jsonl has {sample_id_lines} records, expected {num_samples}")

    return FeatureBankVerification(path, not errors, tuple(errors), tuple(warnings), metadata)


def raise_if_invalid(result: FeatureBankVerification) -> None:
    if result.ok:
        return
    joined = "; ".join(result.errors)
    raise ValueError(f"Invalid feature bank at {result.path}: {joined}")
