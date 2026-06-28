from __future__ import annotations

import json
import math

import numpy as np

from mrb.baselines.feature_data import FeatureToyDataset, TaskFeatureSplit
from mrb.baselines.router_error_anatomy import (
    REQUIRED_SUMMARY_KEYS,
    build_confidence_bin_table,
    build_confusion_count_matrix,
    build_confusion_rate_matrix,
    build_diagnostics_confusion_pairs,
    build_task_gap_decomposition_table,
    build_top_confusions_table,
    compute_confidence_auc_for_correct_route,
    evaluate_router_error_anatomy,
    assign_topk_error_buckets,
    write_router_error_anatomy_outputs,
)


def test_confusion_matrix_shape_and_counts_sum_to_samples() -> None:
    true = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)
    pred = np.asarray([0, 1, 1, 2, 0], dtype=np.int64)

    counts = build_confusion_count_matrix(true, pred, task_ids=[0, 1, 2])

    assert counts.shape == (3, 4)
    assert int(counts.drop(columns=["true_task"]).to_numpy().sum()) == true.shape[0]


def test_row_normalized_confusion_rows_sum_to_one() -> None:
    true = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    pred = np.asarray([0, 1, 1, 2, 0, 2], dtype=np.int64)
    counts = build_confusion_count_matrix(true, pred, task_ids=[0, 1, 2])

    rates = build_confusion_rate_matrix(counts)

    assert np.allclose(rates.drop(columns=["true_task"]).sum(axis=1).to_numpy(), 1.0)


def test_topk_bucket_assignment() -> None:
    topk = np.asarray(
        [
            [0, 1, 2, 3, 4],
            [9, 1, 2, 3, 4],
            [9, 8, 2, 3, 4],
            [9, 8, 7, 3, 4],
            [9, 8, 7, 6, 5],
        ],
        dtype=np.int64,
    )
    true = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)

    buckets = assign_topk_error_buckets(topk, true)

    assert buckets.tolist() == [
        "top1_correct",
        "in_top2_not_top1",
        "in_top3_not_top2",
        "in_top5_not_top3",
        "not_in_top5",
    ]


def test_confidence_bin_statistics() -> None:
    bins = build_confidence_bin_table(
        np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        np.asarray([True, False, True, True]),
        np.asarray([True, False, False, True]),
        np.asarray([True, False, False, False]),
        np.asarray([True, False, True, True]),
        np.asarray([False, False, True, True]),
        n_bins=2,
    )

    assert bins["num_samples"].tolist() == [2, 2]
    assert bins.loc[0, "route_accuracy"] == 0.5
    assert bins.loc[1, "fallback_rate"] == 0.0
    assert math.isclose(float(bins.loc[0, "mean_confidence"]), 0.15, rel_tol=1e-6)


def test_confidence_auc_for_correct_route_is_computable() -> None:
    auc = compute_confidence_auc_for_correct_route(
        np.asarray([0.1, 0.4, 0.8, 0.9], dtype=np.float32),
        np.asarray([False, False, True, True]),
    )

    assert auc == 1.0


def test_diagnostics_confusion_join_handles_missing_pair() -> None:
    true = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    pred = np.asarray([0, 1, 1, 2, 0, 2], dtype=np.int64)
    rates = build_confusion_rate_matrix(build_confusion_count_matrix(true, pred, [0, 1, 2]))
    diagnostics = _diagnostics_frame()

    pairs, summary = build_diagnostics_confusion_pairs(rates, diagnostics, [0, 1, 2])

    assert pairs.shape[0] == 3
    missing = pairs[(pairs["task_i"] == 0) & (pairs["task_j"] == 2)].iloc[0]
    assert math.isnan(float(missing["diagnostics_linear_probe_auc_symmetric"]))
    assert "pearson_auc_vs_confusion" in summary


def test_per_task_gap_decomposition_fields_exist() -> None:
    rows = build_task_gap_decomposition_table(
        {
            "merge_all": {0: 0.5},
            "oracle_task_mask": {0: 0.9},
            "hard_linear_route": {0: 0.6},
            "topk_k2": {0: 0.7},
            "fallback": {0: 0.8},
        }
    )

    assert {
        "oracle_gap",
        "hard_route_error_cost",
        "topk_error_cost",
        "fallback_error_cost",
    }.issubset(rows.columns)


def test_summary_json_contains_required_keys_and_no_sample_csv_when_disabled(tmp_path) -> None:
    dataset = _make_synthetic_dataset()
    evaluation = evaluate_router_error_anatomy(
        dataset,
        router_config={"max_iter": 200, "random_state": 0},
        calibration_config={"temperatures": [1.0], "ece_bins": 4},
        fallback_config={"tau_grid": [0.0, 0.5]},
        analysis_config={"confidence_bins": 4, "save_sample_level": False},
        config={"synthetic": True},
    )
    paths = write_router_error_anatomy_outputs(
        evaluation,
        output_dir=tmp_path,
        stream_id="synthetic",
        seed=0,
        backbone_id="toy",
    )

    assert set(REQUIRED_SUMMARY_KEYS).issubset(evaluation.summary)
    assert "sample_level_csv" not in paths
    assert not list(tmp_path.glob("*sample_level*"))
    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert "interpretation" in payload


def test_router_prediction_errors_do_not_crash_top_confusion_analysis() -> None:
    true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    pred = np.asarray([1, 1, 0, 0], dtype=np.int64)
    counts = build_confusion_count_matrix(true, pred, [0, 1])
    rates = build_confusion_rate_matrix(counts)
    per_task_accs = {
        "merge_all": {0: 0.5, 1: 0.5},
        "hard_linear_route": {0: 0.0, 1: 0.0},
        "topk_k2": {0: 1.0, 1: 1.0},
        "fallback": {0: 0.5, 1: 0.5},
    }

    top = build_top_confusions_table(counts, rates, per_task_accs)

    assert top["count"].tolist() == [2, 2]


def _diagnostics_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "task_i": [0],
            "task_j": [1],
            "linear_probe_auc_symmetric": [0.9],
            "knn_domain_acc": [0.8],
            "separation_ratio": [0.2],
        }
    )


def _make_synthetic_dataset() -> FeatureToyDataset:
    tasks = []
    centers = {
        0: {10: [1.0, 0.0], 11: [0.7, 0.3]},
        1: {20: [-1.0, 0.0], 21: [-0.7, -0.3]},
        2: {30: [0.0, 1.0], 31: [0.2, 0.8]},
    }
    for task_id, class_centers in centers.items():
        train_features, train_labels = _class_clouds(class_centers, seed=task_id)
        val_features, val_labels = _class_clouds(class_centers, seed=task_id + 10)
        test_features, test_labels = _class_clouds(class_centers, seed=task_id + 20)
        tasks.append(
            _task(
                task_id=task_id,
                classes=tuple(class_centers),
                train_features=train_features,
                train_labels=train_labels,
                val_features=val_features,
                val_labels=val_labels,
                test_features=test_features,
                test_labels=test_labels,
            )
        )
    return FeatureToyDataset(
        stream_id="synthetic",
        seed=0,
        backbone_id="toy",
        tasks=tuple(tasks),
        manifest={"tasks": []},
        manifest_path=None,
        feature_bank_metadata={},
    )


def _class_clouds(classes: dict[int, list[float]], *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label, center in classes.items():
        values = rng.normal(loc=center, scale=0.05, size=(12, 2)).astype(np.float32)
        features.append(values)
        labels.append(np.full(values.shape[0], label, dtype=np.int64))
    return np.vstack(features), np.concatenate(labels)


def _task(
    *,
    task_id: int,
    classes: tuple[int, ...],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> TaskFeatureSplit:
    train_indices = np.arange(train_labels.shape[0], dtype=np.int64)
    val_indices = np.arange(val_labels.shape[0], dtype=np.int64)
    test_indices = np.arange(test_labels.shape[0], dtype=np.int64)
    return TaskFeatureSplit(
        task_id=task_id,
        dataset_id="synthetic",
        classes=classes,
        train_features=train_features,
        train_labels=train_labels,
        train_indices=train_indices,
        val_features=val_features,
        val_labels=val_labels,
        val_indices=val_indices,
        test_features=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
        feature_bank_splits={"train": "train", "val": "train", "test": "test"},
    )
