from __future__ import annotations

import math

import numpy as np
import pandas as pd

from mrb.baselines.feature_data import FeatureToyDataset, TaskFeatureSplit
from mrb.baselines.pairwise_policy import (
    ACTION_FALLBACK,
    ACTION_TOP1_MASK,
    ACTION_TOP2_MASK,
    PAIRWISE_PER_TASK_COLUMNS,
    PAIRWISE_RESULT_COLUMNS,
    PairwiseDiagnosticsLookup,
    PairwiseSignals,
    PolicyThresholds,
    SplitPredictions,
    choose_pairwise_actions,
    compute_action_summary,
    compute_oracle_gap_closure,
    evaluate_pairwise_policy,
    select_best_thresholds,
)


def test_pairwise_diagnostics_lookup_is_symmetric() -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "task_i": 0,
                "task_j": 1,
                "linear_probe_auc_symmetric": 0.88,
                "separation_ratio": 0.25,
            }
        ]
    )

    lookup = PairwiseDiagnosticsLookup.from_dataframe(diagnostics)
    forward = lookup.lookup(0, 1)
    reverse = lookup.lookup(1, 0)

    assert forward.pair_auc == 0.88
    assert reverse.pair_auc == 0.88
    assert reverse.pair_separation_ratio == 0.25


def test_pairwise_policy_selects_fallback_for_low_confidence() -> None:
    actions = choose_pairwise_actions(
        _signals(confidence=[0.19], margin=[0.5], pair_auc=[0.99], prior=[0.0]),
        PolicyThresholds(0.2, 0.1, 0.95, 0.04),
    )

    assert actions.tolist() == [ACTION_FALLBACK]


def test_pairwise_policy_selects_top2_for_low_margin_and_low_pair_auc() -> None:
    actions = choose_pairwise_actions(
        _signals(confidence=[0.8], margin=[0.04], pair_auc=[0.90], prior=[0.0]),
        PolicyThresholds(0.2, 0.1, 0.95, 0.04),
    )

    assert actions.tolist() == [ACTION_TOP2_MASK]


def test_pairwise_policy_selects_top1_for_high_confidence_margin_and_pair_auc() -> None:
    actions = choose_pairwise_actions(
        _signals(confidence=[0.8], margin=[0.2], pair_auc=[0.99], prior=[0.0]),
        PolicyThresholds(0.2, 0.1, 0.95, 0.04),
    )

    assert actions.tolist() == [ACTION_TOP1_MASK]


def test_threshold_selection_uses_val_metrics_not_test_metrics() -> None:
    grid = pd.DataFrame(
        [
            {
                "tau_fallback": 0.1,
                "tau_margin": 0.1,
                "tau_pair_auc": 0.95,
                "tau_pair_confusion": 0.04,
                "val_acc": 0.7,
                "test_acc": 0.99,
                "mean_mask_size": 20.0,
                "action_fallback_rate": 0.1,
                "status": "ok",
            },
            {
                "tau_fallback": 0.2,
                "tau_margin": 0.1,
                "tau_pair_auc": 0.95,
                "tau_pair_confusion": 0.04,
                "val_acc": 0.8,
                "test_acc": 0.50,
                "mean_mask_size": 30.0,
                "action_fallback_rate": 0.2,
                "status": "ok",
            },
        ]
    )

    selected = select_best_thresholds(grid)

    assert selected.tau_fallback == 0.2


def test_threshold_tie_break_prefers_lower_mean_mask_size() -> None:
    grid = pd.DataFrame(
        [
            {
                "tau_fallback": 0.1,
                "tau_margin": 0.1,
                "tau_pair_auc": 0.95,
                "tau_pair_confusion": 0.04,
                "val_acc": 0.8,
                "mean_mask_size": 30.0,
                "action_fallback_rate": 0.1,
                "status": "ok",
            },
            {
                "tau_fallback": 0.2,
                "tau_margin": 0.1,
                "tau_pair_auc": 0.95,
                "tau_pair_confusion": 0.04,
                "val_acc": 0.8,
                "mean_mask_size": 20.0,
                "action_fallback_rate": 0.5,
                "status": "ok",
            },
        ]
    )

    selected = select_best_thresholds(grid)

    assert selected.tau_fallback == 0.2


def test_action_rates_and_mean_mask_size_are_computed_correctly() -> None:
    summary = compute_action_summary(
        np.asarray([ACTION_TOP1_MASK, ACTION_TOP2_MASK, ACTION_FALLBACK, ACTION_FALLBACK]),
        np.asarray([10.0, 20.0, 100.0, 100.0], dtype=np.float32),
    )

    assert summary["action_top1_rate"] == 0.25
    assert summary["action_top2_rate"] == 0.25
    assert summary["action_fallback_rate"] == 0.5
    assert summary["mean_mask_size"] == 57.5


def test_oracle_gap_closure_computes_fraction_of_merge_to_oracle_gap() -> None:
    closure = compute_oracle_gap_closure(0.7, 0.6, 0.9)

    assert math.isclose(closure, 1.0 / 3.0)


def test_pairwise_policy_outputs_required_columns_on_synthetic_dataset() -> None:
    evaluation = evaluate_pairwise_policy(
        _make_synthetic_dataset(),
        router_config={"max_iter": 200, "random_state": 0},
        calibration_config={"temperatures": [0.5, 1.0], "ece_bins": 3},
        policy_config={
            "tau_fallback_grid": [0.0, 0.5],
            "tau_margin_grid": [0.05, 0.2],
            "tau_pair_auc_grid": [0.95],
            "tau_pair_confusion_grid": [0.04],
            "topk_controls": [2],
        },
        config={"synthetic": True},
    )

    assert set(PAIRWISE_RESULT_COLUMNS).issubset(evaluation.results.columns)
    assert set(PAIRWISE_PER_TASK_COLUMNS).issubset(evaluation.per_task.columns)
    assert "pairwise_policy_linear_combined" in set(evaluation.results["method"])
    assert not evaluation.results["overall_acc"].isna().any()
    assert not evaluation.pairwise_pairs.empty
    assert "selected_thresholds" in evaluation.summary


def test_missing_diagnostics_pair_uses_safe_default_and_warning() -> None:
    lookup = PairwiseDiagnosticsLookup.from_dataframe(pd.DataFrame(columns=["task_i", "task_j"]))

    value = lookup.lookup(3, 2)

    assert value.pair_auc == 1.0
    assert "safe default" in value.warning or "safe defaults" in value.warning


def _signals(
    *,
    confidence: list[float],
    margin: list[float],
    pair_auc: list[float],
    prior: list[float],
) -> PairwiseSignals:
    n = len(confidence)
    return PairwiseSignals(
        top2_routes=np.tile(np.asarray([[0, 1]], dtype=np.int64), (n, 1)),
        confidence=np.asarray(confidence, dtype=np.float32),
        margin=np.asarray(margin, dtype=np.float32),
        pair_auc=np.asarray(pair_auc, dtype=np.float32),
        pair_separation_ratio=np.full(n, 0.1, dtype=np.float32),
        pair_confusion_prior=np.asarray(prior, dtype=np.float32),
    )


def _make_predictions() -> SplitPredictions:
    return SplitPredictions(
        merge_predictions=np.asarray([0, 1, 2], dtype=np.int64),
        top1_predictions=np.asarray([0, 2, 2], dtype=np.int64),
        top1_mask_sizes=np.asarray([10, 10, 10], dtype=np.float32),
        top2_predictions=np.asarray([0, 1, 2], dtype=np.int64),
        top2_mask_sizes=np.asarray([20, 20, 20], dtype=np.float32),
        full_mask_sizes=np.asarray([100, 100, 100], dtype=np.float32),
    )


def _make_synthetic_dataset() -> FeatureToyDataset:
    task0_train, task0_train_labels = _class_clouds(
        {10: [1.0, 0.0], 11: [0.0, 1.0]},
        seed=0,
    )
    task1_train, task1_train_labels = _class_clouds(
        {20: [-1.0, 0.0], 21: [0.0, -1.0]},
        seed=1,
    )
    task0_val, task0_val_labels = _class_clouds({10: [1.0, 0.0], 11: [0.0, 1.0]}, seed=2)
    task1_val, task1_val_labels = _class_clouds({20: [-1.0, 0.0], 21: [0.0, -1.0]}, seed=3)
    task0_test, task0_test_labels = _class_clouds({10: [1.0, 0.0], 11: [0.0, 1.0]}, seed=4)
    task1_test, task1_test_labels = _class_clouds({20: [-1.0, 0.0], 21: [0.0, -1.0]}, seed=5)
    tasks = (
        _task(
            task_id=0,
            classes=(10, 11),
            train_features=task0_train,
            train_labels=task0_train_labels,
            val_features=task0_val,
            val_labels=task0_val_labels,
            test_features=task0_test,
            test_labels=task0_test_labels,
        ),
        _task(
            task_id=1,
            classes=(20, 21),
            train_features=task1_train,
            train_labels=task1_train_labels,
            val_features=task1_val,
            val_labels=task1_val_labels,
            test_features=task1_test,
            test_labels=task1_test_labels,
        ),
    )
    return FeatureToyDataset(
        stream_id="synthetic",
        seed=0,
        backbone_id="toy",
        tasks=tasks,
        manifest={"tasks": []},
        manifest_path=None,
        feature_bank_metadata={},
    )


def _class_clouds(classes: dict[int, list[float]], *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label, center in classes.items():
        values = rng.normal(loc=center, scale=0.01, size=(8, 2)).astype(np.float32)
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
