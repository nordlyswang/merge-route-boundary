from __future__ import annotations

import numpy as np

from mrb.baselines.calibration import (
    compute_calibration_metrics,
    temperature_scaled_probabilities,
)
from mrb.baselines.classifiers import PrototypeClassifier
from mrb.baselines.feature_data import FeatureToyDataset, TaskFeatureSplit
from mrb.baselines.router_calibration_frontier import (
    evaluate_router_calibration_frontier,
    oracle_topk_route_sets,
    predict_with_calibrated_soft_prior,
)
from mrb.baselines.routing_controls import predict_prototype_with_topk_task_mask


def test_temperature_scaling_preserves_shape_and_probability_rows_sum_to_one() -> None:
    logits = np.asarray([[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]], dtype=np.float32)

    probabilities = temperature_scaled_probabilities(logits, 2.0)

    assert probabilities.shape == logits.shape
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_calibration_metrics_compute_ece_nll_and_brier() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]], dtype=np.float32)
    true = np.asarray([0, 1, 1], dtype=np.int64)
    route_ids = np.asarray([0, 1], dtype=np.int64)

    metrics = compute_calibration_metrics(probabilities, true, route_ids, n_bins=3)

    assert metrics.nll > 0.0
    assert metrics.brier > 0.0
    assert 0.0 <= metrics.ece <= 1.0
    assert metrics.top1_acc == 2 / 3
    assert metrics.top2_recall == 1.0


def test_beta_zero_calibrated_soft_prior_matches_merge_all_prediction() -> None:
    dataset = _make_synthetic_dataset()
    classifier = _fit_dataset_classifier(dataset)
    features = dataset.tasks[0].test_features
    log_prior = np.log(np.asarray([[0.9, 0.1]] * features.shape[0], dtype=np.float32))

    merge_predictions = classifier.predict(features)
    soft_predictions = predict_with_calibrated_soft_prior(
        classifier,
        features,
        log_prior,
        route_ids=np.asarray([0, 1], dtype=np.int64),
        tasks=dataset.tasks,
        beta=0.0,
    )

    assert soft_predictions.tolist() == merge_predictions.tolist()


def test_topk_frontier_k1_and_kt_have_expected_label_spaces() -> None:
    dataset = _make_synthetic_dataset()
    classifier = _fit_dataset_classifier(dataset)
    features = np.vstack([task.test_features for task in dataset.tasks])
    labels = np.concatenate([task.test_labels for task in dataset.tasks])
    task_ids = np.concatenate(
        [np.full(task.test_labels.shape[0], task.task_id, dtype=np.int64) for task in dataset.tasks]
    )

    top1_predictions, top1_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        features,
        task_ids.reshape(-1, 1),
        task_classes={0: (10, 11), 1: (20, 21)},
    )
    top2_predictions, top2_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        features,
        np.tile(np.asarray([[0, 1]], dtype=np.int64), (features.shape[0], 1)),
        task_classes={0: (10, 11), 1: (20, 21)},
    )

    assert np.mean(top1_predictions == labels) >= np.mean(top2_predictions == labels)
    assert top1_mask_sizes.tolist() == [2.0] * features.shape[0]
    assert top2_mask_sizes.tolist() == [4.0] * features.shape[0]
    assert top2_predictions.tolist() == classifier.predict(features).tolist()


def test_oracle_topk_k1_matches_oracle_mask_and_kt_matches_full_union() -> None:
    dataset = _make_synthetic_dataset()
    classifier = _fit_dataset_classifier(dataset)
    features = np.vstack([task.test_features for task in dataset.tasks])
    task_ids = np.concatenate(
        [np.full(task.test_labels.shape[0], task.task_id, dtype=np.int64) for task in dataset.tasks]
    )

    k1_routes = oracle_topk_route_sets(task_ids, [0, 1], k=1, seed=0)
    kt_routes = oracle_topk_route_sets(task_ids, [0, 1], k=2, seed=0)
    k1_predictions, k1_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        features,
        k1_routes,
        task_classes={0: (10, 11), 1: (20, 21)},
    )
    kt_predictions, kt_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        features,
        kt_routes,
        task_classes={0: (10, 11), 1: (20, 21)},
    )

    assert k1_predictions.shape == kt_predictions.shape
    assert k1_mask_sizes.tolist() == [2.0] * features.shape[0]
    assert kt_mask_sizes.tolist() == [4.0] * features.shape[0]
    assert kt_predictions.tolist() == classifier.predict(features).tolist()


def test_router_calibration_evaluation_selects_on_val_and_writes_summary_fields() -> None:
    dataset = _make_synthetic_dataset()

    evaluation = evaluate_router_calibration_frontier(
        dataset,
        router_config={"max_iter": 200, "random_state": 0},
        calibration_config={"temperatures": [0.5, 1.0, 2.0], "ece_bins": 5},
        fallback_config={"tau_grid": [0.0, 0.5, 0.9]},
        soft_prior_config={"beta_grid": [0.0, 0.5, 1.0]},
        topk_config={"ks": [1, 2]},
        oracle_topk_config={"ks": [1, 2], "random_extra_repeats": 2},
        config={"synthetic": True},
    )
    results = evaluation.results

    assert "merge_all_prototype_fallback_linear_calibrated_tau" in set(results["method"])
    assert "merge_all_prototype_soft_prior_linear_calibrated_beta" in set(results["method"])
    assert results[results["control_type"] == "topk_task_mask"]["topk"].tolist() == [1.0, 2.0]
    assert results[results["control_type"] == "oracle_topk_decomposition"]["topk"].tolist() == [
        1.0,
        2.0,
    ]
    assert results.loc[
        results["control_type"].isin(["calibrated_fallback", "calibrated_soft_prior"]),
        "selected_on",
    ].eq("val").all()
    assert not results["overall_acc"].isna().any()
    assert "selected_temperature" in evaluation.summary
    assert "topk_frontier" in evaluation.summary
    assert "oracle_topk_decomposition" in evaluation.summary
    assert "mask_size_cost" in results.columns
    assert "oracle_gap_closure" in results.columns


def _fit_dataset_classifier(dataset: FeatureToyDataset) -> PrototypeClassifier:
    features = np.vstack([task.train_features for task in dataset.tasks])
    labels = np.concatenate([task.train_labels for task in dataset.tasks])
    classes = sorted(int(value) for value in np.unique(labels))
    return PrototypeClassifier(expected_classes=classes).fit(features, labels)


def _make_synthetic_dataset() -> FeatureToyDataset:
    task0_train, task0_train_labels = _class_clouds(
        {10: [0.9, 0.1], 11: [0.0, 1.0]},
        seed=0,
    )
    task1_train, task1_train_labels = _class_clouds(
        {20: [1.0, 0.0], 21: [0.0, -1.0]},
        seed=1,
    )
    task0_val, task0_val_labels = _class_clouds({10: [1.0, 0.0], 11: [0.0, 1.0]}, seed=2)
    task1_val, task1_val_labels = _class_clouds({20: [1.0, 0.0], 21: [0.0, -1.0]}, seed=3)
    task0_test, task0_test_labels = _class_clouds({10: [1.0, 0.0], 11: [0.0, 1.0]}, seed=4)
    task1_test, task1_test_labels = _class_clouds({20: [1.0, 0.0], 21: [0.0, -1.0]}, seed=5)
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
