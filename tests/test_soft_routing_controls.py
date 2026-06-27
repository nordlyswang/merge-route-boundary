from __future__ import annotations

import math

import numpy as np

from mrb.baselines.classifiers import PrototypeClassifier
from mrb.baselines.feature_data import FeatureToyDataset, TaskFeatureSplit
from mrb.baselines.routing_controls import (
    compute_topk_task_recall,
    evaluate_soft_routing_controls,
    predict_prototype_with_topk_task_mask,
    predict_with_fallback_to_merge_all,
    predict_with_soft_task_prior,
)


def test_topk_task_mask_uses_union_of_candidate_task_classes() -> None:
    classifier = _fit_four_class_classifier()
    query = np.asarray([[-1.0, 0.0]], dtype=np.float32)

    top1_predictions, top1_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        query,
        np.asarray([[0]], dtype=np.int64),
        task_classes={0: (10, 11), 1: (20, 21)},
    )
    top2_predictions, top2_mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        query,
        np.asarray([[0, 1]], dtype=np.int64),
        task_classes={0: (10, 11), 1: (20, 21)},
    )

    assert top1_predictions[0] in {10, 11}
    assert top1_mask_sizes.tolist() == [2.0]
    assert top2_predictions.tolist() == [20]
    assert top2_mask_sizes.tolist() == [4.0]


def test_topk_recall_counts_true_task_inside_predicted_set() -> None:
    topk = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    true = np.asarray([1, 9, 4], dtype=np.int64)

    assert compute_topk_task_recall(topk, true) == 2 / 3


def test_fallback_uses_merge_all_for_low_confidence_samples() -> None:
    classifier = _fit_four_class_classifier()
    queries = np.asarray([[-1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    wrong_top1 = np.asarray([[0], [0]], dtype=np.int64)
    confidence = np.asarray([0.1, 0.9], dtype=np.float32)

    predictions, fallback_flags, mask_sizes = predict_with_fallback_to_merge_all(
        classifier,
        queries,
        wrong_top1,
        confidence,
        tau=0.5,
        task_classes={0: (10, 11), 1: (20, 21)},
    )

    assert predictions[0] == 20
    assert predictions[1] in {10, 11}
    assert fallback_flags.tolist() == [True, False]
    assert float(np.mean(fallback_flags)) == 0.5
    assert mask_sizes.tolist() == [4.0, 2.0]


def test_soft_prior_beta_zero_matches_merge_all_prediction() -> None:
    dataset = _make_synthetic_dataset()
    classifier = _fit_dataset_classifier(dataset)
    features = dataset.tasks[0].test_features[:4]
    router_scores = np.asarray([[3.0, -1.0]] * features.shape[0], dtype=np.float32)

    merge_predictions = classifier.predict(features)
    soft_predictions = predict_with_soft_task_prior(
        classifier,
        features,
        router_scores,
        route_ids=np.asarray([0, 1], dtype=np.int64),
        tasks=dataset.tasks,
        beta=0.0,
    )

    assert soft_predictions.tolist() == merge_predictions.tolist()


def test_soft_prior_large_beta_prefers_high_prior_task() -> None:
    dataset = _make_synthetic_dataset()
    classifier = _fit_dataset_classifier(dataset)
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    router_scores = np.asarray([[4.0, 0.0]], dtype=np.float32)

    merge_prediction = classifier.predict(query)
    soft_prediction = predict_with_soft_task_prior(
        classifier,
        query,
        router_scores,
        route_ids=np.asarray([0, 1], dtype=np.int64),
        tasks=dataset.tasks,
        beta=2.0,
    )

    assert merge_prediction.tolist() == [20]
    assert soft_prediction.tolist() == [10]


def test_soft_routing_evaluation_selects_hparams_on_val_and_records_gaps() -> None:
    dataset = _make_synthetic_dataset()

    evaluation = evaluate_soft_routing_controls(
        dataset,
        router_types=("linear", "energy"),
        router_config={"linear": {"max_iter": 200, "random_state": 0}},
        topk_config={"enabled": True, "ks": [2]},
        fallback_config={"enabled": True, "linear_tau_grid": [0.0, 0.5, 0.95]},
        soft_prior_config={"enabled": True, "beta_grid": [0.0, 0.5, 2.0]},
    )
    results = evaluation.results

    assert "oracle_gap_closure" in results.columns
    assert "merge_all_prototype_topk_task_mask_linear_k2" in set(results["method"])
    assert "merge_all_prototype_fallback_linear_val_tau" in set(results["method"])
    assert "merge_all_prototype_soft_prior_linear_beta" in set(results["method"])
    assert not results["overall_acc"].isna().any()
    assert results["status"].eq("ok").all()
    assert results.loc[
        results["control_type"].isin(["fallback_to_merge_all", "soft_prior"]),
        "selected_on",
    ].eq("val").all()
    assert not results["oracle_gap_closure"].dropna().empty
    assert evaluation.selected_hparams["fallback"]["linear"]["selected_tau"] in {
        0.0,
        0.5,
        0.95,
    }


def test_invalid_topk_routes_fall_back_to_unrestricted_prediction_without_crashing() -> None:
    classifier = _fit_four_class_classifier()
    query = np.asarray([[-1.0, 0.0]], dtype=np.float32)

    predictions, mask_sizes = predict_prototype_with_topk_task_mask(
        classifier,
        query,
        np.asarray([[99]], dtype=np.int64),
        task_classes={0: (10, 11), 1: (20, 21)},
    )

    assert predictions.tolist() == [20]
    assert mask_sizes.tolist() == [4.0]


def test_fallback_with_invalid_route_still_reports_fallback_rate() -> None:
    classifier = _fit_four_class_classifier()
    query = np.asarray([[-1.0, 0.0]], dtype=np.float32)

    predictions, fallback_flags, _ = predict_with_fallback_to_merge_all(
        classifier,
        query,
        np.asarray([[99]], dtype=np.int64),
        np.asarray([0.0], dtype=np.float32),
        tau=0.5,
        task_classes={0: (10, 11), 1: (20, 21)},
    )

    assert predictions.tolist() == [20]
    assert float(np.mean(fallback_flags)) == 1.0


def _fit_four_class_classifier() -> PrototypeClassifier:
    features = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    labels = np.asarray([10, 11, 20, 21], dtype=np.int64)
    return PrototypeClassifier().fit(features, labels)


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
    assert not math.isnan(float(train_features.mean()))
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
