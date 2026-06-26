from __future__ import annotations

import numpy as np

from mrb.baselines.classifiers import PrototypeClassifier
from mrb.baselines.evaluate import (
    evaluate_feature_toy_baselines,
    evaluate_task_learned_route,
    predict_prototype_with_task_mask,
)
from mrb.baselines.feature_data import FeatureToyDataset, TaskFeatureSplit


def test_synthetic_two_task_evaluation_runs_and_records_gaps() -> None:
    dataset = _make_synthetic_dataset()

    evaluation = evaluate_feature_toy_baselines(
        dataset,
        cluster_ks=[1, 2],
        cluster_strategies=["sequential"],
    )
    methods = set(evaluation.results["method"].tolist())

    assert "merge_all_prototype" in methods
    assert "merge_all_prototype_oracle_task_mask" in methods
    assert "merge_all_prototype_learned_task_mask_centroid" in methods
    assert "task_oracle_prototype" in methods
    assert "task_learned_centroid_router_prototype" in methods
    assert "route_vs_merge_gap" in evaluation.results.columns
    assert not evaluation.results["overall_acc"].isna().any()


def test_cluster_k_extremes_match_merge_and_task_oracle_for_oracle_route() -> None:
    dataset = _make_synthetic_dataset()

    evaluation = evaluate_feature_toy_baselines(
        dataset,
        cluster_ks=[1, 2],
        cluster_strategies=["sequential"],
    )
    results = evaluation.results
    merge_acc = _row_acc(results, "merge_all_prototype")
    task_acc = _row_acc(results, "task_oracle_prototype")
    cluster_k1 = results[
        (results["method"] == "cluster_oracle_prototype")
        & (results["cluster_strategy"] == "sequential")
        & (results["K"] == 1)
    ].iloc[0]
    cluster_k2 = results[
        (results["method"] == "cluster_oracle_prototype")
        & (results["cluster_strategy"] == "sequential")
        & (results["K"] == 2)
    ].iloc[0]

    assert float(cluster_k1["overall_acc"]) == merge_acc
    assert float(cluster_k2["overall_acc"]) == task_acc


def test_learned_route_errors_do_not_crash_prediction() -> None:
    dataset = _make_synthetic_dataset(force_bad_test_routes=True)

    evaluation = evaluate_task_learned_route(dataset.tasks)

    assert evaluation.result["status"] == "ok"
    assert 0.0 <= float(evaluation.result["overall_acc"]) <= 1.0
    assert 0.0 <= float(evaluation.result["router_acc"]) <= 1.0


def test_task_mask_restricts_global_prototype_predictions_to_route_classes() -> None:
    features = np.asarray(
        [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]],
        dtype=np.float32,
    )
    labels = np.asarray([10, 11, 20, 21], dtype=np.int64)
    classifier = PrototypeClassifier().fit(features, labels)
    query = np.asarray([[-2.0, 0.0]], dtype=np.float32)

    unrestricted = classifier.predict(query)
    masked = predict_prototype_with_task_mask(
        classifier,
        query,
        route_ids=np.asarray([0], dtype=np.int64),
        task_classes={0: (10, 11), 1: (20, 21)},
    )

    assert unrestricted.tolist() == [20]
    assert masked[0] in {10, 11}


def test_decomposition_fields_exist_for_task_mask_controls() -> None:
    dataset = _make_synthetic_dataset()

    evaluation = evaluate_feature_toy_baselines(
        dataset,
        cluster_ks=[1, 2],
        cluster_strategies=["sequential"],
        router_types=["centroid", "linear", "energy"],
        router_config={
            "linear": {"max_iter": 200, "random_state": 0},
            "energy": {"score": "max", "top_k": 2},
        },
    )
    results = evaluation.results
    learned_mask = results[
        results["method"].str.startswith("merge_all_prototype_learned_task_mask_")
    ]

    assert "label_mask_gain" in results.columns
    assert "routing_error_cost" in results.columns
    assert "classifier_specialization_gain" in results.columns
    assert not results["label_mask_gain"].isna().all()
    assert not results["classifier_specialization_gain"].isna().all()
    assert not learned_mask["routing_error_cost"].isna().any()
    assert "task_learned_linear_router_prototype" in set(results["method"])
    assert "task_learned_energy_router_prototype" in set(results["method"])


def _row_acc(results, method: str) -> float:
    return float(results[results["method"] == method].iloc[0]["overall_acc"])


def _make_synthetic_dataset(*, force_bad_test_routes: bool = False) -> FeatureToyDataset:
    task0_train, task0_train_labels = _class_clouds({10: [2.0, 0.0], 11: [0.0, 2.0]}, seed=0)
    task1_train, task1_train_labels = _class_clouds({20: [-2.0, 0.0], 21: [0.0, -2.0]}, seed=1)
    if force_bad_test_routes:
        task0_test, task0_test_labels = _class_clouds({10: [-2.0, 0.0], 11: [0.0, -2.0]}, seed=2)
    else:
        task0_test, task0_test_labels = _class_clouds({10: [2.0, 0.0], 11: [0.0, 2.0]}, seed=2)
    task1_test, task1_test_labels = _class_clouds({20: [-2.0, 0.0], 21: [0.0, -2.0]}, seed=3)
    tasks = (
        _task(
            task_id=0,
            classes=(10, 11),
            train_features=task0_train,
            train_labels=task0_train_labels,
            test_features=task0_test,
            test_labels=task0_test_labels,
        ),
        _task(
            task_id=1,
            classes=(20, 21),
            train_features=task1_train,
            train_labels=task1_train_labels,
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
        values = rng.normal(loc=center, scale=0.03, size=(8, 2)).astype(np.float32)
        features.append(values)
        labels.append(np.full(values.shape[0], label, dtype=np.int64))
    return np.vstack(features), np.concatenate(labels)


def _task(
    *,
    task_id: int,
    classes: tuple[int, ...],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> TaskFeatureSplit:
    train_indices = np.arange(train_labels.shape[0], dtype=np.int64)
    test_indices = np.arange(test_labels.shape[0], dtype=np.int64)
    return TaskFeatureSplit(
        task_id=task_id,
        dataset_id="synthetic",
        classes=classes,
        train_features=train_features,
        train_labels=train_labels,
        train_indices=train_indices,
        val_features=train_features,
        val_labels=train_labels,
        val_indices=train_indices,
        test_features=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
        feature_bank_splits={"train": "train", "val": "train", "test": "test"},
    )
