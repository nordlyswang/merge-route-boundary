from __future__ import annotations

import numpy as np

from mrb.baselines.classifiers import PrototypeClassifier
from scripts.run_feature_toy_baselines import _enabled_classifier_types


def test_prototype_classifier_high_accuracy_on_separable_global_labels() -> None:
    features = np.asarray(
        [
            [3.0, 0.0],
            [2.8, 0.1],
            [0.0, 3.0],
            [0.1, 2.8],
            [-3.0, 0.0],
            [-2.8, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([10, 10, 25, 25, 99, 99], dtype=np.int64)

    classifier = PrototypeClassifier(expected_classes=[10, 25, 99]).fit(features, labels)
    predictions = classifier.predict(features)

    assert np.mean(predictions == labels) >= 0.99
    assert sorted(np.unique(predictions).tolist()) == [10, 25, 99]


def test_prototype_classifier_outputs_global_labels_not_local_indices() -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.asarray([7, 42], dtype=np.int64)

    classifier = PrototypeClassifier().fit(features, labels)

    assert classifier.predict(np.asarray([[0.0, 0.9]], dtype=np.float32)).tolist() == [42]


def test_linear_classifier_disabled_by_default_config_selection() -> None:
    config = {
        "classifiers": {
            "prototype": {"enabled": True},
            "linear": {"enabled": False},
        }
    }

    assert _enabled_classifier_types(config) == ["prototype"]
