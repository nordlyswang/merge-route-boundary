"""Calibration utilities for lightweight feature routers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RouterScorePayload:
    scores: np.ndarray
    route_ids: np.ndarray
    score_source: str


@dataclass(frozen=True)
class CalibrationMetrics:
    nll: float
    ece: float
    brier: float
    top1_acc: float
    top2_recall: float
    top3_recall: float


@dataclass(frozen=True)
class TemperatureSelection:
    temperature: float
    val_metrics: CalibrationMetrics
    metrics_by_temperature: dict[float, CalibrationMetrics]


def extract_linear_router_scores(router: object, features: np.ndarray) -> RouterScorePayload:
    model = getattr(router, "model_", None)
    if model is None:
        raise ValueError("linear router is not fitted")
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"features must be rank 2 [N, D], got shape {values.shape}")
    route_ids = np.asarray(model.classes_, dtype=np.int64)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(values), dtype=np.float32)
        source = "decision_function"
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T.astype(np.float32)
    elif hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(values), dtype=np.float32)
        scores = np.log(np.clip(probabilities, 1e-12, 1.0)).astype(np.float32)
        source = "log_proba"
    else:
        raise ValueError("linear router must expose decision_function or predict_proba")
    if scores.ndim != 2 or scores.shape[1] != route_ids.shape[0]:
        raise ValueError("router score columns must match router classes")
    return RouterScorePayload(scores=scores, route_ids=route_ids, score_source=source)


def temperature_scaled_probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    score_values = _validate_scores(scores)
    return softmax(score_values / float(temperature))


def softmax(scores: np.ndarray) -> np.ndarray:
    score_values = _validate_scores(scores)
    shifted = score_values - np.max(score_values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp, axis=1, keepdims=True), dtype=np.float32)


def log_probabilities(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = _validate_probabilities(probabilities)
    return np.asarray(np.log(np.clip(values, eps, 1.0)), dtype=np.float32)


def predict_topk_from_probabilities(
    probabilities: np.ndarray,
    route_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    values = _validate_probabilities(probabilities)
    route_values = np.asarray(route_ids, dtype=np.int64)
    if route_values.ndim != 1 or route_values.shape[0] != values.shape[1]:
        raise ValueError("route_ids must match probability columns")
    if k <= 0:
        raise ValueError("k must be positive")
    limit = min(int(k), route_values.shape[0])
    order = np.argsort(-values, axis=1)[:, :limit]
    return route_values[order]


def compute_calibration_metrics(
    probabilities: np.ndarray,
    true_route_ids: np.ndarray,
    route_ids: np.ndarray,
    *,
    n_bins: int = 15,
) -> CalibrationMetrics:
    values = _validate_probabilities(probabilities)
    true_values = np.asarray(true_route_ids, dtype=np.int64)
    route_values = np.asarray(route_ids, dtype=np.int64)
    if true_values.ndim != 1 or true_values.shape[0] != values.shape[0]:
        raise ValueError("true_route_ids must be rank 1 and match probabilities rows")
    if route_values.ndim != 1 or route_values.shape[0] != values.shape[1]:
        raise ValueError("route_ids must match probability columns")
    return CalibrationMetrics(
        nll=negative_log_likelihood(values, true_values, route_values),
        ece=expected_calibration_error(values, true_values, route_values, n_bins=n_bins),
        brier=brier_score(values, true_values, route_values),
        top1_acc=topk_recall(values, true_values, route_values, k=1),
        top2_recall=topk_recall(values, true_values, route_values, k=2),
        top3_recall=topk_recall(values, true_values, route_values, k=3),
    )


def select_temperature(
    scores: np.ndarray,
    true_route_ids: np.ndarray,
    route_ids: np.ndarray,
    temperatures: Sequence[float],
    *,
    n_bins: int = 15,
) -> TemperatureSelection:
    candidates = _unique_positive_temperatures(temperatures)
    metrics_by_temperature: dict[float, CalibrationMetrics] = {}
    best_temperature = candidates[0]
    best_metrics: CalibrationMetrics | None = None
    for temperature in candidates:
        probabilities = temperature_scaled_probabilities(scores, temperature)
        metrics = compute_calibration_metrics(
            probabilities,
            true_route_ids,
            route_ids,
            n_bins=n_bins,
        )
        metrics_by_temperature[temperature] = metrics
        if best_metrics is None or _temperature_is_better(metrics, best_metrics):
            best_temperature = temperature
            best_metrics = metrics
    if best_metrics is None:
        raise ValueError("at least one temperature is required")
    return TemperatureSelection(
        temperature=best_temperature,
        val_metrics=best_metrics,
        metrics_by_temperature=metrics_by_temperature,
    )


def negative_log_likelihood(
    probabilities: np.ndarray,
    true_route_ids: np.ndarray,
    route_ids: np.ndarray,
) -> float:
    values = _validate_probabilities(probabilities)
    cols = _true_route_columns(true_route_ids, route_ids)
    return float(-np.mean(np.log(np.clip(values[np.arange(values.shape[0]), cols], 1e-12, 1.0))))


def expected_calibration_error(
    probabilities: np.ndarray,
    true_route_ids: np.ndarray,
    route_ids: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    values = _validate_probabilities(probabilities)
    true_values = np.asarray(true_route_ids, dtype=np.int64)
    route_values = np.asarray(route_ids, dtype=np.int64)
    predictions = route_values[np.argmax(values, axis=1)]
    confidence = np.max(values, axis=1)
    correctness = predictions == true_values
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for bin_index in range(int(n_bins)):
        left = bin_edges[bin_index]
        right = bin_edges[bin_index + 1]
        if bin_index == 0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence > left) & (confidence <= right)
        if not np.any(mask):
            continue
        acc_bin = float(np.mean(correctness[mask]))
        conf_bin = float(np.mean(confidence[mask]))
        ece += abs(acc_bin - conf_bin) * float(np.mean(mask))
    return float(ece)


def brier_score(
    probabilities: np.ndarray,
    true_route_ids: np.ndarray,
    route_ids: np.ndarray,
) -> float:
    values = _validate_probabilities(probabilities)
    cols = _true_route_columns(true_route_ids, route_ids)
    target = np.zeros_like(values, dtype=np.float32)
    target[np.arange(values.shape[0]), cols] = 1.0
    return float(np.mean(np.sum((values - target) ** 2, axis=1)))


def topk_recall(
    probabilities: np.ndarray,
    true_route_ids: np.ndarray,
    route_ids: np.ndarray,
    *,
    k: int,
) -> float:
    topk = predict_topk_from_probabilities(probabilities, route_ids, k)
    true_values = np.asarray(true_route_ids, dtype=np.int64)
    return float(np.mean(np.any(topk == true_values[:, None], axis=1)))


def _temperature_is_better(
    candidate: CalibrationMetrics,
    incumbent: CalibrationMetrics,
) -> bool:
    if candidate.nll < incumbent.nll - 1e-12:
        return True
    if abs(candidate.nll - incumbent.nll) <= 1e-12 and candidate.ece < incumbent.ece:
        return True
    return False


def _true_route_columns(true_route_ids: np.ndarray, route_ids: np.ndarray) -> np.ndarray:
    true_values = np.asarray(true_route_ids, dtype=np.int64)
    route_values = np.asarray(route_ids, dtype=np.int64)
    if true_values.ndim != 1:
        raise ValueError("true_route_ids must be rank 1")
    route_to_col = {int(route_id): col for col, route_id in enumerate(route_values)}
    try:
        return np.asarray([route_to_col[int(route_id)] for route_id in true_values], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"true route id not found in route_ids: {exc}") from exc


def _unique_positive_temperatures(temperatures: Sequence[float]) -> list[float]:
    unique: list[float] = []
    for value in temperatures:
        number = float(value)
        if number <= 0:
            raise ValueError("temperatures must be positive")
        if not any(abs(number - existing) <= 1e-12 for existing in unique):
            unique.append(number)
    if not unique:
        raise ValueError("at least one temperature is required")
    return unique


def _validate_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"scores must be rank 2 [N, R], got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("scores must be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("scores contains NaN or Inf")
    return values


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"probabilities must be rank 2 [N, R], got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("probabilities must be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contains NaN or Inf")
    if np.any(values < -1e-6):
        raise ValueError("probabilities contains negative values")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("probability rows must sum to 1")
    return values
