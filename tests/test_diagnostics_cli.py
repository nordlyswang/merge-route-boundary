from __future__ import annotations

from scripts.compute_boundary_diagnostics import evaluate_quality_gate


def test_quality_gate_all_ok() -> None:
    result = evaluate_quality_gate(
        failed_pairs=0,
        ok_ratio=1.0,
        min_ok_ratio=1.0,
        fail_on_any_failed_pair=True,
    )

    assert result["ok"] is True


def test_quality_gate_fails_on_any_failed_pair() -> None:
    result = evaluate_quality_gate(
        failed_pairs=1,
        ok_ratio=0.9,
        min_ok_ratio=0.8,
        fail_on_any_failed_pair=True,
    )

    assert result["ok"] is False


def test_quality_gate_fails_below_min_ok_ratio() -> None:
    result = evaluate_quality_gate(
        failed_pairs=1,
        ok_ratio=0.5,
        min_ok_ratio=0.8,
        fail_on_any_failed_pair=False,
    )

    assert result["ok"] is False


def test_quality_gate_allows_partial_above_threshold() -> None:
    result = evaluate_quality_gate(
        failed_pairs=1,
        ok_ratio=0.85,
        min_ok_ratio=0.8,
        fail_on_any_failed_pair=False,
    )

    assert result["ok"] is True
