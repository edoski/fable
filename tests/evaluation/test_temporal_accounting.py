from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.evaluation.temporal_accounting import (
    SelectedTemporalDecisionRun,
    summarize_selected_temporal_decision_runs,
)
from spice.prediction.decoded_offsets import DecodedOffsets
from spice.temporal import coerce_execution_policy_config, compile_execution_policy_contract
from spice.temporal.problem_store import CompiledProblemStore


def _store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.zeros((6, 1), dtype=np.float32),
        log_base_fees=np.log(np.array([100, 90, 80, 70, 60, 50], dtype=np.float32)).astype(
            np.float32,
            copy=False,
        ),
        timestamps=(np.arange(6, dtype=np.int64) * 12).astype(np.int64, copy=False),
        anchor_rows=np.array([0, 2], dtype=np.int64),
        context_start_rows=np.array([0, 1], dtype=np.int64),
        candidate_start_rows=np.array([1, 3], dtype=np.int64),
        candidate_end_rows=np.array([3, 5], dtype=np.int64),
        max_candidate_slots=2,
    )


def _overflow_store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.zeros((10, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array([100, 90, 80, 60, 55, 70, 50, 45, 40, 35], dtype=np.float32)
        ).astype(np.float32, copy=False),
        timestamps=(np.arange(10, dtype=np.int64) * 12).astype(np.int64, copy=False),
        anchor_rows=np.array([1, 4], dtype=np.int64),
        context_start_rows=np.array([0, 3], dtype=np.int64),
        candidate_start_rows=np.array([2, 5], dtype=np.int64),
        candidate_end_rows=np.array([3, 7], dtype=np.int64),
        max_candidate_slots=3,
    )


def _execution_policy():
    return compile_execution_policy_contract(
        coerce_execution_policy_config({"id": "strict_deadline_miss"})
    )


def _action_space(store: CompiledProblemStore, sample_indices: np.ndarray):
    return _execution_policy().prepare_action_space(store, sample_indices)


def test_selected_temporal_decisions_compute_exact_event_metrics() -> None:
    store = _store()
    result = summarize_selected_temporal_decision_runs(
        store,
        _execution_policy(),
        DecodedOffsets(torch.tensor([0, 1], dtype=torch.int64)),
        _action_space(store, np.array([0, 1], dtype=np.int64)),
        (
            SelectedTemporalDecisionRun(
                np.array([0, 1], dtype=np.int64),
                metadata={"mode": "unit"},
            ),
        ),
    )
    run = result.runs[0]

    assert run.n_events == 2
    assert run.metrics["profit_over_baseline"] == pytest.approx(1.0 / 14.0, rel=1e-5)
    assert run.metrics["cost_over_optimum"] == pytest.approx(1.0 / 16.0, rel=1e-5)
    assert run.metrics["baseline_cost_over_optimum"] == pytest.approx(7.0 / 48.0, rel=1e-5)
    assert run.metrics["exact_optimum_hit_rate"] == pytest.approx(0.5)
    assert run.metrics["realized_fee_sum"] == pytest.approx(150.0)
    assert run.metrics["baseline_fee_sum"] == pytest.approx(160.0)
    assert run.metrics["optimum_fee_sum"] == pytest.approx(140.0)
    assert run.fee_sums == pytest.approx(
        {
            "realized_fee_sum": 150.0,
            "baseline_fee_sum": 160.0,
            "optimum_fee_sum": 140.0,
        }
    )
    assert run.event_metric_sums["profit_over_baseline"] == pytest.approx(
        1.0 / 7.0, rel=1e-5
    )
    assert run.metadata == {"mode": "unit", "overflow_count": 0}


def test_selected_temporal_decisions_count_policy_overflow() -> None:
    store = _overflow_store()
    result = summarize_selected_temporal_decision_runs(
        store,
        _execution_policy(),
        DecodedOffsets(torch.tensor([2, 1], dtype=torch.int64)),
        _action_space(store, np.array([0, 1], dtype=np.int64)),
        (
            SelectedTemporalDecisionRun(
                np.array([0, 1], dtype=np.int64),
                metadata={"mode": "overflow"},
            ),
        ),
    )
    run = result.runs[0]

    assert run.n_events == 2
    assert run.metrics["profit_over_baseline"] == pytest.approx(15.0 / 56.0, rel=1e-5)
    assert run.metrics["cost_over_optimum"] == pytest.approx(-1.0 / 8.0, rel=1e-5)
    assert run.metrics["baseline_cost_over_optimum"] == pytest.approx(0.2, rel=1e-5)
    assert run.metrics["exact_optimum_hit_rate"] == pytest.approx(0.5)
    assert run.metrics["realized_fee_sum"] == pytest.approx(110.0)
    assert run.metrics["baseline_fee_sum"] == pytest.approx(150.0)
    assert run.metrics["optimum_fee_sum"] == pytest.approx(130.0)
    assert run.metadata == {"mode": "overflow", "overflow_count": 1}


def test_temporal_accounting_aggregates_event_means_and_window_summaries() -> None:
    store = _store()
    execution_policy = _execution_policy()
    decoded_offsets = DecodedOffsets(torch.tensor([0, 1], dtype=torch.int64))
    action_space = _action_space(store, np.array([0, 1], dtype=np.int64))
    combined_result = summarize_selected_temporal_decision_runs(
        store,
        execution_policy,
        decoded_offsets,
        action_space,
        (
            SelectedTemporalDecisionRun(
                np.array([0, 1], dtype=np.int64),
                metadata={"run": "combined"},
            ),
        ),
    )
    result = summarize_selected_temporal_decision_runs(
        store,
        execution_policy,
        decoded_offsets,
        action_space,
        (
            SelectedTemporalDecisionRun(
                np.array([0], dtype=np.int64),
                metadata={"run": "one"},
            ),
            SelectedTemporalDecisionRun(
                np.array([1], dtype=np.int64),
                metadata={"run": "two"},
            ),
        ),
    )

    assert result.total_events == 2
    assert result.metrics == pytest.approx(combined_result.metrics)
    assert {
        key: result.metrics[key]
        for key in ("realized_fee_sum", "baseline_fee_sum", "optimum_fee_sum")
    } == pytest.approx(
        {
            key: sum(run.fee_sums[key] for run in result.runs)
            for key in ("realized_fee_sum", "baseline_fee_sum", "optimum_fee_sum")
        }
    )
    assert result.window_metrics["profit_over_baseline"].mean == pytest.approx(
        sum(run.metrics["profit_over_baseline"] for run in result.runs) / 2.0
    )
    assert result.window_metrics["cost_over_optimum"].mean == pytest.approx(
        sum(run.metrics["cost_over_optimum"] for run in result.runs) / 2.0
    )
