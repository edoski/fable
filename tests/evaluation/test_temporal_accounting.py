from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.evaluation.temporal_accounting import (
    summarize_selected_temporal_decisions,
    summarize_temporal_accounting_runs,
)
from spice.evaluation.temporal_replay_results import (
    TemporalReplayEventMetricSums,
    TemporalReplayMetrics,
    TemporalReplayResult,
    TemporalReplayRunResult,
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


def test_selected_temporal_decisions_compute_exact_event_metrics() -> None:
    run = summarize_selected_temporal_decisions(
        _store(),
        _execution_policy(),
        DecodedOffsets(torch.tensor([0, 1], dtype=torch.int64)),
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        metadata={"mode": "unit"},
    )

    assert isinstance(run, TemporalReplayRunResult)
    assert run.n_events == 2
    assert isinstance(run.metrics, TemporalReplayMetrics)
    assert run.metrics.profit_over_baseline == pytest.approx(1.0 / 14.0, rel=1e-5)
    assert run.metrics.cost_over_optimum == pytest.approx(1.0 / 16.0, rel=1e-5)
    assert run.metrics.baseline_cost_over_optimum == pytest.approx(7.0 / 48.0, rel=1e-5)
    assert run.metrics.exact_optimum_hit_rate == pytest.approx(0.5)
    assert run.metrics.realized_fee_sum == pytest.approx(150.0)
    assert run.metrics.baseline_fee_sum == pytest.approx(160.0)
    assert run.metrics.optimum_fee_sum == pytest.approx(140.0)
    assert run.event_metric_sums.profit_over_baseline == pytest.approx(1.0 / 7.0, rel=1e-5)
    assert run.metadata == {"mode": "unit", "overflow_count": 0}


def test_selected_temporal_decisions_count_policy_overflow() -> None:
    run = summarize_selected_temporal_decisions(
        _overflow_store(),
        _execution_policy(),
        DecodedOffsets(torch.tensor([2, 1], dtype=torch.int64)),
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        metadata={"mode": "overflow"},
    )

    assert run.n_events == 2
    assert run.metrics.profit_over_baseline == pytest.approx(15.0 / 56.0, rel=1e-5)
    assert run.metrics.cost_over_optimum == pytest.approx(-1.0 / 8.0, rel=1e-5)
    assert run.metrics.baseline_cost_over_optimum == pytest.approx(0.2, rel=1e-5)
    assert run.metrics.exact_optimum_hit_rate == pytest.approx(0.5)
    assert run.metrics.realized_fee_sum == pytest.approx(110.0)
    assert run.metrics.baseline_fee_sum == pytest.approx(150.0)
    assert run.metrics.optimum_fee_sum == pytest.approx(130.0)
    assert run.metadata == {"mode": "overflow", "overflow_count": 1}


def test_temporal_accounting_summary_returns_event_weighted_replay_result() -> None:
    runs = [
        _run_result(
            n_events=1,
            profit_sum=0.2,
            cost_sum=0.4,
            baseline_cost_sum=0.6,
            exact_hit_sum=1.0,
            realized_fee_sum=10.0,
            baseline_fee_sum=20.0,
            optimum_fee_sum=8.0,
        ),
        _run_result(
            n_events=3,
            profit_sum=0.3,
            cost_sum=0.6,
            baseline_cost_sum=0.9,
            exact_hit_sum=1.0,
            realized_fee_sum=30.0,
            baseline_fee_sum=60.0,
            optimum_fee_sum=24.0,
        ),
    ]

    result = summarize_temporal_accounting_runs(runs)

    assert isinstance(result, TemporalReplayResult)
    assert result.total_events == 4
    assert result.runs == tuple(runs)
    assert result.metrics.profit_over_baseline == pytest.approx(0.5 / 4)
    assert result.metrics.cost_over_optimum == pytest.approx(1.0 / 4)
    assert result.metrics.baseline_cost_over_optimum == pytest.approx(1.5 / 4)
    assert result.metrics.exact_optimum_hit_rate == pytest.approx(2.0 / 4)
    assert result.metrics.realized_fee_sum == 40.0
    assert result.window_metrics["profit_over_baseline"].mean == pytest.approx(np.mean([0.2, 0.1]))


def _run_result(
    *,
    n_events: int,
    profit_sum: float,
    cost_sum: float,
    baseline_cost_sum: float,
    exact_hit_sum: float,
    realized_fee_sum: float,
    baseline_fee_sum: float,
    optimum_fee_sum: float,
) -> TemporalReplayRunResult:
    event_sums = TemporalReplayEventMetricSums(
        profit_over_baseline=profit_sum,
        cost_over_optimum=cost_sum,
        baseline_cost_over_optimum=baseline_cost_sum,
        exact_optimum_hit_rate=exact_hit_sum,
    )
    return TemporalReplayRunResult(
        n_events=n_events,
        metrics=TemporalReplayMetrics(
            profit_over_baseline=profit_sum / n_events,
            cost_over_optimum=cost_sum / n_events,
            baseline_cost_over_optimum=baseline_cost_sum / n_events,
            exact_optimum_hit_rate=exact_hit_sum / n_events,
            realized_fee_sum=realized_fee_sum,
            baseline_fee_sum=baseline_fee_sum,
            optimum_fee_sum=optimum_fee_sum,
        ),
        event_metric_sums=event_sums,
        metadata={},
    )
