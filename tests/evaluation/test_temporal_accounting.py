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


def _execution_policy():
    return compile_execution_policy_contract(
        coerce_execution_policy_config({"id": "strict_deadline_miss"})
    )


def test_selected_temporal_decisions_return_typed_replay_run_result() -> None:
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
    assert run.metrics.realized_fee_sum > 0.0
    assert run.metrics.baseline_fee_sum > 0.0
    assert run.metadata == {"mode": "unit", "overflow_count": 0}


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
    assert result.window_metrics["profit_over_baseline"].mean == pytest.approx(
        np.mean([0.2, 0.1])
    )


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
