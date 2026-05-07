"""Temporal decision accounting used by the Temporal Replay Runner."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..prediction.decoded_offsets import DecodedOffsets
from ..temporal.execution_policy import CompiledExecutionPolicyContract, PreparedActionSpace
from ..temporal.problem_store import CompiledProblemStore
from ._temporal_replay_metric_catalog import (
    temporal_replay_event_sum_totals,
    temporal_replay_fee_sum_totals,
    temporal_replay_fee_sums,
    temporal_replay_metric_values,
    temporal_replay_window_metrics,
)
from .contracts import EvaluationMetadataValue
from .temporal_replay_results import TemporalReplayResult, TemporalReplayRunResult

IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SelectedTemporalDecisionRun:
    selected_positions: IntVector
    metadata: dict[str, EvaluationMetadataValue]


@dataclass(frozen=True, slots=True)
class _TemporalCostSummary:
    n_events: int
    fee_sums: dict[str, float]
    event_metric_sums: dict[str, float]


def summarize_selected_temporal_decision_runs(
    store: CompiledProblemStore,
    execution_policy: CompiledExecutionPolicyContract,
    decoded_offsets: DecodedOffsets,
    action_space: PreparedActionSpace,
    selected_runs: Iterable[SelectedTemporalDecisionRun],
) -> TemporalReplayResult:
    runs = [
        _summarize_selected_temporal_decision_run(
            store,
            execution_policy,
            decoded_offsets,
            action_space,
            selected_run.selected_positions,
            metadata=selected_run.metadata,
        )
        for selected_run in selected_runs
    ]
    if not runs:
        raise ValueError("evaluation produced no runs")

    total_events = sum(run.n_events for run in runs)
    event_metric_sums = temporal_replay_event_sum_totals(
        run.event_metric_sums for run in runs
    )
    fee_sums = temporal_replay_fee_sum_totals(run.fee_sums for run in runs)
    _validate_fee_totals(fee_sums)
    return TemporalReplayResult(
        metrics=temporal_replay_metric_values(
            event_metric_sums=event_metric_sums,
            fee_sums=fee_sums,
            n_events=total_events,
        ),
        window_metrics=(
            temporal_replay_window_metrics(run.metrics for run in runs)
            if len(runs) > 1
            else {}
        ),
        total_events=total_events,
        runs=tuple(runs),
    )


def _summarize_selected_temporal_decision_run(
    store: CompiledProblemStore,
    execution_policy: CompiledExecutionPolicyContract,
    decoded_offsets: DecodedOffsets,
    action_space: PreparedActionSpace,
    selected_positions: IntVector,
    *,
    metadata: dict[str, EvaluationMetadataValue],
) -> TemporalReplayRunResult:
    realized = execution_policy.realize_selections(
        store,
        decoded_offsets,
        action_space,
        selected_positions,
    )
    realized_fees = np.exp(
        store.log_base_fees[realized.realized_rows].astype(np.float64, copy=False)
    )
    baseline_fees = np.exp(
        store.log_base_fees[realized.baseline_rows].astype(np.float64, copy=False)
    )
    optimum_fees = np.exp(
        store.log_base_fees[realized.optimum_rows].astype(np.float64, copy=False)
    )
    _validate_event_fees(baseline_fees=baseline_fees, optimum_fees=optimum_fees)

    profit_values = (baseline_fees - realized_fees) / baseline_fees
    cost_values = (realized_fees - optimum_fees) / optimum_fees
    baseline_cost_values = (baseline_fees - optimum_fees) / optimum_fees
    exact_hits = realized.realized_rows == realized.optimum_rows
    costs = _TemporalCostSummary(
        n_events=int(selected_positions.shape[0]),
        fee_sums=temporal_replay_fee_sums(
            realized_fee_sum=float(realized_fees.sum()),
            baseline_fee_sum=float(baseline_fees.sum()),
            optimum_fee_sum=float(optimum_fees.sum()),
        ),
        event_metric_sums={
            "profit_over_baseline": float(profit_values.sum()),
            "cost_over_optimum": float(cost_values.sum()),
            "baseline_cost_over_optimum": float(baseline_cost_values.sum()),
            "exact_optimum_hit_rate": float(exact_hits.sum()),
        },
    )
    return TemporalReplayRunResult(
        n_events=costs.n_events,
        metrics=temporal_replay_metric_values(
            event_metric_sums=costs.event_metric_sums,
            fee_sums=costs.fee_sums,
            n_events=costs.n_events,
        ),
        fee_sums=costs.fee_sums,
        event_metric_sums=costs.event_metric_sums,
        metadata={
            **dict(metadata),
            "overflow_count": int(realized.overflow_mask.sum()),
        },
    )


def _validate_event_fees(
    *,
    baseline_fees: NDArray[np.float64],
    optimum_fees: NDArray[np.float64],
) -> None:
    if float(baseline_fees.sum()) <= 0.0:
        raise ValueError("baseline fee total must be positive")
    if float(optimum_fees.sum()) <= 0.0:
        raise ValueError("optimum fee total must be positive")
    if np.any(baseline_fees <= 0.0):
        raise ValueError("baseline fees must be positive")
    if np.any(optimum_fees <= 0.0):
        raise ValueError("optimum fees must be positive")


def _validate_fee_totals(fee_sums: dict[str, float]) -> None:
    if fee_sums["baseline_fee_sum"] <= 0.0:
        raise ValueError("baseline fee sum must be positive")
    if fee_sums["optimum_fee_sum"] <= 0.0:
        raise ValueError("optimum fee sum must be positive")
