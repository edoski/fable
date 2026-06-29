"""Shared temporal replay evaluator runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..metrics import MetricSet
from ..prediction.decoded_offsets import DecodedOffsets, require_decoded_offsets
from ..prediction.decoding import DecodedPredictionResult
from ..temporal.execution_policy import CompiledExecutionPolicyContract, PreparedActionSpace
from ..temporal.problem_store import CompiledProblemStore
from ._temporal_replay_metric_catalog import TEMPORAL_REPLAY_METRIC_DESCRIPTORS
from .config import EvaluatorConfig
from .contracts import CompiledEvaluatorContract, EvaluationRun, EvaluationSummary
from .temporal_accounting import (
    SelectedTemporalDecisionRun,
    summarize_selected_temporal_decision_runs,
)
from .temporal_replay_results import TemporalReplayResult

IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TemporalReplaySelection:
    selected_positions: IntVector
    metadata: dict[str, str | int | float]


@dataclass(frozen=True, slots=True)
class TemporalReplaySampleView:
    sample_positions: IntVector
    sample_timestamps: IntVector
    sample_block_numbers: IntVector
    sample_count: int

    def __post_init__(self) -> None:
        if self.sample_positions.ndim != 1:
            raise ValueError("temporal replay sample_positions must be one-dimensional")
        if self.sample_timestamps.ndim != 1:
            raise ValueError("temporal replay sample_timestamps must be one-dimensional")
        if self.sample_block_numbers.ndim != 1:
            raise ValueError("temporal replay sample_block_numbers must be one-dimensional")
        if not np.issubdtype(self.sample_positions.dtype, np.integer):
            raise ValueError("temporal replay sample_positions must be integer indices")
        if not np.issubdtype(self.sample_timestamps.dtype, np.integer):
            raise ValueError("temporal replay sample_timestamps must be integer timestamps")
        if not np.issubdtype(self.sample_block_numbers.dtype, np.integer):
            raise ValueError("temporal replay sample_block_numbers must be integer block numbers")
        if self.sample_positions.shape[0] != self.sample_timestamps.shape[0]:
            raise ValueError("temporal replay sample view positions and timestamps must align")
        if self.sample_positions.shape[0] != self.sample_block_numbers.shape[0]:
            raise ValueError("temporal replay sample view positions and block numbers must align")
        if self.sample_count != int(self.sample_positions.shape[0]):
            raise ValueError("temporal replay sample_count must match sample_positions")


class TemporalReplayAdapter(Protocol):
    def selections(
        self,
        samples: TemporalReplaySampleView,
    ) -> Iterable[TemporalReplaySelection]: ...


def compile_temporal_replay_evaluator_contract(
    *,
    evaluator_id: str,
    config: EvaluatorConfig,
    adapter: TemporalReplayAdapter,
    no_runs_error: Exception | None = None,
) -> CompiledEvaluatorContract:
    def run_fn(
        store,
        execution_policy,
        decoded_result,
        action_space,
    ):
        return run_temporal_replay(
            store,
            execution_policy,
            decoded_result,
            action_space,
            adapter=adapter,
            no_runs_error=no_runs_error,
        )

    return CompiledEvaluatorContract(
        evaluator_id=evaluator_id,
        metric_descriptors=TEMPORAL_REPLAY_METRIC_DESCRIPTORS,
        config=config,
        accepted_decoded_result_id=DecodedOffsets.decoded_result_id,
        run_fn=run_fn,
    )


def run_temporal_replay(
    store: CompiledProblemStore,
    execution_policy: CompiledExecutionPolicyContract,
    decoded_result: DecodedPredictionResult,
    action_space: PreparedActionSpace,
    *,
    adapter: TemporalReplayAdapter,
    no_runs_error: Exception | None = None,
) -> EvaluationSummary:
    decoded_offsets = require_decoded_offsets(decoded_result)
    if len(decoded_offsets) != int(action_space.sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    samples = temporal_replay_sample_view(store, action_space)

    selected_runs: list[SelectedTemporalDecisionRun] = []
    for selection in adapter.selections(samples):
        selected_positions = _validated_selected_positions(
            selection.selected_positions,
            sample_count=samples.sample_count,
        )
        selected_runs.append(
            SelectedTemporalDecisionRun(
                selected_positions,
                metadata=_validated_metadata(selection.metadata),
            )
        )
    if not selected_runs:
        raise no_runs_error or ValueError("evaluation produced no runs")
    return _temporal_replay_result_to_summary(
        summarize_selected_temporal_decision_runs(
            store,
            execution_policy,
            decoded_offsets,
            action_space,
            selected_runs,
        )
    )


def temporal_replay_sample_view(
    store: CompiledProblemStore,
    action_space: PreparedActionSpace,
) -> TemporalReplaySampleView:
    sample_indices = action_space.sample_indices
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    resolved_sample_indices = sample_indices.astype(np.int64, copy=False)
    sample_count = int(resolved_sample_indices.shape[0])
    return TemporalReplaySampleView(
        sample_positions=np.arange(sample_count, dtype=np.int64),
        sample_timestamps=store.sample_timestamps(resolved_sample_indices),
        sample_block_numbers=store.sample_block_numbers(resolved_sample_indices),
        sample_count=sample_count,
    )


def _validated_selected_positions(
    selected_positions: IntVector,
    *,
    sample_count: int,
) -> IntVector:
    resolved = np.asarray(selected_positions)
    if resolved.ndim != 1:
        raise ValueError("temporal replay selected_positions must be one-dimensional")
    if resolved.size == 0:
        raise ValueError("temporal replay selected_positions must be non-empty")
    if not np.issubdtype(resolved.dtype, np.integer):
        raise ValueError("temporal replay selected_positions must be integer indices")
    positions = resolved.astype(np.int64, copy=False)
    if np.any(positions < 0) or np.any(positions >= sample_count):
        raise ValueError("temporal replay selected_positions are outside sample_indices")
    return positions


def _validated_metadata(metadata: Mapping[str, object]) -> dict[str, str | int | float]:
    validated: dict[str, str | int | float] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("temporal replay metadata keys must be strings")
        if isinstance(value, bool):
            raise ValueError("temporal replay metadata values must be scalar")
        elif isinstance(value, (str, int, float)):
            validated[key] = value
        else:
            raise ValueError("temporal replay metadata values must be scalar")
    return validated


def _temporal_replay_result_to_summary(result: TemporalReplayResult) -> EvaluationSummary:
    return EvaluationSummary(
        metrics=MetricSet(values=dict(result.metrics)),
        window_metrics=dict(result.window_metrics),
        total_events=result.total_events,
        runs=[
            EvaluationRun(
                n_events=run.n_events,
                metrics=dict(run.metrics),
                metadata=dict(run.metadata),
            )
            for run in result.runs
        ],
    )
