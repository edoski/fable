"""Shared temporal replay evaluator runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..prediction.decoded_offsets import DecodedOffsets, require_decoded_offsets
from ..prediction.decoding import DecodedPredictionResult
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore
from .config import EvaluatorConfig
from .contracts import CompiledEvaluatorContract, EvaluationSummary, IntVector
from .metrics import TEMPORAL_REPLAY_METRIC_DESCRIPTORS
from .temporal_accounting import (
    summarize_selected_temporal_decisions,
    summarize_temporal_accounting_runs,
)
from .temporal_replay_results import temporal_replay_result_to_summary


@dataclass(frozen=True, slots=True)
class TemporalReplaySelection:
    selected_positions: IntVector
    metadata: dict[str, str | int | float]


class TemporalReplayAdapter(Protocol):
    def selections(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> Iterable[TemporalReplaySelection]: ...

    def no_runs_error(self) -> Exception: ...


def compile_temporal_replay_evaluator_contract(
    *,
    evaluation_id: str,
    config: EvaluatorConfig,
    adapter: TemporalReplayAdapter,
) -> CompiledEvaluatorContract:
    def run_fn(
        store,
        execution_policy,
        decoded_result,
        sample_indices,
    ):
        return run_temporal_replay(
            store,
            execution_policy,
            decoded_result,
            sample_indices,
            adapter=adapter,
        )

    return CompiledEvaluatorContract(
        evaluation_id=evaluation_id,
        metric_descriptors=TEMPORAL_REPLAY_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        config=config,
        accepted_decoded_result_id=DecodedOffsets.decoded_result_id,
        run_fn=run_fn,
    )


def run_temporal_replay(
    store: CompiledProblemStore,
    execution_policy: CompiledExecutionPolicyContract,
    decoded_result: DecodedPredictionResult,
    sample_indices: IntVector,
    *,
    adapter: TemporalReplayAdapter,
) -> EvaluationSummary:
    decoded_offsets = require_decoded_offsets(decoded_result)
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    runs = []
    for selection in adapter.selections(store, sample_indices):
        selected_positions = _validated_selected_positions(
            selection.selected_positions,
            sample_count=int(sample_indices.shape[0]),
        )
        runs.append(
            summarize_selected_temporal_decisions(
                store,
                execution_policy,
                decoded_offsets,
                sample_indices,
                selected_positions,
                metadata=_validated_metadata(selection.metadata),
            )
        )
    if not runs:
        raise adapter.no_runs_error()
    return temporal_replay_result_to_summary(summarize_temporal_accounting_runs(runs))


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
            validated[key] = int(value)
        elif isinstance(value, (str, int, float)):
            validated[key] = value
        else:
            raise ValueError("temporal replay metadata values must be scalar")
    return validated
