"""Strict bounded-delay realization policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..problem_store import CompiledProblemStore
from ..semantics import BaselineRowMode
from .base import (
    CompiledRealizationPolicyContract,
    DecodedOffsetBatch,
    IntVector,
    PreparedSupervisedRealizationTargets,
    RealizationPolicyConfig,
    RealizedSelectionBatch,
)


class StrictDeadlineMissConfig(RealizationPolicyConfig):
    id: str = "strict_deadline_miss"


@dataclass(frozen=True, slots=True)
class _CandidateWindowSummary:
    baseline_rows: IntVector
    candidate_end_rows: IntVector
    candidate_counts: IntVector
    post_window_rows: IntVector
    optimum_offsets: IntVector
    optimum_log_fees: np.ndarray
    optimum_rows: IntVector


def _candidate_window_summary(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> _CandidateWindowSummary:
    resolved_indices = sample_indices.astype(np.int64, copy=False)
    baseline_rows = store.candidate_start_rows[resolved_indices].astype(np.int64, copy=False)
    candidate_end_rows = store.candidate_end_rows[resolved_indices].astype(np.int64, copy=False)
    candidate_counts = (candidate_end_rows - baseline_rows).astype(np.int64, copy=False)
    if np.any(candidate_counts <= 0):
        raise ValueError("strict_deadline_miss requires at least one future candidate")
    post_window_rows = candidate_end_rows.astype(np.int64, copy=False)
    optimum_offsets = np.empty(resolved_indices.shape[0], dtype=np.int64)
    optimum_log_fees = np.empty(resolved_indices.shape[0], dtype=np.float32)
    optimum_rows = np.empty(resolved_indices.shape[0], dtype=np.int64)
    for row, (start_row, end_row) in enumerate(
        zip(baseline_rows, candidate_end_rows, strict=True)
    ):
        candidate_values = store.log_base_fees[start_row:end_row]
        min_offset = int(np.argmin(candidate_values))
        optimum_offsets[row] = min_offset
        optimum_log_fees[row] = float(candidate_values[min_offset])
        optimum_rows[row] = int(start_row + min_offset)
    return _CandidateWindowSummary(
        baseline_rows=baseline_rows,
        candidate_end_rows=candidate_end_rows,
        candidate_counts=candidate_counts,
        post_window_rows=post_window_rows,
        optimum_offsets=optimum_offsets,
        optimum_log_fees=optimum_log_fees,
        optimum_rows=optimum_rows,
    )


def _prepare_supervised_targets(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> PreparedSupervisedRealizationTargets:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    resolved_indices = sample_indices.astype(np.int64, copy=False)
    window_summary = _candidate_window_summary(store, resolved_indices)
    batch_size = int(resolved_indices.shape[0])
    max_candidate_slots = int(store.max_candidate_slots)
    candidate_mask = np.zeros((batch_size, max_candidate_slots), dtype=np.bool_)
    candidate_log_fees = np.zeros((batch_size, max_candidate_slots), dtype=np.float32)
    optimum_offsets = np.empty(batch_size, dtype=np.int64)
    optimum_log_fees = np.empty(batch_size, dtype=np.float32)
    baseline_candidate_indices = np.zeros(batch_size, dtype=np.int64)
    for row, (start_row, end_row, candidate_count) in enumerate(
        zip(
            window_summary.baseline_rows,
            window_summary.candidate_end_rows,
            window_summary.candidate_counts,
            strict=True,
        )
    ):
        candidate_values = store.log_base_fees[start_row:end_row]
        candidate_mask[row, :] = True
        slot_count = min(int(candidate_count), max_candidate_slots)
        candidate_log_fees[row, :slot_count] = candidate_values[:slot_count]
        reachable_values = candidate_values[:slot_count]
        reachable_offset = int(np.argmin(reachable_values))
        optimum_offsets[row] = reachable_offset
        optimum_log_fees[row] = float(reachable_values[reachable_offset])
        if int(candidate_count) < max_candidate_slots:
            if int(end_row) >= store.n_rows:
                raise ValueError(
                    "strict_deadline_miss requires a post-window row "
                    "for overflow supervision"
                )
            candidate_log_fees[row, candidate_count:] = store.log_base_fees[int(end_row)]
    return PreparedSupervisedRealizationTargets(
        candidate_mask=candidate_mask,
        candidate_log_fees=candidate_log_fees,
        optimum_offsets=optimum_offsets,
        optimum_log_fees=optimum_log_fees,
        baseline_candidate_indices=baseline_candidate_indices,
    )


def _realize_selections(
    store: CompiledProblemStore,
    decoded_offsets: DecodedOffsetBatch,
    sample_indices: IntVector,
    selected_positions: IntVector,
) -> RealizedSelectionBatch:
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if selected_positions.size == 0:
        raise ValueError("selected_positions must be non-empty")
    selected_sample_indices = sample_indices[selected_positions]
    window_summary = _candidate_window_summary(store, selected_sample_indices)
    requested_offsets = decoded_offsets.select(selected_positions).astype(np.int64, copy=False)
    if np.any(requested_offsets < 0):
        raise ValueError("decoded_offsets must be non-negative")
    overflow_mask = requested_offsets >= window_summary.candidate_counts
    overflow_without_post_window = np.any(
        window_summary.post_window_rows[overflow_mask] >= store.n_rows
    )
    if np.any(overflow_mask) and overflow_without_post_window:
        raise ValueError(
            "strict_deadline_miss requires a post-window row "
            "for overflow realization"
        )
    resolved_offsets = requested_offsets.copy()
    realized_rows = window_summary.baseline_rows + requested_offsets
    if np.any(overflow_mask):
        resolved_offsets[overflow_mask] = window_summary.candidate_counts[overflow_mask]
        realized_rows[overflow_mask] = window_summary.post_window_rows[overflow_mask]
    return RealizedSelectionBatch(
        realized_rows=realized_rows,
        baseline_rows=window_summary.baseline_rows,
        optimum_rows=window_summary.optimum_rows,
        requested_offsets=requested_offsets,
        resolved_offsets=resolved_offsets,
        overflow_mask=overflow_mask,
    )


def compile_realization_policy(
    config: StrictDeadlineMissConfig,
) -> CompiledRealizationPolicyContract:
    if config.id != "strict_deadline_miss":
        raise ValueError("realization_policy.id must be strict_deadline_miss")
    return CompiledRealizationPolicyContract(
        realization_policy_id="strict_deadline_miss",
        baseline_row_mode=BaselineRowMode.FIRST_CANDIDATE,
        requires_post_window_row=False,
        prepare_supervised_targets_fn=_prepare_supervised_targets,
        realize_selections_fn=_realize_selections,
    )
