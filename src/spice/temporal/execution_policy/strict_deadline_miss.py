"""Strict bounded-delay execution policy."""

from __future__ import annotations

import numpy as np

from ..problem_store import CompiledProblemStore
from ..semantics import BaselineRowMode
from .base import (
    BoolMatrix,
    CompiledExecutionPolicyContract,
    DecodedOffsetBatch,
    ExecutionPolicyConfig,
    IntVector,
    PreparedActionSpace,
    PreparedSupervisedExecutionTargets,
    RealizedSelectionBatch,
)


class StrictDeadlineMissConfig(ExecutionPolicyConfig):
    id: str = "strict_deadline_miss"


def _action_mask(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> BoolMatrix:
    resolved_sample_indices = sample_indices.astype(np.int64, copy=False)
    return np.ones(
        (resolved_sample_indices.shape[0], store.max_candidate_slots),
        dtype=np.bool_,
    )


def _prepare_action_space(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> PreparedActionSpace:
    return PreparedActionSpace(action_mask=_action_mask(store, sample_indices))


def _reachable_optimum(
    store: CompiledProblemStore,
    *,
    start_row: int,
    reachable_end_row: int,
) -> tuple[int, int, float]:
    candidate_values = store.log_base_fees[start_row:reachable_end_row]
    offset = int(np.argmin(candidate_values))
    return int(start_row + offset), offset, float(candidate_values[offset])


def _prepare_supervised_targets(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> PreparedSupervisedExecutionTargets:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    resolved_indices = sample_indices.astype(np.int64, copy=False)
    window_summary = store.candidate_windows(resolved_indices)
    batch_size = int(resolved_indices.shape[0])
    max_candidate_slots = int(store.max_candidate_slots)
    action_mask = np.zeros((batch_size, max_candidate_slots), dtype=np.bool_)
    candidate_log_fees = np.zeros((batch_size, max_candidate_slots), dtype=np.float32)
    optimum_offsets = np.empty(batch_size, dtype=np.int64)
    optimum_log_fees = np.empty(batch_size, dtype=np.float32)
    baseline_candidate_indices = np.zeros(batch_size, dtype=np.int64)
    for row, (start_row, end_row, candidate_count, reachable_end_row) in enumerate(
        zip(
            window_summary.baseline_rows,
            window_summary.candidate_end_rows,
            window_summary.candidate_counts,
            window_summary.reachable_end_rows,
            strict=True,
        )
    ):
        candidate_values = store.log_base_fees[start_row:end_row]
        action_mask[row, :] = True
        slot_count = min(int(candidate_count), max_candidate_slots)
        candidate_log_fees[row, :slot_count] = candidate_values[:slot_count]
        _, reachable_offset, reachable_log_fee = _reachable_optimum(
            store,
            start_row=int(start_row),
            reachable_end_row=int(reachable_end_row),
        )
        optimum_offsets[row] = reachable_offset
        optimum_log_fees[row] = reachable_log_fee
        if int(candidate_count) < max_candidate_slots:
            if int(end_row) >= store.n_rows:
                raise ValueError(
                    "strict_deadline_miss requires a post-window row "
                    "for overflow supervision"
                )
            candidate_log_fees[row, candidate_count:] = store.log_base_fees[int(end_row)]
    return PreparedSupervisedExecutionTargets(
        action_mask=action_mask,
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
    window_summary = store.candidate_windows(selected_sample_indices)
    requested_offsets = decoded_offsets.select(selected_positions).astype(np.int64, copy=False)
    if np.any(requested_offsets < 0):
        raise ValueError("decoded_offsets must be non-negative")
    if np.any(requested_offsets >= int(store.max_candidate_slots)):
        raise ValueError("decoded_offsets must be smaller than max_candidate_slots")
    overflow_mask = requested_offsets >= window_summary.candidate_counts
    overflow_without_post_window = np.any(
        window_summary.candidate_end_rows[overflow_mask] >= store.n_rows
    )
    if np.any(overflow_mask) and overflow_without_post_window:
        raise ValueError(
            "strict_deadline_miss requires a post-window row "
            "for overflow execution"
        )
    resolved_offsets = requested_offsets.copy()
    realized_rows = window_summary.baseline_rows + requested_offsets
    if np.any(overflow_mask):
        resolved_offsets[overflow_mask] = window_summary.candidate_counts[overflow_mask]
        realized_rows[overflow_mask] = window_summary.candidate_end_rows[overflow_mask]
    optimum_rows = np.empty(selected_sample_indices.shape[0], dtype=np.int64)
    for row, (start_row, reachable_end_row) in enumerate(
        zip(window_summary.baseline_rows, window_summary.reachable_end_rows, strict=True)
    ):
        optimum_rows[row], _, _ = _reachable_optimum(
            store,
            start_row=int(start_row),
            reachable_end_row=int(reachable_end_row),
        )
    return RealizedSelectionBatch(
        realized_rows=realized_rows,
        baseline_rows=window_summary.baseline_rows,
        optimum_rows=optimum_rows,
        requested_offsets=requested_offsets,
        resolved_offsets=resolved_offsets,
        overflow_mask=overflow_mask,
    )


def compile_execution_policy(
    config: StrictDeadlineMissConfig,
) -> CompiledExecutionPolicyContract:
    if config.id != "strict_deadline_miss":
        raise ValueError("execution_policy.id must be strict_deadline_miss")
    return CompiledExecutionPolicyContract(
        execution_policy_id="strict_deadline_miss",
        baseline_row_mode=BaselineRowMode.FIRST_CANDIDATE,
        requires_post_window_row=True,
        prepare_action_space_fn=_prepare_action_space,
        prepare_supervised_targets_fn=_prepare_supervised_targets,
        realize_selections_fn=_realize_selections,
    )
