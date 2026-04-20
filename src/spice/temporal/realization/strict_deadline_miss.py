"""Strict bounded-delay realization policy."""

from __future__ import annotations

import numpy as np

from ..problem_store import CompiledProblemStore
from .base import (
    CompiledRealizationPolicyContract,
    IntVector,
    PreparedSupervisedRealizationTargets,
    RealizationPolicyConfig,
    RealizedSelectionBatch,
)


class StrictDeadlineMissConfig(RealizationPolicyConfig):
    id: str = "strict_deadline_miss"


def _prepare_supervised_targets(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> PreparedSupervisedRealizationTargets:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    resolved_indices = sample_indices.astype(np.int64, copy=False)
    batch_size = int(resolved_indices.shape[0])
    max_candidate_slots = int(store.max_candidate_slots)
    candidate_mask = np.zeros((batch_size, max_candidate_slots), dtype=np.bool_)
    optimum_offsets = np.empty(batch_size, dtype=np.int64)
    optimum_log_fees = np.empty(batch_size, dtype=np.float32)
    for row, sample_index in enumerate(resolved_indices):
        anchor_row = int(store.anchor_rows[sample_index])
        candidate_end = int(store.candidate_end_rows[sample_index])
        candidate_values = store.log_base_fees[anchor_row + 1 : candidate_end]
        candidate_count = int(candidate_values.shape[0])
        if candidate_count <= 0:
            raise ValueError("strict_deadline_miss requires at least one future candidate")
        if candidate_count > max_candidate_slots:
            raise ValueError(
                "strict_deadline_miss requires fixed action space to upper-bound realized "
                "future candidates"
            )
        candidate_mask[row, :candidate_count] = True
        min_offset = int(np.argmin(candidate_values))
        optimum_offsets[row] = min_offset
        optimum_log_fees[row] = float(candidate_values[min_offset])
    return PreparedSupervisedRealizationTargets(
        candidate_mask=candidate_mask,
        optimum_offsets=optimum_offsets,
        optimum_log_fees=optimum_log_fees,
    )


def _realize_selections(
    store: CompiledProblemStore,
    decoded_offsets,
    sample_indices: IntVector,
    selected_positions: IntVector,
) -> RealizedSelectionBatch:
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if selected_positions.size == 0:
        raise ValueError("selected_positions must be non-empty")
    selected_sample_indices = sample_indices[selected_positions]
    requested_offsets = np.asarray(decoded_offsets, dtype=np.int64)[selected_positions]
    baseline_rows = store.anchor_rows[selected_sample_indices] + 1
    candidate_ends = store.candidate_end_rows[selected_sample_indices]
    candidate_counts = candidate_ends - baseline_rows
    valid_mask = requested_offsets < candidate_counts
    overflow_mask = ~valid_mask
    if np.any(overflow_mask & (candidate_ends >= store.n_rows)):
        raise ValueError(
            "strict_deadline_miss requires one realized post-window block for overflow "
            "realization"
        )
    realized_rows = baseline_rows + requested_offsets
    realized_rows = realized_rows.astype(np.int64, copy=False)
    realized_rows[overflow_mask] = candidate_ends[overflow_mask]
    optimum_rows = np.empty(selected_sample_indices.shape[0], dtype=np.int64)
    for row, (start_row, end_row) in enumerate(zip(baseline_rows, candidate_ends, strict=True)):
        optimum_rows[row] = int(start_row + np.argmin(store.log_base_fees[start_row:end_row]))
    resolved_offsets = requested_offsets.copy()
    resolved_offsets[overflow_mask] = -1
    return RealizedSelectionBatch(
        realized_rows=realized_rows,
        baseline_rows=baseline_rows.astype(np.int64, copy=False),
        optimum_rows=optimum_rows,
        requested_offsets=requested_offsets,
        resolved_offsets=resolved_offsets,
        overflow_mask=overflow_mask.astype(np.bool_, copy=False),
    )


def compile_realization_policy(
    config: StrictDeadlineMissConfig,
) -> CompiledRealizationPolicyContract:
    if config.id != "strict_deadline_miss":
        raise ValueError("realization_policy.id must be strict_deadline_miss")
    return CompiledRealizationPolicyContract(
        realization_policy_id="strict_deadline_miss",
        prepare_supervised_targets_fn=_prepare_supervised_targets,
        realize_selections_fn=_realize_selections,
    )
