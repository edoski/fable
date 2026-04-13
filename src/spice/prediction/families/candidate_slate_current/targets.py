"""Current-family target realization."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ....temporal.problem_store import CompiledProblemStore
from .batch import PreparedCandidateSlateTargets

IntVector = NDArray[np.int64]


def prepare_candidate_slate_targets(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> PreparedCandidateSlateTargets:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    sample_indices = sample_indices.astype(np.int64, copy=False)
    anchor_rows = store.anchor_rows[sample_indices]
    candidate_ends = store.candidate_end_rows[sample_indices]
    candidate_counts = candidate_ends - (anchor_rows + 1)
    batch_size = int(sample_indices.shape[0])
    max_candidate_slots = int(candidate_counts.max())
    candidate_log_fees = np.zeros((batch_size, max_candidate_slots), dtype=np.float32)
    candidate_mask = np.zeros((batch_size, max_candidate_slots), dtype=np.bool_)
    for row, sample_index in enumerate(sample_indices):
        anchor_row = int(store.anchor_rows[sample_index])
        candidate_end = int(store.candidate_end_rows[sample_index])
        candidate_values = store.log_base_fees[anchor_row + 1 : candidate_end]
        candidate_log_fees[row, : candidate_values.shape[0]] = candidate_values
        candidate_mask[row, : candidate_values.shape[0]] = True
    return PreparedCandidateSlateTargets(
        candidate_log_fees=torch.from_numpy(candidate_log_fees),
        candidate_mask=torch.from_numpy(candidate_mask),
    )
