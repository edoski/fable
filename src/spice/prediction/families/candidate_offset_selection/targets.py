"""Current-family target realization."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ....temporal.problem_store import CompiledProblemStore
from ....temporal.realization import CompiledRealizationPolicyContract
from .batch import PreparedCandidateSlateTargets

IntVector = NDArray[np.int64]


def prepare_candidate_slate_targets(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    realization_policy: CompiledRealizationPolicyContract,
) -> PreparedCandidateSlateTargets:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    sample_indices = sample_indices.astype(np.int64, copy=False)
    supervised = realization_policy.prepare_supervised_targets(store, sample_indices)
    batch_size = int(sample_indices.shape[0])
    max_candidate_slots = int(store.max_candidate_slots)
    candidate_log_fees = np.zeros((batch_size, max_candidate_slots), dtype=np.float32)
    candidate_mask = supervised.candidate_mask
    for row, sample_index in enumerate(sample_indices):
        anchor_row = int(store.anchor_rows[sample_index])
        candidate_count = int(candidate_mask[row].sum())
        candidate_values = store.log_base_fees[anchor_row + 1 : anchor_row + 1 + candidate_count]
        candidate_log_fees[row, : candidate_values.shape[0]] = candidate_values
    return PreparedCandidateSlateTargets(
        candidate_log_fees=torch.from_numpy(candidate_log_fees),
        candidate_mask=torch.from_numpy(candidate_mask),
    )
