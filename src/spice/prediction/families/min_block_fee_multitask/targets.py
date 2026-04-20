"""Paper-family target realization."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ....temporal.problem_store import CompiledProblemStore
from ....temporal.realization import CompiledRealizationPolicyContract
from .batch import PreparedMinBlockFeeTargets

IntVector = NDArray[np.int64]


def prepare_min_block_fee_targets(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    realization_policy: CompiledRealizationPolicyContract,
) -> PreparedMinBlockFeeTargets:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    sample_indices = sample_indices.astype(np.int64, copy=False)
    supervised = realization_policy.prepare_supervised_targets(store, sample_indices)
    return PreparedMinBlockFeeTargets(
        candidate_mask=torch.from_numpy(supervised.candidate_mask),
        min_block_offsets=torch.from_numpy(supervised.optimum_offsets),
        min_block_log_fees=torch.from_numpy(supervised.optimum_log_fees),
    )
