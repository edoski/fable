from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.prediction import DecodedOffsets
from spice.temporal import (
    coerce_realization_policy_config,
    compile_realization_policy_contract,
)
from spice.temporal.problem_store import CompiledProblemStore


def _realization_policy():
    return compile_realization_policy_contract(
        coerce_realization_policy_config({"id": "strict_deadline_miss"})
    )


def _store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.zeros((8, 1), dtype=np.float32),
        log_base_fees=np.log(np.array([100, 95, 90, 80, 75, 70, 60, 55], dtype=np.float32)),
        timestamps=np.arange(8, dtype=np.int64),
        anchor_rows=np.array([1, 4], dtype=np.int64),
        context_start_rows=np.array([0, 3], dtype=np.int64),
        candidate_start_rows=np.array([2, 5], dtype=np.int64),
        candidate_end_rows=np.array([4, 7], dtype=np.int64),
        max_candidate_slots=3,
    )


def test_strict_deadline_miss_rejects_negative_decoded_offsets() -> None:
    store = _store()

    with pytest.raises(ValueError, match="non-negative"):
        _realization_policy().realize_selections(
            store,
            DecodedOffsets(torch.tensor([-1, 0], dtype=torch.int64)),
            np.arange(store.n_samples, dtype=np.int64),
            np.array([0], dtype=np.int64),
        )


def test_fixed_ex_ante_overflow_realizes_first_post_window_row() -> None:
    store = _store()

    realized = _realization_policy().realize_selections(
        store,
        DecodedOffsets(torch.tensor([2, 2], dtype=torch.int64)),
        np.arange(store.n_samples, dtype=np.int64),
        np.arange(store.n_samples, dtype=np.int64),
    )

    np.testing.assert_array_equal(realized.realized_rows, np.array([4, 7], dtype=np.int64))
    np.testing.assert_array_equal(realized.resolved_offsets, np.array([2, 2], dtype=np.int64))
    np.testing.assert_array_equal(realized.overflow_mask, np.array([True, True]))


def test_supervised_targets_use_best_reachable_fixed_slot() -> None:
    store = CompiledProblemStore(
        feature_matrix=np.zeros((6, 1), dtype=np.float32),
        log_base_fees=np.log(np.array([100, 50, 80, 10, 5, 70], dtype=np.float32)),
        timestamps=np.arange(6, dtype=np.int64),
        anchor_rows=np.array([0], dtype=np.int64),
        context_start_rows=np.array([0], dtype=np.int64),
        candidate_start_rows=np.array([0], dtype=np.int64),
        candidate_end_rows=np.array([5], dtype=np.int64),
        max_candidate_slots=3,
    )

    targets = _realization_policy().prepare_supervised_targets(
        store,
        np.array([0], dtype=np.int64),
    )

    np.testing.assert_array_equal(targets.optimum_offsets, np.array([1], dtype=np.int64))
    np.testing.assert_allclose(targets.optimum_log_fees, np.log(np.array([50], dtype=np.float32)))
