from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.prediction.decoded_offsets import DecodedOffsets
from spice.temporal import (
    coerce_execution_policy_config,
    compile_execution_policy_contract,
)
from spice.temporal.problem_store import CompiledProblemStore


def _execution_policy():
    return compile_execution_policy_contract(
        coerce_execution_policy_config({"id": "strict_deadline_miss"})
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


def test_strict_deadline_miss_owns_post_window_requirement() -> None:
    assert _execution_policy().requires_post_window_row is True


def test_strict_deadline_miss_rejects_negative_decoded_offsets() -> None:
    store = _store()

    with pytest.raises(ValueError, match="non-negative"):
        _execution_policy().realize_selections(
            store,
            DecodedOffsets(torch.tensor([-1, 0], dtype=torch.int64)),
            np.arange(store.n_samples, dtype=np.int64),
            np.array([0], dtype=np.int64),
        )


def test_strict_deadline_miss_rejects_offsets_outside_action_width() -> None:
    store = _store()

    with pytest.raises(ValueError, match="max_candidate_slots"):
        _execution_policy().realize_selections(
            store,
            DecodedOffsets(torch.tensor([3, 0], dtype=torch.int64)),
            np.arange(store.n_samples, dtype=np.int64),
            np.array([0], dtype=np.int64),
        )


def test_fixed_ex_ante_overflow_realizes_first_post_window_row() -> None:
    store = _store()

    realized = _execution_policy().realize_selections(
        store,
        DecodedOffsets(torch.tensor([2, 2], dtype=torch.int64)),
        np.arange(store.n_samples, dtype=np.int64),
        np.arange(store.n_samples, dtype=np.int64),
    )

    np.testing.assert_array_equal(realized.realized_rows, np.array([4, 7], dtype=np.int64))
    np.testing.assert_array_equal(realized.resolved_offsets, np.array([2, 2], dtype=np.int64))
    np.testing.assert_array_equal(realized.overflow_mask, np.array([True, True]))


def test_strict_deadline_miss_action_mask_preserves_full_overflow_semantics() -> None:
    store = _store()

    mask = _execution_policy().prepare_action_space(
        store,
        np.arange(store.n_samples, dtype=np.int64),
    ).action_mask

    assert mask.shape == (2, 3)
    assert mask.dtype == np.bool_
    assert mask.all()


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

    targets = _execution_policy().prepare_supervised_targets(
        store,
        np.array([0], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        targets.action_mask,
        _execution_policy().prepare_action_space(
            store,
            np.array([0], dtype=np.int64),
        ).action_mask,
    )
    np.testing.assert_array_equal(targets.optimum_offsets, np.array([1], dtype=np.int64))
    np.testing.assert_allclose(targets.optimum_log_fees, np.log(np.array([50], dtype=np.float32)))


def test_evaluation_optimum_uses_best_reachable_fixed_slot() -> None:
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

    realized = _execution_policy().realize_selections(
        store,
        DecodedOffsets(torch.tensor([1], dtype=torch.int64)),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
    )

    np.testing.assert_array_equal(realized.optimum_rows, np.array([1], dtype=np.int64))
