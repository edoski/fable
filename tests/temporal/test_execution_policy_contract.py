from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from spice.prediction.decoded_offsets import DecodedOffsets
from spice.temporal import (
    PreparedActionSpace,
    coerce_execution_policy_config,
    compile_execution_policy_contract,
)
from spice.temporal.execution_policy.strict_deadline_miss import StrictDeadlineMissConfig
from spice.temporal.problem_store import CompiledProblemStore


def _execution_policy():
    return compile_execution_policy_contract(
        coerce_execution_policy_config({"id": "strict_deadline_miss"})
    )


def test_execution_policy_coercer_preserves_concrete_config_identity() -> None:
    config = StrictDeadlineMissConfig()

    assert coerce_execution_policy_config(config) is config


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


def test_prepare_action_space_rejects_reordered_sample_indices() -> None:
    def prepare_action_space(store: CompiledProblemStore, sample_indices: np.ndarray):
        reordered = sample_indices[::-1].copy()
        return PreparedActionSpace(
            sample_indices=reordered,
            max_candidate_slots=store.max_candidate_slots,
            action_mask=np.ones(
                (sample_indices.shape[0], store.max_candidate_slots),
                dtype=np.bool_,
            ),
        )

    policy = replace(_execution_policy(), prepare_action_space_fn=prepare_action_space)

    with pytest.raises(ValueError, match="sample_indices"):
        policy.prepare_action_space(_store(), np.arange(2, dtype=np.int64))


def test_prepare_action_space_rejects_action_width_mismatch() -> None:
    def prepare_action_space(store: CompiledProblemStore, sample_indices: np.ndarray):
        action_width = store.max_candidate_slots + 1
        return PreparedActionSpace(
            sample_indices=sample_indices,
            max_candidate_slots=action_width,
            action_mask=np.ones((sample_indices.shape[0], action_width), dtype=np.bool_),
        )

    policy = replace(_execution_policy(), prepare_action_space_fn=prepare_action_space)

    with pytest.raises(ValueError, match="action width"):
        policy.prepare_action_space(_store(), np.arange(2, dtype=np.int64))


def test_prepare_action_space_rejects_action_mask_shape_mismatch() -> None:
    def prepare_action_space(store: CompiledProblemStore, sample_indices: np.ndarray):
        action_space = PreparedActionSpace(
            sample_indices=sample_indices,
            max_candidate_slots=store.max_candidate_slots,
            action_mask=np.ones(
                (sample_indices.shape[0], store.max_candidate_slots),
                dtype=np.bool_,
            ),
        )
        object.__setattr__(
            action_space,
            "action_mask",
            np.ones((sample_indices.shape[0], store.max_candidate_slots + 1), dtype=np.bool_),
        )
        return action_space

    policy = replace(_execution_policy(), prepare_action_space_fn=prepare_action_space)

    with pytest.raises(ValueError, match="action_mask shape"):
        policy.prepare_action_space(_store(), np.arange(2, dtype=np.int64))


def test_prepared_action_space_rejects_all_false_action_rows() -> None:
    with pytest.raises(ValueError, match="at least one action"):
        PreparedActionSpace(
            sample_indices=np.arange(2, dtype=np.int64),
            max_candidate_slots=3,
            action_mask=np.array(
                [
                    [True, False, False],
                    [False, False, False],
                ],
                dtype=np.bool_,
            ),
        )


def test_realize_selections_validates_selected_positions_before_policy_call() -> None:
    store = _store()
    policy = _execution_policy()
    action_space = policy.prepare_action_space(store, np.arange(2, dtype=np.int64))

    with pytest.raises(ValueError, match="outside sample_indices"):
        policy.realize_selections(
            store,
            DecodedOffsets(torch.tensor([0, 1], dtype=torch.int64)),
            action_space,
            np.array([2], dtype=np.int64),
        )


def test_realize_selections_rejects_unavailable_actions() -> None:
    store = _store()
    policy = _execution_policy()
    action_space = PreparedActionSpace(
        sample_indices=np.array([0], dtype=np.int64),
        max_candidate_slots=store.max_candidate_slots,
        action_mask=np.array([[True, False, True]], dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="unavailable actions"):
        policy.realize_selections(
            store,
            DecodedOffsets(torch.tensor([1], dtype=torch.int64)),
            action_space,
            np.array([0], dtype=np.int64),
        )
