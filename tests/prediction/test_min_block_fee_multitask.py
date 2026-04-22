from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from spice.config import PredictionConfig
from spice.modeling.models import ModelOutputs
from spice.prediction import ActionSpaceDecodeContext, compile_prediction_contract
from spice.prediction.families.min_block_fee_multitask.batch import MinBlockFeeTrainingState
from spice.prediction.families.min_block_fee_multitask.outputs import (
    MIN_LOG_FEE_HEAD_ID,
    OFFSET_LOGITS_HEAD_ID,
)
from spice.temporal import (
    coerce_realization_policy_config,
    compile_realization_policy_contract,
)
from spice.temporal.problem_store import CompiledProblemStore
from spice.temporal.semantics import ActionSpaceMode


def _build_store() -> CompiledProblemStore:
    fees = np.log(
        np.array(
            [
                11.0,
                10.0,
                12.0,
                12.0,
                5.0,
                13.0,
                9.0,
                8.0,
                4.0,
                14.0,
                6.0,
                7.0,
            ],
            dtype=np.float32,
        )
    )
    return CompiledProblemStore(
        feature_matrix=np.zeros((12, 1), dtype=np.float32),
        log_base_fees=fees.astype(np.float32, copy=False),
        timestamps=np.arange(12, dtype=np.int64),
        anchor_rows=np.array([0, 2, 5, 9], dtype=np.int64),
        context_start_rows=np.zeros(4, dtype=np.int64),
        candidate_start_rows=np.array([1, 3, 6, 10], dtype=np.int64),
        candidate_end_rows=np.array([2, 5, 9, 12], dtype=np.int64),
        action_space_mode=ActionSpaceMode.REALIZED_PER_SAMPLE,
        max_candidate_slots=3,
    )


def _contract():
    prediction = PredictionConfig.model_validate(
        {
            "id": "icdcs_2026",
            "family_id": "min_block_fee_multitask",
        }
    )
    return compile_prediction_contract(
        prediction_id=prediction.id,
        family_id=prediction.family_id,
    )


def _realization_policy():
    return compile_realization_policy_contract(
        coerce_realization_policy_config({"id": "strict_deadline_miss"})
    )


def test_min_block_fee_multitask_targets_weights_loss_and_decode() -> None:
    store = _build_store()
    contract = _contract()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)

    prepared_targets = contract.prepare_targets(
        store,
        sample_indices,
        realization_policy=_realization_policy(),
    )
    batch = prepared_targets.build_batch(torch.arange(store.n_samples, dtype=torch.int64))

    np.testing.assert_array_equal(
        batch.min_block_offsets.cpu().numpy(),
        np.array([0, 1, 2, 0], dtype=np.int64),
    )
    assert batch.min_block_log_fees[1].item() == pytest_approx_log(5.0)
    assert batch.min_block_log_fees[2].item() == pytest_approx_log(4.0)

    training_state = contract.fit_training_state(
        store,
        sample_indices,
        realization_policy=_realization_policy(),
    )
    assert isinstance(training_state, MinBlockFeeTrainingState)
    class_weights = training_state.class_weights.cpu().numpy()
    assert class_weights[0] < class_weights[1]
    assert class_weights[1] == pytest_approx(class_weights[2])
    assert training_state.fee_std.item() > 0.0

    outputs = ModelOutputs(
        heads={
            OFFSET_LOGITS_HEAD_ID: torch.tensor(
                [
                    [3.0, 100.0, 100.0],
                    [0.5, 3.0, 100.0],
                    [0.1, 0.2, 2.5],
                    [2.2, 1.0, 100.0],
                ],
                dtype=torch.float32,
            ),
            MIN_LOG_FEE_HEAD_ID: (
                (batch.min_block_log_fees - training_state.fee_mean) / training_state.fee_std
            )
            .unsqueeze(-1)
            .clone(),
        }
    )

    loss, state = contract.compute_batch_loss_and_state(
        outputs,
        batch,
        training_state=training_state,
    )
    accumulator = contract.create_epoch_accumulator()
    accumulator.update(state)
    metrics = accumulator.finalize()
    assert loss.item() < 0.5
    assert metrics.require("offset_accuracy") == pytest_approx(1.0)
    assert metrics.require("regression_loss") == pytest_approx(0.0)

    predictions = contract.allocate_decoded_offsets(store.n_samples)
    contract.decode_selected_offsets_into(
        predictions,
        outputs,
        ActionSpaceDecodeContext(
            sample_positions=torch.arange(store.n_samples, dtype=torch.int64),
            action_mask=batch.candidate_mask,
        ),
    )
    assert torch.equal(
        predictions.tensor,
        torch.tensor([0, 1, 2, 0], dtype=torch.int64),
    )


def test_min_block_fee_multitask_uses_realization_policy_targets() -> None:
    store = CompiledProblemStore(
        feature_matrix=np.zeros((8, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0], dtype=np.float32)
        ),
        timestamps=np.arange(8, dtype=np.int64),
        anchor_rows=np.array([0, 1, 2], dtype=np.int64),
        context_start_rows=np.zeros(3, dtype=np.int64),
        candidate_start_rows=np.array([1, 2, 3], dtype=np.int64),
        candidate_end_rows=np.array([2, 5, 6], dtype=np.int64),
        action_space_mode=ActionSpaceMode.REALIZED_PER_SAMPLE,
        max_candidate_slots=4,
    )
    contract = _contract()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)

    prepared_targets = contract.prepare_targets(
        store,
        sample_indices,
        realization_policy=_realization_policy(),
    )
    batch = prepared_targets.build_batch(torch.arange(store.n_samples, dtype=torch.int64))

    np.testing.assert_array_equal(
        batch.min_block_offsets.cpu().numpy(),
        np.array([0, 2, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        batch.candidate_mask.cpu().numpy(),
        np.array(
            [
                [True, False, False, False],
                [True, True, True, False],
                [True, True, True, False],
            ],
            dtype=np.bool_,
        ),
    )
    assert batch.min_block_log_fees[1].item() == pytest_approx_log(6.0)


def test_min_block_fee_multitask_masks_short_realized_candidate_windows() -> None:
    store = CompiledProblemStore(
        feature_matrix=np.zeros((8, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0], dtype=np.float32)
        ),
        timestamps=np.arange(8, dtype=np.int64),
        anchor_rows=np.array([0, 2, 4], dtype=np.int64),
        context_start_rows=np.zeros(3, dtype=np.int64),
        candidate_start_rows=np.array([1, 3, 5], dtype=np.int64),
        candidate_end_rows=np.array([3, 5, 6], dtype=np.int64),
        action_space_mode=ActionSpaceMode.REALIZED_PER_SAMPLE,
        max_candidate_slots=3,
    )
    contract = _contract()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)

    prepared_targets = contract.prepare_targets(
        store,
        sample_indices,
        realization_policy=_realization_policy(),
    )
    batch = prepared_targets.build_batch(torch.arange(store.n_samples, dtype=torch.int64))

    np.testing.assert_array_equal(
        batch.candidate_mask.cpu().numpy(),
        np.array(
            [
                [True, True, False],
                [True, True, False],
                [True, False, False],
            ],
            dtype=np.bool_,
        ),
    )


def test_min_block_fee_multitask_uses_full_action_mask_for_fixed_ex_ante_windows() -> None:
    store = CompiledProblemStore(
        feature_matrix=np.zeros((9, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0], dtype=np.float32)
        ),
        timestamps=np.arange(9, dtype=np.int64),
        anchor_rows=np.array([0, 2, 4], dtype=np.int64),
        context_start_rows=np.zeros(3, dtype=np.int64),
        candidate_start_rows=np.array([1, 3, 5], dtype=np.int64),
        candidate_end_rows=np.array([3, 5, 6], dtype=np.int64),
        action_space_mode=ActionSpaceMode.FIXED_EX_ANTE,
        max_candidate_slots=3,
    )
    contract = _contract()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)

    prepared_targets = contract.prepare_targets(
        store,
        sample_indices,
        realization_policy=_realization_policy(),
    )
    batch = prepared_targets.build_batch(torch.arange(store.n_samples, dtype=torch.int64))

    assert batch.candidate_mask.cpu().numpy().all()


def pytest_approx(value: float):
    return pytest.approx(value)


def pytest_approx_log(value: float):
    return pytest.approx(math.log(value))
