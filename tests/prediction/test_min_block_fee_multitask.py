from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import torch

from spice.config import PredictionConfig
from spice.modeling.models import ModelOutputs
from spice.prediction import compile_prediction_contract
from spice.prediction.decoded_offsets import DecodedOffsets
from spice.prediction.decoding import ActionSpaceDecodeContext
from spice.prediction.families.min_block_fee_multitask.batch import (
    MinBlockFeeTargetBatch,
    MinBlockFeeTrainingState,
)
from spice.prediction.families.min_block_fee_multitask.metrics import (
    compute_batch_loss_and_state,
)
from spice.prediction.families.min_block_fee_multitask.outputs import (
    MIN_LOG_FEE_HEAD_ID,
    OFFSET_LOGITS_HEAD_ID,
)
from spice.temporal import (
    PreparedSupervisedExecutionTargets,
    coerce_execution_policy_config,
    compile_execution_policy_contract,
)
from spice.temporal.problem_store import CompiledProblemStore


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
                8.0,
            ],
            dtype=np.float32,
        )
    )
    return CompiledProblemStore(
        feature_matrix=np.zeros((13, 1), dtype=np.float32),
        log_base_fees=fees.astype(np.float32, copy=False),
        timestamps=np.arange(13, dtype=np.int64),
        anchor_rows=np.array([0, 2, 5, 9], dtype=np.int64),
        context_start_rows=np.zeros(4, dtype=np.int64),
        candidate_start_rows=np.array([1, 3, 6, 10], dtype=np.int64),
        candidate_end_rows=np.array([2, 5, 9, 12], dtype=np.int64),
        max_candidate_slots=3,
    )


def _contract():
    prediction = PredictionConfig.model_validate(
        {
            "id": "current_row_fee_dynamics",
            "family_id": "min_block_fee_multitask",
        }
    )
    return compile_prediction_contract(
        prediction_id=prediction.id,
        family_id=prediction.family_id,
    )


def _execution_policy():
    return compile_execution_policy_contract(
        coerce_execution_policy_config({"id": "strict_deadline_miss"})
    )


def _prepare_action_space(store: CompiledProblemStore, sample_indices: np.ndarray):
    return _execution_policy().prepare_action_space(store, sample_indices)


def test_min_block_fee_multitask_targets_weights_loss_and_decode() -> None:
    store = _build_store()
    contract = _contract()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)

    prepared_targets = contract.prepare_targets(
        store,
        sample_indices,
        execution_policy=_execution_policy(),
        action_space=_prepare_action_space(store, sample_indices),
    )
    batch = cast(
        MinBlockFeeTargetBatch,
        prepared_targets.build_batch(torch.arange(store.n_samples, dtype=torch.int64)),
    )

    np.testing.assert_array_equal(
        batch.min_block_offsets.cpu().numpy(),
        np.array([0, 1, 2, 0], dtype=np.int64),
    )
    assert batch.min_block_log_fees[1].item() == pytest_approx_log(5.0)
    assert batch.min_block_log_fees[2].item() == pytest_approx_log(4.0)

    training_state = contract.fit_training_state(
        store,
        sample_indices,
        execution_policy=_execution_policy(),
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
                    [3.0, 0.0, 0.0],
                    [0.5, 3.0, 0.0],
                    [0.1, 0.2, 2.5],
                    [2.2, 1.0, 0.0],
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
    assert metrics.require("macro_f1") == pytest_approx(1.0)
    assert metrics.require("regression_loss") == pytest_approx(0.0)
    assert metrics.require("log_fee_mae") == pytest_approx(0.0)
    assert metrics.require("log_fee_mse") == pytest_approx(0.0)

    predictions = contract.allocate_decoded_result(store.n_samples)
    assert isinstance(predictions, DecodedOffsets)
    contract.decode_batch_result_into(
        predictions,
        outputs,
        ActionSpaceDecodeContext(
            sample_positions=torch.arange(store.n_samples, dtype=torch.int64),
            action_mask=batch.action_mask,
        ),
    )
    assert torch.equal(
        predictions.tensor,
        torch.tensor([0, 1, 2, 0], dtype=torch.int64),
    )


def test_min_block_fee_multitask_uses_execution_policy_targets() -> None:
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
        max_candidate_slots=4,
    )
    contract = _contract()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)

    prepared_targets = contract.prepare_targets(
        store,
        sample_indices,
        execution_policy=_execution_policy(),
        action_space=_prepare_action_space(store, sample_indices),
    )
    batch = cast(
        MinBlockFeeTargetBatch,
        prepared_targets.build_batch(torch.arange(store.n_samples, dtype=torch.int64)),
    )

    np.testing.assert_array_equal(
        batch.min_block_offsets.cpu().numpy(),
        np.array([0, 2, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        batch.action_mask.cpu().numpy(),
        np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
            ],
            dtype=np.bool_,
        ),
    )
    assert batch.min_block_log_fees[1].item() == pytest_approx_log(6.0)


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
        max_candidate_slots=3,
    )
    contract = _contract()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)

    prepared_targets = contract.prepare_targets(
        store,
        sample_indices,
        execution_policy=_execution_policy(),
        action_space=_prepare_action_space(store, sample_indices),
    )
    batch = cast(
        MinBlockFeeTargetBatch,
        prepared_targets.build_batch(torch.arange(store.n_samples, dtype=torch.int64)),
    )

    assert batch.action_mask.cpu().numpy().all()


def test_min_block_fee_multitask_rejects_divergent_prepared_action_space() -> None:
    store = _build_store()
    contract = _contract()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    execution_policy = _execution_policy()
    action_space = execution_policy.prepare_action_space(store, sample_indices)

    def divergent_targets(store, sample_indices) -> PreparedSupervisedExecutionTargets:
        targets = execution_policy.prepare_supervised_targets(store, sample_indices)
        mask = targets.action_mask.copy()
        mask[0, 0] = False
        return PreparedSupervisedExecutionTargets(
            action_mask=mask,
            candidate_log_fees=targets.candidate_log_fees,
            optimum_offsets=targets.optimum_offsets,
            optimum_log_fees=targets.optimum_log_fees,
            baseline_candidate_indices=targets.baseline_candidate_indices,
        )

    divergent_policy = replace(
        execution_policy,
        prepare_supervised_targets_fn=divergent_targets,
    )

    with pytest.raises(ValueError, match="does not match prepared Action Space"):
        contract.prepare_targets(
            store,
            sample_indices,
            execution_policy=divergent_policy,
            action_space=action_space,
        )


def test_min_block_fee_multitask_macro_f1_and_log_fee_errors() -> None:
    training_state = MinBlockFeeTrainingState(
        class_weights=torch.ones(3, dtype=torch.float32),
        fee_mean=torch.tensor(1.0, dtype=torch.float32),
        fee_std=torch.tensor(2.0, dtype=torch.float32),
    )
    target_log_fees = torch.tensor([1.0, 2.0, 4.0, 5.0], dtype=torch.float32)
    predicted_log_fees = torch.tensor([1.0, 1.0, 5.0, 3.0], dtype=torch.float32)
    batch = MinBlockFeeTargetBatch(
        action_mask=torch.ones((4, 3), dtype=torch.bool),
        min_block_offsets=torch.tensor([0, 1, 1, 2], dtype=torch.int64),
        min_block_log_fees=target_log_fees,
    )
    offset_logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=torch.float32,
    )
    fee_predictions = (
        predicted_log_fees - training_state.fee_mean
    ) / training_state.fee_std

    _, state = compute_batch_loss_and_state(
        offset_logits,
        fee_predictions,
        batch,
        training_state=training_state,
    )
    accumulator = create_metric_accumulator()
    accumulator.update(state)
    metrics = accumulator.finalize()

    assert metrics.require("macro_f1") == pytest_approx(7.0 / 18.0)
    assert metrics.require("log_fee_mae") == pytest_approx(1.0)
    assert metrics.require("log_fee_mse") == pytest_approx(1.5)


def test_min_block_fee_training_state_resolve_preserves_semantic_tensors() -> None:
    training_state = MinBlockFeeTrainingState(
        class_weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        fee_mean=torch.tensor(1.5, dtype=torch.float32),
        fee_std=torch.tensor(2.5, dtype=torch.float32),
    )
    class_weights = training_state.class_weights.clone()
    fee_mean = training_state.fee_mean.clone()
    fee_std = training_state.fee_std.clone()

    first = training_state.resolve(device=torch.device("cpu"), dtype=torch.float64)
    second = training_state.resolve(device=torch.device("cpu"), dtype=torch.float64)

    assert first is second
    assert torch.equal(training_state.class_weights, class_weights)
    assert torch.equal(training_state.fee_mean, fee_mean)
    assert torch.equal(training_state.fee_std, fee_std)
    assert training_state.class_weights.dtype == torch.float32


def test_reused_training_state_matches_independently_refit_state_loss_and_metrics() -> None:
    store = _build_store()
    contract = _contract()
    execution_policy = _execution_policy()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    prepared_targets = contract.prepare_targets(
        store,
        sample_indices,
        execution_policy=execution_policy,
        action_space=execution_policy.prepare_action_space(store, sample_indices),
    )
    batch = cast(
        MinBlockFeeTargetBatch,
        prepared_targets.build_batch(torch.arange(store.n_samples, dtype=torch.int64)),
    )
    outputs = ModelOutputs(
        heads={
            OFFSET_LOGITS_HEAD_ID: torch.tensor(
                [
                    [3.0, 0.0, 0.0],
                    [0.5, 3.0, 0.0],
                    [0.1, 0.2, 2.5],
                    [2.2, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            MIN_LOG_FEE_HEAD_ID: torch.zeros((store.n_samples, 1), dtype=torch.float32),
        }
    )
    reused_state = contract.fit_training_state(
        store,
        sample_indices,
        execution_policy=execution_policy,
    )
    refit_state = contract.fit_training_state(
        store,
        sample_indices,
        execution_policy=execution_policy,
    )

    reused_loss, reused_batch_state = contract.compute_batch_loss_and_state(
        outputs,
        batch,
        training_state=reused_state,
    )
    refit_loss, refit_batch_state = contract.compute_batch_loss_and_state(
        outputs,
        batch,
        training_state=refit_state,
    )
    reused_accumulator = contract.create_epoch_accumulator()
    refit_accumulator = contract.create_epoch_accumulator()
    reused_accumulator.update(reused_batch_state)
    refit_accumulator.update(refit_batch_state)

    assert reused_loss.item() == pytest_approx(refit_loss.item())
    assert reused_accumulator.finalize().values == refit_accumulator.finalize().values


def create_metric_accumulator():
    contract = _contract()
    return contract.create_epoch_accumulator()


def pytest_approx(value: float):
    return pytest.approx(value)


def pytest_approx_log(value: float):
    return pytest.approx(math.log(value))
