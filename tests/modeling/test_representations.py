from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.config import coerce_prediction_config
from spice.modeling.batch_sources import (
    build_prediction_batch_source,
)
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.representations import RepresentationRuntimeContext, sequence_input_contract
from spice.prediction import compile_prediction_contract
from spice.temporal import (
    coerce_realization_policy_config,
    compile_realization_policy_contract,
)
from spice.temporal.problem_store import CompiledProblemStore


def _test_store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.array(
            [
                [-1.0, 0.0, 0.1],
                [-2.0, 0.1, 0.2],
                [0.5, 0.2, 0.3],
                [1.5, 0.3, 0.4],
                [-0.2, 0.4, 0.5],
                [2.0, 0.5, 0.6],
                [-1.1, 0.6, 0.7],
                [0.3, 0.7, 0.8],
                [1.2, 0.8, 0.9],
                [-0.7, 0.9, 1.0],
            ],
            dtype=np.float32,
        ),
        log_base_fees=np.array(
            [0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.4, 0.12, 0.22, 0.18],
            dtype=np.float32,
        ),
        timestamps=np.array([0, 5, 11, 19, 28, 40, 55, 71, 88, 106], dtype=np.int64),
        anchor_rows=np.array([2, 4, 5, 7], dtype=np.int64),
        context_start_rows=np.array([0, 1, 0, 4], dtype=np.int64),
        candidate_end_rows=np.array([5, 8, 7, 10], dtype=np.int64),
        max_candidate_slots=3,
    )


def _prediction_contract():
    prediction = coerce_prediction_config(
        {
            "id": "candidate_offset_selection",
            "family_id": "candidate_offset_selection",
        }
    )
    return compile_prediction_contract(
        prediction_id=prediction.id,
        family_id=prediction.family_id,
    )


def _model_config() -> LstmModelConfig:
    return LstmModelConfig(
        input_projection_dim=8,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
        head_hidden_dim=8,
    )


def _realization_policy():
    return compile_realization_policy_contract(
        coerce_realization_policy_config({"id": "strict_deadline_miss"})
    )


def test_sequence_input_storage_modes_yield_identical_batches() -> None:
    store = _test_store()
    sample_indices = np.array([3, 0, 2, 1], dtype=np.int64)
    contract = sequence_input_contract()
    streaming = contract.prepare(
        store,
        sample_indices,
        runtime_context=RepresentationRuntimeContext(
            batch_size=2,
            available_host_memory_bytes=1,
        ),
    )
    materialized = contract.prepare(
        store,
        sample_indices,
        runtime_context=RepresentationRuntimeContext(
            batch_size=2,
            available_host_memory_bytes=10**12,
        ),
    )
    sample_positions = (
        torch.as_tensor([0, 2], dtype=torch.int64),
        torch.as_tensor([1, 3], dtype=torch.int64),
    )

    assert streaming.sample_count == materialized.sample_count == 4
    for positions in sample_positions:
        left = streaming.build_batch(positions)
        right = materialized.build_batch(positions)
        assert torch.equal(left.sample_positions, right.sample_positions)
        assert torch.equal(left.inputs, right.inputs)
        assert torch.equal(left.input_mask, right.input_mask)
        assert torch.equal(left.action_mask, right.action_mask)


def test_sequence_input_device_storage_requires_cuda() -> None:
    store = _test_store()
    sample_indices = np.array([3, 0, 2, 1], dtype=np.int64)
    contract = sequence_input_contract()
    prepared = contract.prepare(
        store,
        sample_indices,
        runtime_context=RepresentationRuntimeContext(
            batch_size=2,
            available_host_memory_bytes=1,
        ),
    )

    with pytest.raises(ValueError, match="requires CUDA"):
        prepared.to_device_storage(torch.device("cpu"))
def test_prediction_batch_source_binds_current_family_targets() -> None:
    store = _test_store()
    sample_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    representation_contract = sequence_input_contract()
    batch_source = build_prediction_batch_source(
        store,
        sample_indices,
        representation_contract=representation_contract,
        prediction_contract=_prediction_contract(),
        realization_policy=_realization_policy(),
        runtime_context=RepresentationRuntimeContext(
            batch_size=2,
            available_host_memory_bytes=10**12,
        ),
        resolved_device=torch.device("cuda"),
        seed=2026,
    )
    first_batch = next(iter(batch_source))

    assert first_batch.inputs.sample_positions.tolist() == [0, 1]
    assert tuple(first_batch.targets.candidate_log_fees.shape) == (2, 3)
    assert tuple(first_batch.targets.candidate_mask.shape) == (2, 3)
    assert first_batch.targets.candidate_mask.tolist() == [
        [True, True, False],
        [True, True, True],
    ]
