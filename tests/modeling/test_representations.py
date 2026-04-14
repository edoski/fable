from __future__ import annotations

import numpy as np
import torch

from spice.config import coerce_prediction_config
from spice.modeling._runtime import build_prediction_loader
from spice.modeling.families.registry import compile_default_representation_contract
from spice.modeling.inference import predict_with_model
from spice.modeling.models import ModelOutputs, TemporalModel, take_last_valid
from spice.modeling.representations import (
    RepresentationRuntimeContext,
    prepare_representation,
)
from spice.prediction import compile_prediction_contract
from spice.prediction.families.candidate_offset_selection.outputs import (
    CANDIDATE_LOGITS_HEAD_ID,
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
            "family": {
                "id": "candidate_offset_selection",
            },
        }
    )
    return compile_prediction_contract(
        prediction_id=prediction.id,
        family_config=prediction.family,
    )


class _ToyTemporalModel(TemporalModel):
    def __init__(self, n_candidate_slots: int) -> None:
        super().__init__()
        self.n_candidate_slots = n_candidate_slots

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor) -> ModelOutputs:
        last = take_last_valid(inputs, input_mask)
        base = last[:, 0]
        logits = torch.stack(
            (
                base,
                -base,
                torch.zeros_like(base),
            ),
            dim=1,
        )
        return ModelOutputs(
            heads={
                CANDIDATE_LOGITS_HEAD_ID: logits[:, : self.n_candidate_slots],
            }
        )


def test_sequence_input_storage_modes_yield_identical_batches() -> None:
    store = _test_store()
    sample_indices = np.array([3, 0, 2, 1], dtype=np.int64)
    streaming = prepare_representation(
        "sequence_inputs",
        store,
        sample_indices,
        runtime_context=RepresentationRuntimeContext(
            device_type="cpu",
            batch_size=2,
            available_memory_bytes=1,
        ),
    )
    materialized = prepare_representation(
        "sequence_inputs",
        store,
        sample_indices,
        runtime_context=RepresentationRuntimeContext(
            device_type="cpu",
            batch_size=2,
            available_memory_bytes=10**12,
        ),
    )

    streaming_batches = list(streaming.iter_batches(epoch=0, seed=2026, shuffle=False))
    materialized_batches = list(materialized.iter_batches(epoch=0, seed=2026, shuffle=False))

    assert streaming.representation_id == "sequence_inputs"
    assert materialized.representation_id == "sequence_inputs"
    assert len(streaming_batches) == len(materialized_batches) == 2
    for left, right in zip(streaming_batches, materialized_batches, strict=True):
        assert torch.equal(left.sample_positions, right.sample_positions)
        assert torch.equal(left.inputs, right.inputs)
        assert torch.equal(left.input_mask, right.input_mask)


def test_prediction_loader_binds_current_family_targets() -> None:
    store = _test_store()
    sample_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    representation_contract = compile_default_representation_contract("lstm")
    loader = build_prediction_loader(
        store,
        sample_indices,
        representation_contract=representation_contract,
        prediction_contract=_prediction_contract(),
        runtime_context=RepresentationRuntimeContext(
            device_type="cpu",
            batch_size=2,
            available_memory_bytes=10**12,
        ),
        seed=2026,
    )

    first_batch = next(iter(loader))

    assert first_batch.inputs.sample_positions.tolist() == [0, 1]
    assert tuple(first_batch.targets.candidate_log_fees.shape) == (2, 3)
    assert tuple(first_batch.targets.candidate_mask.shape) == (2, 3)
    assert first_batch.targets.candidate_mask.tolist() == [
        [True, True, False],
        [True, True, True],
    ]


def test_predict_with_model_decodes_candidate_offsets() -> None:
    store = _test_store()
    sample_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    representation_contract = compile_default_representation_contract("lstm")
    predictions = predict_with_model(
        _ToyTemporalModel(n_candidate_slots=store.max_candidate_slots),
        prediction_contract=_prediction_contract(),
        representation_contract=representation_contract,
        store=store,
        sample_indices=sample_indices,
        batch_size=2,
        device="cpu",
    )

    assert predictions == [0, 1, 0, 0]
