from __future__ import annotations

import numpy as np
import torch

from spice.config import coerce_prediction_config
from spice.modeling._runtime import build_prediction_batch_source
from spice.modeling.batch_sources import (
    DeviceResidentBatchSource,
    _PositionBatchSampler,
    _should_use_device_resident,
    plan_batch_source,
)
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.families.registry import resolve_model_representation_id
from spice.modeling.inference import predict_with_model
from spice.modeling.models import ModelOutputs, TemporalModel, take_last_valid
from spice.modeling.representations import (
    SEQUENCE_INPUT_REPRESENTATION_ID,
    RepresentationRuntimeContext,
    compile_representation_contract,
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


def _model_config() -> LstmModelConfig:
    return LstmModelConfig(
        input_projection_dim=8,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
        head_hidden_dim=8,
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
        SEQUENCE_INPUT_REPRESENTATION_ID,
        store,
        sample_indices,
        runtime_context=RepresentationRuntimeContext(
            device_type="cpu",
            batch_size=2,
            available_host_memory_bytes=1,
        ),
    )
    materialized = prepare_representation(
        SEQUENCE_INPUT_REPRESENTATION_ID,
        store,
        sample_indices,
        runtime_context=RepresentationRuntimeContext(
            device_type="cpu",
            batch_size=2,
            available_host_memory_bytes=10**12,
        ),
    )
    materialized_from_streaming = streaming.to_device_storage(torch.device("cpu"))

    sample_positions = (
        torch.as_tensor([0, 2], dtype=torch.int64),
        torch.as_tensor([1, 3], dtype=torch.int64),
    )

    assert streaming.representation_id == SEQUENCE_INPUT_REPRESENTATION_ID
    assert materialized.representation_id == SEQUENCE_INPUT_REPRESENTATION_ID
    assert materialized_from_streaming is not None
    assert streaming.sample_count == materialized.sample_count == 4
    for positions in sample_positions:
        left = streaming.build_batch(positions)
        right = materialized.build_batch(positions)
        replay = materialized_from_streaming.build_batch(positions)
        assert torch.equal(left.sample_positions, right.sample_positions)
        assert torch.equal(replay.sample_positions, right.sample_positions)
        assert torch.equal(left.inputs, right.inputs)
        assert torch.equal(replay.inputs, right.inputs)
        assert torch.equal(left.input_mask, right.input_mask)
        assert torch.equal(replay.input_mask, right.input_mask)


def test_device_resident_planner_accepts_streaming_inputs_when_cuda_budget_fits() -> None:
    prepared = type(
        "Prepared",
        (),
        {
            "input_storage_mode_id": "streaming_host",
            "estimated_input_storage_bytes": 1024,
            "estimated_target_storage_bytes": 128,
        },
    )()

    assert _should_use_device_resident(
        prepared,
        runtime_context=RepresentationRuntimeContext(
            device_type="cuda",
            batch_size=2,
            available_host_memory_bytes=1,
            available_device_memory_bytes=10**9,
        ),
        resolved_device=torch.device("cuda"),
    )


def test_plan_batch_source_selects_device_resident_for_streaming_origin_when_cuda_fits() -> None:
    class _Prepared:
        sample_count = 4
        batch_signatures = np.array([2, 1, 2, 1], dtype=np.int64)
        input_storage_mode_id = "streaming_host"
        target_storage_mode_id = "materialized_host"
        batch_planner_id = "signature_bucketed"
        estimated_input_storage_bytes = 1024
        estimated_target_storage_bytes = 128

        def to_device_storage(self, device: torch.device):
            del device
            return self

        def build_batch(self, sample_positions: torch.Tensor):
            raise AssertionError(f"build_batch should not run during planning: {sample_positions}")

    plan = plan_batch_source(
        _Prepared(),
        runtime_context=RepresentationRuntimeContext(
            device_type="cuda",
            batch_size=2,
            available_host_memory_bytes=1,
            available_device_memory_bytes=10**9,
        ),
        resolved_device=torch.device("cuda"),
        seed=2026,
        shuffle=True,
    )

    assert plan.loader_strategy_id == "device_resident"


def test_prediction_batch_source_binds_current_family_targets() -> None:
    store = _test_store()
    sample_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    representation_contract = compile_representation_contract(
        resolve_model_representation_id(_model_config())
    )
    batch_source_plan = build_prediction_batch_source(
        store,
        sample_indices,
        representation_contract=representation_contract,
        prediction_contract=_prediction_contract(),
        runtime_context=RepresentationRuntimeContext(
            device_type="cpu",
            batch_size=2,
            available_host_memory_bytes=10**12,
        ),
        resolved_device=torch.device("cpu"),
        seed=2026,
    )
    loader = batch_source_plan.source

    first_batch = next(iter(loader))

    assert batch_source_plan.loader_strategy_id == "host_dataloader"
    assert first_batch.inputs.sample_positions.tolist() == [0, 1]
    assert tuple(first_batch.targets.candidate_log_fees.shape) == (2, 3)
    assert tuple(first_batch.targets.candidate_mask.shape) == (2, 3)
    assert first_batch.targets.candidate_mask.tolist() == [
        [True, True, False],
        [True, True, True],
    ]


def test_host_and_device_batch_sources_yield_identical_batches() -> None:
    store = _test_store()
    sample_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    representation_contract = compile_representation_contract(
        resolve_model_representation_id(_model_config())
    )
    prepared = representation_contract.prepare(
        store,
        sample_indices,
        runtime_context=RepresentationRuntimeContext(
            device_type="cuda",
            batch_size=2,
            available_host_memory_bytes=10**12,
            available_device_memory_bytes=10**12,
        ),
    )
    prepared_prediction = _prediction_contract().prepare_targets(store, sample_indices)
    from spice.prediction.contracts import bind_prediction_representation

    bound = bind_prediction_representation(prepared, targets=prepared_prediction)
    host_plan = plan_batch_source(
        bound,
        runtime_context=RepresentationRuntimeContext(
            device_type="cpu",
            batch_size=2,
            available_host_memory_bytes=10**12,
        ),
        resolved_device=torch.device("cpu"),
        seed=2026,
        shuffle=False,
    )
    device_prepared = bound.to_device_storage(torch.device("cpu"))
    assert device_prepared is not None
    device_source = DeviceResidentBatchSource(
        prepared=device_prepared,
        batch_sampler=_PositionBatchSampler(
            batch_signatures=device_prepared.batch_signatures,
            batch_size=2,
            seed=2026,
            shuffle=False,
        ),
        input_storage_mode_id=device_prepared.input_storage_mode_id,
        target_storage_mode_id=device_prepared.target_storage_mode_id,
        batch_planner_id=device_prepared.batch_planner_id,
    )

    host_batches = list(host_plan.source)
    device_batches = list(device_source)

    assert host_plan.loader_strategy_id == "host_dataloader"
    assert device_source.loader_strategy_id == "device_resident"
    assert len(host_batches) == len(device_batches) == 2
    for left, right in zip(host_batches, device_batches, strict=True):
        assert torch.equal(left.inputs.sample_positions, right.inputs.sample_positions)
        assert torch.equal(left.inputs.inputs, right.inputs.inputs)
        assert torch.equal(left.inputs.input_mask, right.inputs.input_mask)
        assert torch.equal(left.targets.candidate_log_fees, right.targets.candidate_log_fees)
        assert torch.equal(left.targets.candidate_mask, right.targets.candidate_mask)


def test_predict_with_model_decodes_candidate_offsets() -> None:
    store = _test_store()
    sample_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    representation_contract = compile_representation_contract(
        resolve_model_representation_id(_model_config())
    )
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
