from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from spice.modeling.batch_plan import (
    build_model_input_batch_plan,
    build_prediction_batch_plan,
)
from spice.modeling.representations import RepresentationRuntimeContext
from spice.temporal import PreparedActionSpace


@dataclass(frozen=True, slots=True)
class _Store:
    max_candidate_slots: int = 3


class _ExecutionPolicy:
    def prepare_action_space(
        self, store: _Store, sample_indices: np.ndarray
    ) -> PreparedActionSpace:
        return PreparedActionSpace(
            action_mask=np.ones(
                (sample_indices.shape[0], store.max_candidate_slots), dtype=np.bool_
            )
        )


@dataclass(slots=True)
class _InputBatch:
    sample_positions: torch.Tensor

    def to_device(self, device: torch.device) -> _InputBatch:
        return self

    def model_kwargs(self) -> dict[str, torch.Tensor]:
        return {}

    def pin_memory(self) -> _InputBatch:
        return self


class _Prepared:
    sample_count = 4
    batch_signatures = np.array([2, 1, 2, 1], dtype=np.int64)
    estimated_storage_bytes = 1024

    def __init__(self, *, fail_device_storage: bool = False) -> None:
        self.fail_device_storage = fail_device_storage
        self.device_storage_calls = 0

    def build_batch(self, sample_positions: torch.Tensor) -> _InputBatch:
        return _InputBatch(sample_positions=sample_positions)

    def to_device_storage(self, device: torch.device):
        self.device_storage_calls += 1
        if self.fail_device_storage:
            raise torch.cuda.OutOfMemoryError("oom")
        return self


class _RepresentationContract:
    def __init__(self, prepared: _Prepared) -> None:
        self.prepared = prepared
        self.action_space: PreparedActionSpace | None = None

    def prepare(self, *_args, **_kwargs) -> _Prepared:
        self.action_space = _kwargs["action_space"]
        return self.prepared


class _Targets:
    estimated_storage_bytes = 256

    def __init__(self) -> None:
        self.positions: list[list[int]] = []

    def build_batch(self, sample_positions: torch.Tensor) -> torch.Tensor:
        self.positions.append(sample_positions.tolist())
        return sample_positions + 10

    def to_device_storage(self, device: torch.device):
        return self


class _PredictionContract:
    def __init__(self, targets: _Targets) -> None:
        self.targets = targets
        self.action_space: PreparedActionSpace | None = None

    def prepare_targets(self, *_args, **_kwargs) -> _Targets:
        self.action_space = _kwargs["action_space"]
        return self.targets


def _runtime_context(*, device_budget: int | None = None) -> RepresentationRuntimeContext:
    return RepresentationRuntimeContext(
        batch_size=2,
        available_host_memory_bytes=1024,
        available_device_memory_bytes=device_budget,
    )


def test_host_loader_worker_override_rejects_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("SPICE_DATALOADER_WORKERS", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        build_model_input_batch_plan(
            _Store(),
            np.arange(4, dtype=np.int64),
            representation_contract=_RepresentationContract(_Prepared()),
            execution_policy=_ExecutionPolicy(),
            runtime_context=_runtime_context(),
            resolved_device=torch.device("cpu"),
            seed=2026,
        )


def test_batch_plan_orders_samples_deterministically_by_signature() -> None:
    plan = build_model_input_batch_plan(
        _Store(),
        np.arange(4, dtype=np.int64),
        representation_contract=_RepresentationContract(_Prepared()),
        execution_policy=_ExecutionPolicy(),
        runtime_context=_runtime_context(),
        resolved_device=torch.device("cpu"),
        seed=2026,
    )

    assert plan.storage_mode == "host_materialized"
    assert plan.sample_count == 4
    assert plan.batch_count == 2
    assert plan.estimated_storage_bytes == 1024
    assert [batch.sample_positions.tolist() for batch in plan.source] == [[1, 3], [0, 2]]


def test_shuffled_batch_plan_changes_by_epoch_but_keeps_batch_shape() -> None:
    targets = _Targets()
    plan = build_prediction_batch_plan(
        _Store(),
        np.arange(4, dtype=np.int64),
        representation_contract=_RepresentationContract(_Prepared()),
        prediction_contract=_PredictionContract(targets),
        execution_policy=_ExecutionPolicy(),
        runtime_context=_runtime_context(),
        resolved_device=torch.device("cpu"),
        seed=2026,
        shuffle=True,
    )

    first_epoch = [batch.inputs.sample_positions.tolist() for batch in plan.source]
    second_epoch = [batch.inputs.sample_positions.tolist() for batch in plan.source]

    assert first_epoch != second_epoch
    assert [len(batch) for batch in first_epoch] == [2, 2]
    assert [len(batch) for batch in second_epoch] == [2, 2]
    assert sorted(position for batch in first_epoch for position in batch) == [0, 1, 2, 3]
    assert sorted(position for batch in second_epoch for position in batch) == [0, 1, 2, 3]


def test_prediction_batch_plan_binds_targets_to_input_sample_positions() -> None:
    targets = _Targets()
    representation_contract = _RepresentationContract(_Prepared())
    prediction_contract = _PredictionContract(targets)
    plan = build_prediction_batch_plan(
        _Store(),
        np.arange(4, dtype=np.int64),
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        execution_policy=_ExecutionPolicy(),
        runtime_context=_runtime_context(),
        resolved_device=torch.device("cpu"),
        seed=2026,
    )

    first_batch = next(iter(plan.source))

    assert first_batch.inputs.sample_positions.tolist() == [1, 3]
    assert first_batch.targets.tolist() == [11, 13]
    assert targets.positions == [[1, 3]]
    assert representation_contract.action_space is prediction_contract.action_space
    assert plan.estimated_storage_bytes == 1280


def test_device_resident_plan_exposes_storage_mode() -> None:
    plan = build_model_input_batch_plan(
        _Store(),
        np.arange(4, dtype=np.int64),
        representation_contract=_RepresentationContract(_Prepared()),
        execution_policy=_ExecutionPolicy(),
        runtime_context=_runtime_context(device_budget=2048),
        resolved_device=torch.device("cuda"),
        seed=2026,
    )

    assert plan.storage_mode == "cuda_materialized"
    assert next(iter(plan.source)).sample_positions.tolist() == [1, 3]


def test_zero_device_budget_uses_host_loader_without_device_materialization() -> None:
    prepared = _Prepared()

    plan = build_model_input_batch_plan(
        _Store(),
        np.arange(4, dtype=np.int64),
        representation_contract=_RepresentationContract(prepared),
        execution_policy=_ExecutionPolicy(),
        runtime_context=_runtime_context(device_budget=0),
        resolved_device=torch.device("cuda"),
        seed=2026,
    )

    assert plan.storage_mode == "host_materialized"
    assert prepared.device_storage_calls == 0
    assert next(iter(plan.source)).sample_positions.tolist() == [1, 3]


def test_device_resident_oom_falls_back_to_host_loader(monkeypatch) -> None:
    empty_cache_calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    plan = build_model_input_batch_plan(
        _Store(),
        np.arange(4, dtype=np.int64),
        representation_contract=_RepresentationContract(_Prepared(fail_device_storage=True)),
        execution_policy=_ExecutionPolicy(),
        runtime_context=_runtime_context(device_budget=2048),
        resolved_device=torch.device("cuda"),
        seed=2026,
    )

    assert plan.storage_mode == "host_materialized"
    assert next(iter(plan.source)).sample_positions.tolist() == [1, 3]
    assert empty_cache_calls == [True]
