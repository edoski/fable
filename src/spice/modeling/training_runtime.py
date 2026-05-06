"""Training runtime planning and CUDA memory probing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from ..config.models import TrainingConfig
from ..prediction import CompiledPredictionContract
from ..prediction.contracts import PredictionBatch
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore, IntVector
from ._epoch_execution import execute_training_batch
from ._runtime_probe import build_measured_modeling_runtime_plan, measure_device_resident_budget
from .batch_plan import BatchPlan, build_prediction_batch_plan
from .families.base import ModelConfig
from .models import TemporalModel
from .representations import CompiledRepresentationContract
from .runtime_planning import (
    ModelingRuntimePlan,
    build_training_modeling_runtime_plan,
    modeling_backend_scope,
    prepare_model_for_runtime,
)


@dataclass(frozen=True, slots=True)
class TrainingRuntimePlan:
    runtime_plan: ModelingRuntimePlan
    train_batch_plan: BatchPlan[PredictionBatch]
    validation_batch_plan: BatchPlan[PredictionBatch]
    prediction_training_state: object | None


@dataclass(frozen=True, slots=True)
class PreparedTrainingRuntime:
    fit_model: TemporalModel
    optimizer: torch.optim.Optimizer
    batch_plan: TrainingRuntimePlan


def _unwrap_compiled_model(model: TemporalModel) -> TemporalModel:
    return cast(TemporalModel, getattr(model, "_orig_mod", model))


def _clone_cpu_state(model: TemporalModel) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def plan_training_runtime(
    model: TemporalModel,
    *,
    prediction_contract: CompiledPredictionContract,
    execution_policy: CompiledExecutionPolicyContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    train_sample_indices: IntVector,
    validation_sample_indices: IntVector,
    runtime_plan: ModelingRuntimePlan,
    training_config: TrainingConfig,
) -> TrainingRuntimePlan:
    train_temporal_facts = execution_policy.prepare_temporal_facts(
        store,
        train_sample_indices,
    )
    validation_temporal_facts = execution_policy.prepare_temporal_facts(
        store,
        validation_sample_indices,
    )
    prediction_training_state = prediction_contract.fit_training_state(
        store,
        temporal_facts=train_temporal_facts,
    )
    planned_runtime_plan = build_measured_modeling_runtime_plan(
        runtime_plan,
        build_warmup_plan=lambda warmup_runtime_plan: build_prediction_batch_plan(
            store,
            temporal_facts=train_temporal_facts,
            representation_contract=representation_contract,
            prediction_contract=prediction_contract,
            execution_policy=execution_policy,
            runtime_context=warmup_runtime_plan.representation_runtime_context,
            resolved_device=warmup_runtime_plan.resolved_device,
            seed=warmup_runtime_plan.seed,
            shuffle=False,
        ),
        measure_warmup_budget=lambda warmup_plan, warmup_runtime_plan: (
            _measure_training_batch_budget(
                model,
                warmup_plan=warmup_plan,
                prediction_contract=prediction_contract,
                prediction_training_state=prediction_training_state,
                resolved_device=warmup_runtime_plan.resolved_device,
                training_config=training_config,
                precision=warmup_runtime_plan.precision,
            )
        ),
    )
    planned_runtime_context = planned_runtime_plan.representation_runtime_context
    train_batch_plan = build_prediction_batch_plan(
        store,
        temporal_facts=train_temporal_facts,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        execution_policy=execution_policy,
        runtime_context=planned_runtime_context,
        resolved_device=planned_runtime_plan.resolved_device,
        seed=planned_runtime_plan.seed,
        shuffle=True,
    )
    validation_batch_plan = build_prediction_batch_plan(
        store,
        temporal_facts=validation_temporal_facts,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        execution_policy=execution_policy,
        runtime_context=planned_runtime_context,
        resolved_device=planned_runtime_plan.resolved_device,
        seed=planned_runtime_plan.seed,
        shuffle=False,
    )
    return TrainingRuntimePlan(
        runtime_plan=planned_runtime_plan,
        train_batch_plan=train_batch_plan,
        validation_batch_plan=validation_batch_plan,
        prediction_training_state=prediction_training_state,
    )


def prepare_training_runtime(
    model: TemporalModel,
    *,
    model_config: ModelConfig[str],
    prediction_contract: CompiledPredictionContract,
    execution_policy: CompiledExecutionPolicyContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    train_sample_indices: IntVector,
    validation_sample_indices: IntVector,
    training_config: TrainingConfig,
) -> PreparedTrainingRuntime:
    runtime_plan = build_training_modeling_runtime_plan(
        model_config=model_config,
        training_config=training_config,
    )
    fit_model = prepare_model_for_runtime(model, runtime_plan)
    with modeling_backend_scope(runtime_plan):
        batch_plan = plan_training_runtime(
            fit_model,
            prediction_contract=prediction_contract,
            execution_policy=execution_policy,
            representation_contract=representation_contract,
            store=store,
            train_sample_indices=train_sample_indices,
            validation_sample_indices=validation_sample_indices,
            runtime_plan=runtime_plan,
            training_config=training_config,
        )
    optimizer = torch.optim.AdamW(
        fit_model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    return PreparedTrainingRuntime(
        fit_model=fit_model,
        optimizer=optimizer,
        batch_plan=batch_plan,
    )


def _measure_training_batch_budget(
    model: TemporalModel,
    *,
    warmup_plan: BatchPlan[PredictionBatch],
    prediction_contract: CompiledPredictionContract,
    prediction_training_state: object | None,
    resolved_device: torch.device,
    training_config: TrainingConfig,
    precision: str,
) -> int:
    warmup_state = _clone_cpu_state(_unwrap_compiled_model(model))
    warmup_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    def _run_training_probe() -> None:
        batch = next(iter(warmup_plan.source))
        execute_training_batch(
            model,
            batch,
            resolved_device=resolved_device,
            precision=precision,
            prediction_contract=prediction_contract,
            prediction_training_state=prediction_training_state,
            optimizer=warmup_optimizer,
            gradient_clip_norm=training_config.gradient_clip_norm,
            zero_after_step=True,
        )

    try:
        return measure_device_resident_budget(
            resolved_device=resolved_device,
            run_probe=_run_training_probe,
        )
    finally:
        _unwrap_compiled_model(model).load_state_dict(warmup_state)
        torch.cuda.empty_cache()
