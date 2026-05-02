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
from ._runtime_probe import measure_device_resident_budget, measured_runtime_context
from .batch_plan import BatchPlan, build_prediction_batch_plan
from .evaluation_runtime import EvaluationScoringRuntimePlan
from .models import TemporalModel
from .representations import CompiledRepresentationContract, RepresentationRuntimeContext


@dataclass(frozen=True, slots=True)
class TrainingRuntimePlan:
    runtime_context: RepresentationRuntimeContext
    train_batch_plan: BatchPlan[PredictionBatch]
    validation_batch_plan: BatchPlan[PredictionBatch]
    evaluation_scoring_runtime_plan: EvaluationScoringRuntimePlan
    prediction_training_state: object | None


def _unwrap_compiled_model(model: TemporalModel) -> TemporalModel:
    return cast(TemporalModel, getattr(model, "_orig_mod", model))


def _clone_cpu_state(model: TemporalModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def plan_training_runtime(
    model: TemporalModel,
    *,
    prediction_contract: CompiledPredictionContract,
    execution_policy: CompiledExecutionPolicyContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    train_sample_indices: IntVector,
    validation_sample_indices: IntVector,
    base_runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    training_config: TrainingConfig,
    precision: str,
) -> TrainingRuntimePlan:
    prediction_training_state = prediction_contract.fit_training_state(
        store,
        train_sample_indices,
        execution_policy=execution_policy,
    )
    planned_runtime_context = measured_runtime_context(
        base_runtime_context,
        build_warmup_plan=lambda runtime_context: build_prediction_batch_plan(
            store,
            train_sample_indices,
            representation_contract=representation_contract,
            prediction_contract=prediction_contract,
            execution_policy=execution_policy,
            runtime_context=runtime_context,
            resolved_device=resolved_device,
            seed=training_config.seed,
            shuffle=False,
        ),
        measure_warmup_budget=lambda warmup_plan: _measure_training_batch_budget(
            model,
            warmup_plan=warmup_plan,
            prediction_contract=prediction_contract,
            prediction_training_state=prediction_training_state,
            resolved_device=resolved_device,
            training_config=training_config,
            precision=precision,
        ),
    )
    train_batch_plan = build_prediction_batch_plan(
        store,
        train_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        execution_policy=execution_policy,
        runtime_context=planned_runtime_context,
        resolved_device=resolved_device,
        seed=training_config.seed,
        shuffle=True,
    )
    validation_batch_plan = build_prediction_batch_plan(
        store,
        validation_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        execution_policy=execution_policy,
        runtime_context=planned_runtime_context,
        resolved_device=resolved_device,
        seed=training_config.seed,
        shuffle=False,
    )
    return TrainingRuntimePlan(
        runtime_context=planned_runtime_context,
        train_batch_plan=train_batch_plan,
        validation_batch_plan=validation_batch_plan,
        evaluation_scoring_runtime_plan=EvaluationScoringRuntimePlan(
            resolved_device=resolved_device,
            precision=precision,
            representation_runtime_context=planned_runtime_context,
            deterministic=training_config.deterministic,
            seed=training_config.seed,
        ),
        prediction_training_state=prediction_training_state,
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
