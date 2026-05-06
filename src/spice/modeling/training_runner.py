"""Training-runner fit and metric evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from ..config.models import TrainingConfig
from ..metrics import MetricSet
from ..prediction import CompiledPredictionContract
from ..prediction.contracts import PredictionBatch
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore
from ._epoch_execution import run_epoch
from ._fit_policy import CompletedEpoch, TrainingEpochProgress, TrainingFitPolicy
from .families.base import ModelConfig
from .forward_runtime import run_planned_prediction_forward
from .models import TemporalModel
from .objective_runtime import CompiledObjectiveRuntime, ObjectiveMetricContext
from .representations import CompiledRepresentationContract
from .runtime_planning import (
    build_cuda_modeling_runtime_plan,
    modeling_backend_scope,
    prepare_model_for_runtime,
)
from .training_runtime import prepare_training_runtime

IntVector = NDArray[np.int64]


@dataclass(slots=True)
class TrainingResult:
    best_epoch: int
    objective_metric_id: str
    best_objective_value: float
    train_history: list[MetricSet]
    validation_history: list[MetricSet]
    objective_history: list[MetricSet]
    prediction_training_state: object | None


EpochEndCallback = Callable[[TrainingEpochProgress], None]
EarlyStopCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class TrainingCallbacks:
    on_epoch_end: EpochEndCallback | None = None
    on_early_stop: EarlyStopCallback | None = None


@dataclass(slots=True)
class TrainingFitSpec:
    model: TemporalModel
    model_config: ModelConfig
    prediction_contract: CompiledPredictionContract
    objective_runtime: CompiledObjectiveRuntime
    execution_policy: CompiledExecutionPolicyContract
    representation_contract: CompiledRepresentationContract
    store: CompiledProblemStore
    train_sample_indices: IntVector
    validation_sample_indices: IntVector
    training_config: TrainingConfig


@dataclass(slots=True)
class TrainingMetricEvaluationSpec:
    model: torch.nn.Module
    model_config: ModelConfig
    prediction_contract: CompiledPredictionContract
    execution_policy: CompiledExecutionPolicyContract
    representation_contract: CompiledRepresentationContract
    store: CompiledProblemStore
    sample_indices: IntVector
    prediction_training_state: object | None
    training_config: TrainingConfig


def run_training_fit(
    spec: TrainingFitSpec,
    *,
    callbacks: TrainingCallbacks | None = None,
) -> TrainingResult:
    active_callbacks = callbacks or TrainingCallbacks()
    if spec.train_sample_indices.size == 0 or spec.validation_sample_indices.size == 0:
        raise ValueError("Train and validation sample selections must both be non-empty")

    policy = TrainingFitPolicy.create(
        objective_contract=spec.objective_runtime.contract,
        max_epochs=spec.training_config.max_epochs,
        patience=spec.training_config.early_stopping.patience,
        min_delta=spec.training_config.early_stopping.min_delta,
    )

    prepared_runtime = prepare_training_runtime(
        spec.model,
        model_config=spec.model_config,
        prediction_contract=spec.prediction_contract,
        execution_policy=spec.execution_policy,
        representation_contract=spec.representation_contract,
        store=spec.store,
        train_sample_indices=spec.train_sample_indices,
        validation_sample_indices=spec.validation_sample_indices,
        training_config=spec.training_config,
    )
    fit_model = prepared_runtime.fit_model
    training_runtime_plan = prepared_runtime.batch_plan
    runtime_plan = training_runtime_plan.runtime_plan
    prediction_training_state = training_runtime_plan.prediction_training_state
    optimizer = prepared_runtime.optimizer
    objective_context = ObjectiveMetricContext(
        model=fit_model,
        prediction_contract=spec.prediction_contract,
        representation_contract=spec.representation_contract,
        execution_policy=spec.execution_policy,
        store=spec.store,
        sample_indices=spec.validation_sample_indices,
        runtime_plan=runtime_plan,
    )
    with modeling_backend_scope(runtime_plan):
        for epoch in range(1, spec.training_config.max_epochs + 1):
            train_metrics = run_epoch(
                fit_model,
                loader=training_runtime_plan.train_batch_plan.source,
                resolved_device=runtime_plan.resolved_device,
                precision=runtime_plan.precision,
                prediction_contract=spec.prediction_contract,
                prediction_training_state=prediction_training_state,
                optimizer=optimizer,
                gradient_clip_norm=spec.training_config.gradient_clip_norm,
                training=True,
            )
            train_decision = policy.handle_nonfinite_metrics(
                epoch=epoch,
                phase="train",
                metrics=train_metrics,
            )
            if train_decision is not None:
                if (
                    train_decision.early_stop is not None
                    and active_callbacks.on_early_stop is not None
                ):
                    active_callbacks.on_early_stop(*train_decision.early_stop)
                break
            validation_metrics = run_epoch(
                fit_model,
                loader=training_runtime_plan.validation_batch_plan.source,
                resolved_device=runtime_plan.resolved_device,
                precision=runtime_plan.precision,
                prediction_contract=spec.prediction_contract,
                prediction_training_state=prediction_training_state,
                optimizer=None,
                gradient_clip_norm=None,
                training=False,
            )
            validation_decision = policy.handle_nonfinite_metrics(
                epoch=epoch,
                phase="validation",
                metrics=validation_metrics,
            )
            if validation_decision is not None:
                if (
                    validation_decision.early_stop is not None
                    and active_callbacks.on_early_stop is not None
                ):
                    active_callbacks.on_early_stop(*validation_decision.early_stop)
                break
            objective_metrics = spec.objective_runtime.evaluate_metrics(
                validation_metrics,
                context=objective_context,
            )
            objective_decision = policy.handle_nonfinite_metrics(
                epoch=epoch,
                phase="objective",
                metrics=objective_metrics,
            )
            if objective_decision is not None:
                if (
                    objective_decision.early_stop is not None
                    and active_callbacks.on_early_stop is not None
                ):
                    active_callbacks.on_early_stop(*objective_decision.early_stop)
                break
            fit_decision = policy.record_completed_epoch(
                CompletedEpoch(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    objective_metrics=objective_metrics,
                ),
                model=fit_model,
            )
            if fit_decision.progress is not None and active_callbacks.on_epoch_end is not None:
                active_callbacks.on_epoch_end(fit_decision.progress)

            if fit_decision.should_stop:
                if (
                    fit_decision.early_stop is not None
                    and active_callbacks.on_early_stop is not None
                ):
                    active_callbacks.on_early_stop(*fit_decision.early_stop)
                break

    best_epoch, best_state, best_value = policy.finalized_best(model=fit_model)
    spec.model.load_state_dict(best_state)

    return TrainingResult(
        best_epoch=best_epoch,
        objective_metric_id=spec.objective_runtime.contract.metric_id,
        best_objective_value=best_value,
        train_history=policy.train_history,
        validation_history=policy.validation_history,
        objective_history=policy.objective_history,
        prediction_training_state=prediction_training_state,
    )


def evaluate_training_metrics(spec: TrainingMetricEvaluationSpec) -> MetricSet:
    if spec.sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    runtime_plan = build_cuda_modeling_runtime_plan(
        model_config=spec.model_config,
        batch_size=spec.training_config.batch_size,
        deterministic=spec.training_config.deterministic,
        seed=spec.training_config.seed,
    )
    fit_model = prepare_model_for_runtime(cast(TemporalModel, spec.model), runtime_plan)
    accumulator = spec.prediction_contract.create_epoch_accumulator()

    def _accumulate(batch: PredictionBatch, outputs) -> None:
        _, batch_state = spec.prediction_contract.compute_batch_loss_and_state(
            outputs,
            batch.targets,
            training_state=spec.prediction_training_state,
        )
        accumulator.update(batch_state)

    with modeling_backend_scope(runtime_plan):
        run_planned_prediction_forward(
            fit_model,
            store=spec.store,
            sample_indices=spec.sample_indices,
            representation_contract=spec.representation_contract,
            prediction_contract=spec.prediction_contract,
            execution_policy=spec.execution_policy,
            runtime_plan=runtime_plan,
            on_outputs=_accumulate,
        )
    return accumulator.finalize()
