"""Native PyTorch training and evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from ..config.models import TrainingConfig
from ..core.files import write_path_atomic
from ..objectives import CompiledObjectiveContract, ObjectiveEvaluationContext
from ..prediction import CompiledPredictionContract, MetricSet
from ..prediction.contracts import PredictionBatch
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract
from ._runtime import (
    autocast_context,
    build_cuda_modeling_runtime,
    compute_device_resident_budget,
    configure_torch_backends,
    measure_forward_device_resident_budget,
    peak_cuda_reserved_bytes,
    reset_cuda_peak_memory,
    run_model_forward_pass,
    set_global_seed,
    snapshot_cuda_memory,
)
from .batch_sources import BatchSource, build_prediction_batch_source
from .families.base import ModelConfig
from .families.registry import (
    resolve_model_compile_enabled,
    resolve_model_training_precision,
)
from .models import TemporalModel
from .representations import CompiledRepresentationContract, RepresentationRuntimeContext

IntVector = NDArray[np.int64]


@dataclass(slots=True)
class TrainingResult:
    best_epoch: int
    objective_metric_id: str
    best_objective_value: float
    train_history: list[MetricSet]
    validation_history: list[MetricSet]
    objective_history: list[MetricSet]
    best_checkpoint_path: Path | None
    prediction_training_state: object | None


@dataclass(frozen=True, slots=True)
class TrainingEpochProgress:
    epoch: int
    max_epochs: int
    train_metrics: MetricSet
    validation_metrics: MetricSet
    objective_metrics: MetricSet
    objective_metric_id: str
    direction: str
    best_epoch: int
    best_objective_value: float


EpochEndCallback = Callable[[TrainingEpochProgress], None]
EarlyStopCallback = Callable[[int, int], None]


def _unwrap_compiled_model(model: TemporalModel) -> TemporalModel:
    return cast(TemporalModel, getattr(model, "_orig_mod", model))


def _is_improvement(
    *,
    current_epoch: int,
    best_epoch: int,
    history: list[MetricSet],
    direction: str,
    metric_id: str,
    min_delta: float,
) -> bool:
    if current_epoch == best_epoch:
        return True
    current_value = history[current_epoch - 1].require(metric_id)
    best_value = history[best_epoch - 1].require(metric_id)
    if direction == "maximize":
        return current_value > best_value + min_delta
    return current_value < best_value - min_delta


def _clone_cpu_state(model: TemporalModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _checkpoint_path(
    artifact_dir: Path,
    *,
    epoch: int,
    monitor: str,
    value: float,
) -> Path:
    return artifact_dir / "checkpoints" / f"epoch={epoch:02d}-{monitor}={value:.6f}.pt"


def _write_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    def _write(tmp_path: Path) -> None:
        torch.save(state_dict, tmp_path)

    write_path_atomic(path, _write)


def _build_grad_scaler(
    *,
    resolved_device: torch.device,
    precision: str,
) -> object | None:
    if resolved_device.type != "cuda" or precision != "16-mixed":
        return None
    return torch.cuda.amp.GradScaler()


def _host_streaming_runtime_context(
    runtime_context: RepresentationRuntimeContext,
) -> RepresentationRuntimeContext:
    return runtime_context.with_device_memory_budget(0)


def _run_probe_training_step(
    model: TemporalModel,
    *,
    loader: BatchSource[PredictionBatch],
    resolved_device: torch.device,
    precision: str,
    prediction_contract: CompiledPredictionContract,
    prediction_training_state: object | None,
    optimizer: torch.optim.Optimizer,
    grad_scaler: object | None,
    gradient_clip_norm: float | None,
) -> None:
    batch = next(iter(loader))
    device_batch = batch.to_device(resolved_device)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(resolved_device=resolved_device, precision=precision):
        outputs = model(**device_batch.model_kwargs())
        loss, _ = prediction_contract.compute_batch_loss_and_state(
            outputs,
            device_batch.targets,
            training_state=prediction_training_state,
        )
    if grad_scaler is not None:
        scaler = cast("torch.cuda.amp.GradScaler", grad_scaler)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(resolved_device)


def _planned_training_runtime_context(
    model: TemporalModel,
    *,
    prediction_contract: CompiledPredictionContract,
    realization_policy: CompiledRealizationPolicyContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    train_sample_indices: IntVector,
    base_runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    training_config: TrainingConfig,
    precision: str,
) -> RepresentationRuntimeContext:
    warmup_context = _host_streaming_runtime_context(base_runtime_context)
    warmup_source = build_prediction_batch_source(
        store,
        train_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        realization_policy=realization_policy,
        runtime_context=warmup_context,
        resolved_device=resolved_device,
        seed=training_config.seed,
        shuffle=False,
    )
    warmup_state = _clone_cpu_state(_unwrap_compiled_model(model))
    warmup_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    warmup_grad_scaler = _build_grad_scaler(
        resolved_device=resolved_device,
        precision=precision,
    )
    warmup_prediction_state = prediction_contract.fit_training_state(
        store,
        train_sample_indices,
        realization_policy=realization_policy,
    )
    baseline_memory = snapshot_cuda_memory(resolved_device)
    reset_cuda_peak_memory(resolved_device)
    _run_probe_training_step(
        model,
        loader=warmup_source,
        resolved_device=resolved_device,
        precision=precision,
        prediction_contract=prediction_contract,
        prediction_training_state=warmup_prediction_state,
        optimizer=warmup_optimizer,
        grad_scaler=warmup_grad_scaler,
        gradient_clip_norm=training_config.gradient_clip_norm,
    )
    budget = compute_device_resident_budget(
        free_bytes=baseline_memory.free_bytes,
        baseline_reserved_bytes=baseline_memory.reserved_bytes,
        peak_reserved_bytes=peak_cuda_reserved_bytes(resolved_device),
        total_bytes=baseline_memory.total_bytes,
    )
    _unwrap_compiled_model(model).load_state_dict(warmup_state)
    del warmup_source, warmup_optimizer, warmup_grad_scaler, warmup_prediction_state
    torch.cuda.empty_cache()
    return base_runtime_context.with_device_memory_budget(budget)


def _run_epoch(
    model: TemporalModel,
    *,
    loader: BatchSource[PredictionBatch],
    resolved_device: torch.device,
    precision: str,
    prediction_contract: CompiledPredictionContract,
    prediction_training_state: object | None,
    optimizer: torch.optim.Optimizer | None,
    grad_scaler: object | None,
    gradient_clip_norm: float | None,
    training: bool,
) -> MetricSet:
    accumulator = prediction_contract.create_epoch_accumulator()
    if training:
        model.train()
    else:
        model.eval()
    grad_enabled = training
    with torch.set_grad_enabled(grad_enabled):
        for _batch_idx, batch in enumerate(loader):
            device_batch = batch.to_device(resolved_device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(resolved_device=resolved_device, precision=precision):
                outputs = model(**device_batch.model_kwargs())
                loss, batch_state = prediction_contract.compute_batch_loss_and_state(
                    outputs,
                    device_batch.targets,
                    training_state=prediction_training_state,
                )
            accumulator.update(batch_state)
            if training:
                if optimizer is None:
                    raise RuntimeError("optimizer is required for training epochs")
                if grad_scaler is not None:
                    scaler = cast("torch.cuda.amp.GradScaler", grad_scaler)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
    return accumulator.finalize()


def train_model(
    model: TemporalModel,
    *,
    model_config: ModelConfig,
    prediction_contract: CompiledPredictionContract,
    objective_contract: CompiledObjectiveContract,
    realization_policy: CompiledRealizationPolicyContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    train_sample_indices: IntVector,
    validation_sample_indices: IntVector,
    training_config: TrainingConfig,
    artifact_dir: Path,
    on_epoch_end: EpochEndCallback | None = None,
    on_early_stop: EarlyStopCallback | None = None,
) -> TrainingResult:
    if train_sample_indices.size == 0 or validation_sample_indices.size == 0:
        raise ValueError("Train and validation sample selections must both be non-empty")

    set_global_seed(training_config.seed)
    runtime = build_cuda_modeling_runtime(batch_size=training_config.batch_size)
    precision = resolve_model_training_precision(
        device=runtime.resolved_device,
        model_config=model_config,
    )
    compile_enabled = resolve_model_compile_enabled(
        device=runtime.resolved_device,
        model_config=model_config,
    )
    model.to(runtime.resolved_device)
    fit_model = cast(TemporalModel, torch.compile(model) if compile_enabled else model)

    train_history: list[MetricSet] = []
    validation_history: list[MetricSet] = []
    objective_history: list[MetricSet] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    with configure_torch_backends(
        resolved_device=runtime.resolved_device,
        deterministic=training_config.deterministic,
    ):
        planned_runtime_context = _planned_training_runtime_context(
            fit_model,
            prediction_contract=prediction_contract,
            realization_policy=realization_policy,
            representation_contract=representation_contract,
            store=store,
            train_sample_indices=train_sample_indices,
            base_runtime_context=runtime.representation_runtime_context,
            resolved_device=runtime.resolved_device,
            training_config=training_config,
            precision=precision,
        )
        train_batch_source = build_prediction_batch_source(
            store,
            train_sample_indices,
            representation_contract=representation_contract,
            prediction_contract=prediction_contract,
            realization_policy=realization_policy,
            runtime_context=planned_runtime_context,
            resolved_device=runtime.resolved_device,
            seed=training_config.seed,
            shuffle=True,
        )
        validation_batch_source = build_prediction_batch_source(
            store,
            validation_sample_indices,
            representation_contract=representation_contract,
            prediction_contract=prediction_contract,
            realization_policy=realization_policy,
            runtime_context=planned_runtime_context,
            resolved_device=runtime.resolved_device,
            seed=training_config.seed,
            shuffle=False,
        )
        prediction_training_state = prediction_contract.fit_training_state(
            store,
            train_sample_indices,
            realization_policy=realization_policy,
        )
        optimizer = torch.optim.AdamW(
            fit_model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        grad_scaler = _build_grad_scaler(
            resolved_device=runtime.resolved_device,
            precision=precision,
        )
        for epoch in range(1, training_config.max_epochs + 1):
            train_metrics = _run_epoch(
                fit_model,
                loader=train_batch_source,
                resolved_device=runtime.resolved_device,
                precision=precision,
                prediction_contract=prediction_contract,
                prediction_training_state=prediction_training_state,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                gradient_clip_norm=training_config.gradient_clip_norm,
                training=True,
            )
            validation_metrics = _run_epoch(
                fit_model,
                loader=validation_batch_source,
                resolved_device=runtime.resolved_device,
                precision=precision,
                prediction_contract=prediction_contract,
                prediction_training_state=prediction_training_state,
                optimizer=None,
                grad_scaler=None,
                gradient_clip_norm=None,
                training=False,
            )
            train_history.append(train_metrics)
            validation_history.append(validation_metrics)
            objective_metrics = objective_contract.evaluate_metrics(
                validation_metrics,
                context=ObjectiveEvaluationContext(
                    model=fit_model,
                    model_config=model_config,
                    prediction_contract=prediction_contract,
                    representation_contract=representation_contract,
                    realization_policy=realization_policy,
                    store=store,
                    sample_indices=validation_sample_indices,
                    batch_size=training_config.batch_size,
                ),
            )
            objective_history.append(objective_metrics)

            current_best_epoch = _best_epoch_from_objective_history(
                objective_history,
                objective_contract=objective_contract,
            )
            if _is_improvement(
                current_epoch=epoch,
                best_epoch=current_best_epoch,
                history=objective_history,
                direction=objective_contract.direction,
                metric_id=objective_contract.metric_id,
                min_delta=training_config.early_stopping.min_delta,
            ):
                best_state = _clone_cpu_state(_unwrap_compiled_model(fit_model))
                best_epoch = current_best_epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            best_value = objective_contract.value(objective_history[best_epoch - 1])
            if on_epoch_end is not None:
                on_epoch_end(
                    TrainingEpochProgress(
                        epoch=epoch,
                        max_epochs=training_config.max_epochs,
                        train_metrics=train_metrics,
                        validation_metrics=validation_metrics,
                        objective_metrics=objective_metrics,
                        objective_metric_id=objective_contract.metric_id,
                        direction=objective_contract.direction,
                        best_epoch=best_epoch,
                        best_objective_value=best_value,
                    )
                )

            if epochs_without_improvement >= training_config.early_stopping.patience:
                if on_early_stop is not None:
                    on_early_stop(epoch, best_epoch)
                break

    if best_state is None:
        best_epoch = _best_epoch_from_objective_history(
            objective_history,
            objective_contract=objective_contract,
        )
        best_state = _clone_cpu_state(_unwrap_compiled_model(fit_model))
    else:
        best_epoch = _best_epoch_from_objective_history(
            objective_history,
            objective_contract=objective_contract,
        )
    best_value = objective_contract.value(objective_history[best_epoch - 1])
    model.load_state_dict(best_state)
    best_checkpoint_path = _checkpoint_path(
        artifact_dir,
        epoch=best_epoch,
        monitor=objective_contract.checkpoint_monitor,
        value=best_value,
    )
    _write_checkpoint(best_checkpoint_path, best_state)

    return TrainingResult(
        best_epoch=best_epoch,
        objective_metric_id=objective_contract.metric_id,
        best_objective_value=best_value,
        train_history=train_history,
        validation_history=validation_history,
        objective_history=objective_history,
        best_checkpoint_path=best_checkpoint_path,
        prediction_training_state=prediction_training_state,
    )


def _best_epoch_from_objective_history(
    history: list[MetricSet],
    *,
    objective_contract: CompiledObjectiveContract,
) -> int:
    if not history:
        return 1
    if objective_contract.direction == "maximize":
        winner = max(
            range(len(history)),
            key=lambda index: objective_contract.value(history[index]),
        )
    else:
        winner = min(
            range(len(history)),
            key=lambda index: objective_contract.value(history[index]),
        )
    return winner + 1


def evaluate_model(
    model: torch.nn.Module,
    *,
    model_config: ModelConfig,
    prediction_contract: CompiledPredictionContract,
    realization_policy: CompiledRealizationPolicyContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    sample_indices: IntVector,
    prediction_training_state: object | None,
    training_config: TrainingConfig,
) -> MetricSet:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    runtime = build_cuda_modeling_runtime(batch_size=training_config.batch_size)
    precision = resolve_model_training_precision(
        device=runtime.resolved_device,
        model_config=model_config,
    )
    model.to(runtime.resolved_device)
    accumulator = prediction_contract.create_epoch_accumulator()

    def _accumulate(batch: PredictionBatch, outputs) -> None:
        _, batch_state = prediction_contract.compute_batch_loss_and_state(
            outputs,
            batch.targets,
            training_state=prediction_training_state,
        )
        accumulator.update(batch_state)

    with configure_torch_backends(
        resolved_device=runtime.resolved_device,
        deterministic=training_config.deterministic,
    ):
        warmup_source = build_prediction_batch_source(
            store,
            sample_indices,
            representation_contract=representation_contract,
            prediction_contract=prediction_contract,
            realization_policy=realization_policy,
            runtime_context=_host_streaming_runtime_context(runtime.representation_runtime_context),
            resolved_device=runtime.resolved_device,
            seed=training_config.seed,
        )
        planned_runtime_context = runtime.representation_runtime_context.with_device_memory_budget(
            measure_forward_device_resident_budget(
                cast(TemporalModel, model),
                loader=warmup_source,
                resolved_device=runtime.resolved_device,
                precision=precision,
            )
        )
        batch_source = build_prediction_batch_source(
            store,
            sample_indices,
            representation_contract=representation_contract,
            prediction_contract=prediction_contract,
            realization_policy=realization_policy,
            runtime_context=planned_runtime_context,
            resolved_device=runtime.resolved_device,
            seed=training_config.seed,
        )
        run_model_forward_pass(
            cast(TemporalModel, model),
            loader=batch_source,
            resolved_device=runtime.resolved_device,
            precision=precision,
            on_outputs=_accumulate,
        )
    return accumulator.finalize()
