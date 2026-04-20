"""Native PyTorch training and evaluation utilities."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from ..config import ModelConfig, TrainingConfig
from ..core.files import write_path_atomic
from ..core.reporting import (
    NullReporter,
    Reporter,
    StageMetricValue,
    format_compact_count,
)
from ..prediction import CompiledPredictionContract, MetricSet
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract
from ._runtime import (
    CompiledRepresentationContract,
    build_prediction_batch_source,
    build_representation_runtime_context,
    configure_torch_backends,
    ensure_device_runtime_ready,
    prepare_supervised_prediction_representation,
    resolve_compile_enabled,
    resolve_device,
    resolve_trainer_precision,
    set_global_seed,
)
from .batch_sources import BatchSourcePlan, PreparedBatchSource
from .models import TemporalModel

IntVector = NDArray[np.int64]
_EPOCH_STAGE_METRIC_ID = "epoch"


@dataclass(slots=True)
class TrainingResult:
    best_epoch: int
    train_history: list[MetricSet]
    validation_history: list[MetricSet]
    best_checkpoint_path: Path | None
    resolved_precision: str
    compiled: bool
    resolved_device: str
    representation_id: str
    loader_strategy_id: str
    input_storage_mode_id: str
    target_storage_mode_id: str
    batch_planner_id: str
    prediction_training_state: object | None


def _unwrap_compiled_model(model: TemporalModel) -> TemporalModel:
    return cast(TemporalModel, getattr(model, "_orig_mod", model))


@dataclass(slots=True)
class _ProgressReporter:
    reporter: Reporter
    max_epochs: int
    log_every_n_steps: int
    prediction_contract: CompiledPredictionContract
    task_id: int | None = None
    total_batches: int = 0
    train_batches_per_epoch: int = 0

    def start(self, *, train_batches_per_epoch: int) -> None:
        if train_batches_per_epoch <= 0:
            self.task_id = self.reporter.start_task("train epochs")
            return
        self.train_batches_per_epoch = train_batches_per_epoch
        self.total_batches = train_batches_per_epoch * self.max_epochs
        self.task_id = self.reporter.start_task(
            "train epochs",
            total=self.total_batches,
            unit="batches",
        )

    def on_train_batch_end(
        self,
        *,
        epoch: int,
        batch_idx: int,
        metrics: MetricSet,
    ) -> None:
        if self.task_id is None:
            return
        if (batch_idx + 1) % self.log_every_n_steps != 0 and batch_idx + 1 < max(
            1,
            self.train_batches_per_epoch,
        ):
            return
        completed = None
        if self.train_batches_per_epoch > 0:
            completed = min(
                self.total_batches,
                (epoch - 1) * self.train_batches_per_epoch + batch_idx + 1,
            )
        self.reporter.update_task(
            self.task_id,
            completed=completed,
            message=(
                "batch "
                f"{_format_compact_progress(batch_idx + 1, max(1, self.train_batches_per_epoch))}"
            ),
            metrics=(
                StageMetricValue(
                    id=_EPOCH_STAGE_METRIC_ID,
                    value=f"{epoch}/{self.max_epochs}",
                ),
                *self.prediction_contract.format_progress_metrics(metrics),
            ),
        )

    def on_validation_epoch_end(self, *, epoch: int, metrics: MetricSet) -> None:
        if self.task_id is None:
            return
        completed = None
        if self.train_batches_per_epoch > 0:
            completed = min(self.total_batches, epoch * self.train_batches_per_epoch)
        self.reporter.update_task(
            self.task_id,
            completed=completed,
            message="validation",
            metrics=(
                StageMetricValue(
                    id=_EPOCH_STAGE_METRIC_ID,
                    value=f"{epoch}/{self.max_epochs}",
                ),
                *self.prediction_contract.format_progress_metrics(metrics),
            ),
        )

    def finish(self, *, best_epoch: int) -> None:
        if self.task_id is None:
            return
        self.reporter.finish_task(
            self.task_id,
            message=f"best epoch {best_epoch}",
        )
        self.task_id = None


def _format_compact_progress(completed: int, total: int) -> str:
    return f"{format_compact_count(completed)}/{format_compact_count(total)}"


def _is_improvement(
    *,
    current_epoch: int,
    best_epoch: int,
    history: list[MetricSet],
    direction: str,
    primary_metric_id: str,
    min_delta: float,
) -> bool:
    if current_epoch == best_epoch:
        return True
    current_value = history[current_epoch - 1].require(primary_metric_id)
    best_value = history[best_epoch - 1].require(primary_metric_id)
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


def _autocast_context(
    *,
    resolved_device: torch.device,
    precision: str,
):
    if precision == "32-true":
        return nullcontext()
    if precision == "16-mixed":
        if resolved_device.type != "cuda":
            raise ValueError("fp16 mixed precision is only supported on CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16-mixed":
        if resolved_device.type not in {"cpu", "cuda"}:
            raise ValueError("bf16 mixed precision is only supported on CPU or CUDA")
        return torch.autocast(device_type=resolved_device.type, dtype=torch.bfloat16)
    raise ValueError(f"Unsupported resolved precision: {precision}")


def _build_grad_scaler(
    *,
    resolved_device: torch.device,
    precision: str,
) -> object | None:
    if resolved_device.type != "cuda" or precision != "16-mixed":
        return None
    return torch.cuda.amp.GradScaler()


def _run_epoch(
    model: TemporalModel,
    *,
    loader: PreparedBatchSource,
    resolved_device: torch.device,
    precision: str,
    prediction_contract: CompiledPredictionContract,
    prediction_training_state: object | None,
    optimizer: torch.optim.Optimizer | None,
    grad_scaler: object | None,
    gradient_clip_norm: float | None,
    training: bool,
    progress: _ProgressReporter | None = None,
    epoch: int | None = None,
) -> MetricSet:
    accumulator = prediction_contract.create_epoch_accumulator(
        "train" if training else "validation"
    )
    if training:
        model.train()
    else:
        model.eval()
    grad_enabled = training
    with torch.set_grad_enabled(grad_enabled):
        for batch_idx, batch in enumerate(loader):
            device_batch = batch.to_device(resolved_device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with _autocast_context(resolved_device=resolved_device, precision=precision):
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
                if progress is not None and epoch is not None:
                    progress.on_train_batch_end(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        metrics=accumulator.snapshot(),
                    )
    return accumulator.finalize()


def train_model(
    model: TemporalModel,
    *,
    model_config: ModelConfig,
    prediction_contract: CompiledPredictionContract,
    realization_policy: CompiledRealizationPolicyContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    train_sample_indices: IntVector,
    validation_sample_indices: IntVector,
    training_config: TrainingConfig,
    artifact_dir: Path,
    reporter: Reporter | None = None,
) -> TrainingResult:
    reporter = reporter or NullReporter()
    if train_sample_indices.size == 0 or validation_sample_indices.size == 0:
        raise ValueError("Train and validation sample selections must both be non-empty")

    set_global_seed(training_config.seed)
    resolved_device = resolve_device(training_config.device)
    ensure_device_runtime_ready(
        requested_device=training_config.device,
        resolved_device=resolved_device,
    )
    precision = resolve_trainer_precision(
        training_config,
        device=resolved_device,
        model_config=model_config,
    )
    compile_enabled = resolve_compile_enabled(
        training_config,
        device=resolved_device,
        precision=precision,
        model_config=model_config,
    )
    runtime_context = build_representation_runtime_context(
        device=resolved_device,
        batch_size=training_config.batch_size,
    )
    train_representation = prepare_supervised_prediction_representation(
        store,
        train_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        realization_policy=realization_policy,
        runtime_context=runtime_context,
    )
    validation_representation = prepare_supervised_prediction_representation(
        store,
        validation_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        realization_policy=realization_policy,
        runtime_context=runtime_context,
    )
    train_batch_source_plan = _plan_training_batch_source(
        train_representation,
        runtime_context=runtime_context,
        device=resolved_device,
        seed=training_config.seed,
        shuffle=True,
    )
    validation_batch_source_plan = _plan_training_batch_source(
        validation_representation,
        runtime_context=runtime_context,
        device=resolved_device,
        seed=training_config.seed,
        shuffle=False,
    )

    prediction_training_state = prediction_contract.fit_training_state(
        store,
        train_sample_indices,
        realization_policy=realization_policy,
    )
    model.to(resolved_device)
    fit_model = cast(TemporalModel, torch.compile(model) if compile_enabled else model)
    optimizer = torch.optim.AdamW(
        fit_model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    grad_scaler = _build_grad_scaler(
        resolved_device=resolved_device,
        precision=precision,
    )
    progress = _ProgressReporter(
        reporter=reporter,
        max_epochs=training_config.max_epochs,
        log_every_n_steps=training_config.log_every_n_steps,
        prediction_contract=prediction_contract,
    )
    progress.start(train_batches_per_epoch=len(train_batch_source_plan.source))

    train_history: list[MetricSet] = []
    validation_history: list[MetricSet] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    with configure_torch_backends(
        resolved_device=resolved_device,
        deterministic=training_config.deterministic,
    ):
        for epoch in range(1, training_config.max_epochs + 1):
            train_metrics = _run_epoch(
                fit_model,
                loader=train_batch_source_plan.source,
                resolved_device=resolved_device,
                precision=precision,
                prediction_contract=prediction_contract,
                prediction_training_state=prediction_training_state,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                gradient_clip_norm=training_config.gradient_clip_norm,
                training=True,
                progress=progress,
                epoch=epoch,
            )
            validation_metrics = _run_epoch(
                fit_model,
                loader=validation_batch_source_plan.source,
                resolved_device=resolved_device,
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
            progress.on_validation_epoch_end(epoch=epoch, metrics=validation_metrics)

            current_best_epoch = prediction_contract.best_epoch(validation_history)
            if _is_improvement(
                current_epoch=epoch,
                best_epoch=current_best_epoch,
                history=validation_history,
                direction=prediction_contract.direction,
                primary_metric_id=prediction_contract.primary_metric_id,
                min_delta=training_config.early_stopping.min_delta,
            ):
                best_state = _clone_cpu_state(_unwrap_compiled_model(fit_model))
                best_epoch = current_best_epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= training_config.early_stopping.patience:
                break

    if best_state is None:
        best_epoch = prediction_contract.best_epoch(validation_history)
        best_state = _clone_cpu_state(_unwrap_compiled_model(fit_model))
    else:
        best_epoch = prediction_contract.best_epoch(validation_history)
    best_value = prediction_contract.optimization_value(validation_history[best_epoch - 1])
    model.load_state_dict(best_state)
    best_checkpoint_path = _checkpoint_path(
        artifact_dir,
        epoch=best_epoch,
        monitor=prediction_contract.checkpoint_monitor,
        value=best_value,
    )
    _write_checkpoint(best_checkpoint_path, best_state)
    progress.finish(best_epoch=best_epoch)

    return TrainingResult(
        best_epoch=best_epoch,
        train_history=train_history,
        validation_history=validation_history,
        best_checkpoint_path=best_checkpoint_path,
        resolved_precision=precision,
        compiled=compile_enabled,
        resolved_device=resolved_device.type,
        representation_id=train_representation.representation_id,
        loader_strategy_id=train_batch_source_plan.loader_strategy_id,
        input_storage_mode_id=train_batch_source_plan.input_storage_mode_id,
        target_storage_mode_id=train_batch_source_plan.target_storage_mode_id,
        batch_planner_id=train_batch_source_plan.batch_planner_id,
        prediction_training_state=prediction_training_state,
    )


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
    reporter: Reporter | None = None,
) -> MetricSet:
    reporter = reporter or NullReporter()
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    resolved_device = resolve_device(training_config.device)
    ensure_device_runtime_ready(
        requested_device=training_config.device,
        resolved_device=resolved_device,
    )
    precision = resolve_trainer_precision(
        training_config,
        device=resolved_device,
        model_config=model_config,
    )
    model.to(resolved_device)
    runtime_context = build_representation_runtime_context(
        device=resolved_device,
        batch_size=training_config.batch_size,
    )
    batch_source_plan = build_prediction_batch_source(
        store,
        sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        realization_policy=realization_policy,
        runtime_context=runtime_context,
        resolved_device=resolved_device,
        seed=training_config.seed,
    )
    loader = batch_source_plan.source
    task_id = reporter.start_task("evaluate model", total=len(loader), unit="batches")
    with configure_torch_backends(
        resolved_device=resolved_device,
        deterministic=training_config.deterministic,
    ):
        metrics = _run_evaluation_loader(
            cast(TemporalModel, model),
            loader=loader,
            resolved_device=resolved_device,
            precision=precision,
            prediction_contract=prediction_contract,
            prediction_training_state=prediction_training_state,
            reporter=reporter,
            task_id=task_id,
        )
    reporter.finish_task(task_id)
    return metrics


def _run_evaluation_loader(
    model: TemporalModel,
    *,
    loader: PreparedBatchSource,
    resolved_device: torch.device,
    precision: str,
    prediction_contract: CompiledPredictionContract,
    prediction_training_state: object | None,
    reporter: Reporter,
    task_id: int,
) -> MetricSet:
    model.eval()
    accumulator = prediction_contract.create_epoch_accumulator("evaluation")
    with torch.no_grad():
        for batch in loader:
            device_batch = batch.to_device(resolved_device)
            with _autocast_context(resolved_device=resolved_device, precision=precision):
                outputs = model(**device_batch.model_kwargs())
                _, batch_state = prediction_contract.compute_batch_loss_and_state(
                    outputs,
                    device_batch.targets,
                    training_state=prediction_training_state,
                )
            accumulator.update(batch_state)
            reporter.update_task(task_id, advance=1)
    return accumulator.finalize()


def _plan_training_batch_source(
    prepared,
    *,
    runtime_context,
    device,
    seed,
    shuffle,
) -> BatchSourcePlan:
    from .batch_sources import plan_batch_source

    return plan_batch_source(
        prepared,
        runtime_context=runtime_context,
        resolved_device=device,
        seed=seed,
        shuffle=shuffle,
    )
