"""Training utilities backed by Lightning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from numpy.typing import NDArray

from ..config import ModelConfig, TrainingConfig
from ..core.reporting import (
    NullReporter,
    Reporter,
    StageMetricValue,
)
from ..prediction import CompiledPredictionContract, MetricSet
from ..temporal.problem_store import CompiledProblemStore
from ._runtime import (
    CompiledRepresentationContract,
    build_prediction_loader,
    build_representation_runtime_context,
    prepare_prediction_representation,
    resolve_compile_enabled,
    resolve_device,
    resolve_trainer_precision,
    set_global_seed,
)
from .datamodule import TemporalDataModule
from .lightning_module import TemporalLightningModule
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
    storage_mode_id: str
    batch_planner_id: str
    prediction_training_state: object | None


def _unwrap_compiled_model(model: TemporalModel) -> TemporalModel:
    return cast(TemporalModel, getattr(model, "_orig_mod", model))


class ReporterProgressCallback(L.Callback):
    def __init__(
        self,
        reporter: Reporter,
        *,
        max_epochs: int,
        log_every_n_steps: int,
        prediction_contract: CompiledPredictionContract,
    ) -> None:
        super().__init__()
        self._reporter = reporter
        self._max_epochs = max_epochs
        self._log_every_n_steps = log_every_n_steps
        self._prediction_contract = prediction_contract
        self._task_id: int | None = None
        self._total_batches = 0
        self._train_batches_per_epoch = 0

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        train_batches = trainer.num_training_batches
        if not isinstance(train_batches, int) or train_batches <= 0:
            self._task_id = self._reporter.start_task("train epochs")
            return
        self._train_batches_per_epoch = train_batches
        self._total_batches = train_batches * self._max_epochs
        self._task_id = self._reporter.start_task(
            "train epochs",
            total=self._total_batches,
            unit="batches",
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        del batch, outputs
        if self._task_id is None:
            return
        if (batch_idx + 1) % self._log_every_n_steps != 0 and batch_idx + 1 < max(
            1,
            self._train_batches_per_epoch,
        ):
            return
        if not isinstance(pl_module, TemporalLightningModule):
            return
        snapshot = pl_module.train_progress_snapshot()
        if snapshot is None:
            return
        completed = None
        if self._train_batches_per_epoch > 0:
            completed = min(
                self._total_batches,
                trainer.current_epoch * self._train_batches_per_epoch + batch_idx + 1,
            )
        metrics = [
            StageMetricValue(
                id=_EPOCH_STAGE_METRIC_ID,
                value=f"{trainer.current_epoch + 1}/{self._max_epochs}",
            )
        ]
        message = "batch " + _format_compact_progress(
            batch_idx + 1,
            max(1, self._train_batches_per_epoch),
        )
        metrics.extend(self._prediction_contract.format_progress_metrics(snapshot))
        self._reporter.update_task(
            self._task_id,
            completed=completed,
            message=message,
            metrics=metrics,
        )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._task_id is None:
            return
        metrics = getattr(pl_module, "validation_history", None)
        if not metrics:
            return
        latest = metrics[-1]
        completed = None
        if self._train_batches_per_epoch > 0:
            completed = min(
                self._total_batches,
                (trainer.current_epoch + 1) * self._train_batches_per_epoch,
            )
        metrics = (
            StageMetricValue(
                id=_EPOCH_STAGE_METRIC_ID,
                value=f"{trainer.current_epoch + 1}/{self._max_epochs}",
            ),
            *self._prediction_contract.format_progress_metrics(latest),
        )
        self._reporter.update_task(
            self._task_id,
            completed=completed,
            message="validation",
            metrics=metrics,
        )

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer
        if self._task_id is None:
            return
        validation_history = getattr(pl_module, "validation_history", [])
        self._reporter.finish_task(
            self._task_id,
            message=f"best epoch {_best_epoch(validation_history, self._prediction_contract)}",
        )
        self._task_id = None


def _format_compact_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 100_000:
        return f"{value / 1_000:.0f}k"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    if value >= 1_000:
        return f"{value / 1_000:.2f}k"
    return f"{value:,}"


def _format_compact_progress(completed: int, total: int) -> str:
    return f"{_format_compact_count(completed)}/{_format_compact_count(total)}"


def _trainer_device_args(device_name: str) -> tuple[str, int | str | list[int]]:
    resolved = resolve_device(device_name)
    if resolved.type == "cuda":
        if resolved.index is None:
            return "gpu", 1
        return "gpu", [resolved.index]
    if resolved.type == "mps":
        return "mps", 1
    return "cpu", 1


def _best_epoch(
    validation_history: list[MetricSet],
    prediction_contract: CompiledPredictionContract,
) -> int:
    return prediction_contract.best_epoch(validation_history)


def train_model(
    model: TemporalModel,
    *,
    model_config: ModelConfig,
    prediction_contract: CompiledPredictionContract,
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
    L.seed_everything(training_config.seed, workers=True)
    device = resolve_device(training_config.device)
    precision = resolve_trainer_precision(
        training_config,
        device=device,
        model_config=model_config,
    )
    compile_enabled = resolve_compile_enabled(
        training_config,
        device=device,
        precision=precision,
        model_config=model_config,
    )
    fit_model = cast(TemporalModel, torch.compile(model) if compile_enabled else model)
    runtime_context = build_representation_runtime_context(
        device=device,
        batch_size=training_config.batch_size,
    )
    train_representation = prepare_prediction_representation(
        store,
        train_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        runtime_context=runtime_context,
    )
    validation_representation = prepare_prediction_representation(
        store,
        validation_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        runtime_context=runtime_context,
    )

    data_module = TemporalDataModule(
        train_representation=train_representation,
        validation_representation=validation_representation,
        seed=training_config.seed,
    )
    prediction_training_state = prediction_contract.fit_training_state(
        store,
        train_sample_indices,
    )

    module = TemporalLightningModule(
        fit_model,
        training_config=training_config,
        prediction_contract=prediction_contract,
        prediction_training_state=prediction_training_state,
    )
    monitor = prediction_contract.checkpoint_monitor
    mode = "max" if prediction_contract.direction == "maximize" else "min"
    checkpoint_callback = ModelCheckpoint(
        dirpath=artifact_dir / "checkpoints",
        filename=f"epoch={{epoch:02d}}-{monitor}={{{monitor}:.6f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=training_config.early_stopping.patience,
        min_delta=training_config.early_stopping.min_delta,
    )
    accelerator, devices = _trainer_device_args(training_config.device)
    callbacks: list[L.Callback] = [
        checkpoint_callback,
        early_stopping,
        ReporterProgressCallback(
            reporter,
            max_epochs=training_config.max_epochs,
            log_every_n_steps=training_config.log_every_n_steps,
            prediction_contract=prediction_contract,
        ),
    ]
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=training_config.max_epochs,
        callbacks=callbacks,
        deterministic=training_config.deterministic,
        gradient_clip_val=training_config.gradient_clip_norm,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=training_config.log_every_n_steps,
        num_sanity_val_steps=0,
        default_root_dir=str(artifact_dir),
        precision=cast(Any, precision),
    )
    trainer.fit(module, datamodule=data_module)

    if checkpoint_callback.best_model_path:
        state = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        module.load_state_dict(state["state_dict"])
    trained_model = _unwrap_compiled_model(module.model)
    if trained_model is not model:
        model.load_state_dict(trained_model.state_dict())
    best_epoch = _best_epoch(module.validation_history, prediction_contract)
    return TrainingResult(
        best_epoch=best_epoch,
        train_history=module.train_history,
        validation_history=module.validation_history,
        best_checkpoint_path=(
            Path(checkpoint_callback.best_model_path)
            if checkpoint_callback.best_model_path
            else None
        ),
        resolved_precision=precision,
        compiled=compile_enabled,
        resolved_device=device.type,
        representation_id=data_module.train_dataloader().representation_id,
        storage_mode_id=data_module.train_dataloader().storage_mode_id,
        batch_planner_id=data_module.train_dataloader().batch_planner_id,
        prediction_training_state=prediction_training_state,
    )


def evaluate_model(
    model: torch.nn.Module,
    *,
    prediction_contract: CompiledPredictionContract,
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
    device = resolve_device(training_config.device)
    model.to(device)
    model.eval()
    runtime_context = build_representation_runtime_context(
        device=device,
        batch_size=training_config.batch_size,
    )
    loader = build_prediction_loader(
        store,
        sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        runtime_context=runtime_context,
        seed=training_config.seed,
    )
    task_id = reporter.start_task("evaluate model", total=len(loader), unit="batches")
    accumulator = prediction_contract.create_epoch_accumulator("evaluation")
    with torch.no_grad():
        for batch in loader:
            device_batch = batch.to_device(device)
            outputs = model(**device_batch.model_kwargs())
            _, batch_state = prediction_contract.compute_batch_loss_and_state(
                outputs,
                device_batch.targets,
                training_state=prediction_training_state,
            )
            accumulator.update(batch_state)
            reporter.update_task(task_id, advance=1)
    reporter.finish_task(task_id)
    return accumulator.finalize()
