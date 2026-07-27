"""Concrete model fitting and native Lightning artifacts."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self, cast
from uuid import UUID

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pydantic import Field, TypeAdapter, model_validator
from torch import nn
from torch.utils.data import DataLoader

from . import _runtime
from .addresses import artifact_checkpoint_path
from .config import (
    BaselineSource,
    LstmDefinition,
    Method,
    TrainingDefinition,
    TrainRequest,
    TransformerDefinition,
    TransformerLstmDefinition,
    TuneRequest,
)
from .min_block_fee import (
    MinBlockFeeLoss,
    MinBlockFeeOutput,
    TargetState,
    min_block_fee_loss,
)
from .records import StrictFrozenRecord
from .study import (
    RetainedResult,
    load_selected_method,
)
from .temporal.features import FeatureState
from .temporal.history import HistoricalPreparation


class ArtifactAssociation(StrictFrozenRecord):
    """Scientific facts embedded in one native Lightning artifact."""

    request: TrainRequest
    feature_state: FeatureState
    target_state: TargetState
    study_result_index: Annotated[int, Field(strict=True, ge=0)] | None = None
    method: Method | None = None

    @property
    def training_definition(self) -> TrainingDefinition:
        source = self.request.source
        if isinstance(source, BaselineSource):
            return source.training_definition
        return TrainingDefinition(
            experiment=source.experiment,
            method=cast(Method, self.method),
        )

    @model_validator(mode="after")
    def validate_association(self) -> Self:
        source = self.request.source
        if isinstance(source, BaselineSource):
            if self.study_result_index is not None or self.method is not None:
                raise ValueError("baseline artifacts cannot contain selected Study fields")
        else:
            if self.study_result_index is None or self.method is None:
                raise ValueError("selected Study artifacts require result index and Method")
            if self.study_result_index != source.study_result_index:
                raise ValueError("artifact Study result index must match the TrainRequest")
        if len(self.feature_state.means) != len(
            self.training_definition.experiment.ordered_features
        ):
            raise ValueError("feature state width must match the ordered features")
        return self


class _CandidateAssociation(StrictFrozenRecord):
    request: TuneRequest
    method: Method
    feature_state: FeatureState
    target_state: TargetState

    @property
    def training_definition(self) -> TrainingDefinition:
        return TrainingDefinition(
            experiment=self.request.experiment,
            method=self.method,
        )


_Association = ArtifactAssociation | _CandidateAssociation
_ASSOCIATION_ADAPTER = TypeAdapter(_Association)


def _json_association(association: _Association) -> dict[str, object]:
    return association.model_dump(mode="json", exclude_none=True)


def _hydrate_association(raw: object) -> _Association:
    encoded = json.dumps(raw, allow_nan=False)
    return _ASSOCIATION_ADAPTER.validate_json(encoded, strict=True)


class _Heads(nn.Module):
    def __init__(self, input_width: int, hidden: int, actions: int, dropout: float) -> None:
        super().__init__()
        self.action = _head(input_width, hidden, actions, dropout)
        self.regression = _head(input_width, hidden, 1, dropout)

    def forward(self, state: torch.Tensor) -> MinBlockFeeOutput:
        return MinBlockFeeOutput(
            action_logits=self.action(state),
            minimum_fee_z=self.regression(state).squeeze(-1),
        )


def _head(input_width: int, hidden: int, output_width: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, output_width),
    )


class _LstmModel(nn.Module):
    def __init__(
        self,
        definition: LstmDefinition,
        *,
        feature_count: int,
        actions: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=definition.hidden,
            num_layers=definition.layers,
            dropout=definition.dropout if definition.layers > 1 else 0.0,
            batch_first=True,
        )
        self.heads = _Heads(
            definition.hidden,
            definition.head_hidden,
            actions,
            definition.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        sequence, _ = self.lstm(inputs)
        return self.heads(sequence[:, -1])


def _sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros(length, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


def _encoder(
    *,
    width: int,
    heads: int,
    feedforward: int,
    layers: int,
    dropout: float,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=layers)
    for encoder_layer in encoder.layers:
        for parameter in encoder_layer.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
    return encoder


class _TransformerBackbone(nn.Module):
    def __init__(
        self,
        definition: TransformerDefinition | TransformerLstmDefinition,
        *,
        context_blocks: int,
        feature_count: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_count, definition.model_width)
        self.register_buffer(
            "positions",
            _sinusoidal_positions(context_blocks, definition.model_width),
            persistent=False,
        )
        self.encoder: nn.TransformerEncoder = _encoder(
            width=definition.model_width,
            heads=definition.attention_heads,
            feedforward=definition.feedforward_width,
            layers=definition.transformer_layers,
            dropout=definition.dropout,
        )

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.projection(inputs)
        positions = cast(torch.Tensor, self.positions).to(dtype=projected.dtype)
        return self.encoder(projected + torch.unsqueeze(positions, 0))


class _TransformerModel(_TransformerBackbone):
    def __init__(
        self,
        definition: TransformerDefinition,
        *,
        context_blocks: int,
        feature_count: int,
        actions: int,
    ) -> None:
        super().__init__(definition, context_blocks=context_blocks, feature_count=feature_count)
        self.heads = _Heads(
            definition.model_width, definition.head_hidden, actions, definition.dropout
        )

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        return self.heads(self._encode(inputs)[:, -1])


class _TransformerLstmModel(_TransformerBackbone):
    def __init__(
        self,
        definition: TransformerLstmDefinition,
        *,
        context_blocks: int,
        feature_count: int,
        actions: int,
    ) -> None:
        super().__init__(definition, context_blocks=context_blocks, feature_count=feature_count)
        self.lstm = nn.LSTM(
            input_size=definition.model_width,
            hidden_size=definition.lstm_hidden,
            num_layers=definition.lstm_layers,
            dropout=definition.dropout if definition.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.heads = _Heads(
            definition.lstm_hidden, definition.head_hidden, actions, definition.dropout
        )

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        sequence, _ = self.lstm(self._encode(inputs))
        return self.heads(sequence[:, -1])


class _FitModule(pl.LightningModule):
    def __init__(self, association: dict[str, object]) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.association = _hydrate_association(association)
        self.definition = self.association.training_definition

        experiment = self.definition.experiment
        model = self.definition.method.model
        common = {
            "feature_count": len(experiment.ordered_features),
            "actions": experiment.horizon_blocks,
        }
        match model:
            case LstmDefinition():
                self.model = _LstmModel(model, **common)
            case TransformerDefinition():
                self.model = _TransformerModel(
                    model,
                    context_blocks=experiment.context_blocks,
                    **common,
                )
            case TransformerLstmDefinition():
                self.model = _TransformerLstmModel(
                    model,
                    context_blocks=experiment.context_blocks,
                    **common,
                )

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        return self.model(inputs)

    def _loss(self, batch: Mapping[str, torch.Tensor]) -> MinBlockFeeLoss:
        return min_block_fee_loss(
            self(batch["inputs"]),
            label=batch["label"],
            target=batch["target"],
        )

    def _log_epoch_loss(
        self,
        role: Literal["training", "validation"],
        losses: MinBlockFeeLoss,
    ) -> None:
        loss = losses.total_by_origin.mean(dtype=torch.float64)
        self.log(
            f"{role}_total_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=False,
            batch_size=losses.total_by_origin.numel(),
        )

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        losses = self._loss(batch)
        self._log_epoch_loss("training", losses)
        return losses.mean_total

    def validation_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        del batch_idx
        losses = self._loss(batch)
        self._log_epoch_loss("validation", losses)

    def on_validation_epoch_end(self) -> None:
        metric = self.trainer.callback_metrics["validation_total_loss"]
        value = float(metric.detach().cpu().item())
        if not math.isfinite(value):
            raise FloatingPointError("complete validation_total_loss must be finite")
        print(
            f"epoch={self.trainer.current_epoch + 1} validation_total_loss={value}",
            flush=True,
        )

    def configure_optimizers(self) -> torch.optim.AdamW:
        fit = self.definition.method.fit
        return torch.optim.AdamW(
            self.parameters(),
            lr=fit.learning_rate,
            weight_decay=fit.weight_decay,
        )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        del gradient_clip_algorithm
        parameters = (
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        )
        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=gradient_clip_val or math.inf,
            error_if_nonfinite=True,
        )


@dataclass(frozen=True, slots=True)
class _FitOutcome:
    best_checkpoint: Path
    objective: float
    selected_epoch: int
    completed_epochs: int


def _loaders(
    prepared: HistoricalPreparation,
    generator: torch.Generator,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    workers = _runtime.NUM_WORKERS
    common = {
        "batch_size": _runtime.FIT_BATCH_SIZE,
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": _runtime.PIN_MEMORY,
        "prefetch_factor": _runtime.PREFETCH_FACTOR if workers else None,
        "persistent_workers": _runtime.PERSISTENT_WORKERS if workers else False,
    }
    training = DataLoader(
        prepared.training,
        shuffle=True,
        generator=generator,
        **common,
    )
    validation = DataLoader(
        prepared.validation,
        shuffle=False,
        **common,
    )
    return training, validation


def _callbacks(
    scratch: Path,
    definition: TrainingDefinition,
) -> tuple[EarlyStopping, ModelCheckpoint, ModelCheckpoint]:
    fit = definition.method.fit
    early_stopping = EarlyStopping(
        monitor="validation_total_loss",
        mode="min",
        min_delta=fit.min_delta,
        patience=fit.patience,
        strict=True,
        check_finite=False,
        check_on_train_epoch_end=False,
    )
    best = ModelCheckpoint(
        dirpath=scratch,
        filename="best-{epoch:02d}",
        monitor="validation_total_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    last = ModelCheckpoint(
        dirpath=scratch,
        save_top_k=0,
        save_last=True,
        save_weights_only=False,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    return early_stopping, best, last


def _fit(
    association: _Association,
    prepared: HistoricalPreparation,
    scratch: Path,
) -> _FitOutcome:
    definition = association.training_definition
    scratch.mkdir(parents=True, exist_ok=True)
    _runtime.configure_torch()
    fit = definition.method.fit
    pl.seed_everything(fit.seed, workers=True)
    generator = torch.Generator(device="cpu").manual_seed(fit.seed)

    module = _FitModule(_json_association(association))
    training_loader, validation_loader = _loaders(
        prepared,
        generator,
    )
    early_stopping, best, last = _callbacks(scratch, definition)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        max_epochs=fit.max_epochs,
        check_val_every_n_epoch=fit.validate_every_completed_epoch,
        accumulate_grad_batches=fit.accumulation,
        gradient_clip_val=fit.gradient_clip_norm,
        gradient_clip_algorithm="norm",
        deterministic=_runtime.DETERMINISTIC,
        benchmark=_runtime.BENCHMARK,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[early_stopping, best, last],
    )
    last_checkpoint = scratch / "last.ckpt"
    trainer.fit(
        module,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
        ckpt_path=last_checkpoint if last_checkpoint.exists() else None,
    )

    best_checkpoint = Path(best.best_model_path)
    score = best.best_model_score
    if score is None:
        raise RuntimeError("fit completed without a best validation objective")
    return _FitOutcome(
        best_checkpoint=best_checkpoint,
        objective=float(score),
        selected_epoch=int(best_checkpoint.stem.removeprefix("best-")) + 1,
        completed_epochs=trainer.current_epoch,
    )


def _publish_artifact(
    storage_root: Path,
    artifact_id: UUID,
    scratch: Path,
    outcome: _FitOutcome,
) -> None:
    canonical = artifact_checkpoint_path(storage_root, artifact_id)
    completed = canonical.with_name(f".{canonical.name}")
    outcome.best_checkpoint.rename(completed)
    shutil.rmtree(scratch)
    os.link(completed, canonical)
    try:
        completed.unlink()
    except OSError:
        pass


def train(
    request: TrainRequest,
    prepared: HistoricalPreparation,
    storage_root: Path,
) -> None:
    source = request.source
    if isinstance(source, BaselineSource):
        association = ArtifactAssociation(
            request=request,
            feature_state=prepared.feature_state,
            target_state=prepared.target_state,
        )
    else:
        method = load_selected_method(storage_root, source)
        association = ArtifactAssociation(
            request=request,
            feature_state=prepared.feature_state,
            target_state=prepared.target_state,
            study_result_index=source.study_result_index,
            method=method,
        )

    canonical = artifact_checkpoint_path(storage_root, request.artifact_id)
    if canonical.exists():
        raise FileExistsError(canonical)
    scratch = canonical.parent / f".{request.artifact_id}"
    outcome = _fit(association, prepared, scratch)
    _publish_artifact(storage_root, request.artifact_id, scratch, outcome)


def fit_candidate(
    request: TuneRequest,
    method: Method,
    prepared: HistoricalPreparation,
    candidate_scratch: Path,
) -> RetainedResult:
    association = _CandidateAssociation(
        request=request,
        method=method,
        feature_state=prepared.feature_state,
        target_state=prepared.target_state,
    )
    outcome = _fit(association, prepared, candidate_scratch)
    return RetainedResult(
        method=method,
        objective=outcome.objective,
        selected_epoch=outcome.selected_epoch,
        completed_epochs=outcome.completed_epochs,
    )


def load_artifact(
    storage_root: Path,
    artifact_id: UUID,
) -> tuple[ArtifactAssociation, nn.Module]:
    module = _FitModule.load_from_checkpoint(
        artifact_checkpoint_path(storage_root, artifact_id),
        map_location="cpu",
        weights_only=True,
        strict=True,
    )
    association = module.association
    if not isinstance(association, ArtifactAssociation):
        raise ValueError("canonical artifact must contain a TrainRequest association")
    if association.request.artifact_id != artifact_id:
        raise ValueError("embedded artifact ID does not match the requested artifact")
    module.model.eval()
    return association, module.model
