"""Concrete model fitting and native Lightning artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast
from uuid import UUID

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from ..addresses import artifact_checkpoint_path
from ..config import (
    BaselineSource,
    LstmDefinition,
    Method,
    TrainingDefinition,
    TrainRequest,
    TransformerDefinition,
    TransformerLstmDefinition,
    TuneRequest,
)
from ..min_block_fee import (
    ClassificationLossState,
    MinBlockFeeLoss,
    MinBlockFeeOutput,
    TargetState,
    min_block_fee_loss,
)
from ..study import (
    RetainedResult,
    apply_method,
    materialize_selected_training,
    training_definition_from_method,
)
from ..temporal.features import FeatureState
from ..temporal.history import HistoricalPreparation

_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
    )


class FitDeployment(_FrozenRecord):
    """External host facts consumed by one fit invocation."""

    deterministic: bool | Literal["warn"]
    benchmark: bool
    num_workers: int
    pin_memory: bool
    prefetch_factor: int | None
    persistent_workers: bool
    float32_matmul_precision: Literal["highest", "high"]
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool


class ArtifactAssociation(_FrozenRecord):
    """Scientific facts embedded in one native Lightning artifact."""

    request: TrainRequest
    feature_state: FeatureState
    target_state: TargetState
    classification_state: ClassificationLossState | None = None
    study_result_index: _NonNegativeInt | None = None
    method: Method | None = None

    @model_validator(mode="after")
    def validate_association(self) -> Self:
        source = self.request.source
        if isinstance(source, BaselineSource):
            if self.study_result_index is not None or self.method is not None:
                raise ValueError("baseline artifacts cannot contain selected Study fields")
            definition = source.training_definition
        else:
            if self.study_result_index is None or self.method is None:
                raise ValueError("selected Study artifacts require result index and Method")
            if self.study_result_index != source.study_result_index:
                raise ValueError("artifact Study result index must match the TrainRequest")
            definition = training_definition_from_method(source.experiment, self.method)
        _validate_state_association(
            definition,
            self.feature_state,
            self.classification_state,
        )
        return self


class _CandidateAssociation(_FrozenRecord):
    request: TuneRequest
    method: Method
    feature_state: FeatureState
    target_state: TargetState
    classification_state: ClassificationLossState | None = None

    @model_validator(mode="after")
    def validate_association(self) -> Self:
        _validate_state_association(
            apply_method(self.request, self.method),
            self.feature_state,
            self.classification_state,
        )
        return self


_Association = ArtifactAssociation | _CandidateAssociation


def _validate_state_association(
    definition: TrainingDefinition,
    feature_state: FeatureState,
    classification_state: ClassificationLossState | None,
) -> None:
    experiment = definition.experiment
    if len(feature_state.means) != len(experiment.ordered_features):
        raise ValueError("feature state width must match the ordered features")

    weighting = experiment.loss.classification_weighting
    if weighting == "unweighted":
        if classification_state is not None:
            raise ValueError("unweighted classification requires no classification state")
        return
    if classification_state is None:
        raise ValueError("corrected classification requires classification state")
    if len(classification_state.class_support) != experiment.horizon_blocks:
        raise ValueError("classification support width must match the horizon")


def _training_definition(association: _Association) -> TrainingDefinition:
    if isinstance(association, _CandidateAssociation):
        return apply_method(association.request, association.method)
    source = association.request.source
    if isinstance(source, BaselineSource):
        return source.training_definition
    return training_definition_from_method(
        source.experiment,
        cast(Method, association.method),
    )


def _json_association(association: _Association) -> dict[str, object]:
    return association.model_dump(mode="json", exclude_none=True)


def _hydrate_association(raw: object) -> _Association:
    encoded = json.dumps(raw, allow_nan=False)
    match raw:
        case {"request": {"workflow": "train"}}:
            return ArtifactAssociation.model_validate_json(encoded, strict=True)
        case {"request": {"workflow": "tune"}}:
            return _CandidateAssociation.model_validate_json(encoded, strict=True)
        case _:
            raise ValueError("checkpoint association must contain one train or tune request")


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


def _require_inputs(
    inputs: torch.Tensor,
    *,
    context_blocks: int,
    feature_count: int,
) -> None:
    if inputs.ndim != 3:
        raise ValueError("model inputs must have shape [B, C, F]")
    if inputs.shape[1:] != (context_blocks, feature_count):
        raise ValueError("model input trailing shape must match request C and F")
    if inputs.dtype != torch.float32:
        raise TypeError("model inputs must have dtype float32")


class _LstmModel(nn.Module):
    def __init__(
        self,
        definition: LstmDefinition,
        *,
        context_blocks: int,
        feature_count: int,
        actions: int,
    ) -> None:
        super().__init__()
        self.context_blocks = context_blocks
        self.feature_count = feature_count
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
        _require_inputs(
            inputs,
            context_blocks=self.context_blocks,
            feature_count=self.feature_count,
        )
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


class _TransformerModel(nn.Module):
    def __init__(
        self,
        definition: TransformerDefinition,
        *,
        context_blocks: int,
        feature_count: int,
        actions: int,
    ) -> None:
        super().__init__()
        self.context_blocks = context_blocks
        self.feature_count = feature_count
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
        self.heads = _Heads(
            definition.model_width,
            definition.head_hidden,
            actions,
            definition.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        _require_inputs(
            inputs,
            context_blocks=self.context_blocks,
            feature_count=self.feature_count,
        )
        projected = self.projection(inputs)
        positions = cast(torch.Tensor, self.positions).to(dtype=projected.dtype)
        encoded = self.encoder(projected + torch.unsqueeze(positions, 0))
        return self.heads(encoded[:, -1])


class _TransformerLstmModel(nn.Module):
    def __init__(
        self,
        definition: TransformerLstmDefinition,
        *,
        context_blocks: int,
        feature_count: int,
        actions: int,
    ) -> None:
        super().__init__()
        self.context_blocks = context_blocks
        self.feature_count = feature_count
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
        self.lstm = nn.LSTM(
            input_size=definition.model_width,
            hidden_size=definition.lstm_hidden,
            num_layers=definition.lstm_layers,
            dropout=definition.dropout if definition.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.heads = _Heads(
            definition.lstm_hidden,
            definition.head_hidden,
            actions,
            definition.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        _require_inputs(
            inputs,
            context_blocks=self.context_blocks,
            feature_count=self.feature_count,
        )
        projected = self.projection(inputs)
        positions = cast(torch.Tensor, self.positions).to(dtype=projected.dtype)
        encoded = self.encoder(projected + torch.unsqueeze(positions, 0))
        sequence, _ = self.lstm(encoded)
        return self.heads(sequence[:, -1])


class _FitModule(pl.LightningModule):
    def __init__(self, association: dict[str, object]) -> None:
        super().__init__()
        self.association = _hydrate_association(association)
        self.definition = _training_definition(self.association)
        self.validation_total = MeanMetric(
            nan_strategy="disable",
            sync_on_compute=False,
        ).set_dtype(torch.float64)
        self.save_hyperparameters(
            {"association": _json_association(self.association)},
            logger=False,
        )

        experiment = self.definition.experiment
        model = self.definition.model
        common = {
            "context_blocks": experiment.context_blocks,
            "feature_count": len(experiment.ordered_features),
            "actions": experiment.horizon_blocks,
        }
        match model:
            case LstmDefinition():
                self.model = _LstmModel(model, **common)
            case TransformerDefinition():
                self.model = _TransformerModel(model, **common)
            case TransformerLstmDefinition():
                self.model = _TransformerLstmModel(model, **common)

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        return self.model(inputs)

    def on_fit_start(self) -> None:
        self.validation_total.set_dtype(torch.float64)

    def _loss(self, batch: Mapping[str, torch.Tensor]) -> MinBlockFeeLoss:
        return min_block_fee_loss(
            self(batch["inputs"]),
            label=batch["label"],
            target=batch["target"],
            loss_definition=self.definition.experiment.loss,
            classification_state=self.association.classification_state,
        )

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        loss = self._loss(batch).mean_total
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss must be finite")
        return loss

    def validation_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        del batch_idx
        total = self._loss(batch).total_by_origin.detach().to(torch.float64)
        self.validation_total.update(total)

    def on_validation_epoch_end(self) -> None:
        window = self.definition.experiment.validation_window
        validation_size = window.last_parent_block - window.first_parent_block + 1
        weight = cast(Any, self.validation_total.weight)
        if int(weight.item()) != validation_size:
            raise RuntimeError("validation metric count must equal the validation dataset size")
        mean = self.validation_total.compute()
        if not torch.isfinite(mean):
            raise FloatingPointError("complete validation loss must be finite")
        self.log(
            "validation_total_loss",
            self.validation_total,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=False,
        )

    def configure_optimizers(self) -> torch.optim.AdamW:
        optimizer = self.definition.optimizer
        return torch.optim.AdamW(
            self.parameters(),
            lr=optimizer.learning_rate,
            weight_decay=optimizer.weight_decay,
        )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        del gradient_clip_val, gradient_clip_algorithm
        parameters = (
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        )
        authored_norm = self.definition.fit.gradient_clip_norm
        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=math.inf if authored_norm == 0 else authored_norm,
            error_if_nonfinite=True,
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        hyperparameters = cast(Mapping[str, object], checkpoint["hyper_parameters"])
        loaded = _hydrate_association(hyperparameters["association"])
        if loaded != self.association:
            raise ValueError("checkpoint association does not match the requested fit")


@dataclass(frozen=True, slots=True)
class _FitOutcome:
    best_checkpoint: Path
    objective: float
    selected_epoch: int
    completed_epochs: int


def _configure_numerical_policy(deployment: FitDeployment) -> None:
    torch.set_float32_matmul_precision(deployment.float32_matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = deployment.cuda_matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = deployment.cudnn_allow_tf32


def _loaders(
    prepared: HistoricalPreparation,
    definition: TrainingDefinition,
    deployment: FitDeployment,
    generator: torch.Generator,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    common = {
        "batch_size": definition.training_batch,
        "drop_last": False,
        "num_workers": deployment.num_workers,
        "pin_memory": deployment.pin_memory,
        "prefetch_factor": deployment.prefetch_factor,
        "persistent_workers": deployment.persistent_workers,
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
    fit = definition.fit
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
        filename="last",
        save_top_k=1,
        save_weights_only=False,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    return early_stopping, best, last


def _selected_epoch(best_checkpoint: Path) -> int:
    prefix = "best-"
    stem = best_checkpoint.stem
    if not stem.startswith(prefix):
        raise RuntimeError("best checkpoint does not use the controlled filename")
    return int(stem[len(prefix) :]) + 1


def _fit(
    association: _Association,
    prepared: HistoricalPreparation,
    scratch: Path,
    deployment: FitDeployment,
) -> _FitOutcome:
    definition = _training_definition(association)
    scratch.mkdir(parents=True, exist_ok=True)
    _configure_numerical_policy(deployment)
    pl.seed_everything(definition.fit.seed, workers=True)
    generator = torch.Generator(device="cpu").manual_seed(definition.fit.seed)

    module = _FitModule(_json_association(association))
    training_loader, validation_loader = _loaders(
        prepared,
        definition,
        deployment,
        generator,
    )
    early_stopping, best, last = _callbacks(scratch, definition)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        max_epochs=definition.fit.max_epochs,
        check_val_every_n_epoch=definition.fit.validate_every_completed_epoch,
        accumulate_grad_batches=definition.fit.accumulation,
        gradient_clip_val=definition.fit.gradient_clip_norm,
        gradient_clip_algorithm="norm",
        deterministic=deployment.deterministic,
        benchmark=deployment.benchmark,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[early_stopping, best, last],
    )
    last_checkpoint = scratch / "last.ckpt"
    if last_checkpoint.exists():
        trainer.fit(
            module,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
            ckpt_path=last_checkpoint,
            weights_only=True,
        )
    else:
        trainer.fit(
            module,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
        )

    best_checkpoint = Path(best.best_model_path)
    score = best.best_model_score
    if score is None:
        raise RuntimeError("fit completed without a best validation objective")
    return _FitOutcome(
        best_checkpoint=best_checkpoint,
        objective=float(score),
        selected_epoch=_selected_epoch(best_checkpoint),
        completed_epochs=trainer.current_epoch,
    )


def train(
    request: TrainRequest,
    prepared: HistoricalPreparation,
    storage_root: Path,
    deployment: FitDeployment,
) -> None:
    source = request.source
    if isinstance(source, BaselineSource):
        association = ArtifactAssociation(
            request=request,
            feature_state=prepared.feature_state,
            target_state=prepared.target_state,
            classification_state=prepared.classification_state,
        )
    else:
        selected = materialize_selected_training(storage_root, source)
        association = ArtifactAssociation(
            request=request,
            feature_state=prepared.feature_state,
            target_state=prepared.target_state,
            classification_state=prepared.classification_state,
            study_result_index=selected.study_result_index,
            method=selected.method,
        )

    scratch = storage_root / "artifacts" / f".{request.artifact_id}"
    outcome = _fit(association, prepared, scratch, deployment)
    canonical = artifact_checkpoint_path(storage_root, request.artifact_id)
    if canonical.exists():
        raise FileExistsError(canonical)
    outcome.best_checkpoint.rename(canonical)


def _run_candidate(
    request: TuneRequest,
    method: Method,
    prepared: HistoricalPreparation,
    candidate_scratch: Path,
    deployment: FitDeployment,
) -> RetainedResult:
    association = _CandidateAssociation(
        request=request,
        method=method,
        feature_state=prepared.feature_state,
        target_state=prepared.target_state,
        classification_state=prepared.classification_state,
    )
    outcome = _fit(association, prepared, candidate_scratch, deployment)
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


__all__ = ["ArtifactAssociation", "FitDeployment", "load_artifact", "train"]
