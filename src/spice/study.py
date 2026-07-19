"""Immutable Study publication and selected-training materialization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Self, TypeAlias, assert_never

from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from .addresses import study_json_path
from .config import (
    ExperimentSemantics,
    LstmDefinition,
    LstmMethod,
    LstmMethodSpace,
    Method,
    MethodSpace,
    SelectedStudySource,
    TrainingDefinition,
    TransformerDefinition,
    TransformerLstmDefinition,
    TransformerLstmMethod,
    TransformerLstmMethodSpace,
    TransformerMethod,
    TransformerMethodSpace,
    TuneRequest,
)

__all__ = [
    "RetainedResult",
    "SelectedStudyTraining",
    "Study",
    "apply_method",
    "materialize_selected_training",
    "publish_study",
    "retain_result",
    "training_definition_from_method",
]

_Objective: TypeAlias = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]
_Epoch: TypeAlias = Annotated[int, Field(strict=True, ge=1)]


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
    )


class RetainedResult(_FrozenRecord):
    method: Method
    objective: _Objective
    selected_epoch: _Epoch
    completed_epochs: _Epoch

    @model_validator(mode="after")
    def validate_epochs(self) -> Self:
        if self.selected_epoch > self.completed_epochs:
            raise ValueError("selected_epoch must not exceed completed_epochs")
        if self.completed_epochs > self.method.fit.max_epochs:
            raise ValueError("completed_epochs must not exceed method.fit.max_epochs")
        return self


class Study(_FrozenRecord):
    request: TuneRequest
    trials: Annotated[tuple[RetainedResult, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_methods(self) -> Self:
        for result in self.trials:
            _require_method_in_space(self.request.study_definition.method_space, result.method)
        return self


@dataclass(frozen=True, slots=True)
class SelectedStudyTraining:
    study_result_index: int
    method: Method
    training_definition: TrainingDefinition


def training_definition_from_method(
    experiment: ExperimentSemantics,
    method: Method,
) -> TrainingDefinition:
    """Compose one complete training definition without MethodSpace approval."""

    match method:
        case LstmMethod():
            model = LstmDefinition(
                family="lstm",
                hidden=method.capacity.hidden,
                layers=method.capacity.layers,
                head_hidden=method.capacity.head_hidden,
                dropout=method.dropout,
            )
        case TransformerMethod():
            model = TransformerDefinition(
                family="transformer",
                model_width=method.capacity.model_width,
                attention_heads=method.capacity.attention_heads,
                transformer_layers=method.capacity.transformer_layers,
                feedforward_width=method.capacity.feedforward_width,
                head_hidden=method.capacity.head_hidden,
                dropout=method.dropout,
            )
        case TransformerLstmMethod():
            model = TransformerLstmDefinition(
                family="transformer_lstm",
                model_width=method.capacity.model_width,
                attention_heads=method.capacity.attention_heads,
                transformer_layers=method.capacity.transformer_layers,
                feedforward_width=method.capacity.feedforward_width,
                lstm_hidden=method.capacity.lstm_hidden,
                lstm_layers=method.capacity.lstm_layers,
                head_hidden=method.capacity.head_hidden,
                dropout=method.dropout,
            )
        case _:
            assert_never(method)
    return TrainingDefinition(
        experiment=experiment,
        model=model,
        optimizer=method.optimizer,
        training_batch=method.training_batch,
        fit=method.fit,
    )


def apply_method(request: TuneRequest, method: Method) -> TrainingDefinition:
    """Approve a Method against its TuneRequest and compose its definition."""

    _require_method_in_space(request.study_definition.method_space, method)
    return training_definition_from_method(request.study_definition.experiment, method)


def retain_result(
    storage_root: Path,
    request: TuneRequest,
    result: RetainedResult,
) -> None:
    progress = _progress_path(storage_root, request.study_id)
    if progress.exists():
        current = _load_study(progress)
        if current.request != request:
            raise ValueError("retained result request does not match existing progress")
        study = Study(request=request, trials=(*current.trials, result))
    else:
        study = Study(request=request, trials=(result,))

    progress.parent.mkdir(parents=True, exist_ok=True)
    temporary = progress.with_name(".progress.json.tmp")
    temporary.write_text(study.model_dump_json(), encoding="utf-8")
    os.replace(temporary, progress)


def publish_study(storage_root: Path, study_id: UUID4) -> None:
    progress = _progress_path(storage_root, study_id)
    study = _load_study(progress)
    if study.request.study_id != study_id:
        raise ValueError("progress Study ID does not match requested Study ID")

    canonical = study_json_path(storage_root, study_id)
    if canonical.exists():
        raise FileExistsError(canonical)
    progress.rename(canonical)


def materialize_selected_training(
    storage_root: Path,
    source: SelectedStudySource,
) -> SelectedStudyTraining:
    study = _load_study(study_json_path(storage_root, source.study_id))
    if study.request.study_id != source.study_id:
        raise ValueError("selected source Study ID does not match canonical Study")
    if study.request.corpus_id != source.corpus_id:
        raise ValueError("selected source Corpus ID does not match canonical Study")

    result = study.trials[source.study_result_index]
    return SelectedStudyTraining(
        study_result_index=source.study_result_index,
        method=result.method,
        training_definition=training_definition_from_method(source.experiment, result.method),
    )


def _require_method_in_space(space: MethodSpace, method: Method) -> None:
    match space, method:
        case LstmMethodSpace(), LstmMethod():
            allowed = method in space.methods
        case TransformerMethodSpace(), TransformerMethod():
            allowed = method in space.methods
        case TransformerLstmMethodSpace(), TransformerLstmMethod():
            allowed = method in space.methods
        case _:
            raise ValueError(
                f"Method family {method.family!r} does not match MethodSpace {space.family!r}"
            )

    if not allowed:
        raise ValueError("Method is outside the MethodSpace")


def _progress_path(storage_root: Path, study_id: UUID4) -> Path:
    return storage_root / "studies" / f".{study_id}" / "progress.json"


def _load_study(path: Path) -> Study:
    return Study.model_validate_json(path.read_bytes(), strict=True)
