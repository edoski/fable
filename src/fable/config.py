"""Strict request and definition values."""

from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

from pydantic import UUID4, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

_PositiveInt: TypeAlias = Annotated[int, Field(strict=True, gt=0)]
_NonNegativeInt: TypeAlias = Annotated[int, Field(strict=True, ge=0)]
_PositiveFloat: TypeAlias = Annotated[
    float,
    Field(strict=True, gt=0.0, allow_inf_nan=False),
]
_NonNegativeFloat: TypeAlias = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]
_Dropout: TypeAlias = Annotated[
    float,
    Field(strict=True, ge=0.0, lt=1.0, allow_inf_nan=False),
]
_FeatureName: TypeAlias = Annotated[str, Field(strict=True, min_length=1)]


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_transformer_dimensions(model_width: int, attention_heads: int) -> None:
    if model_width % 2:
        raise ValueError("model_width must be even for sinusoidal positions")
    if model_width % attention_heads:
        raise ValueError("model_width must be divisible by attention_heads")


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")


class CorpusDefinition(_FrozenRecord):
    chain_id: _PositiveInt
    first_block: _NonNegativeInt
    last_block: _NonNegativeInt

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.last_block < self.first_block:
            raise ValueError("last_block must not precede first_block")
        return self


class CorpusRequest(_FrozenRecord):
    corpus_id: UUID4
    definition: CorpusDefinition


class BlockWindow(_FrozenRecord):
    first_parent_block: _NonNegativeInt
    last_parent_block: _NonNegativeInt

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.last_parent_block < self.first_parent_block:
            raise ValueError("last_parent_block must not precede first_parent_block")
        return self


class ExperimentSemantics(_FrozenRecord):
    training_window: BlockWindow
    validation_window: BlockWindow
    context_blocks: _PositiveInt
    horizon_blocks: _PositiveInt
    ordered_features: Annotated[tuple[_FeatureName, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if (
            self.training_window.last_parent_block + self.horizon_blocks
            >= self.validation_window.first_parent_block
        ):
            raise ValueError("validation_window must follow complete training outcomes")
        _require_unique("ordered_features", self.ordered_features)
        return self


class LstmDefinition(_FrozenRecord):
    family: Literal["lstm"]
    hidden: _PositiveInt
    layers: _PositiveInt
    head_hidden: _PositiveInt
    dropout: _Dropout


class TransformerDefinition(_FrozenRecord):
    family: Literal["transformer"]
    model_width: _PositiveInt
    attention_heads: _PositiveInt
    transformer_layers: _PositiveInt
    feedforward_width: _PositiveInt
    head_hidden: _PositiveInt
    dropout: _Dropout

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        _validate_transformer_dimensions(self.model_width, self.attention_heads)
        return self


class TransformerLstmDefinition(_FrozenRecord):
    family: Literal["transformer_lstm"]
    model_width: _PositiveInt
    attention_heads: _PositiveInt
    transformer_layers: _PositiveInt
    feedforward_width: _PositiveInt
    lstm_hidden: _PositiveInt
    lstm_layers: _PositiveInt
    head_hidden: _PositiveInt
    dropout: _Dropout

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        _validate_transformer_dimensions(self.model_width, self.attention_heads)
        return self


ModelDefinition: TypeAlias = Annotated[
    LstmDefinition | TransformerDefinition | TransformerLstmDefinition,
    Field(discriminator="family"),
]


class FitMethod(_FrozenRecord):
    learning_rate: _PositiveFloat
    weight_decay: _NonNegativeFloat
    accumulation: _PositiveInt
    gradient_clip_norm: _NonNegativeFloat
    seed: _NonNegativeInt
    max_epochs: _PositiveInt
    validate_every_completed_epoch: _PositiveInt
    patience: _NonNegativeInt
    min_delta: _NonNegativeFloat


class Method(_FrozenRecord):
    model: ModelDefinition
    fit: FitMethod


class TrainingDefinition(_FrozenRecord):
    experiment: ExperimentSemantics
    method: Method


class BaselineSource(_FrozenRecord):
    kind: Literal["baseline"]
    corpus_id: UUID4
    training_definition: TrainingDefinition

    @property
    def experiment(self) -> ExperimentSemantics:
        return self.training_definition.experiment


class SelectedStudySource(_FrozenRecord):
    kind: Literal["selected_study"]
    corpus_id: UUID4
    study_id: UUID4
    study_result_index: _NonNegativeInt
    experiment: ExperimentSemantics


TrainingSource: TypeAlias = Annotated[
    BaselineSource | SelectedStudySource,
    Field(discriminator="kind"),
]


class TrainRequest(_FrozenRecord):
    workflow: Literal["train"]
    artifact_id: UUID4
    source: TrainingSource


class TuneRequest(_FrozenRecord):
    workflow: Literal["tune"]
    study_id: UUID4
    corpus_id: UUID4
    experiment: ExperimentSemantics
    methods: Annotated[tuple[Method, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_methods(self) -> Self:
        _require_unique("methods", self.methods)
        if len({method.model.family for method in self.methods}) != 1:
            raise ValueError("methods must use one model family")
        return self


class EvaluateRequest(_FrozenRecord):
    workflow: Literal["evaluate"]
    evaluation_id: UUID4
    artifact_id: UUID4
    corpus_id: UUID4
    testing_window: BlockWindow


WorkflowRequest: TypeAlias = Annotated[
    TrainRequest | EvaluateRequest,
    Field(discriminator="workflow"),
]

WORKFLOW_REQUEST_ADAPTER = TypeAdapter(WorkflowRequest)
