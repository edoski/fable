"""DISPOSABLE PROTOTYPE: typed model construction and Method application.

Question: can a TuneRequest-owned MethodSpace plus complete three-family Method values
replace the current registry, lazy loader, Optuna sampling, and partial overlay path?

Synthetic values only. This is planning evidence, not production implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Self, TypeAlias, assert_never

from pydantic import UUID4, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

PositiveInt: TypeAlias = Annotated[int, Field(strict=True, gt=0)]
PositiveFloat: TypeAlias = Annotated[
    float,
    Field(strict=True, gt=0.0, allow_inf_nan=False),
]
NonNegativeFloat: TypeAlias = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]
Dropout: TypeAlias = Annotated[
    float,
    Field(strict=True, ge=0.0, lt=1.0, allow_inf_nan=False),
]
FeatureName: TypeAlias = Annotated[str, Field(strict=True, min_length=1)]


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
    )


def _validate_transformer_dimensions(model_width: int, attention_heads: int) -> None:
    if model_width % 2 != 0:
        raise ValueError("model_width must be even for sinusoidal positions")
    if model_width % attention_heads != 0:
        raise ValueError("model_width must be divisible by attention_heads")


class LstmDefinition(_FrozenRecord):
    family: Literal["lstm"]
    projection: PositiveInt
    hidden: PositiveInt
    layers: PositiveInt
    head_hidden: PositiveInt
    dropout: Dropout


class TransformerDefinition(_FrozenRecord):
    family: Literal["transformer"]
    model_width: PositiveInt
    attention_heads: PositiveInt
    transformer_layers: PositiveInt
    feedforward_width: PositiveInt
    head_hidden: PositiveInt
    dropout: Dropout

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        _validate_transformer_dimensions(self.model_width, self.attention_heads)
        return self


class TransformerLstmDefinition(_FrozenRecord):
    family: Literal["transformer_lstm"]
    model_width: PositiveInt
    attention_heads: PositiveInt
    transformer_layers: PositiveInt
    feedforward_width: PositiveInt
    lstm_hidden: PositiveInt
    lstm_layers: PositiveInt
    head_hidden: PositiveInt
    dropout: Dropout

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        _validate_transformer_dimensions(self.model_width, self.attention_heads)
        return self


ModelDefinition: TypeAlias = Annotated[
    LstmDefinition | TransformerDefinition | TransformerLstmDefinition,
    Field(discriminator="family"),
]


class LstmCapacity(_FrozenRecord):
    projection: PositiveInt
    hidden: PositiveInt
    layers: PositiveInt
    head_hidden: PositiveInt


class TransformerCapacity(_FrozenRecord):
    model_width: PositiveInt
    attention_heads: PositiveInt
    transformer_layers: PositiveInt
    feedforward_width: PositiveInt
    head_hidden: PositiveInt

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        _validate_transformer_dimensions(self.model_width, self.attention_heads)
        return self


class TransformerLstmCapacity(_FrozenRecord):
    model_width: PositiveInt
    attention_heads: PositiveInt
    transformer_layers: PositiveInt
    feedforward_width: PositiveInt
    lstm_hidden: PositiveInt
    lstm_layers: PositiveInt
    head_hidden: PositiveInt

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        _validate_transformer_dimensions(self.model_width, self.attention_heads)
        return self


class AdamWMethod(_FrozenRecord):
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat


class FitMethod(_FrozenRecord):
    accumulation: Literal[1]
    gradient_clip_norm: PositiveFloat
    scheduler: Literal["none"]
    seed: Literal[2026]
    max_epochs: Literal[36]
    validate_every_completed_epoch: Literal[1]
    patience: Literal[8]
    min_delta: NonNegativeFloat
    improvement: Literal["strict_lower"]
    restore: Literal["earliest_best"]
    minimum_epoch_floor: Literal[False]

    @model_validator(mode="after")
    def validate_fixed_floats(self) -> Self:
        if self.gradient_clip_norm != 1.0:
            raise ValueError("gradient_clip_norm must equal 1.0")
        if self.min_delta != 0.0:
            raise ValueError("min_delta must equal 0.0")
        return self


class _MethodFields(_FrozenRecord):
    dropout: Dropout
    optimizer: AdamWMethod
    training_batch: PositiveInt
    fit: FitMethod


class LstmMethod(_MethodFields):
    family: Literal["lstm"]
    capacity: LstmCapacity


class TransformerMethod(_MethodFields):
    family: Literal["transformer"]
    capacity: TransformerCapacity


class TransformerLstmMethod(_MethodFields):
    family: Literal["transformer_lstm"]
    capacity: TransformerLstmCapacity


Method: TypeAlias = Annotated[
    LstmMethod | TransformerMethod | TransformerLstmMethod,
    Field(discriminator="family"),
]
METHOD_ADAPTER = TypeAdapter(Method)


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")


class _MethodSpaceFields(_FrozenRecord):
    dropouts: Annotated[tuple[Dropout, ...], Field(min_length=1)]
    learning_rates: Annotated[tuple[PositiveFloat, ...], Field(min_length=1)]
    weight_decays: Annotated[tuple[NonNegativeFloat, ...], Field(min_length=1)]
    training_batches: Annotated[tuple[PositiveInt, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_scalar_leaves(self) -> Self:
        _require_unique("dropouts", self.dropouts)
        _require_unique("learning_rates", self.learning_rates)
        _require_unique("weight_decays", self.weight_decays)
        _require_unique("training_batches", self.training_batches)
        return self


class LstmMethodSpace(_MethodSpaceFields):
    family: Literal["lstm"]
    capacities: Annotated[tuple[LstmCapacity, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_capacities(self) -> Self:
        _require_unique("capacities", self.capacities)
        return self


class TransformerMethodSpace(_MethodSpaceFields):
    family: Literal["transformer"]
    capacities: Annotated[tuple[TransformerCapacity, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_capacities(self) -> Self:
        _require_unique("capacities", self.capacities)
        return self


class TransformerLstmMethodSpace(_MethodSpaceFields):
    family: Literal["transformer_lstm"]
    capacities: Annotated[tuple[TransformerLstmCapacity, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_capacities(self) -> Self:
        _require_unique("capacities", self.capacities)
        return self


MethodSpace: TypeAlias = Annotated[
    LstmMethodSpace | TransformerMethodSpace | TransformerLstmMethodSpace,
    Field(discriminator="family"),
]


class OriginWindow(_FrozenRecord):
    role: Literal["training", "validation"]
    first_parent_block: PositiveInt
    last_parent_block: PositiveInt

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.last_parent_block < self.first_parent_block:
            raise ValueError("last_parent_block must not precede first_parent_block")
        return self


class ExperimentSemantics(_FrozenRecord):
    training_window: OriginWindow
    validation_window: OriginWindow
    context_blocks: PositiveInt
    horizon_blocks: PositiveInt
    ordered_features: Annotated[tuple[FeatureName, ...], Field(min_length=1)]
    classification_loss: Literal["unweighted", "corrected_inverse_frequency"]

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if self.training_window.role != "training":
            raise ValueError("training_window must carry role='training'")
        if self.validation_window.role != "validation":
            raise ValueError("validation_window must carry role='validation'")
        if self.validation_window.first_parent_block <= self.training_window.last_parent_block:
            raise ValueError("validation_window must follow training_window")
        _require_unique("ordered_features", self.ordered_features)
        return self


class StudyDefinition(_FrozenRecord):
    experiment: ExperimentSemantics
    method_space: MethodSpace


class TuneRequest(_FrozenRecord):
    workflow: Literal["tune"]
    study_id: UUID4
    corpus_id: UUID4
    study_definition: StudyDefinition


TUNE_REQUEST_ADAPTER = TypeAdapter(TuneRequest)


class TrainingDefinition(_FrozenRecord):
    experiment: ExperimentSemantics
    model: ModelDefinition
    optimizer: AdamWMethod
    training_batch: PositiveInt
    fit: FitMethod

    @property
    def input_features(self) -> int:
        return len(self.experiment.ordered_features)

    @property
    def context_blocks(self) -> int:
        return self.experiment.context_blocks

    @property
    def action_count(self) -> int:
        return self.experiment.horizon_blocks


class MethodNotApprovedError(ValueError):
    pass


def _require_common_method(space: _MethodSpaceFields, method: _MethodFields) -> None:
    if method.dropout not in space.dropouts:
        raise MethodNotApprovedError("dropout is outside the MethodSpace")
    if method.optimizer.learning_rate not in space.learning_rates:
        raise MethodNotApprovedError("learning_rate is outside the MethodSpace")
    if method.optimizer.weight_decay not in space.weight_decays:
        raise MethodNotApprovedError("weight_decay is outside the MethodSpace")
    if method.training_batch not in space.training_batches:
        raise MethodNotApprovedError("training_batch is outside the MethodSpace")


def _model_from_approved_method(space: MethodSpace, method: Method) -> ModelDefinition:
    match space, method:
        case LstmMethodSpace(), LstmMethod():
            if method.capacity not in space.capacities:
                raise MethodNotApprovedError("LSTM capacity is outside the MethodSpace")
            _require_common_method(space, method)
            return LstmDefinition(
                family="lstm",
                projection=method.capacity.projection,
                hidden=method.capacity.hidden,
                layers=method.capacity.layers,
                head_hidden=method.capacity.head_hidden,
                dropout=method.dropout,
            )
        case TransformerMethodSpace(), TransformerMethod():
            if method.capacity not in space.capacities:
                raise MethodNotApprovedError("Transformer capacity is outside the MethodSpace")
            _require_common_method(space, method)
            return TransformerDefinition(
                family="transformer",
                model_width=method.capacity.model_width,
                attention_heads=method.capacity.attention_heads,
                transformer_layers=method.capacity.transformer_layers,
                feedforward_width=method.capacity.feedforward_width,
                head_hidden=method.capacity.head_hidden,
                dropout=method.dropout,
            )
        case TransformerLstmMethodSpace(), TransformerLstmMethod():
            if method.capacity not in space.capacities:
                raise MethodNotApprovedError(
                    "Transformer-LSTM capacity is outside the MethodSpace"
                )
            _require_common_method(space, method)
            return TransformerLstmDefinition(
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
            raise MethodNotApprovedError(
                f"Method family {method.family!r} does not match MethodSpace {space.family!r}"
            )


def apply_method(
    request: TuneRequest,
    method: Method,
) -> TrainingDefinition:
    """Use the TuneRequest-owned MethodSpace to compose a TrainingDefinition."""

    validated_request = TUNE_REQUEST_ADAPTER.validate_python(request)
    validated_method = METHOD_ADAPTER.validate_python(method)
    return TrainingDefinition(
        experiment=validated_request.study_definition.experiment,
        model=_model_from_approved_method(
            validated_request.study_definition.method_space,
            validated_method,
        ),
        optimizer=validated_method.optimizer,
        training_batch=validated_method.training_batch,
        fit=validated_method.fit,
    )


@dataclass(frozen=True, slots=True)
class LstmConstruction:
    family: Literal["lstm"]
    input_features: int
    context_blocks: int
    action_count: int
    definition: LstmDefinition


@dataclass(frozen=True, slots=True)
class TransformerConstruction:
    family: Literal["transformer"]
    input_features: int
    context_blocks: int
    action_count: int
    definition: TransformerDefinition


@dataclass(frozen=True, slots=True)
class TransformerLstmConstruction:
    family: Literal["transformer_lstm"]
    input_features: int
    context_blocks: int
    action_count: int
    definition: TransformerLstmDefinition


BuiltModel: TypeAlias = (
    LstmConstruction | TransformerConstruction | TransformerLstmConstruction
)


def construct_fit_module(definition: TrainingDefinition) -> BuiltModel:
    """Prototype the sole exhaustive match inside the Lightning FitModule."""

    validated = TrainingDefinition.model_validate(definition)
    common = {
        "input_features": validated.input_features,
        "context_blocks": validated.context_blocks,
        "action_count": validated.action_count,
    }
    match validated.model:
        case LstmDefinition():
            return LstmConstruction(
                family="lstm",
                definition=validated.model,
                **common,
            )
        case TransformerDefinition():
            return TransformerConstruction(
                family="transformer",
                definition=validated.model,
                **common,
            )
        case TransformerLstmDefinition():
            return TransformerLstmConstruction(
                family="transformer_lstm",
                definition=validated.model,
                **common,
            )
    assert_never(validated.model)


FIXED_FIT = FitMethod(
    accumulation=1,
    gradient_clip_norm=1.0,
    scheduler="none",
    seed=2026,
    max_epochs=36,
    validate_every_completed_epoch=1,
    patience=8,
    min_delta=0.0,
    improvement="strict_lower",
    restore="earliest_best",
    minimum_epoch_floor=False,
)

LSTM_METHOD_SPACE = LstmMethodSpace(
    family="lstm",
    capacities=(
        LstmCapacity(projection=128, hidden=256, layers=1, head_hidden=128),
        LstmCapacity(projection=256, hidden=256, layers=2, head_hidden=256),
        LstmCapacity(projection=256, hidden=384, layers=2, head_hidden=256),
    ),
    dropouts=(0.1, 0.2, 0.3),
    learning_rates=(1e-4, 3e-4, 1e-3),
    weight_decays=(0.0, 1e-4, 1e-3),
    training_batches=(32, 64),
)

TRANSFORMER_METHOD_SPACE = TransformerMethodSpace(
    family="transformer",
    capacities=(
        TransformerCapacity(
            model_width=192,
            attention_heads=4,
            transformer_layers=3,
            feedforward_width=384,
            head_hidden=192,
        ),
        TransformerCapacity(
            model_width=256,
            attention_heads=4,
            transformer_layers=4,
            feedforward_width=512,
            head_hidden=256,
        ),
        TransformerCapacity(
            model_width=384,
            attention_heads=8,
            transformer_layers=4,
            feedforward_width=768,
            head_hidden=256,
        ),
    ),
    dropouts=(0.1, 0.2, 0.3),
    learning_rates=(3e-5, 1e-4, 3e-4),
    weight_decays=(0.0, 1e-4, 1e-3),
    training_batches=(32, 64),
)

TRANSFORMER_LSTM_METHOD_SPACE = TransformerLstmMethodSpace(
    family="transformer_lstm",
    capacities=(
        TransformerLstmCapacity(
            model_width=192,
            attention_heads=4,
            transformer_layers=3,
            feedforward_width=384,
            lstm_hidden=192,
            lstm_layers=1,
            head_hidden=192,
        ),
        TransformerLstmCapacity(
            model_width=256,
            attention_heads=4,
            transformer_layers=4,
            feedforward_width=512,
            lstm_hidden=256,
            lstm_layers=1,
            head_hidden=256,
        ),
        TransformerLstmCapacity(
            model_width=384,
            attention_heads=8,
            transformer_layers=4,
            feedforward_width=768,
            lstm_hidden=384,
            lstm_layers=1,
            head_hidden=256,
        ),
    ),
    dropouts=(0.1, 0.2, 0.3),
    learning_rates=(3e-5, 1e-4, 3e-4),
    weight_decays=(0.0, 1e-4, 1e-3),
    training_batches=(32, 64),
)
