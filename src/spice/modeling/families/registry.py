"""Direct model-family dispatch for the fixed in-repo model set."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import optuna

from ...core.specs import (
    coerce_spec_config,
    lookup_local_spec,
    require_spec_config,
    require_spec_config_from_table,
)
from ...prediction import PredictionOutputSpec
from ..models import TemporalModel
from .base import (
    ModelConfig,
    ModelTuningSpaceConfig,
    TunableFieldSpec,
    TunedModelParams,
    TunedScalar,
)

ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig)
ModelTuningSpaceT = TypeVar("ModelTuningSpaceT", bound=ModelTuningSpaceConfig)
ModelTunedParamsT = TypeVar("ModelTunedParamsT", bound=TunedModelParams)
DeriveTunedValues = Callable[[dict[str, TunedScalar]], Mapping[str, TunedScalar]]


@dataclass(frozen=True, slots=True)
class ModelSpec(Generic[ModelConfigT, ModelTuningSpaceT, ModelTunedParamsT]):
    model_config_type: type[ModelConfigT]
    tuning_space_type: type[ModelTuningSpaceT]
    tuned_params_type: type[ModelTunedParamsT]
    build_model: Callable[[int, PredictionOutputSpec, ModelConfigT], TemporalModel]
    tunable_fields: tuple[TunableFieldSpec, ...] = ()
    derive_tuned_values: DeriveTunedValues | None = None
    validate_tuning_space: Callable[[ModelConfigT, ModelTuningSpaceT], None] | None = None


def _model_spec_loaders() -> dict[str, Callable[[], ModelSpec[Any, Any, Any]]]:
    from .lstm import MODEL_SPEC as lstm_spec
    from .transformer import MODEL_SPEC as transformer_spec
    from .transformer_lstm import MODEL_SPEC as transformer_lstm_spec

    return {
        "lstm": lambda: lstm_spec,
        "transformer": lambda: transformer_spec,
        "transformer_lstm": lambda: transformer_lstm_spec,
    }


def model_spec(model_id: str) -> ModelSpec[Any, Any, Any]:
    return lookup_local_spec(_model_spec_loaders(), model_id, "model.id")()


def coerce_model_config(payload: object) -> ModelConfig[str]:
    return cast(
        ModelConfig[str],
        coerce_spec_config(
            payload,
            owner="model",
            base_config_type=ModelConfig,
            id_label="model.id",
            lookup_spec=model_spec,
            spec_config_type=lambda spec: spec.model_config_type,
        ),
    )


def build_model(
    n_features: int,
    output_spec: PredictionOutputSpec,
    config: ModelConfig[str],
) -> TemporalModel:
    spec = model_spec(config.id)
    concrete_config = require_spec_config_from_table(
        config,
        config_id=config.id,
        lookup_spec=model_spec,
        spec_config_type=lambda entry: entry.model_config_type,
        label="model config",
    )
    return spec.build_model(n_features, output_spec, concrete_config)


def apply_model_tuned_parameters(
    model_config: ModelConfig[str],
    params: TunedModelParams,
) -> ModelConfig[str]:
    spec = model_spec(model_config.id)
    concrete_config = require_spec_config_from_table(
        model_config,
        config_id=model_config.id,
        lookup_spec=model_spec,
        spec_config_type=lambda entry: entry.model_config_type,
        label="model config",
    )
    concrete_params = require_spec_config(
        params,
        spec.tuned_params_type,
        "tuned model params",
    )
    updates = concrete_params.model_dump(exclude={"id"}, exclude_none=True, mode="python")
    config_payload = concrete_config.model_dump(mode="python", exclude_none=True)
    return spec.model_config_type.model_validate({**config_payload, **updates})


def sample_model_tuned_parameters(
    trial: optuna.Trial,
    tuning_space: ModelTuningSpaceConfig[str],
) -> TunedModelParams[str] | None:
    spec = model_spec(tuning_space.id)
    concrete_space = require_spec_config_from_table(
        tuning_space,
        config_id=tuning_space.id,
        lookup_spec=model_spec,
        spec_config_type=lambda entry: entry.tuning_space_type,
        label="tuning_space.model",
    )
    sampled: dict[str, TunedScalar] = {}
    for field in spec.tunable_fields:
        candidates = getattr(concrete_space, field.name)
        if candidates is None:
            continue
        sampled[field.name] = field.coerce_sample(
            trial.suggest_categorical(field.parameter_name, candidates)
        )
    if not sampled:
        return None
    values: Mapping[str, TunedScalar]
    if spec.derive_tuned_values is None:
        values = sampled
    else:
        values = spec.derive_tuned_values(sampled)
    return spec.tuned_params_type.model_validate({"id": concrete_space.id, **values})
