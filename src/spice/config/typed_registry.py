# pyright: strict

"""Context-free typed loaders for named config groups."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig, coerce_evaluator_config
from ..execution.models import ExecutionSpec
from ..modeling.dataset_builders import (
    DatasetBuilderConfig,
    coerce_dataset_builder_config,
)
from ..modeling.families.base import ModelConfig
from ..modeling.families.registry import coerce_model_config
from ..objectives import ObjectiveConfig, coerce_objective_config
from .groups import load_named_group_payload
from .models import (
    ChainSpec,
    DatasetSpec,
    FeaturesConfig,
    PredictionConfig,
    ProblemSpec,
    ProviderSpec,
    SplitConfig,
    TrainingConfig,
    TuningConfig,
    coerce_features_config,
    coerce_problem_spec,
)
from .surfaces import SurfaceFrame

ConfigModelT = TypeVar("ConfigModelT", bound=BaseModel)
ConfigT = TypeVar("ConfigT")


def load_dataset_spec(name: str) -> DatasetSpec:
    return _load_group_model(name, "dataset", DatasetSpec)


def load_chain_spec(name: str) -> ChainSpec:
    return _load_group_model(name, "chain", ChainSpec)


def load_problem_spec(name: str) -> ProblemSpec:
    return _coerce_group(name, "problem", coerce_problem_spec)


def load_features_config(name: str) -> FeaturesConfig:
    return _coerce_group(name, "features", coerce_features_config)


def load_provider_spec(name: str) -> ProviderSpec:
    return _load_group_model(name, "provider", ProviderSpec)


def load_model_config(name: str) -> ModelConfig[str]:
    return _coerce_group(name, "model", coerce_model_config)


def load_dataset_builder_config(name: str) -> DatasetBuilderConfig:
    return _coerce_group(
        name,
        "dataset_builder",
        coerce_dataset_builder_config,
    )


def load_evaluator_config(name: str) -> EvaluatorConfig:
    return _coerce_group(name, "evaluation", coerce_evaluator_config)


def load_objective_config(name: str) -> ObjectiveConfig:
    return _coerce_group(name, "objective", coerce_objective_config)


def load_prediction_config(name: str) -> PredictionConfig:
    return _load_group_model(name, "prediction", PredictionConfig)


def load_training_config(name: str) -> TrainingConfig:
    return _load_group_model(name, "training", TrainingConfig)


def load_split_config(name: str) -> SplitConfig:
    return _load_group_model(name, "split", SplitConfig)


def load_tuning_config(name: str) -> TuningConfig:
    return _load_group_model(name, "tuning", TuningConfig)


def load_execution_spec(name: str) -> ExecutionSpec:
    return _load_group_model(name, "execution", ExecutionSpec)


def load_surface_frame(name: str) -> SurfaceFrame:
    return _load_group_model(name, "surface", SurfaceFrame)


def _load_group_model(
    name: str,
    group: str,
    config_type: type[ConfigModelT],
) -> ConfigModelT:
    return _coerce_group(name, group, config_type.model_validate)


def _coerce_group(
    name: str,
    group: str,
    validate: Callable[[object], ConfigT],
) -> ConfigT:
    try:
        return validate(load_named_group_payload(name, group))
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc
