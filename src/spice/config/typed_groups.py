# pyright: strict

"""Context-free typed loading for named config groups."""

from __future__ import annotations

from typing import TypeVar, cast

from ..evaluation import EvaluatorConfig
from ..execution.models import ExecutionSpec
from ..modeling.dataset_builders import DatasetBuilderConfig
from ..modeling.families.base import ModelConfig
from ..objectives import ObjectiveConfig
from .group_catalog import ConfigGroup, GroupSpec, group_spec, validate_named_group_payload
from .groups import load_named_group_document
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
)
from .surfaces import SurfaceFrame

ConfigT = TypeVar("ConfigT")

CHAIN = cast(GroupSpec[ChainSpec], group_spec(ConfigGroup.CHAIN))
DATASET = cast(GroupSpec[DatasetSpec], group_spec(ConfigGroup.DATASET))
DATASET_BUILDER = cast(
    GroupSpec[DatasetBuilderConfig],
    group_spec(ConfigGroup.DATASET_BUILDER),
)
EVALUATION = cast(GroupSpec[EvaluatorConfig], group_spec(ConfigGroup.EVALUATION))
EXECUTION = cast(GroupSpec[ExecutionSpec], group_spec(ConfigGroup.EXECUTION))
FEATURES = cast(GroupSpec[FeaturesConfig], group_spec(ConfigGroup.FEATURES))
MODEL = cast(GroupSpec[ModelConfig[str]], group_spec(ConfigGroup.MODEL))
OBJECTIVE = cast(GroupSpec[ObjectiveConfig], group_spec(ConfigGroup.OBJECTIVE))
PREDICTION = cast(GroupSpec[PredictionConfig], group_spec(ConfigGroup.PREDICTION))
PROBLEM = cast(GroupSpec[ProblemSpec], group_spec(ConfigGroup.PROBLEM))
PROVIDER = cast(GroupSpec[ProviderSpec], group_spec(ConfigGroup.PROVIDER))
SPLIT = cast(GroupSpec[SplitConfig], group_spec(ConfigGroup.SPLIT))
SURFACE = cast(GroupSpec[SurfaceFrame], group_spec(ConfigGroup.SURFACE))
TRAINING = cast(GroupSpec[TrainingConfig], group_spec(ConfigGroup.TRAINING))
TUNING = cast(GroupSpec[TuningConfig], group_spec(ConfigGroup.TUNING))


def load(spec: GroupSpec[ConfigT], name: str) -> ConfigT:
    payload = load_named_group_document(name, spec.group)
    return cast(
        ConfigT,
        validate_named_group_payload(spec.group, name=name, payload=payload),
    )
