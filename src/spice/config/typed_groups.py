# pyright: strict

"""Context-free typed loading for named config groups."""

from __future__ import annotations

from typing import TypeVar, cast

from ..evaluation import EvaluatorConfig
from ..execution.models import ExecutionSpec
from ..modeling.families.base import ModelConfig
from .group_catalog import ConfigGroup, GroupSpec, group_spec, validate_named_group_payload
from .groups import load_named_group_document
from .models import (
    ChainSpec,
    CorpusSpec,
    EvaluationsSpec,
    ProblemSpec,
    SplitConfig,
    TrainingConfig,
)
from .surfaces import SurfaceFrame

ConfigT = TypeVar("ConfigT")

CHAIN = cast(GroupSpec[ChainSpec], group_spec(ConfigGroup.CHAIN))
CORPUS = cast(GroupSpec[CorpusSpec], group_spec(ConfigGroup.CORPUS))
EVALUATOR = cast(GroupSpec[EvaluatorConfig], group_spec(ConfigGroup.EVALUATOR))
EVALUATIONS = cast(GroupSpec[EvaluationsSpec], group_spec(ConfigGroup.EVALUATIONS))
EXECUTION = cast(GroupSpec[ExecutionSpec], group_spec(ConfigGroup.EXECUTION))
MODEL = cast(GroupSpec[ModelConfig[str]], group_spec(ConfigGroup.MODEL))
PROBLEM = cast(GroupSpec[ProblemSpec], group_spec(ConfigGroup.PROBLEM))
SPLIT = cast(GroupSpec[SplitConfig], group_spec(ConfigGroup.SPLIT))
SURFACE = cast(GroupSpec[SurfaceFrame], group_spec(ConfigGroup.SURFACE))
TRAINING = cast(GroupSpec[TrainingConfig], group_spec(ConfigGroup.TRAINING))


def load(spec: GroupSpec[ConfigT], name: str) -> ConfigT:
    payload = load_named_group_document(name, spec.group)
    return cast(
        ConfigT,
        validate_named_group_payload(spec.group, name=name, payload=payload),
    )
