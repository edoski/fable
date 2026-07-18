# pyright: strict

"""Shared catalog for named config groups."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from ..evaluation import coerce_evaluator_config
from ..modeling.families.registry import coerce_model_config
from .models import (
    ChainSpec,
    EvaluationsSpec,
    SplitConfig,
    TrainingConfig,
    coerce_problem_spec,
)

ConfigT = TypeVar("ConfigT")
_ValidateGroupPayload = Callable[[dict[str, object]], ConfigT]


class ConfigGroup(StrEnum):
    CHAIN = "chain"
    EVALUATOR = "evaluator"
    EVALUATIONS = "evaluations"
    MODEL = "model"
    PROBLEM = "problem"
    SPLIT = "split"
    TRAINING = "training"


@dataclass(frozen=True, slots=True)
class GroupSpec(Generic[ConfigT]):
    group: ConfigGroup
    seed_name: str | None
    validate: _ValidateGroupPayload[ConfigT]
    identity_field: str | None = None
    seed_from_requested_name: bool = False
    public: bool = False

    @property
    def token(self) -> str:
        return self.group.value

    @property
    def directory(self) -> str:
        return self.group.value


GROUP_SPECS: tuple[GroupSpec[object], ...] = (
    GroupSpec(
        group=ConfigGroup.TRAINING,
        seed_name="default",
        validate=TrainingConfig.model_validate,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.SPLIT,
        seed_name="default",
        validate=SplitConfig.model_validate,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.CHAIN,
        seed_name="ethereum",
        validate=ChainSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.PROBLEM,
        seed_name="current_row_nominal",
        validate=coerce_problem_spec,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.EVALUATOR,
        seed_name="poisson_replay",
        validate=coerce_evaluator_config,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.EVALUATIONS,
        seed_name=None,
        validate=EvaluationsSpec.model_validate,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.MODEL,
        seed_name="lstm",
        validate=coerce_model_config,
        seed_from_requested_name=True,
        public=True,
    ),
)
