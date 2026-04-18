"""Closed dispatch for supported evaluators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from ..core.errors import ConfigResolutionError
from .base import EvaluatorConfig
from .contracts import CompiledEvaluatorContract

EvaluatorConfigT = TypeVar("EvaluatorConfigT", bound=EvaluatorConfig)


@dataclass(frozen=True, slots=True)
class EvaluatorSpec(Generic[EvaluatorConfigT]):
    id: str
    config_type: type[EvaluatorConfigT]
    compile: Callable[[EvaluatorConfigT], CompiledEvaluatorContract]


_KNOWN_EVALUATORS = ("paper_fullset", "poisson_replay")


def evaluator_spec(evaluator_id: str) -> EvaluatorSpec[EvaluatorConfig]:
    if evaluator_id == "paper_fullset":
        from .evaluators.paper_fullset import (
            PaperFullsetEvaluatorConfig,
            compile_evaluator as compile_paper_fullset,
        )

        return EvaluatorSpec(
            id="paper_fullset",
            config_type=PaperFullsetEvaluatorConfig,
            compile=compile_paper_fullset,
        )
    if evaluator_id == "poisson_replay":
        from .evaluators.poisson_replay import (
            PoissonReplayEvaluatorConfig,
            compile_evaluator as compile_poisson_replay,
        )

        return EvaluatorSpec(
            id="poisson_replay",
            config_type=PoissonReplayEvaluatorConfig,
            compile=compile_poisson_replay,
        )
    known = ", ".join(_KNOWN_EVALUATORS)
    raise ConfigResolutionError(
        f"Unknown evaluation.evaluator.id: {evaluator_id}. Known values: {known}"
    )


def coerce_evaluator_config(
    raw_config: Mapping[str, object] | EvaluatorConfig,
) -> EvaluatorConfig:
    if isinstance(raw_config, EvaluatorConfig):
        evaluator_id = raw_config.id
        payload = raw_config.model_dump(mode="json")
    elif isinstance(raw_config, Mapping):
        if "id" not in raw_config:
            raise ConfigResolutionError("evaluation.evaluator.id is required")
        evaluator_id = str(raw_config["id"])
        payload = dict(raw_config)
    else:
        raise ConfigResolutionError("evaluation.evaluator must be a mapping")
    return evaluator_spec(evaluator_id).config_type.model_validate(payload)


def compile_evaluator_contract(
    evaluator_config: EvaluatorConfig,
) -> CompiledEvaluatorContract:
    spec = evaluator_spec(evaluator_config.id)
    return spec.compile(cast(Any, evaluator_config))
