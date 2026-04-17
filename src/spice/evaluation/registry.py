"""Open registry for evaluator specs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from ..core.components import ComponentCatalog
from ..core.errors import ConfigResolutionError
from .base import EvaluatorConfig
from .contracts import CompiledEvaluatorContract

EvaluatorConfigT = TypeVar("EvaluatorConfigT", bound=EvaluatorConfig)


@dataclass(frozen=True, slots=True)
class EvaluatorSpec(Generic[EvaluatorConfigT]):
    id: str
    config_type: type[EvaluatorConfigT]
    compile: Callable[[EvaluatorConfigT], CompiledEvaluatorContract]


_EVALUATOR_SPECS = ComponentCatalog[EvaluatorSpec[Any]](
    kind_label="evaluator",
    entry_point_group="spice.evaluators",
)


def register_evaluator_spec(spec: EvaluatorSpec[Any]) -> None:
    _EVALUATOR_SPECS.register(spec.id, spec)


def _load_builtin_evaluators() -> None:
    from .evaluators import paper_fullset, poisson_replay  # noqa: F401


_EVALUATOR_SPECS.configure_builtin_loader(_load_builtin_evaluators)


def evaluator_spec(evaluator_id: str) -> EvaluatorSpec[Any]:
    try:
        return _EVALUATOR_SPECS.get(evaluator_id)
    except ConfigResolutionError as exc:
        raise ConfigResolutionError(
            str(exc).replace("evaluator", "evaluation.evaluator.id")
        ) from exc


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
