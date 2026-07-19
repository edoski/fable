"""Historical evaluator config coercion retained for read-only decoding."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.specs import coerce_spec_config, lookup_local_spec
from .config import (
    BLOCK_POISSON_REPLAY_EVALUATOR_IDS,
    BlockPoissonReplayEvaluatorConfig,
    EvaluatorConfig,
    PoissonReplayEvaluatorConfig,
)


@dataclass(frozen=True, slots=True)
class _EvaluatorSpec:
    config_type: type[EvaluatorConfig]


_BLOCK_POISSON_REPLAY_SPEC = _EvaluatorSpec(
    config_type=BlockPoissonReplayEvaluatorConfig,
)

_EVALUATOR_SPECS: dict[str, _EvaluatorSpec] = {
    **{
        evaluator_id: _BLOCK_POISSON_REPLAY_SPEC
        for evaluator_id in BLOCK_POISSON_REPLAY_EVALUATOR_IDS
    },
    "poisson_replay": _EvaluatorSpec(config_type=PoissonReplayEvaluatorConfig),
}


def coerce_evaluator_config(payload: object) -> EvaluatorConfig:
    return coerce_spec_config(
        payload,
        owner="evaluator",
        base_config_type=EvaluatorConfig,
        id_label="evaluator.id",
        lookup_spec=evaluator_spec,
        spec_config_type=lambda spec: spec.config_type,
    )


def evaluator_spec(evaluator_id: str) -> _EvaluatorSpec:
    return lookup_local_spec(_EVALUATOR_SPECS, evaluator_id, "evaluator.id")
