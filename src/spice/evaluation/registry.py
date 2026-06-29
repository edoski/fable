"""Evaluator config coercion and contract compiler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.specs import (
    coerce_spec_config,
    lookup_local_spec,
    require_spec_config,
    require_spec_config_from_table,
)
from .block_poisson_replay import compile_block_poisson_replay_evaluator_contract
from .config import (
    BlockPoissonReplayEvaluatorConfig,
    EvaluatorConfig,
    PoissonReplayEvaluatorConfig,
)
from .contracts import CompiledEvaluatorContract
from .poisson_replay import compile_poisson_replay_evaluator_contract


@dataclass(frozen=True, slots=True)
class _EvaluatorSpec:
    config_type: type[EvaluatorConfig]
    compile_contract: Callable[[EvaluatorConfig], CompiledEvaluatorContract]


_EVALUATOR_SPECS: dict[str, _EvaluatorSpec] = {
    "block_poisson_replay": _EvaluatorSpec(
        config_type=BlockPoissonReplayEvaluatorConfig,
        compile_contract=lambda config: compile_block_poisson_replay_evaluator_contract(
            require_spec_config(
                config,
                BlockPoissonReplayEvaluatorConfig,
                "evaluator config",
            )
        ),
    ),
    "poisson_replay": _EvaluatorSpec(
        config_type=PoissonReplayEvaluatorConfig,
        compile_contract=lambda config: compile_poisson_replay_evaluator_contract(
            require_spec_config(config, PoissonReplayEvaluatorConfig, "evaluator config")
        ),
    ),
}


def coerce_evaluator_config(
    payload: object,
) -> EvaluatorConfig:
    return coerce_spec_config(
        payload,
        owner="evaluator",
        base_config_type=EvaluatorConfig,
        id_label="evaluator.id",
        lookup_spec=evaluator_spec,
        spec_config_type=lambda spec: spec.config_type,
    )


def compile_evaluator_contract(
    evaluator_config: EvaluatorConfig,
) -> CompiledEvaluatorContract:
    spec = evaluator_spec(evaluator_config.id)
    concrete_config = require_spec_config_from_table(
        evaluator_config,
        config_id=evaluator_config.id,
        lookup_spec=evaluator_spec,
        spec_config_type=lambda entry: entry.config_type,
        label="evaluator config",
    )
    return spec.compile_contract(concrete_config)


def evaluator_spec(evaluator_id: str) -> _EvaluatorSpec:
    return lookup_local_spec(_EVALUATOR_SPECS, evaluator_id, "evaluator.id")
