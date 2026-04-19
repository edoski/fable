"""Closed dispatch for supported evaluators."""

from __future__ import annotations

from collections.abc import Mapping

from ..core.closed_dispatch import config_payload_and_id, unknown_id_error
from .base import CompiledEvaluatorContract, EvaluatorConfig


def _coerce_known_evaluator_config(
    evaluator_id: str,
    payload: Mapping[str, object],
) -> EvaluatorConfig:
    if evaluator_id == "paper_fullset":
        from .evaluators.paper_fullset import PaperFullsetEvaluatorConfig

        return PaperFullsetEvaluatorConfig.model_validate(payload)
    if evaluator_id == "poisson_replay":
        from .evaluators.poisson_replay import PoissonReplayEvaluatorConfig

        return PoissonReplayEvaluatorConfig.model_validate(payload)
    raise unknown_id_error(
        field_name="evaluation.evaluator.id",
        component_id=evaluator_id,
        known_ids=("paper_fullset", "poisson_replay"),
    )


def coerce_evaluator_config(
    raw_config: Mapping[str, object] | EvaluatorConfig,
) -> EvaluatorConfig:
    payload, evaluator_id = config_payload_and_id(
        raw_config,
        config_type=EvaluatorConfig,
        field_name="evaluation.evaluator.id",
        mapping_label="evaluation.evaluator",
    )
    return _coerce_known_evaluator_config(evaluator_id, payload)


def compile_evaluator_contract(
    evaluator_config: EvaluatorConfig,
) -> CompiledEvaluatorContract:
    if evaluator_config.id == "paper_fullset":
        from .evaluators.paper_fullset import (
            PaperFullsetEvaluatorConfig,
            compile_evaluator,
        )

        if not isinstance(evaluator_config, PaperFullsetEvaluatorConfig):
            raise TypeError("paper_fullset evaluator config has unexpected type")
        return compile_evaluator(evaluator_config)
    if evaluator_config.id == "poisson_replay":
        from .evaluators.poisson_replay import (
            PoissonReplayEvaluatorConfig,
            compile_evaluator,
        )

        if not isinstance(evaluator_config, PoissonReplayEvaluatorConfig):
            raise TypeError("poisson_replay evaluator config has unexpected type")
        return compile_evaluator(evaluator_config)
    raise unknown_id_error(
        field_name="evaluation.evaluator.id",
        component_id=evaluator_config.id,
        known_ids=("paper_fullset", "poisson_replay"),
    )
