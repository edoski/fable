"""Evaluator contract registry."""

from __future__ import annotations

from ..prediction import DecodedOffsets
from .config import EvaluationAggregationId, EvaluationEngine, EvaluationSampler, EvaluatorConfig
from .contracts import CompiledEvaluatorContract, RunEvaluatorFn
from .mechanical import run_anchor_basefee_fullset, run_zero_stop_rollout_fullset
from .metrics import (
    ANCHOR_BASEFEE_METRIC_DESCRIPTORS,
    REPLAY_METRIC_DESCRIPTORS,
    ZERO_STOP_ROLLOUT_METRIC_DESCRIPTORS,
)
from .replay import run_fullset, run_poisson_arrivals


def compile_evaluator_contract(
    evaluator_config: EvaluatorConfig,
) -> CompiledEvaluatorContract:
    if evaluator_config.engine is EvaluationEngine.ZERO_STOP_ROLLOUT:
        return CompiledEvaluatorContract(
            evaluation_id=evaluator_config.id,
            metric_descriptors=ZERO_STOP_ROLLOUT_METRIC_DESCRIPTORS,
            primary_metric_id="profit_over_baseline",
            direction="maximize",
            config_payload=evaluator_config.model_dump(mode="json", exclude_none=True),
            accepted_decoded_result_id=DecodedOffsets.decoded_result_id,
            run_fn=run_zero_stop_rollout_fullset,
        )
    if evaluator_config.engine is EvaluationEngine.ANCHOR_BASEFEE:
        return CompiledEvaluatorContract(
            evaluation_id=evaluator_config.id,
            metric_descriptors=ANCHOR_BASEFEE_METRIC_DESCRIPTORS,
            primary_metric_id="fee_delta_over_anchor",
            direction="maximize",
            config_payload=evaluator_config.model_dump(mode="json", exclude_none=True),
            accepted_decoded_result_id=DecodedOffsets.decoded_result_id,
            run_fn=run_anchor_basefee_fullset,
        )
    if evaluator_config.sampler is EvaluationSampler.FULLSET:
        def run_fn(
            store,
            realization_policy,
            decoded_result,
            sample_indices,
        ):
            return run_fullset(
                store,
                realization_policy,
                decoded_result,
                sample_indices,
                aggregation=_aggregation(evaluator_config),
            )

    elif evaluator_config.sampler is EvaluationSampler.POISSON_ARRIVALS:

        def run_fn(
            store,
            realization_policy,
            decoded_result,
            sample_indices,
        ):
            return run_poisson_arrivals(
                store,
                realization_policy,
                decoded_result,
                sample_indices,
                config=evaluator_config,
            )

    else:
        raise ValueError("replay evaluator requires evaluation.sampler")
    resolved_run_fn: RunEvaluatorFn = run_fn
    return CompiledEvaluatorContract(
        evaluation_id=evaluator_config.id,
        metric_descriptors=REPLAY_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        config_payload=evaluator_config.model_dump(mode="json", exclude_none=True),
        accepted_decoded_result_id=DecodedOffsets.decoded_result_id,
        run_fn=resolved_run_fn,
    )


def _aggregation(config: EvaluatorConfig) -> EvaluationAggregationId:
    if config.aggregation is None:
        raise ValueError("Missing required field: evaluation.aggregation")
    return config.aggregation.id
