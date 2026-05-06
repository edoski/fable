from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from spice.core.errors import ConfigResolutionError
from spice.evaluation import EvaluationSummary
from spice.metrics import MetricDescriptor, MetricSet
from spice.modeling.objective_runtime import ObjectiveMetricContext, compile_objective_runtime
from spice.modeling.scoring import EvaluationScoringRuntimePlan
from spice.objectives import ObjectiveConfig, ObjectiveDirection


def _objective_config(
    *,
    objective_id: str,
    evaluator_id: str | None,
    metric_id: str = "score",
) -> ObjectiveConfig:
    return ObjectiveConfig(
        id=objective_id,
        metric_id=metric_id,
        direction=ObjectiveDirection.MAXIMIZE,
        evaluator_id=evaluator_id,
    )


def test_validation_objective_runtime_returns_validation_metrics_unchanged() -> None:
    metrics = MetricSet({"score": 1.0})
    runtime = compile_objective_runtime(
        _objective_config(objective_id="validation", evaluator_id=None),
        evaluator_contract=None,
        prediction_metric_descriptors=(
            MetricDescriptor(id="score", label="score", role="primary"),
        ),
    )

    result = runtime.evaluate_metrics(metrics)

    assert runtime.contract.metric_id == "score"
    assert result is metrics


def test_evaluation_objective_runtime_scores_with_same_runtime_facts(
    monkeypatch,
) -> None:
    evaluator_contract = SimpleNamespace(
        evaluator_id="poisson_replay_2h",
        metric_descriptors=(
            MetricDescriptor(
                id="score",
                label="score",
                role="primary",
                direction="maximize",
            ),
        ),
    )
    summary = EvaluationSummary(
        metrics=MetricSet({"score": 2.0}),
        window_metrics={},
        total_events=1,
        runs=[],
    )
    seen_contexts: list[tuple[EvaluationScoringRuntimePlan, object]] = []

    def fake_score_evaluation(*, scoring_plan, evaluator_contract):
        seen_contexts.append((scoring_plan, evaluator_contract))
        return summary

    monkeypatch.setattr(
        "spice.modeling.objective_runtime.score_evaluation",
        fake_score_evaluation,
    )
    runtime = compile_objective_runtime(
        _objective_config(
            objective_id="evaluation",
            evaluator_id="poisson_replay_2h",
        ),
        evaluator_contract=cast(Any, evaluator_contract),
        prediction_metric_descriptors=(
            MetricDescriptor(id="total_loss", label="total loss", role="primary"),
        ),
    )
    context = ObjectiveMetricContext(
        model=cast(Any, SimpleNamespace(name="model")),
        prediction_contract=cast(Any, SimpleNamespace(name="prediction")),
        representation_contract=cast(Any, SimpleNamespace(name="representation")),
        execution_policy=cast(Any, SimpleNamespace(name="policy")),
        store=cast(Any, SimpleNamespace(name="store")),
        sample_indices=cast(Any, SimpleNamespace(name="samples")),
        runtime_plan=cast(Any, SimpleNamespace(name="runtime_plan")),
    )

    result = runtime.evaluate_metrics(
        MetricSet({"score": 1.0}),
        context=context,
    )

    assert runtime.contract.evaluator_id == "poisson_replay_2h"
    assert result == summary.metrics
    assert len(seen_contexts) == 1
    scoring_plan, seen_evaluator_contract = seen_contexts[0]
    assert scoring_plan.model is context.model
    assert scoring_plan.prediction_contract is context.prediction_contract
    assert scoring_plan.representation_contract is context.representation_contract
    assert scoring_plan.execution_policy is context.execution_policy
    assert scoring_plan.store is context.store
    assert scoring_plan.sample_indices is context.sample_indices
    assert scoring_plan.runtime_plan is context.runtime_plan
    assert seen_evaluator_contract is evaluator_contract


def test_evaluation_objective_runtime_requires_metric_context() -> None:
    evaluator_contract = SimpleNamespace(
        evaluator_id="poisson_replay_2h",
        metric_descriptors=(
            MetricDescriptor(
                id="score",
                label="score",
                role="primary",
                direction="maximize",
            ),
        ),
    )
    runtime = compile_objective_runtime(
        _objective_config(
            objective_id="evaluation",
            evaluator_id="poisson_replay_2h",
        ),
        evaluator_contract=cast(Any, evaluator_contract),
        prediction_metric_descriptors=(
            MetricDescriptor(id="total_loss", label="total loss", role="primary"),
        ),
    )

    with pytest.raises(ValueError, match="objective metric context"):
        runtime.evaluate_metrics(MetricSet({"score": 1.0}))


def test_validation_objective_runtime_rejects_unknown_prediction_metric() -> None:
    with pytest.raises(ConfigResolutionError, match="objective metric missing"):
        compile_objective_runtime(
            _objective_config(objective_id="validation", evaluator_id=None, metric_id="missing"),
            evaluator_contract=None,
            prediction_metric_descriptors=(
                MetricDescriptor(id="score", label="score", role="primary"),
            ),
        )


def test_evaluation_objective_runtime_rejects_unknown_evaluator_metric() -> None:
    evaluator_contract = SimpleNamespace(
        evaluator_id="poisson_replay_2h",
        metric_descriptors=(MetricDescriptor(id="profit", label="profit", role="primary"),),
    )

    with pytest.raises(ConfigResolutionError, match="objective metric score"):
        compile_objective_runtime(
            _objective_config(
                objective_id="evaluation",
                evaluator_id="poisson_replay_2h",
            ),
            evaluator_contract=cast(Any, evaluator_contract),
            prediction_metric_descriptors=(
                MetricDescriptor(id="total_loss", label="total loss", role="primary"),
            ),
        )


def test_evaluation_objective_runtime_rejects_metric_direction_mismatch() -> None:
    evaluator_contract = SimpleNamespace(
        evaluator_id="poisson_replay_2h",
        metric_descriptors=(
            MetricDescriptor(
                id="score",
                label="score",
                role="primary",
                direction="minimize",
            ),
        ),
    )

    with pytest.raises(ConfigResolutionError, match="direction"):
        compile_objective_runtime(
            _objective_config(
                objective_id="evaluation",
                evaluator_id="poisson_replay_2h",
            ),
            evaluator_contract=cast(Any, evaluator_contract),
            prediction_metric_descriptors=(
                MetricDescriptor(id="total_loss", label="total loss", role="primary"),
            ),
        )
