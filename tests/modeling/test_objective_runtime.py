from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from spice.core.errors import ConfigResolutionError
from spice.evaluation import EvaluationSummary
from spice.metrics import MetricDescriptor, MetricSet
from spice.modeling.objective_runtime import compile_objective_runtime
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
        evaluator_id="poisson_replay",
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
    seen_scoring_calls: list[tuple[EvaluationScoringRuntimePlan, object]] = []

    def fake_score_evaluation(*, scoring_plan, evaluator_contract):
        seen_scoring_calls.append((scoring_plan, evaluator_contract))
        return summary

    monkeypatch.setattr(
        "spice.modeling.objective_runtime.score_evaluation",
        fake_score_evaluation,
    )
    runtime = compile_objective_runtime(
        _objective_config(
            objective_id="evaluation",
            evaluator_id="poisson_replay",
        ),
        evaluator_contract=cast(Any, evaluator_contract),
        prediction_metric_descriptors=(
            MetricDescriptor(id="total_loss", label="total loss", role="primary"),
        ),
    )
    scoring_plan = EvaluationScoringRuntimePlan(
        model=cast(Any, SimpleNamespace(name="model")),
        prediction_contract=cast(Any, SimpleNamespace(name="prediction")),
        representation_contract=cast(Any, SimpleNamespace(name="representation")),
        execution_policy=cast(Any, SimpleNamespace(name="policy")),
        store=cast(Any, SimpleNamespace(name="store")),
        action_space=cast(Any, SimpleNamespace(name="action_space")),
        runtime_plan=cast(Any, SimpleNamespace(name="runtime_plan")),
    )

    result = runtime.evaluate_metrics(
        MetricSet({"score": 1.0}),
        scoring_plan=scoring_plan,
    )

    assert runtime.contract.evaluator_id == "poisson_replay"
    assert result == summary.metrics
    assert len(seen_scoring_calls) == 1
    seen_scoring_plan, seen_evaluator_contract = seen_scoring_calls[0]
    assert seen_scoring_plan is scoring_plan
    assert seen_evaluator_contract is evaluator_contract


def test_evaluation_objective_runtime_requires_scoring_plan() -> None:
    evaluator_contract = SimpleNamespace(
        evaluator_id="poisson_replay",
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
            evaluator_id="poisson_replay",
        ),
        evaluator_contract=cast(Any, evaluator_contract),
        prediction_metric_descriptors=(
            MetricDescriptor(id="total_loss", label="total loss", role="primary"),
        ),
    )

    with pytest.raises(ValueError, match="evaluation scoring runtime plan"):
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
        evaluator_id="poisson_replay",
        metric_descriptors=(MetricDescriptor(id="profit", label="profit", role="primary"),),
    )

    with pytest.raises(ConfigResolutionError, match="objective metric score"):
        compile_objective_runtime(
            _objective_config(
                objective_id="evaluation",
                evaluator_id="poisson_replay",
            ),
            evaluator_contract=cast(Any, evaluator_contract),
            prediction_metric_descriptors=(
                MetricDescriptor(id="total_loss", label="total loss", role="primary"),
            ),
        )


def test_evaluation_objective_runtime_rejects_metric_direction_mismatch() -> None:
    evaluator_contract = SimpleNamespace(
        evaluator_id="poisson_replay",
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
                evaluator_id="poisson_replay",
            ),
            evaluator_contract=cast(Any, evaluator_contract),
            prediction_metric_descriptors=(
                MetricDescriptor(id="total_loss", label="total loss", role="primary"),
            ),
        )
