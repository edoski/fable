from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from spice.core.errors import ConfigResolutionError
from spice.evaluation import EvaluationSummary
from spice.modeling.evaluation_runtime import EvaluationScoringRuntimePlan
from spice.modeling.objective_runtime import compile_objective_runtime
from spice.modeling.representations import RepresentationRuntimeContext
from spice.modeling.scoring import ModelScoringInput
from spice.objectives import ObjectiveConfig, ObjectiveDirection
from spice.prediction import MetricDescriptor, MetricSet


def _objective_config(
    *,
    objective_id: str,
    benchmark_id: str | None,
    metric_id: str = "score",
) -> ObjectiveConfig:
    return ObjectiveConfig(
        id=objective_id,
        metric_id=metric_id,
        direction=ObjectiveDirection.MAXIMIZE,
        benchmark_id=benchmark_id,
    )


def test_validation_objective_runtime_returns_validation_metrics_unchanged() -> None:
    metrics = MetricSet({"score": 1.0})
    runtime = compile_objective_runtime(
        _objective_config(objective_id="validation", benchmark_id=None),
        evaluation=None,
        prediction_metric_descriptors=(
            MetricDescriptor(id="score", label="score", role="primary"),
        ),
    )

    result = runtime.evaluate_metrics(
        metrics,
        context=cast(Any, SimpleNamespace()),
    )

    assert runtime.contract.metric_id == "score"
    assert result is metrics


def test_evaluation_objective_runtime_scores_with_same_runtime_facts(
    monkeypatch,
) -> None:
    evaluator_contract = SimpleNamespace(
        evaluation_id="poisson_replay_2h",
        metric_descriptors=(MetricDescriptor(id="score", label="score", role="primary"),),
    )
    evaluation = SimpleNamespace(id="poisson_replay_2h")
    summary = EvaluationSummary(
        metrics=MetricSet({"score": 2.0}),
        window_metrics={},
        total_events=1,
        runs=[],
    )
    seen_contexts = []
    monkeypatch.setattr(
        "spice.modeling.objective_runtime.compile_evaluator_contract",
        lambda config: evaluator_contract,
    )

    def fake_score_evaluation(*, model_input, evaluator_contract):
        seen_contexts.append((model_input, evaluator_contract))
        return summary

    monkeypatch.setattr(
        "spice.modeling.objective_runtime.score_evaluation",
        fake_score_evaluation,
    )
    runtime = compile_objective_runtime(
        _objective_config(
            objective_id="evaluation",
            benchmark_id="poisson_replay_2h",
        ),
        evaluation=cast(Any, evaluation),
        prediction_metric_descriptors=(
            MetricDescriptor(id="total_loss", label="total loss", role="primary"),
        ),
    )
    runtime_plan = EvaluationScoringRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="fp32",
        representation_runtime_context=RepresentationRuntimeContext(
            batch_size=8,
            available_host_memory_bytes=1024,
        ),
        deterministic=None,
        seed=0,
    )
    runtime_context = ModelScoringInput(
        model=cast(Any, SimpleNamespace(name="model")),
        model_config=cast(Any, SimpleNamespace(name="model_config")),
        prediction_contract=cast(Any, SimpleNamespace(name="prediction")),
        representation_contract=cast(Any, SimpleNamespace(name="representation")),
        execution_policy=cast(Any, SimpleNamespace(name="execution")),
        store=cast(Any, SimpleNamespace(name="store")),
        sample_indices=np.array([2, 4], dtype=np.int64),
        runtime_plan=runtime_plan,
    )

    result = runtime.evaluate_metrics(
        MetricSet({"score": 1.0}),
        context=runtime_context,
    )

    assert runtime.contract.benchmark_id == "poisson_replay_2h"
    assert result == summary.metrics
    assert len(seen_contexts) == 1
    model_input, seen_evaluator_contract = seen_contexts[0]
    assert model_input is runtime_context
    assert seen_evaluator_contract is evaluator_contract


def test_validation_objective_runtime_rejects_unknown_prediction_metric() -> None:
    with pytest.raises(ConfigResolutionError, match="objective metric missing"):
        compile_objective_runtime(
            _objective_config(objective_id="validation", benchmark_id=None, metric_id="missing"),
            evaluation=None,
            prediction_metric_descriptors=(
                MetricDescriptor(id="score", label="score", role="primary"),
            ),
        )


def test_evaluation_objective_runtime_rejects_unknown_evaluator_metric(monkeypatch) -> None:
    evaluator_contract = SimpleNamespace(
        evaluation_id="poisson_replay_2h",
        metric_descriptors=(MetricDescriptor(id="profit", label="profit", role="primary"),),
    )
    monkeypatch.setattr(
        "spice.modeling.objective_runtime.compile_evaluator_contract",
        lambda config: evaluator_contract,
    )

    with pytest.raises(ConfigResolutionError, match="objective metric score"):
        compile_objective_runtime(
            _objective_config(
                objective_id="evaluation",
                benchmark_id="poisson_replay_2h",
            ),
            evaluation=cast(Any, SimpleNamespace(id="poisson_replay_2h")),
            prediction_metric_descriptors=(
                MetricDescriptor(id="total_loss", label="total loss", role="primary"),
            ),
        )
