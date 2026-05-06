from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from spice.evaluation import EvaluationRun, EvaluationSummary
from spice.metrics import MetricSet
from spice.modeling.representations import RepresentationRuntimeContext
from spice.modeling.runtime_planning import ModelingRuntimePlan
from spice.modeling.scoring import EvaluationScoringRuntimePlan, score_evaluation


def test_score_evaluation_validates_predicts_and_runs_evaluator(monkeypatch) -> None:
    decoded = SimpleNamespace(decoded_result_id="offsets")
    summary = EvaluationSummary(
        metrics=MetricSet(values={"profit": 1.0}),
        window_metrics={},
        total_events=1,
        runs=[EvaluationRun(n_events=1, metrics={"profit": 1.0}, metadata={})],
    )
    calls: list[str] = []
    runtime_plan = ModelingRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="32-true",
        representation_runtime_context=RepresentationRuntimeContext(
            batch_size=8,
            available_host_memory_bytes=1024,
        ),
        deterministic=None,
        seed=0,
    )

    def fake_predict_with_model(*_args, **kwargs):
        calls.append(
            f"predict:{kwargs['runtime_plan'].representation_runtime_context.batch_size}:"
            f"{kwargs['execution_policy'].name}"
        )
        return decoded

    class FakeEvaluator:
        def validate_prediction_contract(self, prediction_contract) -> None:
            calls.append(f"validate:{prediction_contract.decoded_result_id}")

        def run(self, store, execution_policy, decoded_result, *, sample_indices):
            del store, execution_policy
            calls.append(f"run:{decoded_result.decoded_result_id}:{sample_indices.tolist()}")
            return summary

    monkeypatch.setattr("spice.modeling.scoring.predict_with_model", fake_predict_with_model)

    result = score_evaluation(
        scoring_plan=EvaluationScoringRuntimePlan(
            model=cast(Any, SimpleNamespace()),
            prediction_contract=cast(Any, SimpleNamespace(decoded_result_id="offsets")),
            representation_contract=cast(Any, SimpleNamespace()),
            execution_policy=cast(Any, SimpleNamespace(name="policy")),
            store=cast(Any, SimpleNamespace()),
            sample_indices=np.array([2, 4], dtype=np.int64),
            runtime_plan=runtime_plan,
        ),
        evaluator_contract=cast(Any, FakeEvaluator()),
    )

    assert result is summary
    assert calls == ["validate:offsets", "predict:8:policy", "run:offsets:[2, 4]"]
