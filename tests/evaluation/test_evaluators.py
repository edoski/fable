from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.core.errors import ConfigResolutionError, SpiceOperatorError
from spice.evaluation import (
    CompiledEvaluatorContract,
    EvaluationRun,
    EvaluationSummary,
    EvaluatorConfig,
    coerce_evaluator_config,
    compile_evaluator_contract,
)
from spice.metrics import MetricDescriptor, MetricSet
from spice.prediction.decoded_offsets import OFFSET_DECODED_RESULT_ID, DecodedOffsets
from spice.temporal import (
    coerce_execution_policy_config,
    compile_execution_policy_contract,
)
from spice.temporal.problem_store import CompiledProblemStore


class _OtherDecodedResult:
    decoded_result_id = "other"

    def __len__(self) -> int:
        return 4


def _store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.zeros((16, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array(
                [100, 95, 90, 80, 75, 70, 60, 55, 50, 40, 35, 30, 25, 20, 18, 16],
                dtype=np.float32,
            )
        ).astype(np.float32, copy=False),
        timestamps=(np.arange(16, dtype=np.int64) * 1800).astype(np.int64, copy=False),
        anchor_rows=np.array([1, 4, 7, 10], dtype=np.int64),
        context_start_rows=np.array([0, 3, 6, 9], dtype=np.int64),
        candidate_start_rows=np.array([2, 5, 8, 11], dtype=np.int64),
        candidate_end_rows=np.array([4, 7, 10, 13], dtype=np.int64),
        max_candidate_slots=2,
    )


def _overflow_store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.zeros((10, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array([100, 90, 80, 60, 55, 70, 50, 45, 40, 35], dtype=np.float32)
        ).astype(np.float32, copy=False),
        timestamps=(np.arange(10, dtype=np.int64) * 1800).astype(np.int64, copy=False),
        anchor_rows=np.array([1, 4], dtype=np.int64),
        context_start_rows=np.array([0, 3], dtype=np.int64),
        candidate_start_rows=np.array([2, 5], dtype=np.int64),
        candidate_end_rows=np.array([3, 7], dtype=np.int64),
        max_candidate_slots=3,
    )


def _poisson_config() -> dict[str, str | int | float]:
    return {
        "id": "poisson_replay",
        "window_seconds": 7200,
        "arrival_rate_per_second": 0.01,
        "repetitions": 3,
        "seed": 2026,
    }


def _full_config() -> dict[str, str]:
    return {"id": "full_temporal_replay"}


def _execution_policy():
    return compile_execution_policy_contract(
        coerce_execution_policy_config({"id": "strict_deadline_miss"})
    )


def test_evaluator_config_supports_explicit_temporal_replay_specs() -> None:
    config = coerce_evaluator_config(_poisson_config())
    contract = compile_evaluator_contract(config)

    assert config.id == "poisson_replay"
    assert contract.evaluator_id == "poisson_replay"
    assert contract.config == config
    assert contract.accepted_decoded_result_id == OFFSET_DECODED_RESULT_ID
    assert contract.primary_metric_id == "profit_over_baseline"
    assert contract.primary_metric_descriptor.direction == "maximize"

    full_config = coerce_evaluator_config(_full_config())
    full_contract = compile_evaluator_contract(full_config)

    assert full_config.id == "full_temporal_replay"
    assert full_contract.evaluator_id == "full_temporal_replay"
    assert full_contract.config == full_config
    assert full_contract.accepted_decoded_result_id == OFFSET_DECODED_RESULT_ID
    assert full_contract.metric_descriptors == contract.metric_descriptors

    with pytest.raises(ConfigResolutionError, match="Extra inputs"):
        coerce_evaluator_config({**_poisson_config(), "engine": "replay"})

    with pytest.raises(ConfigResolutionError, match="Known values"):
        coerce_evaluator_config({**_poisson_config(), "id": "other_evaluation"})


def test_evaluator_compile_requires_concrete_poisson_config() -> None:
    with pytest.raises(ConfigResolutionError, match="Known values"):
        compile_evaluator_contract(EvaluatorConfig(id="base"))


def test_metric_descriptors_and_evaluator_contract_validate_primary_metric() -> None:
    def unused_run_fn(
        _store: object,
        _execution_policy: object,
        _decoded_result: object,
        _sample_indices: object,
    ) -> EvaluationSummary:
        return EvaluationSummary(
            metrics=MetricSet(values={"profit": 0.0}),
            window_metrics={},
            total_events=0,
            runs=[EvaluationRun(n_events=0, metrics={"profit": 0.0}, metadata={})],
        )

    with pytest.raises(ValueError, match="metric.label must be non-empty"):
        MetricDescriptor(id="profit", label="", role="primary")

    with pytest.raises(ValueError, match="exactly one primary"):
        CompiledEvaluatorContract(
            evaluator_id="bad",
            metric_descriptors=(MetricDescriptor(id="profit", label="Profit", role="secondary"),),
            config=EvaluatorConfig(id="bad"),
            accepted_decoded_result_id="offsets",
            run_fn=unused_run_fn,
        )


def test_evaluator_rejects_incompatible_decoded_result_semantics() -> None:
    store = _store()
    evaluator = compile_evaluator_contract(coerce_evaluator_config(_poisson_config()))

    with pytest.raises(TypeError, match="decoded-result requirement"):
        evaluator.run(
            store,
            _execution_policy(),
            _OtherDecodedResult(),
            np.arange(store.n_samples, dtype=np.int64),
        )


def test_poisson_replay_reports_event_summary_metadata() -> None:
    store = _store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    offsets = DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64))
    evaluator = compile_evaluator_contract(coerce_evaluator_config(_poisson_config()))

    summary = evaluator.run(store, _execution_policy(), offsets, sample_indices)

    assert summary.total_events == sum(run.n_events for run in summary.runs)
    assert summary.total_events > 0
    assert set(summary.metrics.values) == {
        descriptor.id for descriptor in evaluator.metric_descriptors
    }
    assert all(run.n_events == run.metadata["n_arrivals"] for run in summary.runs)
    assert all(
        run.metadata["window_start_timestamp"] < run.metadata["window_end_timestamp"]
        for run in summary.runs
    )
    assert all("overflow_count" in run.metadata for run in summary.runs)


def test_full_temporal_replay_scores_every_supplied_sample_once() -> None:
    store = _store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    offsets = DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64))
    evaluator = compile_evaluator_contract(coerce_evaluator_config(_full_config()))

    summary = evaluator.run(store, _execution_policy(), offsets, sample_indices)

    assert summary.total_events == store.n_samples
    assert summary.runs[0].n_events == store.n_samples
    assert summary.runs[0].metadata["mode"] == "full_temporal_replay"
    assert summary.runs[0].metadata["sample_count"] == store.n_samples
    assert set(summary.metrics.values) == {
        descriptor.id for descriptor in evaluator.metric_descriptors
    }


def test_full_temporal_replay_accounting_resolves_policy_overflow() -> None:
    store = _overflow_store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    offsets = DecodedOffsets(torch.tensor([2, 1], dtype=torch.int64))
    execution_policy = _execution_policy()
    evaluator = compile_evaluator_contract(coerce_evaluator_config(_full_config()))

    summary = evaluator.run(store, execution_policy, offsets, sample_indices)

    assert summary.runs[0].metadata["overflow_count"] == 1
    assert summary.total_events == store.n_samples
    assert summary.metrics.values["realized_fee_sum"] > 0.0


def test_poisson_replay_handles_non_chronological_sample_indices() -> None:
    store = _store()
    forward_indices = np.arange(store.n_samples, dtype=np.int64)
    reversed_indices = forward_indices[::-1].copy()
    forward_offsets = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    reversed_offsets = DecodedOffsets(forward_offsets.flip(0))
    evaluator = compile_evaluator_contract(coerce_evaluator_config(_poisson_config()))

    summary = evaluator.run(
        store,
        _execution_policy(),
        DecodedOffsets(forward_offsets),
        forward_indices,
    )
    reversed_summary = evaluator.run(
        store,
        _execution_policy(),
        reversed_offsets,
        reversed_indices,
    )

    assert reversed_summary.metrics.values == pytest.approx(summary.metrics.values)
    assert reversed_summary.total_events == summary.total_events
    assert [run.n_events for run in reversed_summary.runs] == [run.n_events for run in summary.runs]


def test_poisson_replay_rejects_window_larger_than_sample_coverage() -> None:
    store = _store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    config = coerce_evaluator_config(
        {
            **_poisson_config(),
            "window_seconds": int(store.timestamps[-1] - store.timestamps[0] + 1),
        }
    )
    evaluator = compile_evaluator_contract(config)

    with pytest.raises(ValueError, match="requested replay window"):
        evaluator.run(
            store,
            _execution_policy(),
            DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64)),
            sample_indices,
        )


def test_poisson_replay_rejects_all_empty_arrival_repetitions() -> None:
    store = _store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    evaluator = compile_evaluator_contract(
        coerce_evaluator_config(
            {
                **_poisson_config(),
                "arrival_rate_per_second": 1e-12,
                "repetitions": 1,
            }
        )
    )

    with pytest.raises(SpiceOperatorError, match="no valid arrivals"):
        evaluator.run(
            store,
            _execution_policy(),
            DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64)),
            sample_indices,
        )
