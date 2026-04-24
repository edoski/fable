from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.evaluation import EvaluatorConfig, compile_evaluator_contract
from spice.prediction import DecodedOffsets
from spice.temporal import (
    coerce_realization_policy_config,
    compile_realization_policy_contract,
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


def _realization_policy():
    return compile_realization_policy_contract(
        coerce_realization_policy_config({"id": "strict_deadline_miss"})
    )


def _current_row_store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.zeros((9, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array([110, 95, 90, 80, 70, 60, 50, 40, 30], dtype=np.float32)
        ).astype(np.float32, copy=False),
        timestamps=(np.arange(9, dtype=np.int64) * 1800).astype(np.int64, copy=False),
        anchor_rows=np.array([1, 2, 3, 4], dtype=np.int64),
        context_start_rows=np.array([0, 1, 2, 3], dtype=np.int64),
        candidate_start_rows=np.array([1, 2, 3, 4], dtype=np.int64),
        candidate_end_rows=np.array([3, 4, 5, 6], dtype=np.int64),
        max_candidate_slots=2,
    )


def test_evaluator_rejects_incompatible_decoded_result_semantics() -> None:
    store = _store()
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "fullset",
                "sampler": "fullset",
                "aggregation": {"id": "total_ratio"},
            }
        )
    )

    with pytest.raises(TypeError, match="decoded-result requirement"):
        evaluator.run(
            store,
            _realization_policy(),
            _OtherDecodedResult(),
            np.arange(store.n_samples, dtype=np.int64),
        )

def test_poisson_replay_handles_non_chronological_sample_indices() -> None:
    store = _store()
    forward_indices = np.arange(store.n_samples, dtype=np.int64)
    reversed_indices = forward_indices[::-1].copy()
    forward_offsets = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    reversed_offsets = DecodedOffsets(forward_offsets.flip(0))
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "poisson_replay_2h_mean",
                "sampler": "poisson_arrivals",
                "aggregation": {"id": "event_mean"},
                "window_seconds": 7200,
                "arrival_rate_per_second": 0.01,
                "repetitions": 3,
                "seed": 2026,
            }
        )
    )

    summary = evaluator.run(
        store,
        _realization_policy(),
        DecodedOffsets(forward_offsets),
        forward_indices,
    )
    reversed_summary = evaluator.run(
        store,
        _realization_policy(),
        reversed_offsets,
        reversed_indices,
    )

    assert reversed_summary.metrics.values == pytest.approx(summary.metrics.values)
    assert reversed_summary.total_events == summary.total_events
    assert [run.n_events for run in reversed_summary.runs] == [run.n_events for run in summary.runs]


def test_replay_total_ratio_uses_fee_sums() -> None:
    store = _store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "fullset",
                "sampler": "fullset",
                "aggregation": {"id": "total_ratio"},
            }
        )
    )

    summary = evaluator.run(
        store,
        _realization_policy(),
        DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64)),
        sample_indices,
    )

    baseline_total = 90.0 + 70.0 + 50.0 + 30.0
    realized_total = 90.0 + 60.0 + 50.0 + 25.0
    optimum_total = 80.0 + 60.0 + 40.0 + 25.0

    assert summary.total_events == 4
    assert summary.metrics.values == pytest.approx(
        {
            "profit_over_baseline": (baseline_total - realized_total) / baseline_total,
            "cost_over_optimum": (realized_total - optimum_total) / optimum_total,
            "baseline_cost_over_optimum": (baseline_total - optimum_total) / optimum_total,
            "realized_fee_sum": realized_total,
            "baseline_fee_sum": baseline_total,
            "optimum_fee_sum": optimum_total,
        }
    )


def test_replay_event_mean_uses_per_event_ratios() -> None:
    store = _store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "fullset_mean",
                "sampler": "fullset",
                "aggregation": {"id": "event_mean"},
            }
        )
    )

    summary = evaluator.run(
        store,
        _realization_policy(),
        DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64)),
        sample_indices,
    )

    baseline_fees = np.array([90.0, 70.0, 50.0, 30.0])
    realized_fees = np.array([90.0, 60.0, 50.0, 25.0])
    optimum_fees = np.array([80.0, 60.0, 40.0, 25.0])

    assert summary.metrics.values == pytest.approx(
        {
            "profit_over_baseline": float(
                np.mean((baseline_fees - realized_fees) / baseline_fees)
            ),
            "cost_over_optimum": float(np.mean((realized_fees - optimum_fees) / optimum_fees)),
            "baseline_cost_over_optimum": float(
                np.mean((baseline_fees - optimum_fees) / optimum_fees)
            ),
            "realized_fee_sum": float(realized_fees.sum()),
            "baseline_fee_sum": float(baseline_fees.sum()),
            "optimum_fee_sum": float(optimum_fees.sum()),
        }
    )


def test_zero_stop_rollout_stops_on_zero_and_truncates_tail_windows() -> None:
    store = _current_row_store()
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "zero_stop_rollout_fullset",
                "engine": "zero_stop_rollout",
            }
        )
    )
    summary = evaluator.run(
        store,
        _realization_policy(),
        DecodedOffsets(torch.tensor([1, 0, 1, 1], dtype=torch.int64)),
        np.arange(store.n_samples, dtype=np.int64),
    )

    baseline_total = 95.0 + 90.0 + 80.0 + 70.0
    realized_total = 90.0 + 90.0 + 70.0 + 70.0
    optimum_total = 90.0 + 80.0 + 70.0 + 60.0

    assert summary.total_events == 4
    assert summary.runs[0].metadata == {
        "mode": "zero_stop_rollout_fullset",
        "zero_stop_count": 2,
        "terminal_without_zero_count": 2,
        "truncated_window_count": 1,
    }
    assert summary.metrics.values == pytest.approx(
        {
            "profit_over_baseline": (baseline_total - realized_total) / baseline_total,
            "cost_over_optimum": (realized_total - optimum_total) / optimum_total,
            "baseline_cost_over_optimum": (baseline_total - optimum_total) / optimum_total,
            "realized_fee_sum": realized_total,
            "baseline_fee_sum": baseline_total,
            "optimum_fee_sum": optimum_total,
            "mean_steps_to_stop": 0.5,
            "zero_stop_rate": 0.5,
            "terminal_without_zero_count": 2.0,
        }
    )


def test_anchor_basefee_compares_anchor_and_realized_fees() -> None:
    store = _current_row_store()
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "anchor_basefee_fullset",
                "engine": "anchor_basefee",
            }
        )
    )
    summary = evaluator.run(
        store,
        _realization_policy(),
        DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64)),
        np.arange(store.n_samples, dtype=np.int64),
    )

    anchor_total = 95.0 + 90.0 + 80.0 + 70.0
    realized_total = 95.0 + 80.0 + 80.0 + 60.0

    assert summary.runs[0].metadata == {
        "mode": "anchor_basefee_fullset",
        "overflow_count": 0,
        "zero_action_count": 2,
    }
    assert summary.metrics.values == pytest.approx(
        {
            "fee_delta_over_anchor": (anchor_total - realized_total) / anchor_total,
            "realized_fee_sum": realized_total,
            "anchor_fee_sum": anchor_total,
            "overflow_count": 0.0,
            "zero_action_rate": 0.5,
        }
    )
