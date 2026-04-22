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
from spice.temporal.semantics import ActionSpaceMode


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
        action_space_mode=ActionSpaceMode.REALIZED_PER_SAMPLE,
        max_candidate_slots=2,
    )


def _realization_policy():
    return compile_realization_policy_contract(
        coerce_realization_policy_config({"id": "strict_deadline_miss"})
    )


def test_paper_windowed_falls_back_to_fullset_for_short_spans() -> None:
    store = _store()
    decoded_offsets = DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64))
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    windowed = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "paper_windowed_2h",
                "sampler": "uniform_window",
                "window_seconds": 99_999,
                "repetitions": 3,
                "seed": 2026,
            }
        )
    )
    fullset = compile_evaluator_contract(
        EvaluatorConfig.model_validate({"id": "paper_fullset", "sampler": "fullset"})
    )

    summary = windowed.run(
        store,
        _realization_policy(),
        decoded_offsets,
        sample_indices,
    )
    reference = fullset.run(
        store,
        _realization_policy(),
        decoded_offsets,
        sample_indices,
    )

    assert len(summary.runs) == 1
    assert summary.runs[0].metadata["mode"] == "fullset_fallback"
    assert summary.metrics.values == pytest.approx(reference.metrics.values)


def test_paper_windowed_falls_back_to_fullset_for_exact_spans() -> None:
    store = _store()
    decoded_offsets = DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64))
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    sample_timestamps = store.timestamps[store.anchor_rows[sample_indices]]
    exact_window_seconds = int(sample_timestamps[-1] - sample_timestamps[0])
    windowed = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "paper_windowed_2h",
                "sampler": "uniform_window",
                "window_seconds": exact_window_seconds,
                "repetitions": 3,
                "seed": 2026,
            }
        )
    )
    fullset = compile_evaluator_contract(
        EvaluatorConfig.model_validate({"id": "paper_fullset", "sampler": "fullset"})
    )

    summary = windowed.run(
        store,
        _realization_policy(),
        decoded_offsets,
        sample_indices,
    )
    reference = fullset.run(
        store,
        _realization_policy(),
        decoded_offsets,
        sample_indices,
    )

    assert len(summary.runs) == 1
    assert summary.runs[0].metadata["mode"] == "fullset_fallback"
    assert summary.metrics.values == pytest.approx(reference.metrics.values)


def test_paper_windowed_samples_requested_number_of_runs() -> None:
    store = _store()
    decoded_offsets = DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64))
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "paper_windowed_2h",
                "sampler": "uniform_window",
                "window_seconds": 3600,
                "repetitions": 3,
                "seed": 2026,
            }
        )
    )

    summary = evaluator.run(
        store,
        _realization_policy(),
        decoded_offsets,
        sample_indices,
    )

    assert len(summary.runs) == 3
    assert summary.window_metrics
    assert all(run.metadata["mode"] == "windowed" for run in summary.runs)


def test_paper_windowed_handles_sparse_non_empty_windows_without_retry_failure() -> None:
    store = _store()
    decoded_offsets = DecodedOffsets(torch.tensor([0, 1, 0, 1], dtype=torch.int64))
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "paper_windowed_2h",
                "sampler": "uniform_window",
                "window_seconds": 1,
                "repetitions": 8,
                "seed": 2026,
            }
        )
    )

    summary = evaluator.run(
        store,
        _realization_policy(),
        decoded_offsets,
        sample_indices,
    )

    assert len(summary.runs) == 8
    assert all(run.n_events == 1 for run in summary.runs)


def test_poisson_replay_handles_non_chronological_sample_indices() -> None:
    store = _store()
    forward_indices = np.arange(store.n_samples, dtype=np.int64)
    reversed_indices = forward_indices[::-1].copy()
    forward_offsets = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    reversed_offsets = DecodedOffsets(forward_offsets.flip(0))
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate(
            {
                "id": "paper_replay_2h",
                "sampler": "poisson_arrivals",
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


def test_realization_rejects_negative_decoded_offsets() -> None:
    store = _store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)

    with pytest.raises(ValueError, match="non-negative"):
        _realization_policy().realize_selections(
            store,
            DecodedOffsets(torch.tensor([-1, 0, 0, 0], dtype=torch.int64)),
            sample_indices,
            np.array([0], dtype=np.int64),
        )


def test_fullset_uses_next_block_baseline_and_future_window_optimum() -> None:
    store = _store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate({"id": "paper_fullset", "sampler": "fullset"})
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


def test_fixed_ex_ante_overflow_realizes_first_post_window_row() -> None:
    store = CompiledProblemStore(
        feature_matrix=np.zeros((16, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array(
                [100, 95, 90, 80, 75, 70, 60, 55, 50, 40, 35, 30, 25, 20, 18, 16],
                dtype=np.float32,
            )
        ).astype(np.float32, copy=False),
        timestamps=(np.arange(16, dtype=np.int64) * 1800).astype(np.int64, copy=False),
        anchor_rows=np.array([1, 4], dtype=np.int64),
        context_start_rows=np.array([0, 3], dtype=np.int64),
        candidate_start_rows=np.array([2, 5], dtype=np.int64),
        candidate_end_rows=np.array([4, 7], dtype=np.int64),
        action_space_mode=ActionSpaceMode.FIXED_EX_ANTE,
        max_candidate_slots=3,
    )
    evaluator = compile_evaluator_contract(
        EvaluatorConfig.model_validate({"id": "paper_fullset", "sampler": "fullset"})
    )

    summary = evaluator.run(
        store,
        _realization_policy(),
        DecodedOffsets(torch.tensor([2, 2], dtype=torch.int64)),
        np.arange(store.n_samples, dtype=np.int64),
    )

    baseline_total = 90.0 + 70.0
    realized_total = 75.0 + 55.0
    optimum_total = 80.0 + 60.0

    assert summary.runs[0].metadata["overflow_count"] == 2
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
