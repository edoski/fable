from __future__ import annotations

import numpy as np
import pytest

from spice.evaluation import coerce_evaluator_config, compile_evaluator_contract
from spice.temporal import (
    coerce_realization_policy_config,
    compile_realization_policy_contract,
)
from spice.temporal.problem_store import CompiledProblemStore


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
        candidate_end_rows=np.array([4, 7, 10, 13], dtype=np.int64),
        max_candidate_slots=2,
    )


def _realization_policy():
    return compile_realization_policy_contract(
        coerce_realization_policy_config({"id": "strict_deadline_miss"})
    )


def test_paper_windowed_falls_back_to_fullset_for_short_spans() -> None:
    store = _store()
    decoded_offsets = [0, 1, 0, 1]
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    windowed = compile_evaluator_contract(
        coerce_evaluator_config(
            {
                "id": "paper_windowed",
                "window_seconds": 99_999,
                "repetitions": 3,
                "seed": 2026,
            }
        )
    )
    fullset = compile_evaluator_contract(coerce_evaluator_config({"id": "paper_fullset"}))

    summary = windowed.run(
        store,
        _realization_policy(),
        decoded_offsets,
        sample_indices,
        reporter=None,
    )
    reference = fullset.run(
        store,
        _realization_policy(),
        decoded_offsets,
        sample_indices,
        reporter=None,
    )

    assert len(summary.runs) == 1
    assert summary.runs[0].metadata["mode"] == "fullset_fallback"
    assert summary.metrics.values == pytest.approx(reference.metrics.values)


def test_paper_windowed_samples_requested_number_of_runs() -> None:
    store = _store()
    decoded_offsets = [0, 1, 0, 1]
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    evaluator = compile_evaluator_contract(
        coerce_evaluator_config(
            {
                "id": "paper_windowed",
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
        reporter=None,
    )

    assert len(summary.runs) == 3
    assert summary.window_metrics
    assert all(run.metadata["mode"] == "windowed" for run in summary.runs)
