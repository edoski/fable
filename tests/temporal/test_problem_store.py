from __future__ import annotations

import numpy as np

from spice.temporal.problem_store import CompiledProblemStore


def _store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.zeros((8, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array([10, 9, 4, 3, 8, 6, 2, 7], dtype=np.float32)
        ).astype(np.float32, copy=False),
        timestamps=(np.arange(8, dtype=np.int64) * 10).astype(np.int64, copy=False),
        anchor_rows=np.array([2, 4, 6], dtype=np.int64),
        context_start_rows=np.array([0, 2, 4], dtype=np.int64),
        candidate_start_rows=np.array([2, 4, 6], dtype=np.int64),
        candidate_end_rows=np.array([5, 7, 8], dtype=np.int64),
        max_candidate_slots=2,
    )


def test_candidate_windows_include_generic_reachable_geometry() -> None:
    store = _store()

    windows = store.candidate_windows(np.array([0, 1, 2], dtype=np.int64))

    np.testing.assert_array_equal(windows.anchor_rows, [2, 4, 6])
    np.testing.assert_array_equal(windows.baseline_rows, [2, 4, 6])
    np.testing.assert_array_equal(windows.candidate_counts, [3, 3, 2])
    np.testing.assert_array_equal(windows.reachable_end_rows, [4, 6, 8])
    np.testing.assert_array_equal(windows.last_candidate_rows, [3, 5, 7])


def test_sample_timestamp_filtering_uses_anchor_timestamps() -> None:
    store = _store()

    selected = store.sample_indices_by_timestamp_window(
        start_timestamp=40,
        end_timestamp=70,
    )

    np.testing.assert_array_equal(selected, [1, 2])
    np.testing.assert_array_equal(store.sample_timestamps(selected), [40, 60])


def test_fixed_context_filtering_rewrites_store_sample_rows() -> None:
    store = _store()

    fixed = store.with_fixed_context_length(
        context_length=2,
        history_seconds=20,
        warmup_rows=2,
    )

    np.testing.assert_array_equal(fixed.anchor_rows, [4, 6])
    np.testing.assert_array_equal(fixed.context_start_rows, [3, 5])
    np.testing.assert_array_equal(fixed.candidate_start_rows, [4, 6])


def test_context_windows_and_selected_span_use_selected_samples() -> None:
    store = _store()
    sample_indices = np.array([1, 2], dtype=np.int64)

    context = store.context_windows(sample_indices)

    np.testing.assert_array_equal(context.context_start_rows, [2, 4])
    np.testing.assert_array_equal(context.anchor_rows, [4, 6])
    np.testing.assert_array_equal(context.context_lengths, [3, 3])
    assert store.selected_row_span(sample_indices) == (2, 8)
