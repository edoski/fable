from __future__ import annotations

import numpy as np
import pytest

from spice.temporal.problem_store import CompiledProblemStore


def _store_kwargs():
    return {
        "feature_matrix": np.zeros((8, 1), dtype=np.float32),
        "log_base_fees": np.log(
            np.array([10, 9, 4, 3, 8, 6, 2, 7], dtype=np.float32)
        ).astype(np.float32, copy=False),
        "timestamps": (np.arange(8, dtype=np.int64) * 10).astype(np.int64, copy=False),
        "anchor_rows": np.array([2, 4, 6], dtype=np.int64),
        "context_start_rows": np.array([0, 2, 4], dtype=np.int64),
        "candidate_start_rows": np.array([2, 4, 6], dtype=np.int64),
        "candidate_end_rows": np.array([5, 7, 8], dtype=np.int64),
        "max_candidate_slots": 2,
    }


def _store(**overrides) -> CompiledProblemStore:
    kwargs = _store_kwargs()
    kwargs.update(overrides)
    return CompiledProblemStore(**kwargs)


def test_store_constructor_coerces_arrays_to_runtime_dtypes() -> None:
    store = _store(
        feature_matrix=[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]],
        log_base_fees=np.arange(8, dtype=np.float64),
        timestamps=np.arange(8),
        anchor_rows=[2, 4, 6],
    )

    assert store.feature_matrix.dtype == np.float32
    assert store.log_base_fees.dtype == np.float32
    assert store.timestamps.dtype == np.int64
    assert store.anchor_rows.dtype == np.int64


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"feature_matrix": np.zeros((8,), dtype=np.float32)}, "two-dimensional"),
        ({"log_base_fees": np.zeros(7, dtype=np.float32)}, "log_base_fees"),
        ({"timestamps": np.array([0, 2, 1, 3, 4, 5, 6, 7])}, "timestamps"),
        ({"anchor_rows": np.array([2, 4], dtype=np.int64)}, "sample row arrays"),
        ({"context_start_rows": np.array([0, 5, 4], dtype=np.int64)}, "<= anchor"),
        ({"candidate_start_rows": np.array([2, 3, 6], dtype=np.int64)}, ">= anchor"),
        ({"candidate_end_rows": np.array([5, 7, 9], dtype=np.int64)}, "row count"),
        ({"max_candidate_slots": 0}, "positive"),
    ],
)
def test_store_constructor_rejects_invalid_row_geometry(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _store(**overrides)


def test_candidate_windows_include_generic_reachable_geometry() -> None:
    store = _store()

    windows = store.candidate_windows(np.array([0, 1, 2], dtype=np.int64))

    np.testing.assert_array_equal(windows.anchor_rows, [2, 4, 6])
    np.testing.assert_array_equal(windows.candidate_start_rows, [2, 4, 6])
    np.testing.assert_array_equal(windows.candidate_counts, [3, 3, 2])
    np.testing.assert_array_equal(windows.reachable_end_rows, [4, 6, 8])


def test_sample_timestamps_use_anchor_rows() -> None:
    store = _store()

    np.testing.assert_array_equal(
        store.sample_timestamps(np.array([1, 2], dtype=np.int64)),
        [40, 60],
    )


def test_sample_block_numbers_use_anchor_rows() -> None:
    store = _store(block_numbers=np.arange(100, 108, dtype=np.int64))

    np.testing.assert_array_equal(
        store.sample_block_numbers(np.array([1, 2], dtype=np.int64)),
        [104, 106],
    )


def test_store_constructor_rejects_non_increasing_block_numbers() -> None:
    with pytest.raises(ValueError, match="block_numbers"):
        _store(block_numbers=np.array([1, 2, 2, 4, 5, 6, 7, 8], dtype=np.int64))


@pytest.mark.parametrize(
    "sample_indices",
    [
        np.array([-1], dtype=np.int64),
        np.array([3], dtype=np.int64),
        np.array([[0]], dtype=np.int64),
    ],
)
def test_sample_views_reject_invalid_sample_indices(sample_indices) -> None:
    with pytest.raises(ValueError, match="sample_indices"):
        _store().sample_timestamps(sample_indices)


def test_context_row_multiplicities_count_selected_windows() -> None:
    store = _store()

    np.testing.assert_array_equal(
        store.context_row_multiplicities(np.array([0, 1], dtype=np.int64)),
        np.array([1, 1, 2, 1, 1, 0, 0, 0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="sample_indices"):
        store.context_row_multiplicities(np.array([], dtype=np.int64))


def test_context_windows_and_selected_span_use_selected_samples() -> None:
    store = _store()
    sample_indices = np.array([2, 1], dtype=np.int64)

    context = store.context_windows(sample_indices)

    np.testing.assert_array_equal(context.context_start_rows, [4, 2])
    np.testing.assert_array_equal(context.anchor_rows, [6, 4])
    np.testing.assert_array_equal(context.context_lengths, [3, 3])
    assert store.selected_row_span(sample_indices) == (2, 8)
