from __future__ import annotations

import numpy as np

from spice.temporal.input_normalization import (
    ScalerStats,
    coerce_input_normalization_config,
    compile_input_normalization_contract,
    transform_feature_matrix,
)
from spice.temporal.problem_store import CompiledProblemStore


def _store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        log_base_fees=np.zeros(3, dtype=np.float32),
        timestamps=np.arange(3, dtype=np.int64),
        anchor_rows=np.array([0, 2], dtype=np.int64),
        context_start_rows=np.array([0, 0], dtype=np.int64),
        candidate_start_rows=np.array([0, 2], dtype=np.int64),
        candidate_end_rows=np.array([1, 3], dtype=np.int64),
        max_candidate_slots=1,
    )


def test_row_standard_and_window_weighted_standard_fit_different_statistics() -> None:
    store = _store()
    sample_indices = np.array([0, 1], dtype=np.int64)

    row_contract = compile_input_normalization_contract(
        coerce_input_normalization_config({"id": "row_standard"})
    )
    weighted_contract = compile_input_normalization_contract(
        coerce_input_normalization_config({"id": "window_weighted_standard"})
    )

    row_scaler = row_contract.fit_scaler(
        store,
        sample_indices=sample_indices,
    )
    weighted_scaler = weighted_contract.fit_scaler(
        store,
        sample_indices=sample_indices,
    )

    assert row_scaler.means == [1.0]
    assert weighted_scaler.means == [0.75]
    np.testing.assert_allclose(row_scaler.scales, [np.sqrt(2.0 / 3.0)])
    np.testing.assert_allclose(weighted_scaler.scales, [np.sqrt(0.6875)])


def test_standard_scaler_stats_use_unit_scale_for_constant_features() -> None:
    store = _store()
    constant_store = CompiledProblemStore(
        feature_matrix=np.ones((3, 1), dtype=np.float32),
        log_base_fees=store.log_base_fees,
        timestamps=store.timestamps,
        anchor_rows=store.anchor_rows,
        context_start_rows=store.context_start_rows,
        candidate_start_rows=store.candidate_start_rows,
        candidate_end_rows=store.candidate_end_rows,
        max_candidate_slots=store.max_candidate_slots,
    )
    row_contract = compile_input_normalization_contract(
        coerce_input_normalization_config({"id": "row_standard"})
    )

    scaler = row_contract.fit_scaler(
        constant_store,
        sample_indices=np.array([0, 1], dtype=np.int64),
    )

    assert scaler.means == [1.0]
    assert scaler.scales == [1.0]


def test_transform_feature_matrix_uses_safe_scales_and_float32() -> None:
    transformed = transform_feature_matrix(
        np.array([[2.0, 4.0]], dtype=np.float32),
        ScalerStats(means=[1.0, 1.0], scales=[0.0, -2.0]),
    )

    assert transformed.dtype == np.float32
    np.testing.assert_allclose(transformed, np.array([[1.0, 3.0]], dtype=np.float32))
