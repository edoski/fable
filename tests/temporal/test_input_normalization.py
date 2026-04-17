from __future__ import annotations

import numpy as np

from spice.temporal.input_normalization import (
    coerce_input_normalization_config,
    compile_input_normalization_contract,
)


def test_input_normalization_config_coerces_and_compiles() -> None:
    config = coerce_input_normalization_config({"id": "row_standard"})
    contract = compile_input_normalization_contract(config)

    assert config.id == "row_standard"
    assert contract.semantics.input_normalization_id == "row_standard"


def test_row_standard_and_window_weighted_standard_fit_different_statistics() -> None:
    feature_matrix = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    context_start_rows = np.array([0, 0], dtype=np.int64)
    anchor_rows = np.array([0, 2], dtype=np.int64)
    sample_indices = np.array([0, 1], dtype=np.int64)

    row_contract = compile_input_normalization_contract(
        coerce_input_normalization_config({"id": "row_standard"})
    )
    weighted_contract = compile_input_normalization_contract(
        coerce_input_normalization_config({"id": "window_weighted_standard"})
    )

    row_scaler = row_contract.fit_scaler(
        feature_matrix,
        context_start_rows=context_start_rows,
        anchor_rows=anchor_rows,
        sample_indices=sample_indices,
    )
    weighted_scaler = weighted_contract.fit_scaler(
        feature_matrix,
        context_start_rows=context_start_rows,
        anchor_rows=anchor_rows,
        sample_indices=sample_indices,
    )

    assert row_scaler.means == [1.0]
    assert weighted_scaler.means == [0.75]
