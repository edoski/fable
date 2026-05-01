from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from spice.config import coerce_features_config, coerce_problem_spec
from spice.config.models import ChainRuntimeSpec
from spice.features import compile_feature_contract
from spice.temporal.contracts import (
    compile_problem_contract,
    problem_runtime_metadata_from_compiler_payload,
    problem_runtime_metadata_payload,
)


def _feature_contract():
    return compile_feature_contract(
        features=coerce_features_config(
            {
                "id": "core_fee_dynamics",
                "outputs": [
                    "log_base_fee_per_gas",
                    "log_prev_gas_used",
                    "prev_fee_history_gas_used_ratio",
                ],
            }
        )
    )


def _blocks(row_count: int = 80) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "block_number": np.arange(400, 400 + row_count, dtype=np.int64),
            "timestamp": np.arange(row_count, dtype=np.int64) * 12,
            "base_fee_per_gas": np.full(row_count, 1_000_000_000, dtype=np.int64),
            "gas_used": np.full(row_count, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(row_count, 30_000_000, dtype=np.int64),
            "tx_count": np.full(row_count, 100, dtype=np.int64),
            "block_size_bytes": [None] * row_count,
            "blob_gas_used": [None] * row_count,
            "excess_blob_gas": [None] * row_count,
            "priority_fee_p10": np.full(row_count, 1_000_000, dtype=np.int64),
            "priority_fee_p50": np.full(row_count, 2_000_000, dtype=np.int64),
            "priority_fee_p90": np.full(row_count, 4_000_000, dtype=np.int64),
            "priority_fee_spread": np.full(row_count, 3_000_000, dtype=np.int64),
            "fee_history_gas_used_ratio": np.full(row_count, 0.6, dtype=np.float64),
            "chain_id": np.ones(row_count, dtype=np.int64),
        }
    )


def _problem(slot_spacing_id: str = "nominal"):
    return coerce_problem_spec(
        {
            "id": f"test_{slot_spacing_id}",
            "lookback_seconds": 24,
            "sample_count": 4,
            "max_delay_seconds": 36,
            "compiler": {
                "id": "observed_time_window",
                "slot_spacing": {"id": slot_spacing_id},
            },
            "execution_policy": {"id": "strict_deadline_miss"},
        }
    )


def _chain(nominal_block_time_seconds: float = 12.0) -> ChainRuntimeSpec:
    return ChainRuntimeSpec(
        chain_id=1,
        uses_poa_extra_data=False,
        nominal_block_time_seconds=nominal_block_time_seconds,
    )


def test_observed_time_window_builds_timestamp_bounded_windows() -> None:
    feature_contract = _feature_contract()
    feature_table = feature_contract.build_table(_blocks())
    contract = compile_problem_contract(
        problem=_problem("nominal"),
        feature_contract=feature_contract,
        chain_runtime=_chain(),
    )

    assert contract.execution_policy.requires_post_window_row is True
    store, runtime_metadata = contract.build_capability_store(feature_table)

    assert runtime_metadata.slot_spacing_id == "nominal"
    assert runtime_metadata.slot_spacing_seconds == 12.0
    assert runtime_metadata.capability_action_count == 4
    assert store.max_candidate_slots == 4
    np.testing.assert_array_equal(store.candidate_start_rows, store.anchor_rows)
    assert np.all(store.context_start_rows <= store.anchor_rows)
    assert np.all(store.candidate_end_rows > store.candidate_start_rows)


def test_observed_time_window_runtime_metadata_round_trips() -> None:
    feature_contract = _feature_contract()
    feature_table = feature_contract.build_table(_blocks())
    contract = compile_problem_contract(
        problem=_problem("nominal"),
        feature_contract=feature_contract,
        chain_runtime=_chain(),
    )
    _, runtime_metadata = contract.build_capability_store(feature_table)

    payload = problem_runtime_metadata_payload("observed_time_window", runtime_metadata)
    assert problem_runtime_metadata_from_compiler_payload(
        "observed_time_window",
        payload,
    ) == runtime_metadata


def test_observed_time_window_delay_store_rejects_artifact_action_width_mismatch() -> None:
    feature_contract = _feature_contract()
    feature_table = feature_contract.build_table(_blocks())
    contract = compile_problem_contract(
        problem=_problem("nominal"),
        feature_contract=feature_contract,
        chain_runtime=_chain(),
    )
    _, runtime_metadata = contract.build_capability_store(feature_table)

    with pytest.raises(ValueError, match="artifact action width"):
        contract.build_delay_store(
            feature_table,
            delay_seconds=12,
            compiler_runtime_metadata=runtime_metadata,
            max_candidate_slots=runtime_metadata.capability_action_count + 1,
        )


def test_recent_median_slot_spacing_is_scoped_to_slot_spacing() -> None:
    feature_contract = _feature_contract()
    blocks = _blocks(10).with_columns(
        pl.Series("timestamp", [0, 10, 21, 31, 42, 52, 63, 73, 85, 97])
    )
    feature_table = feature_contract.build_table(blocks)
    contract = compile_problem_contract(
        problem=_problem("recent_median"),
        feature_contract=feature_contract,
        chain_runtime=_chain(),
    )

    _, runtime_metadata = contract.build_capability_store(feature_table)

    assert runtime_metadata.slot_spacing_id == "recent_median"
    assert runtime_metadata.slot_spacing_seconds == 11.0
