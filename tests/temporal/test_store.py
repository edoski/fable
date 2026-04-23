from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from spice.config import coerce_feature_set_config, coerce_problem_spec
from spice.config.models import ChainRuntimeSpec
from spice.features import compile_feature_contract
from spice.temporal.contracts import compile_problem_contract


def _realization_policy_config() -> dict[str, object]:
    return {"id": "strict_deadline_miss"}


def test_temporal_store_uses_real_timestamps_for_context_and_candidates() -> None:
    feature_contract = compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": "test_timestamp_native",
                "family": {"id": "time_native"},
                "outputs": ["seconds_since_previous_block", "elapsed_seconds"],
            }
        )
    )
    blocks = pl.DataFrame(
        {
            "block_number": np.arange(100, 107, dtype=np.int64),
            "timestamp": np.array([0, 5, 11, 18, 27, 29, 40], dtype=np.int64),
            "base_fee_per_gas": np.full(7, 1_000_000_000, dtype=np.int64),
            "gas_used": np.full(7, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(7, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(7, dtype=np.int64),
        }
    )
    feature_table = feature_contract.build_table(blocks)
    contract = compile_problem_contract(
        problem=coerce_problem_spec(
            {
                "id": "test_timestamp_native",
                "lookback_seconds": 10,
                "sample_count": 4,
                "max_delay_seconds": 12,
                "compiler": {"id": "timestamp_native"},
                "realization_policy": _realization_policy_config(),
            }
        ),
        feature_contract=feature_contract,
        chain_runtime=ChainRuntimeSpec(
            chain_id=1,
            uses_poa_extra_data=False,
            nominal_block_time_seconds=12.0,
        ),
    )
    store, _ = contract.build_capability_store(feature_table)

    np.testing.assert_array_equal(
        store.anchor_rows,
        np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        store.context_start_rows,
        np.array([0, 0, 1, 2, 3, 4, 6], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        store.candidate_end_rows - store.candidate_start_rows,
        np.array([3, 2, 2, 3, 2, 2, 1], dtype=np.int64),
    )
    assert store.max_candidate_slots == 3


def test_estimated_block_store_uses_corpus_calibration_for_row_geometry() -> None:
    feature_contract = compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": "test_estimated_block",
                "family": {"id": "time_native"},
                "outputs": ["seconds_since_previous_block", "elapsed_seconds"],
            }
        )
    )
    blocks = pl.DataFrame(
        {
            "block_number": np.arange(200, 208, dtype=np.int64),
            "timestamp": np.array([0, 7, 15, 26, 34, 45, 53, 65], dtype=np.int64),
            "base_fee_per_gas": np.full(8, 1_000_000_000, dtype=np.int64),
            "gas_used": np.full(8, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(8, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(8, dtype=np.int64),
        }
    )
    feature_table = feature_contract.build_table(blocks)
    contract = compile_problem_contract(
        problem=coerce_problem_spec(
            {
                "id": "test_estimated_block",
                "lookback_seconds": 20,
                "sample_count": 3,
                "max_delay_seconds": 18,
                "compiler": {"id": "estimated_block"},
                "realization_policy": _realization_policy_config(),
            }
        ),
        feature_contract=feature_contract,
        chain_runtime=ChainRuntimeSpec(
            chain_id=1,
            uses_poa_extra_data=False,
            nominal_block_time_seconds=12.0,
        ),
    )

    store, runtime_metadata = contract.build_capability_store(feature_table)

    assert runtime_metadata.calibrated_interval_seconds == 8.0
    assert runtime_metadata.lookback_interval_seconds == 8.0
    assert runtime_metadata.candidate_interval_seconds == 8.0
    np.testing.assert_array_equal(store.anchor_rows, np.array([1, 2, 3, 4, 5], dtype=np.int64))
    np.testing.assert_array_equal(
        store.context_start_rows,
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        store.candidate_end_rows,
        np.array([4, 5, 6, 7, 8], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        store.candidate_counts,
        np.array([2, 2, 2, 2, 2], dtype=np.int64),
    )
    assert store.max_candidate_slots == 2


def test_estimated_block_supports_nominal_lookback_and_mean_calibrated_candidates() -> None:
    feature_contract = compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": "test_estimated_block_mixed_policy",
                "family": {"id": "time_native"},
                "outputs": ["seconds_since_previous_block", "elapsed_seconds"],
            }
        )
    )
    blocks = pl.DataFrame(
        {
            "block_number": np.arange(300, 309, dtype=np.int64),
            "timestamp": np.array([0, 7, 15, 26, 34, 45, 53, 65, 74], dtype=np.int64),
            "base_fee_per_gas": np.full(9, 1_000_000_000, dtype=np.int64),
            "gas_used": np.full(9, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(9, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(9, dtype=np.int64),
        }
    )
    feature_table = feature_contract.build_table(blocks)
    contract = compile_problem_contract(
        problem=coerce_problem_spec(
            {
                "id": "test_estimated_block_mixed_policy",
                "lookback_seconds": 24,
                "sample_count": 3,
                "max_delay_seconds": 28,
                "compiler": {
                    "id": "estimated_block",
                    "lookback_interval_source": "nominal_chain_runtime",
                    "candidate_interval_source": "calibrated",
                    "calibrated_interval_statistic": "mean",
                },
                "realization_policy": _realization_policy_config(),
            }
        ),
        feature_contract=feature_contract,
        chain_runtime=ChainRuntimeSpec(
            chain_id=1,
            uses_poa_extra_data=False,
            nominal_block_time_seconds=12.0,
        ),
    )

    store, runtime_metadata = contract.build_capability_store(feature_table)

    assert runtime_metadata.lookback_interval_seconds == 12.0
    assert runtime_metadata.lookback_steps == 2
    assert runtime_metadata.candidate_interval_seconds == np.mean(
        np.array([7, 8, 11, 8, 11, 8, 12, 9], dtype=np.float64)
    )
    np.testing.assert_array_equal(store.context_start_rows, store.anchor_rows - 1)
    assert store.max_candidate_slots == 3


def test_timestamp_future_window_uses_future_only_fixed_action_windows() -> None:
    feature_contract = compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": "test_timestamp_future_window",
                "family": {"id": "time_native"},
                "outputs": ["seconds_since_previous_block", "elapsed_seconds"],
            }
        )
    )
    blocks = pl.DataFrame(
        {
            "block_number": np.arange(400, 408, dtype=np.int64),
            "timestamp": np.array([0, 11, 23, 35, 46, 58, 71, 84], dtype=np.int64),
            "base_fee_per_gas": np.full(8, 1_000_000_000, dtype=np.int64),
            "gas_used": np.full(8, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(8, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(8, dtype=np.int64),
        }
    )
    feature_table = feature_contract.build_table(blocks)
    contract = compile_problem_contract(
        problem=coerce_problem_spec(
            {
                "id": "test_timestamp_future_window",
                "lookback_seconds": 24,
                "sample_count": 4,
                "max_delay_seconds": 36,
                "compiler": {
                    "id": "timestamp_future_window",
                    "action_interval_estimator": {"id": "nominal"},
                },
                "realization_policy": _realization_policy_config(),
            }
        ),
        feature_contract=feature_contract,
        chain_runtime=ChainRuntimeSpec(
            chain_id=1,
            uses_poa_extra_data=False,
            nominal_block_time_seconds=12.0,
        ),
    )

    store, runtime_metadata = contract.build_capability_store(feature_table)

    np.testing.assert_array_equal(
        store.anchor_rows,
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        store.context_start_rows,
        np.array([0, 0, 0, 1, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        store.candidate_counts,
        np.array([3, 3, 3, 3, 2], dtype=np.int64),
    )
    assert runtime_metadata.action_interval_seconds == 12.0
    assert runtime_metadata.action_interval_estimator_id == "nominal"
    assert runtime_metadata.capability_action_count == 3
    assert store.max_candidate_slots == 3
    assert store.action_space_mode.value == "fixed_ex_ante"


def test_timestamp_future_window_supports_current_row_fixed_action_windows() -> None:
    feature_contract = compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": "test_timestamp_future_window_current",
                "family": {"id": "time_native"},
                "outputs": ["seconds_since_previous_block", "elapsed_seconds"],
            }
        )
    )
    blocks = pl.DataFrame(
        {
            "block_number": np.arange(500, 508, dtype=np.int64),
            "timestamp": np.array([0, 11, 23, 35, 46, 58, 71, 84], dtype=np.int64),
            "base_fee_per_gas": np.full(8, 1_000_000_000, dtype=np.int64),
            "gas_used": np.full(8, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(8, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(8, dtype=np.int64),
        }
    )
    feature_table = feature_contract.build_table(blocks)
    contract = compile_problem_contract(
        problem=coerce_problem_spec(
            {
                "id": "test_timestamp_future_window_current",
                "lookback_seconds": 24,
                "sample_count": 4,
                "max_delay_seconds": 36,
                "compiler": {
                    "id": "timestamp_future_window",
                    "action_interval_estimator": {"id": "nominal"},
                    "candidate_start_mode": "current_row",
                },
                "realization_policy": _realization_policy_config(),
            }
        ),
        feature_contract=feature_contract,
        chain_runtime=ChainRuntimeSpec(
            chain_id=1,
            uses_poa_extra_data=False,
            nominal_block_time_seconds=12.0,
        ),
    )

    store, runtime_metadata = contract.build_capability_store(feature_table)

    np.testing.assert_array_equal(
        store.anchor_rows,
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        store.candidate_start_rows,
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        store.candidate_counts,
        np.array([4, 4, 4, 4, 3], dtype=np.int64),
    )
    assert runtime_metadata.capability_action_count == 4
    assert store.max_candidate_slots == 4
    assert store.action_space_mode.value == "fixed_ex_ante"


def test_timestamp_future_window_fixed_ex_ante_derives_width_from_realized_training_window(
) -> None:
    feature_contract = compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": "test_timestamp_future_window_avalanche_like",
                "family": {"id": "time_native"},
                "outputs": ["seconds_since_previous_block", "elapsed_seconds"],
            }
        )
    )
    blocks = pl.DataFrame(
        {
            "block_number": np.arange(600, 650, dtype=np.int64),
            "timestamp": np.arange(50, dtype=np.int64),
            "base_fee_per_gas": np.full(50, 1_000_000_000, dtype=np.int64),
            "gas_used": np.full(50, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(50, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(50, dtype=np.int64),
        }
    )
    feature_table = feature_contract.build_table(blocks)
    contract = compile_problem_contract(
        problem=coerce_problem_spec(
            {
                "id": "test_timestamp_future_window_avalanche_like",
                "lookback_seconds": 24,
                "sample_count": 4,
                "max_delay_seconds": 36,
                "compiler": {
                    "id": "timestamp_future_window",
                    "action_interval_estimator": {"id": "nominal"},
                    "candidate_start_mode": "current_row",
                },
                "realization_policy": _realization_policy_config(),
            }
        ),
        feature_contract=feature_contract,
        chain_runtime=ChainRuntimeSpec(
            chain_id=43114,
            uses_poa_extra_data=False,
            nominal_block_time_seconds=1.6,
        ),
    )

    store, runtime_metadata = contract.build_capability_store(feature_table)

    assert runtime_metadata.capability_action_count == 23
    assert store.max_candidate_slots > runtime_metadata.capability_action_count
    assert store.max_candidate_slots == int(store.candidate_counts.max())

    delay_store = contract.build_delay_store(
        feature_table,
        36,
        compiler_runtime_metadata=runtime_metadata,
        max_candidate_slots=runtime_metadata.capability_action_count,
    )

    assert delay_store.max_candidate_slots == runtime_metadata.capability_action_count
    assert int(delay_store.candidate_counts.max()) > delay_store.max_candidate_slots


def test_timestamp_future_window_supports_recent_delta_interval_estimator() -> None:
    feature_contract = compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": "test_timestamp_future_window_recent_deltas",
                "family": {"id": "time_native"},
                "outputs": ["seconds_since_previous_block", "elapsed_seconds"],
            }
        )
    )
    blocks = pl.DataFrame(
        {
            "block_number": np.arange(700, 710, dtype=np.int64),
            "timestamp": np.array([0, 10, 21, 31, 42, 52, 63, 73, 85, 97], dtype=np.int64),
            "base_fee_per_gas": np.full(10, 1_000_000_000, dtype=np.int64),
            "gas_used": np.full(10, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(10, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(10, dtype=np.int64),
        }
    )
    feature_table = feature_contract.build_table(blocks)
    contract = compile_problem_contract(
        problem=coerce_problem_spec(
            {
                "id": "test_timestamp_future_window_recent_deltas",
                "lookback_seconds": 24,
                "sample_count": 4,
                "max_delay_seconds": 36,
                "compiler": {
                    "id": "timestamp_future_window",
                    "action_interval_estimator": {
                        "id": "recent_deltas",
                        "window_blocks": 4,
                        "statistic": "median",
                    },
                },
                "realization_policy": _realization_policy_config(),
            }
        ),
        feature_contract=feature_contract,
        chain_runtime=ChainRuntimeSpec(
            chain_id=1,
            uses_poa_extra_data=False,
            nominal_block_time_seconds=12.0,
        ),
    )

    _, runtime_metadata = contract.build_capability_store(feature_table)

    assert runtime_metadata.action_interval_estimator_id == "recent_deltas"
    assert runtime_metadata.action_interval_seconds == 11.5
    assert runtime_metadata.capability_action_count == 3


def test_timestamp_future_window_rejects_recent_delta_quantile_without_quantile() -> None:
    feature_contract = compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": "test_timestamp_future_window_recent_deltas_quantile",
                "family": {"id": "time_native"},
                "outputs": ["seconds_since_previous_block", "elapsed_seconds"],
            }
        )
    )

    with pytest.raises(ValidationError, match="recent_deltas quantile statistic requires quantile"):
        compile_problem_contract(
            problem=coerce_problem_spec(
                {
                    "id": "test_timestamp_future_window_recent_deltas_quantile",
                    "lookback_seconds": 24,
                    "sample_count": 4,
                    "max_delay_seconds": 36,
                    "compiler": {
                        "id": "timestamp_future_window",
                        "action_interval_estimator": {
                            "id": "recent_deltas",
                            "window_blocks": 4,
                            "statistic": "quantile",
                        },
                    },
                    "realization_policy": _realization_policy_config(),
                }
            ),
            feature_contract=feature_contract,
            chain_runtime=ChainRuntimeSpec(
                chain_id=1,
                uses_poa_extra_data=False,
                nominal_block_time_seconds=12.0,
            ),
        )
