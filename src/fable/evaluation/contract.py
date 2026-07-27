"""Canonical durable evaluation schemas."""

import polars as pl

OBSERVATION_SCHEMA = pl.Schema(
    {
        "origin_block": pl.Int64,
        "predicted_action_k": pl.Int64,
        "predicted_minimum_log_base_fee": pl.Float64,
        "minimum_action_k": pl.Int64,
        "immediate_base_fee_per_gas": pl.Int64,
        "immediate_effective_priority_fee_per_gas_p50": pl.Int64,
        "selected_base_fee_per_gas": pl.Int64,
        "selected_effective_priority_fee_per_gas_p50": pl.Int64,
        "minimum_base_fee_per_gas": pl.Int64,
    }
)
