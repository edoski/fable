"""Canonical schemas for raw and enriched block datasets."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

ENRICHED_BLOCK_SCHEMA = {
    "block_number": pl.Int64,
    "timestamp": pl.Int64,
    "base_fee_per_gas": pl.Int64,
    "gas_used": pl.Int64,
    "chain_id": pl.Int64,
    "gas_limit": pl.Int64,
}
ENRICHED_BLOCK_COLUMNS = tuple(ENRICHED_BLOCK_SCHEMA)
RAW_BLOCK_COLUMNS = tuple(column for column in ENRICHED_BLOCK_COLUMNS if column != "gas_limit")


def canonicalize_enriched_block_frame(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in ENRICHED_BLOCK_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Block dataset is missing required columns for canonical enriched output: "
            + ", ".join(missing)
        )

    return frame.select(
        [
            pl.col(column).cast(dtype, strict=True).alias(column)
            for column, dtype in ENRICHED_BLOCK_SCHEMA.items()
        ]
    )


def validate_enriched_block_schema(
    schema: Mapping[str, pl.DataType],
    *,
    context: str = "Block dataset",
) -> None:
    missing = [column for column in ENRICHED_BLOCK_COLUMNS if column not in schema]
    extra = [column for column in schema if column not in ENRICHED_BLOCK_SCHEMA]
    mismatched = [
        f"{column}: expected {ENRICHED_BLOCK_SCHEMA[column]}, got {schema[column]}"
        for column in ENRICHED_BLOCK_COLUMNS
        if column in schema and schema[column] != ENRICHED_BLOCK_SCHEMA[column]
    ]
    problems: list[str] = []
    if missing:
        problems.append("missing columns: " + ", ".join(missing))
    if extra:
        problems.append("extra columns: " + ", ".join(extra))
    if mismatched:
        problems.append("dtype mismatches: " + ", ".join(mismatched))
    if problems:
        raise ValueError(
            f"{context} does not match canonical enriched block schema ("
            + "; ".join(problems)
            + ")"
        )


def validate_enriched_block_frame(frame: pl.DataFrame) -> None:
    validate_enriched_block_schema(frame.schema)
    if frame.height == 0:
        raise ValueError("Block dataset is empty")

    null_counts = frame.null_count().row(0, named=True)
    null_columns = [column for column in ENRICHED_BLOCK_COLUMNS if null_counts[column] > 0]
    if null_columns:
        raise ValueError(
            "Block dataset contains null values in required columns: " + ", ".join(null_columns)
        )

    duplicated = frame.select(pl.col("block_number").is_duplicated().any()).item()
    if duplicated:
        raise ValueError("Block dataset contains duplicate block_number values")

    chain_ids = frame["chain_id"].unique()
    if len(chain_ids) != 1:
        raise ValueError("Block dataset must contain exactly one chain_id")
