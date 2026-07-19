"""Canonical Corpus values and block schema."""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from ..config import CorpusRequest


class FinalizedAnchor(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    block_number: int = Field(ge=0)
    block_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]+$")


class Corpus(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    request: CorpusRequest
    finalized_anchor: FinalizedAnchor
    blocks: pl.DataFrame


_BLOCK_SCHEMA = pl.Schema(
    {
        "block_number": pl.Int64,
        "timestamp": pl.Int64,
        "chain_id": pl.Int64,
        "base_fee_per_gas": pl.Int64,
        "gas_used": pl.Int64,
        "gas_limit": pl.Int64,
        "tx_count": pl.Int64,
    }
)

_CANONICAL_BLOCK_SCHEMA = pl.Schema(
    {
        "block_number": pl.Int64,
        "timestamp": pl.Int64,
        "base_fee_per_gas": pl.Int64,
        "gas_used": pl.Int64,
        "chain_id": pl.Int64,
        "gas_limit": pl.Int64,
        "tx_count": pl.Int64,
        "block_size_bytes": pl.Int64,
        "blob_gas_used": pl.Int64,
        "excess_blob_gas": pl.Int64,
        "priority_fee_p10": pl.Int64,
        "priority_fee_p50": pl.Int64,
        "priority_fee_p90": pl.Int64,
        "priority_fee_spread": pl.Int64,
    }
)
_CANONICAL_BLOCK_COLUMNS = tuple(_CANONICAL_BLOCK_SCHEMA)


def _select_canonical_columns(
    frame: pl.DataFrame,
    *,
    strict_columns: bool,
) -> pl.DataFrame:
    missing = [column for column in _CANONICAL_BLOCK_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Block corpus is missing required columns for canonical output: " + ", ".join(missing)
        )
    if strict_columns:
        unexpected = [column for column in frame.columns if column not in _CANONICAL_BLOCK_COLUMNS]
        if unexpected:
            raise ValueError(
                "Block corpus contains unexpected columns for canonical output: "
                + ", ".join(unexpected)
            )

    return frame.select(
        [
            pl.col(column).cast(dtype, strict=True).alias(column)
            for column, dtype in _CANONICAL_BLOCK_SCHEMA.items()
        ]
    )
