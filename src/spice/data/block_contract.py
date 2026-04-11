"""Canonical block dataset contract shared by acquisition and consumers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandera.polars as pa
import polars as pl

from ..core.config import ChainConfig

RpcBlock = Mapping[str, object]
BlockFieldExtractor = Callable[[RpcBlock, ChainConfig], int]


@dataclass(frozen=True, slots=True)
class CanonicalBlockFieldSpec:
    name: str
    dtype: Any
    extract: BlockFieldExtractor


def _as_int(value: object) -> int:
    return int(cast(Any, value))


CANONICAL_BLOCK_FIELDS = (
    CanonicalBlockFieldSpec(
        name="block_number",
        dtype=pl.Int64,
        extract=lambda block, _chain: _as_int(block["number"]),
    ),
    CanonicalBlockFieldSpec(
        name="timestamp",
        dtype=pl.Int64,
        extract=lambda block, _chain: _as_int(block["timestamp"]),
    ),
    CanonicalBlockFieldSpec(
        name="base_fee_per_gas",
        dtype=pl.Int64,
        extract=lambda block, _chain: 0
        if block.get("baseFeePerGas") is None
        else _as_int(block["baseFeePerGas"]),
    ),
    CanonicalBlockFieldSpec(
        name="gas_used",
        dtype=pl.Int64,
        extract=lambda block, _chain: _as_int(block["gasUsed"]),
    ),
    CanonicalBlockFieldSpec(
        name="chain_id",
        dtype=pl.Int64,
        extract=lambda _block, chain: chain.chain_id,
    ),
    CanonicalBlockFieldSpec(
        name="gas_limit",
        dtype=pl.Int64,
        extract=lambda block, _chain: _as_int(block["gasLimit"]),
    ),
)

BLOCK_SCHEMA = {field.name: field.dtype for field in CANONICAL_BLOCK_FIELDS}
BLOCK_COLUMNS = tuple(BLOCK_SCHEMA)

BLOCK_FRAME_SCHEMA = pa.DataFrameSchema(
    {
        column: pa.Column(dtype, nullable=False)
        for column, dtype in BLOCK_SCHEMA.items()
    },
    strict=True,
    unique="block_number",
    checks=[
        pa.Check(
            lambda data: data.lazyframe.collect().height > 0,
            error="Block dataset is empty",
        ),
        pa.Check(
            lambda data: (
                data.lazyframe.select(pl.col("chain_id").n_unique()).collect().item() == 1
            ),
            error="Block dataset must contain exactly one chain_id",
        ),
    ],
)


def _validate_contract() -> None:
    field_names = tuple(field.name for field in CANONICAL_BLOCK_FIELDS)
    if len(field_names) != len(set(field_names)):
        raise RuntimeError(f"Duplicate canonical block fields are not allowed: {field_names}")


def build_canonical_block_row(block: RpcBlock, chain: ChainConfig) -> dict[str, int]:
    row: dict[str, int] = {}
    for field in CANONICAL_BLOCK_FIELDS:
        try:
            value = field.extract(block, chain)
        except KeyError as exc:
            raise KeyError(
                f"Missing RPC block field while extracting canonical field {field.name}: {exc}"
            ) from exc
        row[field.name] = int(value)
    return row


def canonicalize_block_frame(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in BLOCK_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Block dataset is missing required columns for canonical output: "
            + ", ".join(missing)
        )

    return frame.select(
        [
            pl.col(column).cast(dtype, strict=True).alias(column)
            for column, dtype in BLOCK_SCHEMA.items()
        ]
    )


def validate_block_frame(frame: pl.DataFrame) -> None:
    BLOCK_FRAME_SCHEMA.validate(frame)


_validate_contract()
