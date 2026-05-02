"""Canonical block dataset contract shared by acquisition and consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import SupportsInt, TypedDict, cast

import polars as pl
from polars.datatypes.classes import DataTypeClass

from ..config.models import ChainSpec

RpcBlock = Mapping[str, object]


class CanonicalBlockRow(TypedDict):
    block_number: int
    timestamp: int
    base_fee_per_gas: int | None
    gas_used: int
    chain_id: int
    gas_limit: int
    tx_count: int
    block_size_bytes: int | None
    blob_gas_used: int | None
    excess_blob_gas: int | None
    priority_fee_p10: int | None
    priority_fee_p50: int | None
    priority_fee_p90: int | None
    priority_fee_spread: int | None


@dataclass(frozen=True, slots=True)
class CanonicalBlockFieldSpec:
    name: str
    dtype: DataTypeClass


def _as_int(value: object) -> int:
    if isinstance(value, str) and value.startswith("0x"):
        return int(value, 16)
    if isinstance(value, bytes | bytearray) and value.startswith(b"0x"):
        return int(value, 16)
    return int(cast(SupportsInt, value))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _as_int(value)


CANONICAL_BLOCK_FIELDS = (
    CanonicalBlockFieldSpec(
        name="block_number",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="timestamp",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="base_fee_per_gas",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="gas_used",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="chain_id",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="gas_limit",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="tx_count",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="block_size_bytes",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="blob_gas_used",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="excess_blob_gas",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="priority_fee_p10",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="priority_fee_p50",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="priority_fee_p90",
        dtype=pl.Int64,
    ),
    CanonicalBlockFieldSpec(
        name="priority_fee_spread",
        dtype=pl.Int64,
    ),
)

BLOCK_SCHEMA = {field.name: field.dtype for field in CANONICAL_BLOCK_FIELDS}
BLOCK_COLUMNS = tuple(BLOCK_SCHEMA)
REQUIRED_BLOCK_COLUMNS = (
    "block_number",
    "timestamp",
    "chain_id",
    "gas_used",
    "gas_limit",
    "tx_count",
)


def _validate_contract() -> None:
    field_names = tuple(field.name for field in CANONICAL_BLOCK_FIELDS)
    if len(field_names) != len(set(field_names)):
        raise RuntimeError(f"Duplicate canonical block fields are not allowed: {field_names}")


def build_canonical_block_row(
    block: RpcBlock,
    chain: ChainSpec,
    *,
    priority_fee_p10: int | None = None,
    priority_fee_p50: int | None = None,
    priority_fee_p90: int | None = None,
) -> CanonicalBlockRow:
    try:
        transactions = block.get("transactions")
        if not isinstance(transactions, list | tuple):
            raise TypeError("RPC block transactions field must be a sequence")
        priority_fee_spread = (
            None
            if priority_fee_p10 is None or priority_fee_p90 is None
            else priority_fee_p90 - priority_fee_p10
        )
        return CanonicalBlockRow(
            block_number=_as_int(block["number"]),
            timestamp=_as_int(block["timestamp"]),
            base_fee_per_gas=_optional_int(block.get("baseFeePerGas")),
            gas_used=_as_int(block["gasUsed"]),
            chain_id=chain.runtime.chain_id,
            gas_limit=_as_int(block["gasLimit"]),
            tx_count=len(transactions),
            block_size_bytes=_optional_int(block.get("size")),
            blob_gas_used=_optional_int(block.get("blobGasUsed")),
            excess_blob_gas=_optional_int(block.get("excessBlobGas")),
            priority_fee_p10=priority_fee_p10,
            priority_fee_p50=priority_fee_p50,
            priority_fee_p90=priority_fee_p90,
            priority_fee_spread=priority_fee_spread,
        )
    except KeyError as exc:
        raise KeyError(f"Missing RPC block field while extracting canonical row: {exc}") from exc


def canonicalize_block_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return _select_canonical_columns(frame, strict_columns=False)


def validate_block_frame(frame: pl.DataFrame) -> None:
    if frame.height == 0:
        raise ValueError("Block dataset is empty")
    canonical = _select_canonical_columns(frame, strict_columns=True)
    null_required = [
        column for column in REQUIRED_BLOCK_COLUMNS if canonical[column].null_count() > 0
    ]
    if null_required:
        raise ValueError(
            "Block dataset has null required columns: " + ", ".join(null_required)
        )
    if canonical["block_number"].n_unique() != canonical.height:
        raise ValueError("Block dataset must have unique block_number values")
    if canonical["chain_id"].n_unique() != 1:
        raise ValueError("Block dataset must contain exactly one chain_id")


def _select_canonical_columns(
    frame: pl.DataFrame,
    *,
    strict_columns: bool,
) -> pl.DataFrame:
    missing = [column for column in BLOCK_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Block dataset is missing required columns for canonical output: "
            + ", ".join(missing)
        )
    if strict_columns:
        unexpected = [column for column in frame.columns if column not in BLOCK_COLUMNS]
        if unexpected:
            raise ValueError(
                "Block dataset contains unexpected columns for canonical output: "
                + ", ".join(unexpected)
            )

    return frame.select(
        [
            pl.col(column).cast(dtype, strict=True).alias(column)
            for column, dtype in BLOCK_SCHEMA.items()
        ]
    )


_validate_contract()
