"""Canonical block dataset contract shared by acquisition and consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import SupportsInt, TypedDict, cast

import polars as pl
from polars.datatypes.classes import DataTypeClass

from ..config import ChainSpec

RpcBlock = Mapping[str, object]


class CanonicalBlockRow(TypedDict):
    block_number: int
    timestamp: int
    base_fee_per_gas: int
    gas_used: int
    chain_id: int
    gas_limit: int


@dataclass(frozen=True, slots=True)
class CanonicalBlockFieldSpec:
    name: str
    dtype: DataTypeClass


def _as_int(value: object) -> int:
    return int(cast(SupportsInt, value))


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
)

BLOCK_SCHEMA = {field.name: field.dtype for field in CANONICAL_BLOCK_FIELDS}
BLOCK_COLUMNS = tuple(BLOCK_SCHEMA)


def _validate_contract() -> None:
    field_names = tuple(field.name for field in CANONICAL_BLOCK_FIELDS)
    if len(field_names) != len(set(field_names)):
        raise RuntimeError(f"Duplicate canonical block fields are not allowed: {field_names}")


def build_canonical_block_row(block: RpcBlock, chain: ChainSpec) -> CanonicalBlockRow:
    try:
        return CanonicalBlockRow(
            block_number=_as_int(block["number"]),
            timestamp=_as_int(block["timestamp"]),
            base_fee_per_gas=(
                0 if block.get("baseFeePerGas") is None else _as_int(block["baseFeePerGas"])
            ),
            gas_used=_as_int(block["gasUsed"]),
            chain_id=chain.runtime.chain_id,
            gas_limit=_as_int(block["gasLimit"]),
        )
    except KeyError as exc:
        raise KeyError(f"Missing RPC block field while extracting canonical row: {exc}") from exc


def canonicalize_block_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return _select_canonical_columns(frame, strict_columns=False)


def validate_block_frame(frame: pl.DataFrame) -> None:
    if frame.height == 0:
        raise ValueError("Block dataset is empty")
    canonical = _select_canonical_columns(frame, strict_columns=True)
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
