"""Owned canonical block rows."""

from __future__ import annotations

import polars as pl

from ..config import CorpusDefinition

_SCHEMA = pl.Schema(
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


class BlockFrame:
    """One isolated, validated frame of contiguous canonical block rows."""

    __slots__ = ("_definition", "_frame")

    def __init__(self, frame: pl.DataFrame, definition: CorpusDefinition) -> None:
        if frame.schema != _SCHEMA:
            raise ValueError(f"Block schema must be exactly {_SCHEMA}, got {frame.schema}")
        if any(frame.null_count().row(0)):
            raise ValueError("Block columns must be non-null")

        expected_count = definition.last_block - definition.first_block + 1
        if frame.height != expected_count:
            raise ValueError(f"Block row count must be {expected_count}, got {frame.height}")

        block_numbers = frame["block_number"]
        if (
            int(block_numbers[0]) != definition.first_block
            or int(block_numbers[-1]) != definition.last_block
            or (frame.height > 1 and not (block_numbers.diff().drop_nulls() == 1).all())
        ):
            raise ValueError("Block numbers must exactly match the definition range")
        if not (frame["chain_id"] == definition.chain_id).all():
            raise ValueError("Block chain_id values must match the definition")

        timestamps = frame["timestamp"]
        if not (timestamps >= 0).all():
            raise ValueError("Block timestamps must be nonnegative")
        if frame.height > 1 and not (timestamps.diff().drop_nulls() >= 0).all():
            raise ValueError("Block timestamps must be nondecreasing")
        if not (frame["base_fee_per_gas"] > 0).all():
            raise ValueError("Block base_fee_per_gas values must be positive")
        if not (frame["gas_limit"] > 0).all():
            raise ValueError("Block gas_limit values must be positive")
        if (
            not (frame["gas_used"] >= 0).all()
            or not (frame["gas_used"] <= frame["gas_limit"]).all()
        ):
            raise ValueError("Block gas_used values must be between zero and gas_limit")
        if not (frame["tx_count"] >= 0).all():
            raise ValueError("Block tx_count values must be nonnegative")

        self._frame = frame.clone()
        self._definition = definition

    @property
    def definition(self) -> CorpusDefinition:
        return self._definition

    def select_range(self, first_block: int, last_block: int) -> BlockFrame:
        if first_block > last_block:
            raise ValueError("Selected range must not be inverted")
        if first_block < self._definition.first_block or last_block > self._definition.last_block:
            raise ValueError("Selected range must be within the BlockFrame definition")

        definition = CorpusDefinition(
            chain_id=self._definition.chain_id,
            first_block=first_block,
            last_block=last_block,
        )
        selected: BlockFrame = object.__new__(BlockFrame)
        selected._frame = self._frame.slice(
            first_block - self._definition.first_block,
            last_block - first_block + 1,
        ).clone()
        selected._definition = definition
        return selected

    def to_polars(self) -> pl.DataFrame:
        return self._frame.clone()


__all__ = ["BlockFrame"]
