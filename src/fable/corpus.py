"""Canonical block rows, Corpus values, and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import polars as pl
from pydantic import UUID4, ConfigDict, Field, model_validator

from .addresses import corpus_blocks_path, corpus_json_path
from .config import CorpusDefinition, CorpusRequest
from .records import StrictFrozenRecord

_SCHEMA = pl.Schema(
    {
        "block_number": pl.Int64,
        "timestamp": pl.Int64,
        "chain_id": pl.Int64,
        "base_fee_per_gas": pl.Int64,
        "gas_used": pl.Int64,
        "gas_limit": pl.Int64,
        "tx_count": pl.Int64,
        "effective_priority_fee_per_gas_p50": pl.Int64,
        "effective_priority_fee_per_gas_p90": pl.Int64,
    }
)


class BlockFrame:
    """One isolated, validated frame of contiguous canonical block facts."""

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
        if not (frame["effective_priority_fee_per_gas_p50"] >= 0).all():
            raise ValueError("Block effective_priority_fee_per_gas_p50 values must be nonnegative")

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
        )
        selected._definition = definition
        return selected

    def to_polars(self) -> pl.DataFrame:
        return self._frame.clone()


class FinalizedAnchor(StrictFrozenRecord):
    block_number: int = Field(ge=0)
    block_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]+$")


class Corpus(StrictFrozenRecord):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: CorpusRequest
    finalized_anchor: FinalizedAnchor
    blocks: BlockFrame

    @model_validator(mode="after")
    def validate_ownership(self) -> Self:
        if self.blocks.definition != self.request.definition:
            raise ValueError("BlockFrame definition must match the Corpus request")
        if self.finalized_anchor.block_number < self.request.definition.last_block:
            raise ValueError("Finalized anchor precedes the requested last block")
        return self


class _CorpusDocument(StrictFrozenRecord):
    request: CorpusRequest
    finalized_anchor: FinalizedAnchor


def _load_corpus_document(storage_root: Path, corpus_id: UUID4) -> _CorpusDocument:
    document = _CorpusDocument.model_validate_json(
        corpus_json_path(storage_root, corpus_id).read_text(encoding="utf-8"),
        strict=True,
    )
    if document.request.corpus_id != corpus_id:
        raise ValueError("Corpus request UUID does not match the requested corpus")
    return document


def load_corpus_request(storage_root: Path, corpus_id: UUID4) -> CorpusRequest:
    return _load_corpus_document(storage_root, corpus_id).request


def load_corpus(storage_root: Path, corpus_id: UUID4) -> Corpus:
    document = _load_corpus_document(storage_root, corpus_id)
    return Corpus(
        request=document.request,
        finalized_anchor=document.finalized_anchor,
        blocks=BlockFrame(
            pl.read_parquet(corpus_blocks_path(storage_root, corpus_id)),
            document.request.definition,
        ),
    )
