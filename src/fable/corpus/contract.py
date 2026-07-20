"""Canonical Corpus values."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import CorpusRequest
from .blocks import BlockFrame


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
    blocks: BlockFrame

    @model_validator(mode="after")
    def validate_ownership(self) -> Self:
        if self.blocks.definition != self.request.definition:
            raise ValueError("BlockFrame definition must match the Corpus request")
        if self.finalized_anchor.block_number < self.request.definition.last_block:
            raise ValueError("Finalized anchor precedes the requested last block")
        return self
