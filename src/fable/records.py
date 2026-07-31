"""Shared strict immutable record base."""

from pydantic import BaseModel, ConfigDict


class StrictFrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
