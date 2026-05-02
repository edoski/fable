"""Shared strict Pydantic base for config models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
