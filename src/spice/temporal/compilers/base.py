"""Shared problem-compiler config type."""

from __future__ import annotations

from pydantic import field_validator

from ...core.validation import validate_path_segment
from ...modeling.families.base import ConfigModel

class ProblemCompilerConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="problem.compiler.id")
