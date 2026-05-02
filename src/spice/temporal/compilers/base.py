"""Shared problem-compiler config type."""

from __future__ import annotations

from pydantic import field_validator

from ...core.config_model import ConfigModel
from ...core.validation import validate_path_segment


class ProblemCompilerConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="problem.compiler.id")
