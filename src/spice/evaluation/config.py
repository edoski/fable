"""Evaluation config models."""

from __future__ import annotations

from pydantic import ConfigDict, Field, field_validator

from ..core.config_model import ConfigModel
from ..core.validation import validate_path_segment


class EvaluatorConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    id: str

    @field_validator("id")
    @classmethod
    def validate_id(_cls, value: str) -> str:
        return validate_path_segment(value, label="evaluation.id")


class PoissonReplayEvaluatorConfig(EvaluatorConfig):
    window_seconds: int = Field(gt=0)
    repetitions: int = Field(gt=0)
    seed: int = Field(ge=0)
    arrival_rate_per_second: float = Field(gt=0.0)

    @field_validator("id")
    @classmethod
    def validate_poisson_id(_cls, value: str) -> str:
        value = EvaluatorConfig.validate_id(value)
        if value != "poisson_replay":
            raise ValueError("evaluation.id must be poisson_replay")
        return value


class BlockPoissonReplayEvaluatorConfig(EvaluatorConfig):
    window_blocks: int = Field(gt=0)
    repetitions: int = Field(gt=0)
    seed: int = Field(ge=0)
    arrival_rate_per_block: float = Field(gt=0.0)

    @field_validator("id")
    @classmethod
    def validate_block_poisson_id(_cls, value: str) -> str:
        value = EvaluatorConfig.validate_id(value)
        if value != "block_poisson_replay":
            raise ValueError("evaluation.id must be block_poisson_replay")
        return value
