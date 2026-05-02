"""Shared base types for model-family configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar

from pydantic import field_validator

from ...core.config_model import ConfigModel as _ConfigModel
from ...core.validation import validate_path_segment

ModelIdT = TypeVar("ModelIdT", bound=str)
TunedScalar: TypeAlias = int | float
TunableScalarType: TypeAlias = type[int] | type[float]


@dataclass(frozen=True, slots=True)
class TunableFieldSpec:
    name: str
    value_type: TunableScalarType

    @property
    def parameter_name(self) -> str:
        return f"model.{self.name}"

    def coerce_sample(self, value: Any) -> TunedScalar:
        return self.value_type(value)


class ModelConfig(_ConfigModel, Generic[ModelIdT]):
    id: ModelIdT

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="model.id")


class ModelTuningSpaceConfig(_ConfigModel, Generic[ModelIdT]):
    id: ModelIdT

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="tuning_space.model.id")


class TunedModelParams(_ConfigModel, Generic[ModelIdT]):
    id: ModelIdT

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="tuned model params id")
