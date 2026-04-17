"""Shared input-normalization seam types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import field_validator

from ...modeling.families.base import ConfigModel
from ...semantics import InputNormalizationSemantics
from ..scaling import FloatMatrix, IntVector, ScalerStats


def _validate_path_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty path segment")
    return value


class InputNormalizationConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="training.input_normalization.id")


@dataclass(frozen=True, slots=True)
class CompiledInputNormalizationContract:
    input_normalization_id: str
    fit_scaler_fn: Callable[[FloatMatrix, IntVector, IntVector, IntVector], ScalerStats]

    @property
    def semantics(self) -> InputNormalizationSemantics:
        return InputNormalizationSemantics(
            input_normalization_id=self.input_normalization_id,
        )

    def fit_scaler(
        self,
        feature_matrix: FloatMatrix,
        *,
        context_start_rows: IntVector,
        anchor_rows: IntVector,
        sample_indices: IntVector,
    ) -> ScalerStats:
        return self.fit_scaler_fn(
            feature_matrix,
            context_start_rows=context_start_rows,
            anchor_rows=anchor_rows,
            sample_indices=sample_indices,
        )


InputNormalizationConfigT = TypeVar(
    "InputNormalizationConfigT",
    bound=InputNormalizationConfig,
)


@dataclass(frozen=True, slots=True)
class InputNormalizationSpec(Generic[InputNormalizationConfigT]):
    id: str
    config_type: type[InputNormalizationConfigT]
    compile: Callable[[InputNormalizationConfigT], CompiledInputNormalizationContract]
