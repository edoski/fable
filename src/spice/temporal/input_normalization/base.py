"""Input-normalization seam types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import field_validator

from ...core.validation import validate_path_segment
from ...modeling.families.base import ConfigModel
from ...semantics import InputNormalizationSemantics
from ..scaling import FloatMatrix, IntVector, ScalerStats


class InputNormalizationConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="training.input_normalization.id")


class FitScalerFn(Protocol):
    def __call__(
        self,
        feature_matrix: FloatMatrix,
        *,
        context_start_rows: IntVector,
        anchor_rows: IntVector,
        sample_indices: IntVector,
    ) -> ScalerStats: ...


@dataclass(frozen=True, slots=True)
class CompiledInputNormalizationContract:
    input_normalization_id: str
    fit_scaler_fn: FitScalerFn

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


def coerce_input_normalization_config(
    payload: Mapping[str, object] | InputNormalizationConfig,
) -> InputNormalizationConfig:
    from .row_standard import RowStandardConfig
    from .window_weighted_standard import WindowWeightedStandardConfig

    if isinstance(payload, InputNormalizationConfig):
        raw_payload = payload.model_dump(mode="json")
        normalization_id = payload.id
    elif isinstance(payload, Mapping):
        raw_payload = dict(payload)
        normalization_id = raw_payload.get("id")
    else:
        raise TypeError("training.input_normalization must be a mapping or config model")
    if normalization_id == "row_standard":
        return RowStandardConfig.model_validate(raw_payload)
    if normalization_id == "window_weighted_standard":
        return WindowWeightedStandardConfig.model_validate(raw_payload)
    raise ValueError(
        "training.input_normalization.id must be one of: row_standard, window_weighted_standard"
    )


def compile_input_normalization_contract(
    config: InputNormalizationConfig,
) -> CompiledInputNormalizationContract:
    from .row_standard import RowStandardConfig, compile_input_normalization as compile_row_standard
    from .window_weighted_standard import (
        WindowWeightedStandardConfig,
        compile_input_normalization as compile_window_weighted_standard,
    )

    if config.id == "row_standard":
        return compile_row_standard(RowStandardConfig.model_validate(config))
    if config.id == "window_weighted_standard":
        return compile_window_weighted_standard(
            WindowWeightedStandardConfig.model_validate(config)
        )
    raise ValueError(
        "training.input_normalization.id must be one of: row_standard, window_weighted_standard"
    )
