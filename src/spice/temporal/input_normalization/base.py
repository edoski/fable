"""Shared input-normalization seam types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import field_validator

from ...core.closed_dispatch import (
    config_payload_and_id,
    unknown_id_error,
    validate_path_segment,
)
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


@dataclass(frozen=True, slots=True)
class InputNormalizationSpec:
    id: str
    config_type: type[InputNormalizationConfig]
    compile: Callable[[InputNormalizationConfig], CompiledInputNormalizationContract]


def _compile_row_standard(
    config: InputNormalizationConfig,
) -> CompiledInputNormalizationContract:
    from .row_standard import RowStandardConfig, compile_input_normalization

    return compile_input_normalization(cast(RowStandardConfig, config))


def _compile_window_weighted_standard(
    config: InputNormalizationConfig,
) -> CompiledInputNormalizationContract:
    from .window_weighted_standard import (
        WindowWeightedStandardConfig,
        compile_input_normalization,
    )

    return compile_input_normalization(cast(WindowWeightedStandardConfig, config))


def input_normalization_spec(normalization_id: str) -> InputNormalizationSpec:
    if normalization_id == "row_standard":
        from .row_standard import RowStandardConfig

        return InputNormalizationSpec(
            id="row_standard",
            config_type=RowStandardConfig,
            compile=_compile_row_standard,
        )
    if normalization_id == "window_weighted_standard":
        from .window_weighted_standard import WindowWeightedStandardConfig

        return InputNormalizationSpec(
            id="window_weighted_standard",
            config_type=WindowWeightedStandardConfig,
            compile=_compile_window_weighted_standard,
        )
    raise unknown_id_error(
        field_name="training.input_normalization.id",
        component_id=normalization_id,
        known_ids=("row_standard", "window_weighted_standard"),
    )


def coerce_input_normalization_config(
    payload: Mapping[str, object] | InputNormalizationConfig,
) -> InputNormalizationConfig:
    raw_payload, normalization_id = config_payload_and_id(
        payload,
        config_type=InputNormalizationConfig,
        field_name="training.input_normalization.id",
        mapping_label="training.input_normalization",
    )
    spec = input_normalization_spec(normalization_id)
    return spec.config_type.model_validate(raw_payload)


def compile_input_normalization_contract(
    config: InputNormalizationConfig,
) -> CompiledInputNormalizationContract:
    return input_normalization_spec(config.id).compile(config)
