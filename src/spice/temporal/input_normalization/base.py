"""Input-normalization seam types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import field_validator

from ...core.config_model import ConfigModel
from ...core.specs import (
    coerce_spec_config,
    lookup_local_spec,
    require_spec_config_from_table,
)
from ...core.validation import validate_path_segment
from ...semantics import InputNormalizationSemantics
from ..problem_store import CompiledProblemStore
from .scaling import IntVector, ScalerStats


class InputNormalizationConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="training.input_normalization.id")


class FitScalerFn(Protocol):
    def __call__(
        self,
        store: CompiledProblemStore,
        *,
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
        store: CompiledProblemStore,
        *,
        sample_indices: IntVector,
    ) -> ScalerStats:
        return self.fit_scaler_fn(
            store,
            sample_indices=sample_indices,
        )


@dataclass(frozen=True, slots=True)
class InputNormalizationSpec:
    config_type: type[InputNormalizationConfig]
    compile_contract: Callable[[Any], CompiledInputNormalizationContract]


def _compile_row_standard(
    config: Any,
) -> CompiledInputNormalizationContract:
    from .row_standard import compile_input_normalization

    return compile_input_normalization(config)


def _compile_window_weighted_standard(
    config: Any,
) -> CompiledInputNormalizationContract:
    from .window_weighted_standard import compile_input_normalization

    return compile_input_normalization(config)


def _input_normalization_specs() -> dict[str, InputNormalizationSpec]:
    from .row_standard import RowStandardConfig
    from .window_weighted_standard import WindowWeightedStandardConfig

    return {
        "row_standard": InputNormalizationSpec(
            config_type=RowStandardConfig,
            compile_contract=_compile_row_standard,
        ),
        "window_weighted_standard": InputNormalizationSpec(
            config_type=WindowWeightedStandardConfig,
            compile_contract=_compile_window_weighted_standard,
        ),
    }


def input_normalization_spec(normalization_id: str) -> InputNormalizationSpec:
    return lookup_local_spec(
        _input_normalization_specs(),
        normalization_id,
        "training.input_normalization.id",
    )


def coerce_input_normalization_config(
    payload: object,
) -> InputNormalizationConfig:
    return coerce_spec_config(
        payload,
        owner="training.input_normalization",
        base_config_type=InputNormalizationConfig,
        id_label="training.input_normalization.id",
        lookup_spec=input_normalization_spec,
        spec_config_type=lambda spec: spec.config_type,
    )


def compile_input_normalization_contract(
    config: InputNormalizationConfig,
) -> CompiledInputNormalizationContract:
    spec = input_normalization_spec(config.id)
    concrete_config = require_spec_config_from_table(
        config,
        config_id=config.id,
        lookup_spec=input_normalization_spec,
        spec_config_type=lambda entry: entry.config_type,
        label="input normalization config",
    )
    return spec.compile_contract(concrete_config)
