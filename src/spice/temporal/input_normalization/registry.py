"""Closed dispatch for supported input-normalization modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from ...core.errors import ConfigResolutionError
from .base import (
    CompiledInputNormalizationContract,
    InputNormalizationConfig,
    InputNormalizationSpec,
)
from .row_standard import RowStandardConfig, compile_input_normalization as compile_row_standard
from .window_weighted_standard import (
    WindowWeightedStandardConfig,
    compile_input_normalization as compile_window_weighted_standard,
)

_INPUT_NORMALIZATION_SPECS: dict[str, InputNormalizationSpec[InputNormalizationConfig]] = {
    "row_standard": InputNormalizationSpec(
        id="row_standard",
        config_type=RowStandardConfig,
        compile=compile_row_standard,
    ),
    "window_weighted_standard": InputNormalizationSpec(
        id="window_weighted_standard",
        config_type=WindowWeightedStandardConfig,
        compile=compile_window_weighted_standard,
    ),
}


def input_normalization_spec(
    normalization_id: str,
) -> InputNormalizationSpec[InputNormalizationConfig]:
    try:
        return _INPUT_NORMALIZATION_SPECS[normalization_id]
    except KeyError as exc:
        known = ", ".join(sorted(_INPUT_NORMALIZATION_SPECS))
        raise ConfigResolutionError(
            "Unknown training.input_normalization.id: "
            f"{normalization_id}. Known values: {known}"
        ) from exc


def coerce_input_normalization_config(
    payload: Mapping[str, object] | InputNormalizationConfig,
) -> InputNormalizationConfig:
    if isinstance(payload, InputNormalizationConfig):
        raw_payload = payload.model_dump(mode="json")
        normalization_id = payload.id
    else:
        raw_payload = dict(payload)
        normalization_id = _mapping_input_normalization_id(raw_payload)
    spec = input_normalization_spec(normalization_id)
    return spec.config_type.model_validate(raw_payload)


def compile_input_normalization_contract(
    config: InputNormalizationConfig,
) -> CompiledInputNormalizationContract:
    return input_normalization_spec(config.id).compile(config)


def _mapping_input_normalization_id(payload: Mapping[str, object]) -> str:
    value = payload.get("id")
    if not isinstance(value, str):
        raise ConfigResolutionError("training.input_normalization.id is required")
    return cast(str, value)
