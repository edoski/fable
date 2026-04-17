"""Open registry for input-normalization specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ...core.components import ComponentCatalog
from ...core.errors import ConfigResolutionError
from .base import (
    CompiledInputNormalizationContract,
    InputNormalizationConfig,
    InputNormalizationSpec,
)

_INPUT_NORMALIZATION_SPECS = ComponentCatalog[InputNormalizationSpec[Any]](
    kind_label="input normalization",
    entry_point_group="spice.input_normalizations",
)


def register_input_normalization_spec(spec: InputNormalizationSpec[Any]) -> None:
    _INPUT_NORMALIZATION_SPECS.register(spec.id, spec)


def _load_builtin_input_normalizations() -> None:
    from . import row_standard, window_weighted_standard  # noqa: F401


_INPUT_NORMALIZATION_SPECS.configure_builtin_loader(_load_builtin_input_normalizations)


def input_normalization_spec(normalization_id: str) -> InputNormalizationSpec[Any]:
    try:
        return _INPUT_NORMALIZATION_SPECS.get(normalization_id)
    except ConfigResolutionError as exc:
        raise ConfigResolutionError(
            str(exc).replace("input normalization", "training.input_normalization.id")
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
