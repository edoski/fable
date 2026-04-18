"""Closed dispatch for supported dataset builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from ...core.errors import ConfigResolutionError
from .base import (
    CompiledDatasetBuilderContract,
    DatasetBuilderConfig,
    DatasetBuilderSpec,
    StandardTemporalDatasetBuilderConfig,
)


def _compile_standard_temporal(
    config: StandardTemporalDatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    from .standard_temporal import compile_dataset_builder

    return compile_dataset_builder(config)

_DATASET_BUILDERS: dict[str, DatasetBuilderSpec[DatasetBuilderConfig]] = {
    "standard_temporal": DatasetBuilderSpec(
        id="standard_temporal",
        config_type=StandardTemporalDatasetBuilderConfig,
        compile=_compile_standard_temporal,
    ),
}


def dataset_builder_spec(builder_id: str) -> DatasetBuilderSpec[DatasetBuilderConfig]:
    try:
        return _DATASET_BUILDERS[builder_id]
    except KeyError as exc:
        known = ", ".join(sorted(_DATASET_BUILDERS))
        raise ConfigResolutionError(
            f"Unknown dataset_builder.id: {builder_id}. Known values: {known}"
        ) from exc


def coerce_dataset_builder_config(
    payload: Mapping[str, object] | DatasetBuilderConfig,
) -> DatasetBuilderConfig:
    if isinstance(payload, DatasetBuilderConfig):
        raw_payload = payload.model_dump(mode="json")
        builder_id = payload.id
    else:
        raw_payload = dict(payload)
        builder_id = _mapping_builder_id(raw_payload)
    spec = dataset_builder_spec(builder_id)
    return spec.config_type.model_validate(raw_payload)


def compile_dataset_builder_contract(
    config: DatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    return dataset_builder_spec(config.id).compile(config)


def _mapping_builder_id(payload: Mapping[str, object]) -> str:
    value = payload.get("id")
    if not isinstance(value, str):
        raise ConfigResolutionError("dataset_builder.id is required")
    return cast(str, value)
