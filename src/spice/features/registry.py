"""Closed dispatch for supported feature families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from ..core.errors import ConfigResolutionError
from .core import validate_feature_names
from .families.base import FeatureFamilyConfig, FeatureFamilySpec

_KNOWN_FEATURE_FAMILIES = ("block_native", "time_native")


def feature_family_spec(family_id: str) -> FeatureFamilySpec:
    if family_id == "block_native":
        from .families.block_native import FEATURE_FAMILY_SPEC

        return FEATURE_FAMILY_SPEC
    if family_id == "time_native":
        from .families.time_native import FEATURE_FAMILY_SPEC

        return FEATURE_FAMILY_SPEC
    known = ", ".join(_KNOWN_FEATURE_FAMILIES)
    raise ConfigResolutionError(
        f"Unknown feature_set.family.id: {family_id}. Known values: {known}"
    )


def coerce_feature_family_config(
    raw_config: Mapping[str, object] | FeatureFamilyConfig,
) -> FeatureFamilyConfig:
    if isinstance(raw_config, FeatureFamilyConfig):
        family_id = raw_config.id
        payload = raw_config.model_dump(mode="json")
    elif isinstance(raw_config, Mapping):
        if "id" not in raw_config:
            raise ConfigResolutionError("feature_set.family.id is required")
        family_id = str(raw_config["id"])
        payload = dict(raw_config)
    else:
        raise ConfigResolutionError("feature_set.family must be a mapping")
    return cast(
        FeatureFamilyConfig,
        feature_family_spec(family_id).config_type.model_validate(payload),
    )


def validate_feature_selection(
    feature_set_id: str,
    feature_family_id: str,
    feature_names: tuple[str, ...],
) -> None:
    spec = feature_family_spec(feature_family_id)
    validate_feature_names(
        feature_set_id,
        feature_names,
        known_feature_names=tuple(spec.features),
    )
