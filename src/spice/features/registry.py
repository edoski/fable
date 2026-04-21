"""Feature-family lookup for the fixed in-repo families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.errors import ConfigResolutionError
from .core import validate_feature_names
from .families.base import FeatureFamily, FeatureFamilyConfig
from .families.block_native import (
    BLOCK_NATIVE_FAMILY,
    BlockNativeFeatureFamilyConfig,
)
from .families.time_native import TIME_NATIVE_FAMILY, TimeNativeFeatureFamilyConfig


@dataclass(frozen=True, slots=True)
class _FeatureFamilyEntry:
    config_type: type[FeatureFamilyConfig]
    family: FeatureFamily


_FEATURE_FAMILIES: dict[str, _FeatureFamilyEntry] = {
    "block_native": _FeatureFamilyEntry(
        config_type=BlockNativeFeatureFamilyConfig,
        family=BLOCK_NATIVE_FAMILY,
    ),
    "time_native": _FeatureFamilyEntry(
        config_type=TimeNativeFeatureFamilyConfig,
        family=TIME_NATIVE_FAMILY,
    ),
}


def feature_family(family_id: str) -> FeatureFamily:
    entry = _FEATURE_FAMILIES.get(family_id)
    if entry is None:
        known = ", ".join(sorted(_FEATURE_FAMILIES))
        raise ConfigResolutionError(
            f"Unknown feature_set.family.id: {family_id}. Known values: {known}"
        )
    return entry.family


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
    entry = _FEATURE_FAMILIES.get(family_id)
    if entry is None:
        known = ", ".join(sorted(_FEATURE_FAMILIES))
        raise ConfigResolutionError(
            f"Unknown feature_set.family.id: {family_id}. Known values: {known}"
        )
    return entry.config_type.model_validate(payload)


def validate_feature_selection(
    feature_set_id: str,
    feature_family_id: str,
    feature_names: tuple[str, ...],
) -> None:
    family = feature_family(feature_family_id)
    validate_feature_names(
        feature_set_id,
        feature_names,
        known_feature_names=tuple(family.features),
    )
