"""Time-native alternate feature family."""

from __future__ import annotations

from typing import Literal

from ..base import FeatureFamilyConfig, FeatureFamilySpec, tagged_feature_prerequisites
from ..registry import register_feature_family_spec
from . import base, rolling, trend


class TimeNativeFeatureFamilyConfig(FeatureFamilyConfig):
    id: Literal["time_native"] = "time_native"


register_feature_family_spec(
    FeatureFamilySpec[TimeNativeFeatureFamilyConfig](
        id="time_native",
        config_type=TimeNativeFeatureFamilyConfig,
        modules=(base, rolling, trend),
        resolve_prerequisites=tagged_feature_prerequisites,
    )
)
