"""Feature compilation and table execution."""

from .contracts import (
    CompiledFeatureContract,
    compile_feature_contract,
)
from .core import (
    CanonicalBlockSeries,
    ResolvedFeatureTable,
    build_feature_table,
)
from .families import (
    FeatureFamilyConfig,
    FeatureFamilySpec,
    FeaturePrerequisites,
)
from .registry import coerce_feature_family_config, feature_family_spec, validate_feature_selection

__all__ = [
    "CanonicalBlockSeries",
    "CompiledFeatureContract",
    "FeatureFamilyConfig",
    "FeatureFamilySpec",
    "FeaturePrerequisites",
    "ResolvedFeatureTable",
    "build_feature_table",
    "compile_feature_contract",
    "coerce_feature_family_config",
    "feature_family_spec",
    "validate_feature_selection",
]
