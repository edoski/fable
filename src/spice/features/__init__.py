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
    FeaturePrerequisites,
)
from .registry import coerce_feature_family_config, feature_family, validate_feature_selection

__all__ = [
    "CanonicalBlockSeries",
    "CompiledFeatureContract",
    "FeatureFamilyConfig",
    "FeaturePrerequisites",
    "ResolvedFeatureTable",
    "build_feature_table",
    "compile_feature_contract",
    "coerce_feature_family_config",
    "feature_family",
    "validate_feature_selection",
]
