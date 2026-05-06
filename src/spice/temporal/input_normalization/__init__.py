"""Registry-backed input-normalization seam."""

from .base import (
    CompiledInputNormalizationContract,
    InputNormalizationConfig,
    coerce_input_normalization_config,
    compile_input_normalization_contract,
)
from .scaling import ScalerStats, transform_feature_matrix, transform_problem_store_features

__all__ = [
    "CompiledInputNormalizationContract",
    "InputNormalizationConfig",
    "ScalerStats",
    "coerce_input_normalization_config",
    "compile_input_normalization_contract",
    "transform_feature_matrix",
    "transform_problem_store_features",
]
