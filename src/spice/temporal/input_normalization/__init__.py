"""Registry-backed input-normalization seam."""

from .base import (
    CompiledInputNormalizationContract,
    InputNormalizationConfig,
)
from .registry import (
    coerce_input_normalization_config,
    compile_input_normalization_contract,
    input_normalization_spec,
)

__all__ = [
    "CompiledInputNormalizationContract",
    "InputNormalizationConfig",
    "coerce_input_normalization_config",
    "compile_input_normalization_contract",
    "input_normalization_spec",
]
