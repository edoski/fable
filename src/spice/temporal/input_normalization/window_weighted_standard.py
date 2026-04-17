"""Window-weighted standardization matching the legacy pipeline."""

from __future__ import annotations

from typing import Literal

from ..scaling import fit_window_weighted_standard_scaler
from .base import CompiledInputNormalizationContract, InputNormalizationConfig, InputNormalizationSpec
from .registry import register_input_normalization_spec


class WindowWeightedStandardConfig(InputNormalizationConfig):
    id: Literal["window_weighted_standard"] = "window_weighted_standard"


def _compile(config: WindowWeightedStandardConfig) -> CompiledInputNormalizationContract:
    del config
    return CompiledInputNormalizationContract(
        input_normalization_id="window_weighted_standard",
        fit_scaler_fn=fit_window_weighted_standard_scaler,
    )


register_input_normalization_spec(
    InputNormalizationSpec[WindowWeightedStandardConfig](
        id="window_weighted_standard",
        config_type=WindowWeightedStandardConfig,
        compile=_compile,
    )
)
