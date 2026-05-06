"""Window-weighted standardization."""

from pydantic import field_validator

from .base import CompiledInputNormalizationContract, InputNormalizationConfig
from .scaling import fit_window_weighted_standard_scaler


class WindowWeightedStandardConfig(InputNormalizationConfig):
    id: str = "window_weighted_standard"

    @field_validator("id")
    @classmethod
    def validate_window_weighted_standard_id(cls, value: str) -> str:
        value = InputNormalizationConfig.validate_id(value)
        if value != "window_weighted_standard":
            raise ValueError("training.input_normalization.id must be window_weighted_standard")
        return value


def compile_input_normalization(
    config: WindowWeightedStandardConfig,
) -> CompiledInputNormalizationContract:
    del config
    return CompiledInputNormalizationContract(
        input_normalization_id="window_weighted_standard",
        fit_scaler_fn=fit_window_weighted_standard_scaler,
    )
