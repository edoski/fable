"""Unweighted train-row standardization."""

from pydantic import field_validator

from .base import CompiledInputNormalizationContract, InputNormalizationConfig
from .scaling import fit_row_standard_scaler


class RowStandardConfig(InputNormalizationConfig):
    id: str = "row_standard"

    @field_validator("id")
    @classmethod
    def validate_row_standard_id(cls, value: str) -> str:
        value = InputNormalizationConfig.validate_id(value)
        if value != "row_standard":
            raise ValueError("training.input_normalization.id must be row_standard")
        return value


def compile_input_normalization(config: RowStandardConfig) -> CompiledInputNormalizationContract:
    del config
    return CompiledInputNormalizationContract(
        input_normalization_id="row_standard",
        fit_scaler_fn=fit_row_standard_scaler,
    )
