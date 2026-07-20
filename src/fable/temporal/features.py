"""Exact causal feature construction and scaling."""

from __future__ import annotations

import math
from typing import Annotated, Self

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..corpus import BlockFrame

_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
_PositiveFiniteFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


class FeatureState(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    means: Annotated[tuple[_FiniteFloat, ...], Field(min_length=1)]
    standard_deviations: Annotated[
        tuple[_PositiveFiniteFloat, ...],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def validate_widths(self) -> Self:
        if len(self.means) != len(self.standard_deviations):
            raise ValueError("means and standard_deviations must have equal widths")
        return self


def fit_feature_state(
    training_support: BlockFrame,
    *,
    ordered_features: tuple[str, ...],
) -> FeatureState:
    raw = _raw_feature_rows(
        training_support.to_polars(),
        ordered_features=ordered_features,
    )
    if not np.isfinite(raw).all():
        raise ValueError("training_support must produce finite raw features")

    means = raw.mean(axis=0, dtype=np.float64)
    standard_deviations = raw.std(axis=0, ddof=0, dtype=np.float64)
    return FeatureState(
        means=tuple(float(value) for value in means),
        standard_deviations=tuple(float(value) for value in standard_deviations),
    )


def transform_feature_rows(
    blocks: BlockFrame,
    *,
    ordered_features: tuple[str, ...],
    state: FeatureState,
) -> NDArray[np.float32]:
    if len(state.means) != len(ordered_features):
        raise ValueError("state width must equal ordered_features width")

    raw = _raw_feature_rows(blocks.to_polars(), ordered_features=ordered_features)
    means = np.asarray(state.means, dtype=np.float64)
    standard_deviations = np.asarray(state.standard_deviations, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.ascontiguousarray(
            (raw - means) / standard_deviations,
            dtype=np.float32,
        )
    if not np.isfinite(transformed).all():
        raise ValueError("transformed features must be finite float32 values")
    return transformed


def _raw_feature_rows(
    blocks: pl.DataFrame,
    *,
    ordered_features: tuple[str, ...],
) -> NDArray[np.float64]:
    _validate_feature_uniqueness(ordered_features)
    columns = [_feature_values(blocks, feature_name) for feature_name in ordered_features]
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)


def _validate_feature_uniqueness(ordered_features: tuple[str, ...]) -> None:
    if len(set(ordered_features)) != len(ordered_features):
        raise ValueError("ordered_features must not contain duplicates")


def _feature_values(blocks: pl.DataFrame, feature_name: str) -> NDArray[np.float64]:
    if feature_name == "log_base_fee_per_gas":
        base_fees = _float_column(blocks, "base_fee_per_gas")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(base_fees)
    if feature_name == "gas_utilization":
        gas_limits = _float_column(blocks, "gas_limit")
        return _float_column(blocks, "gas_used") / gas_limits
    if feature_name == "log_exact_forming_base_fee_per_gas":
        return _forming_base_fee_logs(blocks)
    if feature_name == "log_gas_limit":
        gas_limits = _float_column(blocks, "gas_limit")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(gas_limits)
    if feature_name == "log1p_tx_count":
        tx_counts = _float_column(blocks, "tx_count")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log1p(tx_counts)
    if feature_name == "hour_sin":
        return np.sin(_hour_angles(blocks))
    if feature_name == "hour_cos":
        return np.cos(_hour_angles(blocks))
    raise ValueError(f"Unsupported feature: {feature_name}")


def _hour_angles(blocks: pl.DataFrame) -> NDArray[np.float64]:
    timestamps = blocks["timestamp"].to_numpy().astype(np.int64, copy=False)
    hours = (timestamps // 3_600) % 24
    return 2.0 * math.pi * hours.astype(np.float64, copy=False) / 24.0


def _forming_base_fee_logs(blocks: pl.DataFrame) -> NDArray[np.float64]:
    base_fees = blocks["base_fee_per_gas"].to_list()
    gas_used_values = blocks["gas_used"].to_list()
    gas_limits = blocks["gas_limit"].to_list()
    return np.fromiter(
        (
            _forming_base_fee_log(base_fee, gas_used, gas_limit)
            for base_fee, gas_used, gas_limit in zip(
                base_fees,
                gas_used_values,
                gas_limits,
                strict=True,
            )
        ),
        dtype=np.float64,
        count=blocks.height,
    )


def _forming_base_fee_log(
    base_fee_per_gas: int,
    gas_used: int,
    gas_limit: int,
) -> float:
    child_base_fee = _forming_child_base_fee(base_fee_per_gas, gas_used, gas_limit)
    if child_base_fee <= 0:
        raise ValueError("forming base fee must be positive")
    return math.log(child_base_fee)


def _forming_child_base_fee(
    base_fee_per_gas: int,
    gas_used: int,
    gas_limit: int,
) -> int:
    gas_target = gas_limit // 2
    if gas_target <= 0:
        raise ValueError("gas_target must be positive")
    if gas_used == gas_target:
        return base_fee_per_gas
    if gas_used > gas_target:
        return base_fee_per_gas + max(
            base_fee_per_gas * (gas_used - gas_target) // gas_target // 8,
            1,
        )
    return base_fee_per_gas - (base_fee_per_gas * (gas_target - gas_used) // gas_target // 8)


def _float_column(blocks: pl.DataFrame, name: str) -> NDArray[np.float64]:
    return blocks[name].to_numpy().astype(np.float64, copy=False)


__all__ = ["FeatureState", "fit_feature_state", "transform_feature_rows"]
