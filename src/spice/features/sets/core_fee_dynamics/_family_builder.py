"""Shared builder for core fee-dynamics feature catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from ...core import FeatureCatalog, FeatureSpec, SourceSpec
from . import _base_fee, _block_facts, _fee_context, _time, _transforms
from ._base_fee import (
    BASE_FEE_TREND_OUTPUTS,
    CORE_FEE_LEVEL_OUTPUTS,
    base_fee_trend_features,
    core_fee_level_features,
    current_base_fee_sources,
)
from ._block_facts import (
    PREVIOUS_BLOCK_FACT_OUTPUTS,
    PREVIOUS_GAS_UTILIZATION_ROLLING_OUTPUTS,
    PREVIOUS_GAS_UTILIZATION_TREND_OUTPUTS,
    gas_utilization_rolling_features,
    gas_utilization_trend_features,
    previous_block_fact_features,
    previous_block_fact_sources,
)
from ._fee_context import (
    EXTENDED_ROLLING_FEE_CONTEXT_OUTPUTS,
    LOCAL_FEE_CONTEXT_OUTPUTS,
    extended_rolling_fee_context_features,
    local_fee_context_features,
)
from ._time import CADENCE_CALENDAR_OUTPUTS, cadence_calendar_features


def base_outputs() -> tuple[str, ...]:
    return (
        *CORE_FEE_LEVEL_OUTPUTS,
        *PREVIOUS_BLOCK_FACT_OUTPUTS,
        *CADENCE_CALENDAR_OUTPUTS,
        *LOCAL_FEE_CONTEXT_OUTPUTS,
        *BASE_FEE_TREND_OUTPUTS,
        *PREVIOUS_GAS_UTILIZATION_TREND_OUTPUTS,
        *EXTENDED_ROLLING_FEE_CONTEXT_OUTPUTS,
        *PREVIOUS_GAS_UTILIZATION_ROLLING_OUTPUTS,
    )


def build_catalog(
    *,
    variant_module_path: Path,
    allowed_outputs: tuple[str, ...],
    extra_sources: Mapping[str, SourceSpec] | None = None,
    extra_features: Mapping[str, FeatureSpec] | None = None,
    extra_fingerprint_modules: tuple[ModuleType, ...] = (),
) -> FeatureCatalog:
    return FeatureCatalog(
        sources={
            **current_base_fee_sources(),
            **previous_block_fact_sources(),
            **(dict(extra_sources) if extra_sources is not None else {}),
        },
        features={
            **core_fee_level_features(),
            **previous_block_fact_features(),
            **cadence_calendar_features(),
            **local_fee_context_features(),
            **base_fee_trend_features(),
            **gas_utilization_trend_features(
                "prev_gas_utilization",
                base_warmup_rows=1,
            ),
            **extended_rolling_fee_context_features(),
            **gas_utilization_rolling_features(
                "prev_gas_utilization",
                base_warmup_rows=1,
            ),
            **(dict(extra_features) if extra_features is not None else {}),
        },
        allowed_outputs=allowed_outputs,
        fingerprint_sources=fingerprint_sources(
            variant_module_path,
            extra_modules=extra_fingerprint_modules,
        ),
    )


def fingerprint_sources(
    variant_module_path: Path,
    *,
    extra_modules: tuple[ModuleType, ...] = (),
) -> tuple[Path, ...]:
    return (
        variant_module_path,
        Path(__file__).resolve(),
        Path(_transforms.__file__).resolve(),
        Path(_time.__file__).resolve(),
        Path(_base_fee.__file__).resolve(),
        Path(_block_facts.__file__).resolve(),
        Path(_fee_context.__file__).resolve(),
        *(_module_path(module) for module in extra_modules),
        Path(__file__).resolve().parents[2] / "core.py",
    )


def _module_path(module: ModuleType) -> Path:
    module_file = module.__file__
    if module_file is None:
        raise ValueError(f"Feature catalog fingerprint module has no file: {module.__name__}")
    return Path(module_file).resolve()
