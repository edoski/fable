"""Prediction-family registry and compiled contracts."""

from .base import (
    MetricDescriptor,
    MetricSet,
    PredictionFamilyConfig,
    PredictionHeadSpec,
    PredictionOutputSpec,
    PredictionSimulationRun,
    PredictionSimulationSummary,
    WindowMetricSummary,
)
from .contracts import (
    CompiledPredictionContract,
    ModelInputBatch,
    PredictionBatch,
    PredictionTargetBatch,
    bind_prediction_representation,
)
from .registry import (
    PredictionFamilySpec,
    coerce_prediction_family_config,
    compile_prediction_contract,
    prediction_family_spec,
)

__all__ = [
    "CompiledPredictionContract",
    "MetricDescriptor",
    "MetricSet",
    "ModelInputBatch",
    "PredictionBatch",
    "PredictionFamilyConfig",
    "PredictionFamilySpec",
    "PredictionHeadSpec",
    "PredictionOutputSpec",
    "PredictionSimulationRun",
    "PredictionSimulationSummary",
    "PredictionTargetBatch",
    "WindowMetricSummary",
    "bind_prediction_representation",
    "coerce_prediction_family_config",
    "compile_prediction_contract",
    "prediction_family_spec",
]
