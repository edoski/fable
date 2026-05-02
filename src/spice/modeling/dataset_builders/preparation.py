"""Dataset-builder preparation Interface types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config.models import SplitConfig
    from ...features import CompiledFeatureContract
    from ...temporal.capability import TemporalCapability
    from ...temporal.contracts import CompiledProblemContract
    from ...temporal.execution_policy import CompiledExecutionPolicyContract
    from ...temporal.input_normalization import CompiledInputNormalizationContract
    from ...temporal.problem_store import (
        CompiledProblemStore,
        DatasetSplitIndices,
        IntVector,
    )
    from ...temporal.scaling import ScalerStats
    from .base import BuilderRuntimeMetadata, DatasetBuilderConfig


@dataclass(slots=True)
class TrainingDatasetPreparationSpec:
    dataset_builder: DatasetBuilderConfig
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract
    sample_count: int
    lookback_seconds: int
    split: SplitConfig
    input_normalization_contract: CompiledInputNormalizationContract


@dataclass(slots=True)
class ArtifactInferencePreparationSpec:
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract
    delay_seconds: int
    builder_runtime_metadata: BuilderRuntimeMetadata
    scaler: ScalerStats
    temporal_capability: TemporalCapability
    evaluation_start_timestamp: int
    evaluation_end_timestamp: int


@dataclass(slots=True)
class CompiledInferenceDatasetPreparationSpec:
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract
    delay_seconds: int
    builder_runtime_metadata: BuilderRuntimeMetadata
    scaler: ScalerStats
    temporal_capability: TemporalCapability
    window_start_timestamp: int
    window_end_timestamp: int


@dataclass(slots=True)
class PreparedTrainingDataset:
    n_rows_available: int
    n_rows_used: int
    sample_count: int
    execution_policy: CompiledExecutionPolicyContract
    store: CompiledProblemStore
    split_indices: DatasetSplitIndices
    scaler: ScalerStats
    builder_runtime_metadata: BuilderRuntimeMetadata
    temporal_capability: TemporalCapability

    @property
    def n_features(self) -> int:
        return self.store.n_features

@dataclass(slots=True)
class PreparedInferenceDataset:
    n_history_rows: int
    n_evaluation_rows: int
    sample_count: int
    execution_policy: CompiledExecutionPolicyContract
    store: CompiledProblemStore
    sample_indices: IntVector

    @property
    def n_features(self) -> int:
        return self.store.n_features
