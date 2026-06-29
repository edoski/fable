"""Dataset-builder preparation Interface types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config.models import SplitConfig
    from ...features import CompiledFeatureContract
    from ...temporal.capability import TemporalCapability
    from ...temporal.contracts import CompiledProblemContract
    from ...temporal.execution_policy import (
        CompiledExecutionPolicyContract,
        PreparedActionSpace,
        PreparedTemporalFacts,
    )
    from ...temporal.input_normalization import ScalerStats
    from ...temporal.problem_store import (
        CompiledProblemStore,
        IntVector,
    )
    from .base import SequenceRuntimeMetadata


@dataclass(slots=True)
class TrainingDatasetPreparationContext:
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract


@dataclass(slots=True)
class TrainingDatasetPreparationFacts:
    split: SplitConfig
    training_cutoff_timestamp: int | None = None


@dataclass(slots=True)
class DatasetSplitIndices:
    train: IntVector
    validation: IntVector
    test: IntVector


@dataclass(frozen=True, slots=True)
class PreparedTrainingSampleSelection:
    temporal_facts: PreparedTemporalFacts

    @property
    def action_space(self) -> PreparedActionSpace:
        return self.temporal_facts.action_space

    @property
    def sample_indices(self) -> IntVector:
        return self.action_space.sample_indices


@dataclass(frozen=True, slots=True)
class PreparedTrainingSampleRoles:
    train: PreparedTrainingSampleSelection
    validation: PreparedTrainingSampleSelection
    test: PreparedTrainingSampleSelection


@dataclass(frozen=True, slots=True)
class PreparedInferenceSampleSelection:
    action_space: PreparedActionSpace

    def __post_init__(self) -> None:
        if self.action_space.sample_indices.size == 0:
            raise ValueError("inference sample selection must be non-empty")

    @property
    def sample_indices(self) -> IntVector:
        return self.action_space.sample_indices


def validate_temporal_facts_alignment(
    role: str,
    temporal_facts: PreparedTemporalFacts,
) -> None:
    action_space = temporal_facts.action_space
    sample_count = int(action_space.sample_indices.shape[0])
    if sample_count == 0:
        raise ValueError(f"{role} sample selection must be non-empty")
    if temporal_facts.outcome_facts.baseline_rows.shape != (sample_count,):
        raise ValueError(f"{role} temporal facts do not align with selected samples")


def training_sample_selection(
    role: str,
    temporal_facts: PreparedTemporalFacts,
) -> PreparedTrainingSampleSelection:
    validate_temporal_facts_alignment(role, temporal_facts)
    return PreparedTrainingSampleSelection(temporal_facts=temporal_facts)


@dataclass(frozen=True, slots=True)
class EvaluationCoverageWindow:
    first_timestamp: int
    last_timestamp: int

    def __post_init__(self) -> None:
        if self.last_timestamp < self.first_timestamp:
            raise ValueError("evaluation coverage last_timestamp must be >= first_timestamp")

    def to_sample_timestamp_window(self) -> SampleTimestampWindow:
        return SampleTimestampWindow(
            start_timestamp_inclusive=self.first_timestamp,
            end_timestamp_exclusive=self.last_timestamp + 1,
        )


@dataclass(frozen=True, slots=True)
class SampleTimestampWindow:
    start_timestamp_inclusive: int
    end_timestamp_exclusive: int

    def __post_init__(self) -> None:
        if self.end_timestamp_exclusive <= self.start_timestamp_inclusive:
            raise ValueError(
                "sample timestamp window end_timestamp_exclusive must be greater "
                "than start_timestamp_inclusive"
            )


@dataclass(frozen=True, slots=True)
class SampleBlockWindow:
    start_block_inclusive: int
    end_block_exclusive: int

    def __post_init__(self) -> None:
        if self.end_block_exclusive <= self.start_block_inclusive:
            raise ValueError(
                "sample block window end_block_exclusive must be greater "
                "than start_block_inclusive"
            )

    @property
    def block_count(self) -> int:
        return self.end_block_exclusive - self.start_block_inclusive


@dataclass(slots=True)
class ArtifactInferenceDatasetPreparationFacts:
    delay_seconds: int
    evaluation_coverage: EvaluationCoverageWindow


@dataclass(slots=True)
class ArtifactInferenceDatasetPreparationContext:
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract
    sequence_runtime_metadata: SequenceRuntimeMetadata
    scaler: ScalerStats
    temporal_capability: TemporalCapability


@dataclass(slots=True)
class CompiledInferenceDatasetPreparationRequest:
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract
    delay_seconds: int
    sequence_runtime_metadata: SequenceRuntimeMetadata
    scaler: ScalerStats
    temporal_capability: TemporalCapability
    sample_timestamp_window: SampleTimestampWindow | None = None
    sample_block_window: SampleBlockWindow | None = None

    def __post_init__(self) -> None:
        if (self.sample_timestamp_window is None) == (self.sample_block_window is None):
            raise ValueError(
                "inference preparation requires exactly one sample timestamp or block window"
            )


@dataclass(slots=True)
class PreparedTrainingDataset:
    n_rows_available: int
    n_rows_used: int
    sample_count: int
    execution_policy: CompiledExecutionPolicyContract
    store: CompiledProblemStore
    samples: PreparedTrainingSampleRoles
    scaler: ScalerStats
    sequence_runtime_metadata: SequenceRuntimeMetadata
    temporal_capability: TemporalCapability

    @property
    def n_features(self) -> int:
        return self.store.n_features


@dataclass(slots=True)
class PreparedInferenceDataset:
    n_history_rows: int
    n_evaluation_rows: int
    execution_policy: CompiledExecutionPolicyContract
    store: CompiledProblemStore
    samples: PreparedInferenceSampleSelection

    @property
    def sample_count(self) -> int:
        return int(self.samples.sample_indices.shape[0])

    @property
    def n_features(self) -> int:
        return self.store.n_features
