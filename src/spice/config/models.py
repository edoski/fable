"""Explicit runtime configuration models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self, TypeAlias
from urllib.parse import urlparse

from pydantic import (
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from ..core.config_model import ConfigModel as _ConfigModel
from ..core.errors import ConfigResolutionError
from ..core.specs import owner_payload, validate_owner_config
from ..core.validation import validate_path_segment
from ..evaluation import EvaluatorConfig
from ..features import validate_feature_selection
from ..modeling.families.base import (
    ModelConfig,
    ModelTuningSpaceConfig,
    TunedModelParams,
)
from ..prediction import validate_prediction_family_id
from ..temporal.compilers import ProblemCompilerConfig
from ..temporal.execution_policy import ExecutionPolicyConfig


class WorkflowTask(StrEnum):
    ACQUIRE = "acquire"
    TUNE = "tune"
    TRAIN = "train"
    EVALUATE = "evaluate"


class ArtifactVariant(StrEnum):
    BASELINE = "baseline"
    TUNED = "tuned"


class TimestampWindowSpec(_ConfigModel):
    start: datetime
    end: datetime | None = None
    duration_seconds: int | None = Field(default=None, gt=0)

    @field_validator("start", "end")
    @classmethod
    def validate_utc_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp windows require timezone-aware UTC datetimes")
        resolved = value.astimezone(UTC)
        if resolved.utcoffset() != UTC.utcoffset(resolved):
            raise ValueError("timestamp windows require UTC datetimes")
        return resolved

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if (self.end is None) == (self.duration_seconds is None):
            raise ValueError("timestamp window must define exactly one of end or duration_seconds")
        if self.end is not None and self.end <= self.start:
            raise ValueError("timestamp window end must be greater than start")
        return self

    @property
    def start_timestamp(self) -> int:
        return int(self.start.timestamp())

    @property
    def end_timestamp(self) -> int:
        if self.end is not None:
            return int(self.end.timestamp())
        if self.duration_seconds is None:
            raise ValueError("timestamp window duration_seconds was not resolved")
        return self.start_timestamp + self.duration_seconds


class BlockWindowSpec(_ConfigModel):
    start_block: int = Field(gt=0)
    block_count: int = Field(gt=0)

    @property
    def end_block_exclusive(self) -> int:
        return self.start_block + self.block_count


RuntimeEvaluationWindowSpec: TypeAlias = TimestampWindowSpec | BlockWindowSpec


class EvaluationWindowSpecBase(_ConfigModel):
    id: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="evaluation_window.id")

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("evaluation_window.tags must not contain duplicates")
        for tag in value:
            validate_path_segment(tag, label="evaluation_window.tags")
        return value


class TimestampEvaluationWindowSpec(EvaluationWindowSpecBase, TimestampWindowSpec):
    pass


class BlockEvaluationWindowSpec(EvaluationWindowSpecBase, BlockWindowSpec):
    pass


EvaluationWindowSpec: TypeAlias = (
    TimestampEvaluationWindowSpec | BlockEvaluationWindowSpec
)


class EvaluationsSpec(_ConfigModel):
    id: str
    items: list[EvaluationWindowSpec] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="evaluations.id")

    @model_validator(mode="after")
    def validate_items(self) -> Self:
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("evaluations.items must not contain duplicate ids")
        return self

    @property
    def training_cutoff_timestamp(self) -> int:
        timestamp_items = [
            item for item in self.items if isinstance(item, TimestampEvaluationWindowSpec)
        ]
        if len(timestamp_items) != len(self.items):
            raise ValueError(
                "training cutoff resolution requires timestamp evaluation windows"
            )
        return min(item.start_timestamp for item in timestamp_items)


def _validate_http_endpoint_url(value: str, *, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an http:// or https:// URL")
    return value


class ChainRuntimeSpec(_ConfigModel):
    chain_id: int = Field(gt=0)
    uses_poa_extra_data: bool
    nominal_block_time_seconds: float = Field(gt=0.0)


class ChainSpec(_ConfigModel):
    name: str
    runtime: ChainRuntimeSpec

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_path_segment(value, label="chain.name")


class CorpusSpec(_ConfigModel):
    name: str
    window: TimestampWindowSpec

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_path_segment(value, label="corpus.name")


class StorageSpec(_ConfigModel):
    root: Path = Path("outputs")


class ProblemSpec(_ConfigModel):
    id: str
    lookback_seconds: int = Field(gt=0)
    max_delay_seconds: int = Field(gt=0)
    compiler: SerializeAsAny[ProblemCompilerConfig]
    execution_policy: SerializeAsAny[ExecutionPolicyConfig]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="problem.id")


def coerce_problem_spec(payload: object) -> ProblemSpec:
    from ..temporal.compilers import coerce_problem_compiler_config

    raw_payload = owner_payload(payload, owner="problem", config_type=ProblemSpec)
    raw_compiler = (
        payload.compiler if isinstance(payload, ProblemSpec) else raw_payload.get("compiler")
    )
    if raw_compiler is None:
        raise ConfigResolutionError("problem.compiler is required")
    raw_execution_policy = (
        payload.execution_policy
        if isinstance(payload, ProblemSpec)
        else raw_payload.get("execution_policy")
    )
    if raw_execution_policy is None:
        raise ConfigResolutionError("problem.execution_policy is required")
    from ..temporal.execution_policy import coerce_execution_policy_config

    compiler = coerce_problem_compiler_config(raw_compiler)
    execution_policy = coerce_execution_policy_config(raw_execution_policy)
    if (
        isinstance(payload, ProblemSpec)
        and compiler is payload.compiler
        and execution_policy is payload.execution_policy
    ):
        return payload
    raw_payload["compiler"] = compiler
    raw_payload["execution_policy"] = execution_policy
    return validate_owner_config(raw_payload, ProblemSpec)


class SplitConfig(_ConfigModel):
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_split(self) -> Self:
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
        return self


class EarlyStoppingConfig(_ConfigModel):
    patience: int = Field(gt=0)
    min_delta: float = Field(ge=0.0)


class SequenceConfig(_ConfigModel):
    min_length: int = Field(gt=0)
    max_length: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.max_length < self.min_length:
            raise ValueError("training.sequence.max_length must be >= min_length")
        return self


class TrainingConfig(_ConfigModel):
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    early_stopping: EarlyStoppingConfig
    gradient_clip_norm: float = Field(gt=0.0)
    seed: int = Field(ge=0)
    deterministic: bool
    sequence: SequenceConfig


class AcquisitionRpcConfig(_ConfigModel):
    batch_size: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    min_batch_size: int = Field(gt=0)
    concurrency_rungs: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        if self.min_batch_size > self.batch_size:
            raise ValueError("acquisition.rpc.min_batch_size must be <= batch_size")
        if sorted(self.concurrency_rungs) != self.concurrency_rungs:
            raise ValueError("acquisition.rpc.concurrency_rungs must be sorted ascending")
        if len(set(self.concurrency_rungs)) != len(self.concurrency_rungs):
            raise ValueError("acquisition.rpc.concurrency_rungs must not contain duplicates")
        if any(value <= 0 for value in self.concurrency_rungs):
            raise ValueError("acquisition.rpc.concurrency_rungs values must be positive")
        if self.concurrency not in self.concurrency_rungs:
            raise ValueError("acquisition.rpc.concurrency must be present in concurrency_rungs")
        return self


class AcquisitionConfig(_ConfigModel):
    dry_run: bool = False
    chunk_size: int = Field(gt=0)
    rpc: AcquisitionRpcConfig


class FeaturesConfig(_ConfigModel):
    id: str
    outputs: list[str] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="features.id")

    @field_validator("outputs")
    @classmethod
    def validate_outputs(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("features.outputs must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_feature_selection(self) -> Self:
        validate_feature_selection(self.id, tuple(self.outputs))
        return self


def coerce_features_config(payload: object) -> FeaturesConfig:
    if isinstance(payload, FeaturesConfig):
        return payload
    return validate_owner_config(
        owner_payload(payload, owner="features", config_type=FeaturesConfig),
        FeaturesConfig,
    )


class PredictionConfig(_ConfigModel):
    id: str
    family_id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="prediction.id")

    @field_validator("family_id")
    @classmethod
    def validate_family_id(cls, value: str) -> str:
        return validate_prediction_family_id(
            validate_path_segment(value, label="prediction.family_id")
        )


class StudyConfig(_ConfigModel):
    name: str = "default"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_path_segment(value, label="study.name")


class ArtifactConfig(_ConfigModel):
    variant: ArtifactVariant = ArtifactVariant.BASELINE


class TuningTrainingSearchSpace(_ConfigModel):
    learning_rate: list[float] | None = Field(default=None, min_length=1)
    weight_decay: list[float] | None = Field(default=None, min_length=1)
    batch_size: list[int] | None = Field(default=None, min_length=1)

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate_candidates(cls, values: list[float] | None) -> list[float] | None:
        if values is not None and any(value <= 0.0 for value in values):
            raise ValueError("tuning_space.training.learning_rate values must be positive")
        return values

    @field_validator("weight_decay")
    @classmethod
    def validate_weight_decay_candidates(cls, values: list[float] | None) -> list[float] | None:
        if values is not None and any(value < 0.0 for value in values):
            raise ValueError("tuning_space.training.weight_decay values must be non-negative")
        return values

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size_candidates(cls, values: list[int] | None) -> list[int] | None:
        if values is not None and any(value <= 0 for value in values):
            raise ValueError("tuning_space.training.batch_size values must be positive")
        return values

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if (
            self.learning_rate is None
            and self.weight_decay is None
            and self.batch_size is None
        ):
            raise ValueError("tuning_space.training must declare at least one field")
        return self


class TuningProblemSearchSpace(_ConfigModel):
    lookback_seconds: list[int] | None = Field(default=None, min_length=1)

    @field_validator("lookback_seconds")
    @classmethod
    def validate_lookback_seconds_candidates(
        cls,
        values: list[int] | None,
    ) -> list[int] | None:
        if values is not None and any(value <= 0 for value in values):
            raise ValueError("tuning_space.problem.lookback_seconds values must be positive")
        return values

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.lookback_seconds is None:
            raise ValueError("tuning_space.problem must declare at least one field")
        return self


class TuningSpaceConfig(_ConfigModel):
    training: TuningTrainingSearchSpace | None = None
    problem: TuningProblemSearchSpace | None = None
    model: SerializeAsAny[ModelTuningSpaceConfig]

    def has_candidates(self) -> bool:
        model_candidates = self.model.model_dump(exclude={"id"}, exclude_none=True)
        return (
            self.training is not None
            or self.problem is not None
            or bool(model_candidates)
        )


class TunedTrainingParams(_ConfigModel):
    learning_rate: float | None = Field(default=None, gt=0.0)
    weight_decay: float | None = Field(default=None, ge=0.0)
    batch_size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if (
            self.learning_rate is None
            and self.weight_decay is None
            and self.batch_size is None
        ):
            raise ValueError("tuned training params must declare at least one field")
        return self


class TunedProblemParams(_ConfigModel):
    lookback_seconds: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.lookback_seconds is None:
            raise ValueError("tuned problem params must declare at least one field")
        return self


class TunedParameterSet(_ConfigModel):
    training: TunedTrainingParams | None = None
    problem: TunedProblemParams | None = None
    model: SerializeAsAny[TunedModelParams] | None = None

    @model_validator(mode="after")
    def validate_non_empty_param_set(self) -> Self:
        if (
            self.training is None
            and self.problem is None
            and self.model is None
        ):
            raise ValueError("tuned parameter set must declare at least one parameter group")
        return self


class TuningConfig(_ConfigModel):
    trial_count: int = Field(gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    sampler_seed: int = Field(ge=0)
    enable_pruning: bool

    @property
    def search(self) -> TuningSearchConfig:
        return TuningSearchConfig(
            sampler_seed=self.sampler_seed,
            enable_pruning=self.enable_pruning,
        )


class TuningSearchConfig(_ConfigModel):
    sampler_seed: int = Field(ge=0)
    enable_pruning: bool


class ProviderEndpointConfig(_ConfigModel):
    url: str
    reference: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_http_endpoint_url(value, label="provider.endpoints.url")


class ProviderTransportConfig(_ConfigModel):
    timeout_seconds: float = Field(gt=0.0)
    retry_count: int = Field(ge=0)
    backoff_factor: float = Field(ge=0.0)


class ResolvedRpcEndpointConfig(_ConfigModel):
    provider_name: str
    url: str
    reference: str
    timeout_seconds: float = Field(gt=0.0)
    retry_count: int = Field(ge=0)
    backoff_factor: float = Field(ge=0.0)

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, value: str) -> str:
        return validate_path_segment(value, label="rpc_endpoint.provider_name")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_http_endpoint_url(value, label="rpc_endpoint.url")


class ProviderSpec(_ConfigModel):
    name: str
    transport: ProviderTransportConfig
    acquisition: AcquisitionConfig | None = None
    endpoints: dict[str, ProviderEndpointConfig]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_path_segment(value, label="provider.name")

    @model_validator(mode="after")
    def validate_chain_coverage(self) -> Self:
        if not self.endpoints:
            raise ValueError("provider.endpoints must not be empty")
        for name in self.endpoints:
            validate_path_segment(name, label="provider.endpoints key")
        return self

    def endpoint_config_for(self, chain_name: str) -> ProviderEndpointConfig:
        try:
            return self.endpoints[chain_name]
        except KeyError as exc:
            raise ConfigResolutionError(
                f"provider {self.name} does not define endpoint for {chain_name}"
            ) from exc


class WorkflowConfig(_ConfigModel):
    workflow: WorkflowTask
    chain: ChainSpec
    corpus: CorpusSpec
    storage: StorageSpec

    @property
    def corpus_window_start_timestamp(self) -> int:
        return self.corpus.window.start_timestamp

    @property
    def corpus_window_end_timestamp(self) -> int:
        return self.corpus.window.end_timestamp


class AcquireConfig(WorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.ACQUIRE
    problem: ProblemSpec
    features: FeaturesConfig
    rpc_endpoint: ResolvedRpcEndpointConfig
    acquisition: AcquisitionConfig


class ModelWorkflowConfig(WorkflowConfig):
    problem: ProblemSpec
    model: SerializeAsAny[ModelConfig]
    features: FeaturesConfig
    prediction: PredictionConfig
    study: StudyConfig = Field(default_factory=StudyConfig)
    artifact: ArtifactConfig = Field(default_factory=ArtifactConfig)


class TrainConfig(ModelWorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.TRAIN
    corpus_id: str | None = None
    study_id: str | None = None
    training_cutoff_timestamp: int | None = Field(default=None, gt=0)
    split: SplitConfig
    training: TrainingConfig
    tuning: TuningConfig | None = None
    tuning_space: TuningSpaceConfig | None = None

    @model_validator(mode="after")
    def validate_root_selector(self) -> Self:
        if self.artifact.variant is ArtifactVariant.TUNED:
            if self.study_id is None:
                raise ConfigResolutionError("tuned training requires study_id")
            if self.corpus_id is not None:
                raise ConfigResolutionError("tuned training must not define corpus_id")
            return self
        if self.corpus_id is None:
            raise ConfigResolutionError("baseline training requires corpus_id")
        if self.study_id is not None:
            raise ConfigResolutionError("baseline training must not define study_id")
        return self


class TuneConfig(ModelWorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.TUNE
    corpus_id: str
    training_cutoff_timestamp: int | None = Field(default=None, gt=0)
    split: SplitConfig
    training: TrainingConfig
    tuning: TuningConfig
    tuning_space: TuningSpaceConfig

    @model_validator(mode="after")
    def validate_tuning_space(self) -> Self:
        if self.tuning_space.model.id != self.model.id:
            raise ConfigResolutionError("tuning_space.model.id must match model.id")
        if not self.tuning_space.has_candidates():
            raise ConfigResolutionError("tuning_space must declare at least one tunable parameter")
        return self


class EvaluateConfig(_ConfigModel):
    workflow: WorkflowTask = WorkflowTask.EVALUATE
    storage: StorageSpec = Field(default_factory=StorageSpec)
    artifact_id: str
    corpus_id: str
    evaluation_window: RuntimeEvaluationWindowSpec
    evaluator: SerializeAsAny[EvaluatorConfig]
    delay_seconds: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=256, gt=0)

    @model_validator(mode="after")
    def validate_required_evaluator(self) -> Self:
        if self.evaluator is None:
            raise ConfigResolutionError("evaluation workflow requires evaluator")
        return self
