"""Explicit runtime configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from pydantic import (
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from ..core.validation import validate_path_segment
from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig
from ..features import FeatureFamilyConfig, validate_feature_selection
from ..modeling.dataset_builders import (
    DatasetBuilderConfig,
    coerce_dataset_builder_config,
)
from ..modeling.families.base import (
    ConfigModel,
    ModelConfig,
    ModelTuningSpaceConfig,
    TunedModelParams,
)
from ..objectives import ObjectiveConfig
from ..prediction import validate_prediction_family_id
from ..temporal.compilers import ProblemCompilerConfig
from ..temporal.input_normalization import InputNormalizationConfig
from ..temporal.realization import RealizationPolicyConfig


class WorkflowTask(StrEnum):
    ACQUIRE = "acquire"
    TUNE = "tune"
    TRAIN = "train"
    EVALUATE = "evaluate"


class ArtifactVariant(StrEnum):
    BASELINE = "baseline"
    TUNED = "tuned"


def _utc_midnight_timestamp(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=UTC).timestamp())


def _validate_http_endpoint_url(value: str, *, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an http:// or https:// URL")
    return value


class ChainRuntimeSpec(ConfigModel):
    chain_id: int = Field(gt=0)
    uses_poa_extra_data: bool
    nominal_block_time_seconds: float = Field(gt=0.0)


class ChainSpec(ConfigModel):
    name: str
    runtime: ChainRuntimeSpec

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_path_segment(value, label="chain.name")


class DatasetSpec(ConfigModel):
    name: str
    evaluation_date: date

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_path_segment(value, label="dataset.name")


class StorageSpec(ConfigModel):
    root: Path = Path("outputs")


class ProblemSpec(ConfigModel):
    id: str
    lookback_seconds: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    max_delay_seconds: int = Field(gt=0)
    compiler: SerializeAsAny[ProblemCompilerConfig]
    realization_policy: SerializeAsAny[RealizationPolicyConfig]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="problem.id")


def coerce_problem_spec(payload: Mapping[str, object] | ProblemSpec) -> ProblemSpec:
    from ..temporal.compilers import coerce_problem_compiler_config

    raw_payload = (
        payload.model_dump(mode="json") if isinstance(payload, ProblemSpec) else dict(payload)
    )
    raw_compiler = raw_payload.get("compiler")
    if raw_compiler is None:
        raise ConfigResolutionError("problem.compiler is required")
    if not isinstance(raw_compiler, Mapping) and not isinstance(
        raw_compiler,
        ProblemCompilerConfig,
    ):
        raise ConfigResolutionError("problem.compiler must be a mapping")
    raw_realization_policy = raw_payload.get("realization_policy")
    if raw_realization_policy is None:
        raise ConfigResolutionError("problem.realization_policy is required")
    if not isinstance(raw_realization_policy, Mapping) and not isinstance(
        raw_realization_policy,
        RealizationPolicyConfig,
    ):
        raise ConfigResolutionError("problem.realization_policy must be a mapping")
    from ..temporal.realization import coerce_realization_policy_config

    raw_payload["compiler"] = coerce_problem_compiler_config(raw_compiler)
    raw_payload["realization_policy"] = coerce_realization_policy_config(raw_realization_policy)
    return ProblemSpec.model_validate(raw_payload)


class SplitConfig(ConfigModel):
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_split(self) -> Self:
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
        return self


class EarlyStoppingConfig(ConfigModel):
    patience: int = Field(gt=0)
    min_delta: float = Field(ge=0.0)


class TrainingConfig(ConfigModel):
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    early_stopping: EarlyStoppingConfig
    gradient_clip_norm: float = Field(gt=0.0)
    seed: int = Field(ge=0)
    deterministic: bool
    log_every_n_steps: int = Field(gt=0)
    input_normalization: SerializeAsAny[InputNormalizationConfig] = Field(
        default_factory=lambda: _default_input_normalization_config()
    )

    @field_validator("input_normalization", mode="before")
    @classmethod
    def validate_input_normalization(
        cls,
        value: object,
    ) -> InputNormalizationConfig:
        from ..temporal.input_normalization import coerce_input_normalization_config

        if value is None:
            return _default_input_normalization_config()
        if isinstance(value, Mapping):
            return coerce_input_normalization_config(value)
        if isinstance(value, InputNormalizationConfig):
            return coerce_input_normalization_config(value)
        raise TypeError("training.input_normalization must be a mapping or config model")


def _default_input_normalization_config() -> InputNormalizationConfig:
    from ..temporal.input_normalization import coerce_input_normalization_config

    return coerce_input_normalization_config({"id": "window_weighted_standard"})


class AcquisitionRpcConfig(ConfigModel):
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


class AcquisitionConfig(ConfigModel):
    dry_run: bool = False
    chunk_size: int = Field(gt=0)
    rpc: AcquisitionRpcConfig


class FeatureSetConfig(ConfigModel):
    id: str
    family: SerializeAsAny[FeatureFamilyConfig]
    outputs: list[str] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="feature_set.id")

    @field_validator("outputs")
    @classmethod
    def validate_outputs(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("feature_set.outputs must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_feature_selection(self) -> Self:
        validate_feature_selection(self.id, self.family.id, tuple(self.outputs))
        return self


def coerce_feature_set_config(payload: Mapping[str, object] | FeatureSetConfig) -> FeatureSetConfig:
    from ..features import coerce_feature_family_config

    raw_payload = (
        payload.model_dump(mode="json") if isinstance(payload, FeatureSetConfig) else dict(payload)
    )
    raw_family = raw_payload.get("family")
    if raw_family is None:
        raise ConfigResolutionError("feature_set.family is required")
    if not isinstance(raw_family, Mapping) and not isinstance(raw_family, FeatureFamilyConfig):
        raise ConfigResolutionError("feature_set.family must be a mapping")
    raw_payload["family"] = coerce_feature_family_config(raw_family)
    return FeatureSetConfig.model_validate(raw_payload)


class PredictionConfig(ConfigModel):
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


def coerce_prediction_config(payload: Mapping[str, object] | PredictionConfig) -> PredictionConfig:
    if isinstance(payload, PredictionConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise ConfigResolutionError("prediction must be a mapping")
    return PredictionConfig.model_validate(dict(payload))


class StudyConfig(ConfigModel):
    name: str = "default"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_path_segment(value, label="study.name")


class ArtifactConfig(ConfigModel):
    variant: ArtifactVariant = ArtifactVariant.BASELINE


class TuningTrainingSearchSpace(ConfigModel):
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


class TuningProblemSearchSpace(ConfigModel):
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


class TuningSpaceConfig(ConfigModel):
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


class TunedTrainingParams(ConfigModel):
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


class TunedProblemParams(ConfigModel):
    lookback_seconds: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.lookback_seconds is None:
            raise ValueError("tuned problem params must declare at least one field")
        return self


class TunedParameterSet(ConfigModel):
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


class TuningConfig(ConfigModel):
    trial_count: int = Field(gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    sampler_seed: int = Field(ge=0)
    enable_pruning: bool


class ProviderEndpointConfig(ConfigModel):
    url: str
    reference: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_http_endpoint_url(value, label="provider.endpoints.url")


class ProviderTransportConfig(ConfigModel):
    timeout_seconds: float = Field(gt=0.0)
    retry_count: int = Field(ge=0)
    backoff_factor: float = Field(ge=0.0)


class ResolvedRpcEndpointConfig(ConfigModel):
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


class ProviderSpec(ConfigModel):
    name: str
    transport: ProviderTransportConfig
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


class WorkflowConfig(ConfigModel):
    workflow: WorkflowTask
    chain: ChainSpec
    dataset: DatasetSpec
    storage: StorageSpec

    @property
    def evaluation_window_start_timestamp(self) -> int:
        return _utc_midnight_timestamp(self.dataset.evaluation_date)

    @property
    def evaluation_window_end_timestamp(self) -> int:
        return _utc_midnight_timestamp(self.dataset.evaluation_date + timedelta(days=1))

    @property
    def history_window_end_timestamp(self) -> int:
        return self.evaluation_window_start_timestamp


class AcquireConfig(WorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.ACQUIRE
    problem: ProblemSpec
    feature_set: FeatureSetConfig
    rpc_endpoint: ResolvedRpcEndpointConfig
    acquisition: AcquisitionConfig


class ModelWorkflowConfig(WorkflowConfig):
    problem: ProblemSpec
    model: SerializeAsAny[ModelConfig]
    dataset_builder: SerializeAsAny[DatasetBuilderConfig]
    feature_set: FeatureSetConfig
    prediction: PredictionConfig
    study: StudyConfig = Field(default_factory=StudyConfig)
    artifact: ArtifactConfig = Field(default_factory=ArtifactConfig)


class ObjectiveModelWorkflowConfig(ModelWorkflowConfig):
    objective: SerializeAsAny[ObjectiveConfig]
    evaluation: SerializeAsAny[EvaluatorConfig] | None = None


class TrainConfig(ObjectiveModelWorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.TRAIN
    split: SplitConfig
    training: TrainingConfig
    tuning: TuningConfig | None = None
    tuning_space: TuningSpaceConfig | None = None


class TuneConfig(ObjectiveModelWorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.TUNE
    split: SplitConfig
    training: TrainingConfig
    tuning: TuningConfig
    tuning_space: TuningSpaceConfig

    @model_validator(mode="after")
    def validate_required_objective_and_tuning_space(self) -> Self:
        if self.tuning_space.model.id != self.model.id:
            raise ConfigResolutionError("tuning_space.model.id must match model.id")
        if not self.tuning_space.has_candidates():
            raise ConfigResolutionError("tuning_space must declare at least one tunable parameter")
        return self


class EvaluateConfig(ObjectiveModelWorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.EVALUATE
    split: SplitConfig
    training: TrainingConfig
    delay_seconds: int = Field(gt=0)
    tuning: TuningConfig | None = None
    tuning_space: TuningSpaceConfig | None = None

    @model_validator(mode="after")
    def validate_required_objective_and_delay(self) -> Self:
        if self.evaluation is None:
            raise ConfigResolutionError("evaluation workflow requires evaluation")
        if self.delay_seconds > self.problem.max_delay_seconds:
            raise ConfigResolutionError("delay_seconds must be <= problem.max_delay_seconds")
        return self
