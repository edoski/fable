"""Explicit runtime configuration models."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import (
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from ..features import FeatureFamilyConfig, validate_feature_selection
from ..modeling.families.base import (
    ConfigModel,
    ModelConfig,
    ModelTuningSpaceConfig,
    TunedModelParams,
)
from ..prediction import PredictionFamilyConfig
from ..temporal.compilers import ProblemCompilerConfig

if TYPE_CHECKING:
    from ..storage.layout import PathLayout


class WorkflowTask(StrEnum):
    ACQUIRE = "acquire"
    TUNE = "tune"
    TRAIN = "train"
    SIMULATE = "simulate"


class TrainingPrecision(StrEnum):
    AUTO = "auto"
    FP32 = "fp32"
    FP16_MIXED = "fp16-mixed"
    BF16_MIXED = "bf16-mixed"


class CompileMode(StrEnum):
    AUTO = "auto"
    OFF = "off"
    ON = "on"


class ArtifactVariant(StrEnum):
    BASELINE = "baseline"
    TUNED = "tuned"


def _validate_path_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty path segment")
    return value


def _utc_midnight_timestamp(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=UTC).timestamp())


class ChainRuntimeSpec(ConfigModel):
    chain_id: int = Field(gt=0)
    uses_poa_extra_data: bool


class ChainSpec(ConfigModel):
    name: str
    runtime: ChainRuntimeSpec

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_path_segment(value, label="chain.name")


class DatasetSpec(ConfigModel):
    name: str
    evaluation_date: date

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_path_segment(value, label="dataset.name")


class StorageSpec(ConfigModel):
    root: Path = Path("outputs")


class ProblemSpec(ConfigModel):
    id: str
    lookback_seconds: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    max_supported_delay_seconds: int = Field(gt=0)
    compiler: SerializeAsAny[ProblemCompilerConfig]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="problem.id")


def coerce_problem_spec(payload: Mapping[str, object] | ProblemSpec) -> ProblemSpec:
    from ..temporal.compilers import coerce_problem_compiler_config

    raw_payload = (
        payload.model_dump(mode="json")
        if isinstance(payload, ProblemSpec)
        else dict(payload)
    )
    raw_compiler = raw_payload.get("compiler")
    if raw_compiler is None:
        raise ValueError("problem.compiler is required")
    if not isinstance(raw_compiler, Mapping) and not isinstance(
        raw_compiler,
        ProblemCompilerConfig,
    ):
        raise TypeError("problem.compiler must be a mapping")
    raw_payload["compiler"] = coerce_problem_compiler_config(raw_compiler)
    return ProblemSpec.model_validate(raw_payload)


class ExecutionSpec(ConfigModel):
    id: str
    requested_delay_seconds: int = Field(gt=0)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="execution.id")


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
    device: str
    seed: int = Field(ge=0)
    deterministic: bool
    log_every_n_steps: int = Field(gt=0)
    precision: TrainingPrecision
    compile: CompileMode


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


class SimulationConfig(ConfigModel):
    window_seconds: int = Field(gt=0)
    arrival_rate_per_second: float = Field(gt=0.0)
    repetitions: int = Field(gt=0)
    seed: int = Field(ge=0)


class FeatureSetConfig(ConfigModel):
    id: str
    family: SerializeAsAny[FeatureFamilyConfig]
    outputs: list[str] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="feature_set.id")

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
        payload.model_dump(mode="json")
        if isinstance(payload, FeatureSetConfig)
        else dict(payload)
    )
    raw_family = raw_payload.get("family")
    if raw_family is None:
        raise ValueError("feature_set.family is required")
    if not isinstance(raw_family, Mapping) and not isinstance(raw_family, FeatureFamilyConfig):
        raise TypeError("feature_set.family must be a mapping")
    raw_payload["family"] = coerce_feature_family_config(raw_family)
    return FeatureSetConfig.model_validate(raw_payload)


class PredictionConfig(ConfigModel):
    id: str
    family: SerializeAsAny[PredictionFamilyConfig]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="prediction.id")


def coerce_prediction_config(payload: Mapping[str, object] | PredictionConfig) -> PredictionConfig:
    from ..prediction import coerce_prediction_family_config

    raw_payload = (
        payload.model_dump(mode="json")
        if isinstance(payload, PredictionConfig)
        else dict(payload)
    )
    raw_family = raw_payload.get("family")
    if raw_family is None:
        raise ValueError("prediction.family is required")
    if not isinstance(raw_family, Mapping) and not isinstance(raw_family, PredictionFamilyConfig):
        raise TypeError("prediction.family must be a mapping")
    raw_payload["family"] = coerce_prediction_family_config(raw_family)
    return PredictionConfig.model_validate(raw_payload)


class StudyConfig(ConfigModel):
    name: str = "default"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_path_segment(value, label="study.name")


class ArtifactConfig(ConfigModel):
    variant: ArtifactVariant = ArtifactVariant.BASELINE


class TuningTrainingSearchSpace(ConfigModel):
    learning_rate: list[float] | None = Field(default=None, min_length=1)
    weight_decay: list[float] | None = Field(default=None, min_length=1)

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

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.learning_rate is None and self.weight_decay is None:
            raise ValueError("tuning_space.training must declare at least one field")
        return self


class TuningSpaceConfig(ConfigModel):
    training: TuningTrainingSearchSpace | None = None
    model: SerializeAsAny[ModelTuningSpaceConfig]

    def has_candidates(self) -> bool:
        model_candidates = self.model.model_dump(exclude={"id"}, exclude_none=True)
        return self.training is not None or bool(model_candidates)


class TunedTrainingParams(ConfigModel):
    learning_rate: float | None = Field(default=None, gt=0.0)
    weight_decay: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.learning_rate is None and self.weight_decay is None:
            raise ValueError("tuned training params must declare at least one field")
        return self


class TunedParameterSet(ConfigModel):
    training: TunedTrainingParams | None = None
    model: SerializeAsAny[TunedModelParams] | None = None

    @model_validator(mode="after")
    def validate_non_empty_param_set(self) -> Self:
        if self.training is None and self.model is None:
            raise ValueError("tuned parameter set must declare at least one parameter group")
        return self


class TuningConfig(ConfigModel):
    trial_count: int = Field(gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    sampler_seed: int = Field(ge=0)
    enable_pruning: bool


class ProviderEndpointSpec(ConfigModel):
    url: str | None = None
    url_template: str | None = None
    env_var: str | None = None
    reference: str | None = None
    reference_template: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        source_count = sum(
            value is not None for value in (self.url, self.url_template, self.env_var)
        )
        if source_count != 1:
            raise ValueError(
                "provider endpoint spec must declare exactly one of url, url_template, env_var"
            )
        if self.reference is not None and self.reference_template is not None:
            raise ValueError(
                "provider endpoint spec cannot declare both reference and reference_template"
            )
        return self

    def resolve(self, *, api_key_envvar: str | None = None) -> tuple[str, str]:
        if self.env_var is not None:
            value = os.getenv(self.env_var, "")
            if not value:
                raise ValueError(f"Missing RPC endpoint env var: {self.env_var}")
            return value, f"${self.env_var}"

        if self.url is not None:
            return self.url, self.reference or self.url

        assert self.url_template is not None
        if not api_key_envvar:
            raise ValueError("Provider endpoint template requires api_key_envvar")
        api_key = os.getenv(api_key_envvar, "")
        if not api_key:
            raise ValueError(f"Missing RPC API key env var: {api_key_envvar}")
        endpoint = self.url_template.format(key=api_key)
        reference_template = self.reference_template or self.url_template
        return endpoint, reference_template.format(key=f"${api_key_envvar}")


class ProviderRpcConfig(ConfigModel):
    timeout_seconds: float = Field(gt=0.0)
    retry_count: int = Field(ge=0)
    backoff_factor: float = Field(ge=0.0)
    api_key_envvar: str | None = None


class ProviderChainSpec(ConfigModel):
    endpoint: ProviderEndpointSpec


class ProviderAcquisitionRpcOverrides(ConfigModel):
    batch_size: int | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, gt=0)
    min_batch_size: int | None = Field(default=None, gt=0)
    concurrency_rungs: list[int] | None = None

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        if self.min_batch_size is not None and self.batch_size is not None:
            if self.min_batch_size > self.batch_size:
                raise ValueError(
                    "provider acquisition rpc override min_batch_size must be <= batch_size"
                )
        if self.concurrency_rungs is not None:
            if sorted(self.concurrency_rungs) != self.concurrency_rungs:
                raise ValueError(
                    "provider acquisition rpc override concurrency_rungs must be sorted ascending"
                )
            if len(set(self.concurrency_rungs)) != len(self.concurrency_rungs):
                raise ValueError(
                    "provider acquisition rpc override "
                    "concurrency_rungs must not contain duplicates"
                )
            if any(value <= 0 for value in self.concurrency_rungs):
                raise ValueError(
                    "provider acquisition rpc override concurrency_rungs values must be positive"
                )
            if self.concurrency is not None and self.concurrency not in self.concurrency_rungs:
                raise ValueError(
                    "provider acquisition rpc override "
                    "concurrency must be present in concurrency_rungs"
                )
        return self


class ProviderAcquisitionOverrides(ConfigModel):
    chunk_size: int | None = Field(default=None, gt=0)
    rpc: ProviderAcquisitionRpcOverrides | None = None


class ProviderAcquisitionConfig(ConfigModel):
    overrides: ProviderAcquisitionOverrides


class ProviderSpec(ConfigModel):
    name: str
    rpc: ProviderRpcConfig
    chains: dict[str, ProviderChainSpec]
    acquisition: ProviderAcquisitionConfig | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_path_segment(value, label="provider.name")

    @model_validator(mode="after")
    def validate_chain_coverage(self) -> Self:
        if not self.chains:
            raise ValueError("provider.chains must not be empty")
        for name in self.chains:
            _validate_path_segment(name, label="provider.chains key")
        return self

    def endpoint_spec_for(self, chain_name: str) -> ProviderEndpointSpec:
        try:
            return self.chains[chain_name].endpoint
        except KeyError as exc:
            raise ValueError(
                f"provider {self.name} does not define chain endpoint for {chain_name}"
            ) from exc

    def endpoint_for(self, chain_name: str) -> str:
        endpoint, _ = self.endpoint_spec_for(chain_name).resolve(
            api_key_envvar=self.rpc.api_key_envvar
        )
        return endpoint

    def reference_for(self, chain_name: str) -> str:
        _, reference = self.endpoint_spec_for(chain_name).resolve(
            api_key_envvar=self.rpc.api_key_envvar
        )
        return reference


def apply_provider_acquisition_overrides(
    *,
    provider: ProviderSpec,
    acquisition: AcquisitionConfig,
) -> AcquisitionConfig:
    if provider.acquisition is None:
        return acquisition
    overrides = provider.acquisition.overrides.model_dump(mode="json", exclude_none=True)
    if not overrides:
        return acquisition
    merged = acquisition.model_dump(mode="json")
    if "rpc" in overrides and isinstance(merged.get("rpc"), dict):
        merged["rpc"] = {
            **merged["rpc"],
            **overrides.pop("rpc"),
        }
    merged.update(overrides)
    return AcquisitionConfig.model_validate(merged)


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
    provider: ProviderSpec
    acquisition: AcquisitionConfig

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        self.provider.endpoint_for(self.chain.name)
        return self

    @property
    def paths(self) -> PathLayout:
        from ..storage.layout import build_path_layout

        return build_path_layout(storage=self.storage, chain=self.chain, dataset=self.dataset)


class ModelWorkflowConfig(WorkflowConfig):
    problem: ProblemSpec
    model: SerializeAsAny[ModelConfig]
    feature_set: FeatureSetConfig
    prediction: PredictionConfig
    study: StudyConfig = StudyConfig()
    artifact: ArtifactConfig = ArtifactConfig()

    @property
    def paths(self) -> PathLayout:
        from ..storage.layout import build_path_layout

        return build_path_layout(
            storage=self.storage,
            chain=self.chain,
            dataset=self.dataset,
            feature_set_name=self.feature_set.id,
            model_name=self.model.id,
            problem_name=self.problem.id,
            feature_set_payload=self.feature_set.model_dump(mode="json", exclude_none=True),
            model_payload=self.model.model_dump(mode="json", exclude_none=True),
            problem_payload=self.problem.model_dump(mode="json", exclude_none=True),
            prediction_name=self.prediction.id,
            prediction_payload=self.prediction.model_dump(mode="json", exclude_none=True),
            variant=self.artifact.variant,
            study_name=self.study.name,
            include_artifacts=True,
            tuning_mode=self.workflow is WorkflowTask.TUNE,
        )


class TrainConfig(ModelWorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.TRAIN
    split: SplitConfig
    training: TrainingConfig


class TuneConfig(ModelWorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.TUNE
    split: SplitConfig
    training: TrainingConfig
    tuning: TuningConfig
    tuning_space: TuningSpaceConfig

    @model_validator(mode="after")
    def validate_tuning_space(self) -> Self:
        if self.tuning_space.model.id != self.model.id:
            raise ValueError("tuning_space.model.id must match model.id")
        if not self.tuning_space.has_candidates():
            raise ValueError("tuning_space must declare at least one tunable parameter")
        return self


class SimulateConfig(ModelWorkflowConfig):
    workflow: WorkflowTask = WorkflowTask.SIMULATE
    training: TrainingConfig
    simulation: SimulationConfig
    execution: ExecutionSpec

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        if self.execution.requested_delay_seconds > self.problem.max_supported_delay_seconds:
            raise ValueError(
                "execution.requested_delay_seconds must be <= problem.max_supported_delay_seconds"
            )
        return self


class PresetSpec(ConfigModel):
    dataset: str | None = None
    problem: str | None = None
    execution: str | None = None
    chain: str | None = None
    provider: str | None = None
    model: str | None = None
    feature_set: str | None = None
    prediction: str | None = None
    acquisition: str | None = None
    training: str | None = None
    split: str | None = None
    simulation: str | None = None
    tuning: str | None = None
    tuning_space: str | None = None
    storage: StorageSpec | None = None
    study: StudyConfig | None = None
    artifact: ArtifactConfig | None = None
