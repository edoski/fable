"""Explicit runtime configuration models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from ..features import validate_feature_selection
from ..identifiers import artifact_storage_id, corpus_storage_id, study_storage_id


class WorkflowTask(StrEnum):
    ACQUIRE = "acquire"
    TUNE = "tune"
    TRAIN = "train"
    SIMULATE = "simulate"


class StudyDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class TuningObjective(StrEnum):
    VALIDATION_LOSS = "validation_loss"
    VALIDATION_ACCURACY = "validation_accuracy"
    VALIDATION_COST_OVER_OPTIMUM = "validation_cost_over_optimum"
    VALIDATION_PROFIT_OVER_BASELINE = "validation_profit_over_baseline"

    @property
    def metric_name(self) -> str:
        return self.value.removeprefix("validation_")


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


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


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


class TaskSpec(ConfigModel):
    id: str
    lookback_seconds: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    max_supported_delay_seconds: int = Field(gt=0)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="task.id")


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
    action_loss_weight: float = Field(gt=0.0)
    fee_loss_weight: float = Field(gt=0.0)
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


class ModelConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="model.id")


class FeatureSetConfig(ConfigModel):
    id: str
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
        validate_feature_selection(self.id, tuple(self.outputs))
        return self


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
            raise ValueError(
                "tuning_space.training.weight_decay values must be non-negative"
            )
        return values

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.learning_rate is None and self.weight_decay is None:
            raise ValueError("tuning_space.training must declare at least one field")
        return self


class ModelTuningSpaceConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="tuning_space.model.id")


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


class TunedModelParams(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="tuned model params id")


class TunedParameterSet(ConfigModel):
    training: TunedTrainingParams | None = None
    model: SerializeAsAny[TunedModelParams] | None = None

    @model_validator(mode="after")
    def validate_non_empty_param_set(self) -> Self:
        if self.training is None and self.model is None:
            raise ValueError("tuned parameter set must declare at least one parameter group")
        return self


class TuningConfig(ConfigModel):
    direction: StudyDirection
    trial_count: int = Field(gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    objective_metric: TuningObjective
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
                "provider endpoint spec cannot declare both reference and "
                "reference_template"
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
            if (
                self.concurrency is not None
                and self.concurrency not in self.concurrency_rungs
            ):
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


@dataclass(frozen=True, slots=True)
class PathLayout:
    output_root: Path
    catalog_db: Path
    corpus_id: str
    corpus_root: Path
    history_dir: Path
    evaluation_dir: Path
    corpus_state_db: Path
    artifact_id: str | None = None
    artifact_root: Path | None = None
    checkpoint_dir: Path | None = None
    artifact_state_db: Path | None = None
    study_id: str | None = None
    study_root: Path | None = None
    study_state_db: Path | None = None


def build_path_layout(
    *,
    storage: StorageSpec,
    chain: ChainSpec,
    dataset: DatasetSpec,
    feature_set_name: str | None = None,
    model_name: str | None = None,
    task_name: str | None = None,
    feature_set_payload: dict[str, object] | None = None,
    model_payload: dict[str, object] | None = None,
    task_payload: dict[str, object] | None = None,
    variant: ArtifactVariant = ArtifactVariant.BASELINE,
    study_name: str = "default",
    include_artifacts: bool = False,
    tuning_mode: bool = False,
) -> PathLayout:
    output_root = storage.root
    catalog_db = output_root / ".spice" / "catalog.sqlite"
    corpus_id = corpus_storage_id(chain_name=chain.name, dataset_name=dataset.name)
    corpus_root = output_root / "corpora" / chain.name / corpus_id
    artifact_id: str | None = None
    artifact_root: Path | None = None
    checkpoint_dir: Path | None = None
    artifact_state_db: Path | None = None
    study_id: str | None = None
    study_root: Path | None = None
    study_state_db: Path | None = None

    if include_artifacts:
        if feature_set_name is None or model_name is None or task_name is None:
            raise ValueError("artifact paths require feature_set_name, model_name, task_name")
        if feature_set_payload is None or model_payload is None or task_payload is None:
            raise ValueError(
                "artifact paths require feature_set_payload, model_payload, task_payload"
            )
        resolved_feature_set_payload = feature_set_payload
        resolved_model_payload = model_payload
        resolved_task_payload = task_payload
        if tuning_mode or variant is ArtifactVariant.TUNED:
            study_id = study_storage_id(
                chain_name=chain.name,
                corpus_id=corpus_id,
                feature_set=resolved_feature_set_payload,
                model=resolved_model_payload,
                task=resolved_task_payload,
                study_name=study_name,
            )
            study_root = output_root / "studies" / chain.name / study_id
            study_state_db = study_root / ".spice" / "state.sqlite"
        if not tuning_mode:
            artifact_id = artifact_storage_id(
                chain_name=chain.name,
                corpus_id=corpus_id,
                feature_set=resolved_feature_set_payload,
                model=resolved_model_payload,
                task=resolved_task_payload,
                variant=variant.value,
                study_id=study_id if variant is ArtifactVariant.TUNED else None,
            )
            artifact_root = output_root / "artifacts" / chain.name / artifact_id
            checkpoint_dir = artifact_root / "checkpoints"
            artifact_state_db = artifact_root / ".spice" / "state.sqlite"

    return PathLayout(
        output_root=output_root,
        catalog_db=catalog_db,
        corpus_id=corpus_id,
        corpus_root=corpus_root,
        history_dir=corpus_root / "history",
        evaluation_dir=corpus_root / "evaluation",
        corpus_state_db=corpus_root / ".spice" / "state.sqlite",
        artifact_id=artifact_id,
        artifact_root=artifact_root,
        checkpoint_dir=checkpoint_dir,
        artifact_state_db=artifact_state_db,
        study_id=study_id,
        study_root=study_root,
        study_state_db=study_state_db,
    )


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
    task: TaskSpec
    feature_set: FeatureSetConfig
    provider: ProviderSpec
    acquisition: AcquisitionConfig

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        self.provider.endpoint_for(self.chain.name)
        return self

    @property
    def paths(self) -> PathLayout:
        return build_path_layout(storage=self.storage, chain=self.chain, dataset=self.dataset)


class ModelWorkflowConfig(WorkflowConfig):
    task: TaskSpec
    model: SerializeAsAny[ModelConfig]
    feature_set: FeatureSetConfig
    study: StudyConfig = StudyConfig()
    artifact: ArtifactConfig = ArtifactConfig()

    @property
    def paths(self) -> PathLayout:
        return build_path_layout(
            storage=self.storage,
            chain=self.chain,
            dataset=self.dataset,
            feature_set_name=self.feature_set.id,
            model_name=self.model.id,
            task_name=self.task.id,
            feature_set_payload=self.feature_set.model_dump(mode="json", exclude_none=True),
            model_payload=self.model.model_dump(mode="json", exclude_none=True),
            task_payload=self.task.model_dump(mode="json", exclude_none=True),
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
        if self.execution.requested_delay_seconds > self.task.max_supported_delay_seconds:
            raise ValueError(
                "execution.requested_delay_seconds must be <= "
                "task.max_supported_delay_seconds"
            )
        return self


class PresetSpec(ConfigModel):
    dataset: str | None = None
    task: str | None = None
    execution: str | None = None
    chain: str | None = None
    provider: str | None = None
    model: str | None = None
    feature_set: str | None = None
    acquisition: str | None = None
    training: str | None = None
    split: str | None = None
    simulation: str | None = None
    tuning: str | None = None
    tuning_space: str | None = None
    storage: StorageSpec | None = None
    study: StudyConfig | None = None
    artifact: ArtifactConfig | None = None
