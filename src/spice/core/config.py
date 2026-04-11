"""Hydra-composed runtime configuration validated by Pydantic."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Self, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from .json import JsonObject


class ChainName(StrEnum):
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"


class ModelFamily(StrEnum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    TRANSFORMER_LSTM = "transformer_lstm"


class RpcProviderName(StrEnum):
    DIRECT = "direct"
    ALCHEMY = "alchemy"
    PUBLICNODE = "publicnode"


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


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _utc_midnight_timestamp(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=UTC).timestamp())


class ChainConfig(ConfigModel):
    name: ChainName
    chain_id: int = Field(gt=0)
    block_time_seconds: float = Field(gt=0)
    uses_poa_extra_data: bool


class DatasetWindowConfig(ConfigModel):
    start_date: date
    end_date: date

    @property
    def start_timestamp(self) -> int:
        return _utc_midnight_timestamp(self.start_date)

    @property
    def end_timestamp(self) -> int:
        return _utc_midnight_timestamp(self.end_date + timedelta(days=1))

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.start_date > self.end_date:
            raise ValueError(
                "dataset.window.start_date must be on or before dataset.window.end_date"
            )
        return self


class DatasetTemporalConfig(ConfigModel):
    max_delay_seconds: int = Field(gt=0)
    lookback_seconds: int = Field(gt=0)


class DatasetSamplingConfig(ConfigModel):
    anchor_count: int = Field(gt=0)
    history_anchor_count: int | None = Field(default=None, gt=0)

    @property
    def effective_history_anchor_count(self) -> int:
        if self.history_anchor_count is None:
            return self.anchor_count
        return self.history_anchor_count

    @model_validator(mode="after")
    def validate_history_anchor_count(self) -> Self:
        if self.effective_history_anchor_count < self.anchor_count:
            raise ValueError(
                "history_anchor_count must be at least dataset.sampling.anchor_count"
            )
        return self


class DatasetConfig(ConfigModel):
    id: str
    window: DatasetWindowConfig
    temporal: DatasetTemporalConfig
    sampling: DatasetSamplingConfig

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value:
            raise ValueError("dataset id must be a non-empty path segment")
        return value


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


class AcquisitionConfig(ConfigModel):
    dry_run: bool
    overwrite: bool
    chunk_size: int = Field(gt=0)
    rpc_batch_size: int = Field(gt=0)


class SimulationConfig(ConfigModel):
    window_seconds: int = Field(gt=0)
    arrival_rate_per_second: float = Field(gt=0.0)
    repetitions: int = Field(gt=0)
    seed: int = Field(ge=0)


class ModelConfig(ConfigModel):
    family: ModelFamily
    input_projection_dim: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    d_model: int = Field(gt=0)
    nhead: int = Field(gt=0)
    transformer_layers: int = Field(gt=0)
    feedforward_dim: int = Field(gt=0)
    head_hidden_dim: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_transformer_dimensions(self) -> Self:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encodings")
        return self


class TrackingConfig(ConfigModel):
    enabled: bool
    experiment_name: str
    tracking_uri: str
    tags: dict[str, str]


class TuningTrainingSearchSpace(ConfigModel):
    learning_rate: list[float] | None = Field(default=None, min_length=1)
    weight_decay: list[float] | None = Field(default=None, min_length=1)

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate_candidates(cls, values: list[float] | None) -> list[float] | None:
        if values is not None and any(value <= 0.0 for value in values):
            raise ValueError("tuning.search_space.training.learning_rate values must be positive")
        return values

    @field_validator("weight_decay")
    @classmethod
    def validate_weight_decay_candidates(cls, values: list[float] | None) -> list[float] | None:
        if values is not None and any(value < 0.0 for value in values):
            raise ValueError(
                "tuning.search_space.training.weight_decay values must be non-negative"
            )
        return values

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.learning_rate is None and self.weight_decay is None:
            raise ValueError("tuning.search_space.training must declare at least one field")
        return self


class TuningModelSearchSpace(ConfigModel):
    hidden_size: list[int] | None = Field(default=None, min_length=1)
    dropout: list[float] | None = Field(default=None, min_length=1)

    @field_validator("hidden_size")
    @classmethod
    def validate_hidden_size_candidates(cls, values: list[int] | None) -> list[int] | None:
        if values is not None and any(value <= 0 for value in values):
            raise ValueError("tuning.search_space.model.hidden_size values must be positive")
        return values

    @field_validator("dropout")
    @classmethod
    def validate_dropout_candidates(cls, values: list[float] | None) -> list[float] | None:
        if values is not None and any(value < 0.0 or value >= 1.0 for value in values):
            raise ValueError("tuning.search_space.model.dropout values must be in [0.0, 1.0)")
        return values

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.hidden_size is None and self.dropout is None:
            raise ValueError("tuning.search_space.model must declare at least one field")
        return self


class TuningSearchSpace(ConfigModel):
    training: TuningTrainingSearchSpace | None = None
    model: TuningModelSearchSpace | None = None

    @model_validator(mode="after")
    def validate_non_empty_search_space(self) -> Self:
        if self.training is None and self.model is None:
            raise ValueError("tuning.search_space must declare at least one parameter group")
        return self


class TunedTrainingParams(ConfigModel):
    learning_rate: float | None = Field(default=None, gt=0.0)
    weight_decay: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.learning_rate is None and self.weight_decay is None:
            raise ValueError("tuned training params must declare at least one field")
        return self


class TunedModelParams(ConfigModel):
    hidden_size: int | None = Field(default=None, gt=0)
    dropout: float | None = Field(default=None, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> Self:
        if self.hidden_size is None and self.dropout is None:
            raise ValueError("tuned model params must declare at least one field")
        return self


class TunedParameterSet(ConfigModel):
    training: TunedTrainingParams | None = None
    model: TunedModelParams | None = None

    @model_validator(mode="after")
    def validate_non_empty_param_set(self) -> Self:
        if self.training is None and self.model is None:
            raise ValueError("tuned parameter set must declare at least one parameter group")
        return self


class TuningConfig(ConfigModel):
    apply_best_params: bool
    study_name: str
    direction: StudyDirection
    trial_count: int = Field(gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    objective_metric: TuningObjective
    sampler_seed: int = Field(ge=0)
    enable_pruning: bool
    search_space: TuningSearchSpace


class RuntimeConfig(ConfigModel):
    output_root: Path
    hydra_run_dir: Path
    hydra_sweep_dir: Path


class PathsConfig(ConfigModel):
    output_root: Path
    dataset_root: Path
    metadata_root: Path
    history_dir: Path
    evaluation_dir: Path
    dataset_metadata_path: Path
    artifact_root: Path
    checkpoint_dir: Path
    train_report_path: Path
    simulation_report_path: Path
    tuning_root: Path
    tuning_best_params_path: Path
    mlruns_dir: Path


class ProviderConfig(ConfigModel):
    name: RpcProviderName
    endpoints: dict[ChainName, str]
    references: dict[ChainName, str]
    timeout_seconds: float = Field(gt=0.0)
    retry_count: int = Field(ge=0)
    backoff_factor: float = Field(ge=0.0)

    def endpoint_for(self, chain_name: ChainName) -> str:
        value = self.endpoints.get(chain_name, "")
        if not value:
            raise ValueError(f"Missing RPC endpoint for chain: {chain_name.value}")
        return value

    def reference_for(self, chain_name: ChainName) -> str:
        reference = self.references.get(chain_name)
        if reference:
            return reference
        return self.endpoint_for(chain_name)


class ExperimentConfig(ConfigModel):
    task: WorkflowTask
    chain: ChainConfig
    dataset: DatasetConfig
    model: ModelConfig
    acquisition: AcquisitionConfig
    split: SplitConfig
    training: TrainingConfig
    simulation: SimulationConfig
    tracking: TrackingConfig
    tuning: TuningConfig
    runtime: RuntimeConfig
    paths: PathsConfig
    provider: ProviderConfig

    @model_validator(mode="after")
    def validate_provider_for_task(self) -> Self:
        if self.task is WorkflowTask.ACQUIRE:
            self.provider.endpoint_for(self.chain.name)
        return self


_EXPERIMENT_CONFIG_ADAPTER = TypeAdapter(ExperimentConfig)


def coerce_config(cfg: DictConfig, *, task: WorkflowTask | str) -> ExperimentConfig:
    resolved_task = WorkflowTask(task)
    working = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(working, False)
    OmegaConf.update(working, "task", resolved_task.value, merge=False)
    OmegaConf.resolve(working)
    payload = OmegaConf.to_container(working, resolve=True, enum_to_str=True)
    if not isinstance(payload, dict):
        raise TypeError("Hydra configuration did not produce a mapping payload")
    payload.pop("hydra", None)
    return _EXPERIMENT_CONFIG_ADAPTER.validate_python(payload)


def config_to_dict(cfg: ExperimentConfig) -> JsonObject:
    payload = _EXPERIMENT_CONFIG_ADAPTER.dump_python(cfg, mode="json")
    if not isinstance(payload, dict):
        raise TypeError("ExperimentConfig did not serialize to a mapping payload")
    return cast(JsonObject, payload)


def revalidate_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return _EXPERIMENT_CONFIG_ADAPTER.validate_python(config_to_dict(cfg))
