"""Hydra-composed runtime configuration validated by Pydantic."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import Any, Self

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


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


class TuningConfig(ConfigModel):
    apply_best_params: bool
    study_name: str
    direction: str
    trial_count: int = Field(gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    objective_metric: str
    sampler_seed: int = Field(ge=0)
    enable_pruning: bool
    search_space: dict[str, list[Any]]


class RuntimeConfig(ConfigModel):
    output_root: str
    hydra_run_dir: str
    hydra_sweep_dir: str


class PathsConfig(ConfigModel):
    output_root: str
    dataset_root: str
    metadata_root: str
    history_dir: str
    evaluation_dir: str
    dataset_metadata_path: str
    artifact_root: str
    checkpoint_dir: str
    train_report_path: str
    simulation_report_path: str
    tuning_root: str
    tuning_best_params_path: str
    mlruns_dir: str


class ProviderConfig(ConfigModel):
    name: RpcProviderName
    endpoints: dict[str, str]
    references: dict[str, str]
    timeout_seconds: float = Field(gt=0.0)
    retry_count: int = Field(ge=0)
    backoff_factor: float = Field(ge=0.0)

    def endpoint_for(self, chain_name: ChainName | str) -> str:
        key = chain_name.value if isinstance(chain_name, ChainName) else str(chain_name)
        value = self.endpoints.get(key, "")
        if not value:
            raise ValueError(f"Missing RPC endpoint for chain: {key}")
        return value

    def reference_for(self, chain_name: ChainName | str) -> str:
        key = chain_name.value if isinstance(chain_name, ChainName) else str(chain_name)
        reference = self.references.get(key)
        if reference:
            return reference
        return self.endpoint_for(key)


class ExperimentConfig(ConfigModel):
    task: str
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
        if self.task == "acquire":
            self.provider.endpoint_for(self.chain.name)
        return self


_EXPERIMENT_CONFIG_ADAPTER = TypeAdapter(ExperimentConfig)


def coerce_config(cfg: DictConfig, *, task: str) -> ExperimentConfig:
    working = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(working, False)
    OmegaConf.update(working, "task", task, merge=False)
    OmegaConf.resolve(working)
    payload = OmegaConf.to_container(working, resolve=True, enum_to_str=True)
    if not isinstance(payload, dict):
        raise TypeError("Hydra configuration did not produce a mapping payload")
    payload.pop("hydra", None)
    return _EXPERIMENT_CONFIG_ADAPTER.validate_python(payload)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    payload = _EXPERIMENT_CONFIG_ADAPTER.dump_python(cfg, mode="json")
    if not isinstance(payload, dict):
        raise TypeError("ExperimentConfig did not serialize to a mapping payload")
    return payload


def revalidate_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return _EXPERIMENT_CONFIG_ADAPTER.validate_python(config_to_dict(cfg))
