"""Hydra-driven runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .constants import EVALUATION_END_TS, EVALUATION_START_TS


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


@dataclass(slots=True)
class ChainConfig:
    name: ChainName = ChainName.ETHEREUM
    chain_id: int = 1
    block_time_seconds: float = 12.0
    uses_poa_extra_data: bool = False


@dataclass(slots=True)
class DatasetConfig:
    id: str = "icdcs_2025_11_09"
    evaluation_start_timestamp: int = EVALUATION_START_TS
    evaluation_end_timestamp: int = EVALUATION_END_TS
    min_history_anchor_count: int = 400_000


@dataclass(slots=True)
class SplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    effective_batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0
    alpha: float = 1.0
    beta: float = 0.25
    device: str = "auto"
    seed: int = 2026
    deterministic: bool = True
    log_every_n_steps: int = 10


@dataclass(slots=True)
class PullConfig:
    requests_per_second: int = 10
    max_concurrent_requests: int = 2
    max_concurrent_chunks: int = 1
    chunk_size: int = 1000
    dry_run: bool = False
    overwrite: bool = False
    enrich_batch_size: int = 100
    max_methods_per_second: float = 20.0


@dataclass(slots=True)
class SimulationConfig:
    window_seconds: int = 7_200
    arrival_rate_per_second: float = 0.05
    repetitions: int = 50
    seed: int = 2026


@dataclass(slots=True)
class ModelConfig:
    family: ModelFamily = ModelFamily.LSTM
    input_projection_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    d_model: int = 128
    nhead: int = 4
    transformer_layers: int = 2
    feedforward_dim: int = 512
    head_hidden_dim: int = 64


@dataclass(slots=True)
class TrackingConfig:
    enabled: bool = True
    experiment_name: str = "spice"
    tracking_uri: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TuningConfig:
    apply_best_params: bool = False
    study_name: str = "spice-study"
    direction: str = "maximize"
    n_trials: int = 20
    timeout_seconds: int | None = None
    metric_name: str = "validation_profit_over_baseline"
    sampler_seed: int = 2026
    prune: bool = True
    search_space: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "training.learning_rate": [1e-4, 3e-4, 1e-3],
            "training.weight_decay": [0.0, 1e-2, 5e-2],
            "model.hidden_size": [64, 128, 256],
            "model.dropout": [0.0, 0.1, 0.2],
        }
    )


@dataclass(slots=True)
class RuntimeConfig:
    output_root: str = "artifacts"
    hydra_run_dir: str = ".hydra/runs/${task}/${now:%Y-%m-%d_%H-%M-%S}"
    hydra_sweep_dir: str = ".hydra/sweeps/${task}/${now:%Y-%m-%d_%H-%M-%S}"


@dataclass(slots=True)
class PathsConfig:
    output_root: str = "${runtime.output_root}"
    dataset_root: str = "${paths.output_root}/datasets/${chain.name}/${dataset.id}"
    metadata_root: str = "${paths.dataset_root}/.spice"
    raw_root: str = "${paths.dataset_root}/raw"
    raw_history_dir: str = "${paths.raw_root}/history"
    raw_evaluation_dir: str = "${paths.raw_root}/evaluation"
    enriched_root: str = "${paths.dataset_root}/enriched"
    enriched_history_dir: str = "${paths.enriched_root}/history"
    enriched_evaluation_dir: str = "${paths.enriched_root}/evaluation"
    dataset_metadata_path: str = "${paths.metadata_root}/metadata.json"
    artifact_root: str = (
        "${paths.output_root}/models/${chain.name}/${dataset.id}/${model.family}/${max_delay_seconds}s"
    )
    checkpoint_dir: str = "${paths.artifact_root}/checkpoints"
    train_report_path: str = "${paths.artifact_root}/train_report.json"
    simulation_report_path: str = "${paths.artifact_root}/simulation_report.json"
    tuning_root: str = "${paths.artifact_root}/tuning"
    tuning_best_params_path: str = "${paths.tuning_root}/best_params.json"
    mlruns_dir: str = "${paths.output_root}/mlruns"


@dataclass(slots=True)
class ProviderConfig:
    name: RpcProviderName = RpcProviderName.PUBLICNODE
    endpoints: dict[str, str] = field(default_factory=dict)
    references: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    retry_count: int = 5
    backoff_factor: float = 0.125

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

    def sensitive_values(self) -> tuple[str, ...]:
        return tuple(value for value in self.endpoints.values() if value)


@dataclass(slots=True)
class ExperimentConfig:
    task: str = "train"
    max_delay_seconds: int = 36
    lookback_seconds: int = 600
    target_anchor_count: int = 400_000
    chain: ChainConfig = field(default_factory=ChainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pull: PullConfig = field(default_factory=PullConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)


def _task_uses_provider(task: str) -> bool:
    return task == "acquire"


def coerce_config(cfg: DictConfig, *, task: str) -> ExperimentConfig:
    base = OmegaConf.create(config_to_dict(ExperimentConfig(task=task)))
    merged = OmegaConf.merge(base, cfg)
    OmegaConf.resolve(merged)
    payload = OmegaConf.to_container(merged, resolve=True, enum_to_str=True)
    if not isinstance(payload, dict):
        raise TypeError("Hydra configuration did not produce a mapping payload")
    config = _instantiate_experiment(cast(dict[str, Any], payload), task=task)
    validate_config(config)
    return config


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    structured = OmegaConf.structured(cfg)
    return OmegaConf.to_container(structured, resolve=True, enum_to_str=True)  # type: ignore[return-value]


def _instantiate_experiment(payload: dict[str, Any], *, task: str) -> ExperimentConfig:
    chain = payload["chain"]
    dataset = payload["dataset"]
    model = payload["model"]
    pull = payload["pull"]
    split = payload["split"]
    training = payload["training"]
    simulation = payload["simulation"]
    tracking = payload["tracking"]
    tuning = payload["tuning"]
    runtime = payload["runtime"]
    paths = payload["paths"]
    provider = payload["provider"]
    return ExperimentConfig(
        task=str(payload.get("task", task)),
        max_delay_seconds=int(payload["max_delay_seconds"]),
        lookback_seconds=int(payload["lookback_seconds"]),
        target_anchor_count=int(payload["target_anchor_count"]),
        chain=ChainConfig(
            name=ChainName(chain["name"]),
            chain_id=int(chain["chain_id"]),
            block_time_seconds=float(chain["block_time_seconds"]),
            uses_poa_extra_data=bool(chain["uses_poa_extra_data"]),
        ),
        dataset=DatasetConfig(
            id=str(dataset["id"]),
            evaluation_start_timestamp=int(dataset["evaluation_start_timestamp"]),
            evaluation_end_timestamp=int(dataset["evaluation_end_timestamp"]),
            min_history_anchor_count=int(dataset["min_history_anchor_count"]),
        ),
        model=ModelConfig(
            family=ModelFamily(model["family"]),
            input_projection_dim=int(model["input_projection_dim"]),
            hidden_size=int(model["hidden_size"]),
            num_layers=int(model["num_layers"]),
            dropout=float(model["dropout"]),
            d_model=int(model["d_model"]),
            nhead=int(model["nhead"]),
            transformer_layers=int(model["transformer_layers"]),
            feedforward_dim=int(model["feedforward_dim"]),
            head_hidden_dim=int(model["head_hidden_dim"]),
        ),
        pull=PullConfig(
            requests_per_second=int(pull["requests_per_second"]),
            max_concurrent_requests=int(pull["max_concurrent_requests"]),
            max_concurrent_chunks=int(pull["max_concurrent_chunks"]),
            chunk_size=int(pull["chunk_size"]),
            dry_run=bool(pull["dry_run"]),
            overwrite=bool(pull["overwrite"]),
            enrich_batch_size=int(pull["enrich_batch_size"]),
            max_methods_per_second=float(pull["max_methods_per_second"]),
        ),
        split=SplitConfig(
            train_fraction=float(split["train_fraction"]),
            validation_fraction=float(split["validation_fraction"]),
        ),
        training=TrainingConfig(
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            effective_batch_size=int(training["effective_batch_size"]),
            max_epochs=int(training["max_epochs"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            early_stopping_min_delta=float(training["early_stopping_min_delta"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            alpha=float(training["alpha"]),
            beta=float(training["beta"]),
            device=str(training["device"]),
            seed=int(training["seed"]),
            deterministic=bool(training["deterministic"]),
            log_every_n_steps=int(training["log_every_n_steps"]),
        ),
        simulation=SimulationConfig(
            window_seconds=int(simulation["window_seconds"]),
            arrival_rate_per_second=float(simulation["arrival_rate_per_second"]),
            repetitions=int(simulation["repetitions"]),
            seed=int(simulation["seed"]),
        ),
        tracking=TrackingConfig(
            enabled=bool(tracking["enabled"]),
            experiment_name=str(tracking["experiment_name"]),
            tracking_uri=str(tracking["tracking_uri"]),
            tags={str(key): str(value) for key, value in tracking["tags"].items()},
        ),
        tuning=TuningConfig(
            apply_best_params=bool(tuning["apply_best_params"]),
            study_name=str(tuning["study_name"]),
            direction=str(tuning["direction"]),
            n_trials=int(tuning["n_trials"]),
            timeout_seconds=(
                None
                if tuning["timeout_seconds"] is None
                else int(tuning["timeout_seconds"])
            ),
            metric_name=str(tuning["metric_name"]),
            sampler_seed=int(tuning["sampler_seed"]),
            prune=bool(tuning["prune"]),
            search_space={
                str(key): list(value)
                for key, value in tuning["search_space"].items()
            },
        ),
        runtime=RuntimeConfig(
            output_root=str(runtime["output_root"]),
            hydra_run_dir=str(runtime["hydra_run_dir"]),
            hydra_sweep_dir=str(runtime["hydra_sweep_dir"]),
        ),
        paths=PathsConfig(
            output_root=str(paths["output_root"]),
            dataset_root=str(paths["dataset_root"]),
            metadata_root=str(paths["metadata_root"]),
            raw_root=str(paths["raw_root"]),
            raw_history_dir=str(paths["raw_history_dir"]),
            raw_evaluation_dir=str(paths["raw_evaluation_dir"]),
            enriched_root=str(paths["enriched_root"]),
            enriched_history_dir=str(paths["enriched_history_dir"]),
            enriched_evaluation_dir=str(paths["enriched_evaluation_dir"]),
            dataset_metadata_path=str(paths["dataset_metadata_path"]),
            artifact_root=str(paths["artifact_root"]),
            checkpoint_dir=str(paths["checkpoint_dir"]),
            train_report_path=str(paths["train_report_path"]),
            simulation_report_path=str(paths["simulation_report_path"]),
            tuning_root=str(paths["tuning_root"]),
            tuning_best_params_path=str(paths["tuning_best_params_path"]),
            mlruns_dir=str(paths["mlruns_dir"]),
        ),
        provider=ProviderConfig(
            name=RpcProviderName(provider["name"]),
            endpoints={
                str(key): str(value)
                for key, value in provider["endpoints"].items()
            },
            references={
                str(key): str(value)
                for key, value in provider["references"].items()
            },
            timeout_seconds=float(provider["timeout_seconds"]),
            retry_count=int(provider["retry_count"]),
            backoff_factor=float(provider["backoff_factor"]),
        ),
    )


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.max_delay_seconds <= 0:
        raise ValueError("max_delay_seconds must be positive")
    if cfg.lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")
    if cfg.target_anchor_count <= 0:
        raise ValueError("target_anchor_count must be positive")
    if not cfg.dataset.id or "/" in cfg.dataset.id or "\\" in cfg.dataset.id:
        raise ValueError("dataset id must be a non-empty path segment")
    if cfg.dataset.evaluation_start_timestamp >= cfg.dataset.evaluation_end_timestamp:
        raise ValueError("evaluation_start_timestamp must be before evaluation_end_timestamp")
    if cfg.dataset.min_history_anchor_count <= 0:
        raise ValueError("min_history_anchor_count must be positive")
    if cfg.chain.chain_id <= 0:
        raise ValueError("chain_id must be positive")
    if cfg.chain.block_time_seconds <= 0:
        raise ValueError("block_time_seconds must be positive")
    if not 0.0 < cfg.split.train_fraction < 1.0:
        raise ValueError("train_fraction must be greater than 0 and less than 1")
    if not 0.0 <= cfg.split.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be non-negative and less than 1")
    if cfg.split.train_fraction + cfg.split.validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if cfg.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if cfg.training.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if cfg.training.effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    if cfg.training.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if cfg.training.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if cfg.training.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if cfg.training.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if cfg.training.alpha <= 0 or cfg.training.beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if cfg.training.seed < 0:
        raise ValueError("training seed must be non-negative")
    if cfg.pull.requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")
    if cfg.pull.max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be positive")
    if cfg.pull.max_concurrent_chunks <= 0:
        raise ValueError("max_concurrent_chunks must be positive")
    if cfg.pull.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if cfg.pull.enrich_batch_size <= 0:
        raise ValueError("enrich_batch_size must be positive")
    if cfg.pull.max_methods_per_second <= 0:
        raise ValueError("max_methods_per_second must be positive")
    if cfg.simulation.window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if cfg.simulation.arrival_rate_per_second <= 0:
        raise ValueError("arrival_rate_per_second must be positive")
    if cfg.simulation.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if cfg.simulation.seed < 0:
        raise ValueError("simulation seed must be non-negative")
    if not 0.0 <= cfg.model.dropout < 1.0:
        raise ValueError("dropout must be between 0 and 1")
    if cfg.model.input_projection_dim <= 0:
        raise ValueError("input_projection_dim must be positive")
    if cfg.model.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if cfg.model.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if cfg.model.d_model <= 0 or cfg.model.nhead <= 0:
        raise ValueError("d_model and nhead must be positive")
    if cfg.model.d_model % cfg.model.nhead != 0:
        raise ValueError("d_model must be divisible by nhead")
    if cfg.model.d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal positional encodings")
    if cfg.model.transformer_layers <= 0:
        raise ValueError("transformer_layers must be positive")
    if cfg.model.feedforward_dim <= 0:
        raise ValueError("feedforward_dim must be positive")
    if cfg.model.head_hidden_dim <= 0:
        raise ValueError("head_hidden_dim must be positive")
    if _task_uses_provider(cfg.task):
        if cfg.provider.timeout_seconds <= 0:
            raise ValueError("provider timeout_seconds must be positive")
        if cfg.provider.retry_count < 0:
            raise ValueError("provider retry_count must be non-negative")
        if cfg.provider.backoff_factor < 0:
            raise ValueError("provider backoff_factor must be non-negative")
        cfg.provider.endpoint_for(cfg.chain.name)
    if cfg.tuning.n_trials <= 0:
        raise ValueError("tuning n_trials must be positive")
    if cfg.tuning.timeout_seconds is not None and cfg.tuning.timeout_seconds <= 0:
        raise ValueError("tuning timeout_seconds must be positive when set")
