"""Supported high-level Python API for the SPICE temporal baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from subprocess import CompletedProcess

from spice_temporal._alchemy_rpc import AlchemyRpcClient
from spice_temporal.artifacts import (
    SIMULATION_REPORT_FILENAME,
    TRAIN_REPORT_FILENAME,
    LoadedTrainingArtifact,
    build_training_artifact_manifest,
    load_training_artifact,
    write_training_artifact,
)
from spice_temporal.config import (
    BlockSegment,
    ChainConfig,
    ChainName,
    ExperimentConfig,
    ModelConfig,
    ModelFamily,
)
from spice_temporal.cryo import (
    CryoCommandPlan,
    TimestampRange,
    build_pull_plan,
    evaluation_range,
    history_range_for_chain,
    run_cryo,
)
from spice_temporal.datasets import derive_dataset_geometry
from spice_temporal.enrich import enrich_path
from spice_temporal.env import load_project_env, resolve_rpc_url
from spice_temporal.inference import predict_class_offsets
from spice_temporal.io import load_block_records
from spice_temporal.pipeline import prepare_inference_dataset, run_training
from spice_temporal.raw_validation import RawPullValidationReport, validate_raw_pull
from spice_temporal.reporting import (
    SimulationReport,
    TrainingRunReport,
    build_simulation_report,
    build_training_run_report,
    write_json_report,
)
from spice_temporal.simulation import run_temporal_simulation
from spice_temporal.specs import SimulationSpec, TrainingSpec

__all__ = [
    "SimulationSpec",
    "TrainingSpec",
    "load_artifact",
    "load_config",
    "run_simulation_workflow",
    "run_training_workflow",
]


@dataclass(slots=True)
class BlockPullResult:
    output_dir: Path
    process: CompletedProcess[str]
    validation: RawPullValidationReport | None


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path)


def load_artifact(artifact_dir: Path) -> LoadedTrainingArtifact:
    return load_training_artifact(artifact_dir)


def build_training_spec(
    config: ExperimentConfig,
    *,
    chain_name: ChainName | str,
    family: ModelFamily | str,
    max_delay_seconds: int,
    device: str | None = None,
) -> TrainingSpec:
    chain = _require_chain(config, chain_name)
    model = ModelConfig(family=ModelFamily(family))
    training = config.training if device is None else replace(config.training, device=device)
    return TrainingSpec(
        chain=chain,
        model=model,
        max_delay_seconds=max_delay_seconds,
        lookback_seconds=config.lookback_seconds,
        target_anchor_count=config.target_anchor_count,
        split=config.split,
        training=training,
    )


def run_training_workflow(
    config_or_path: ExperimentConfig | Path,
    history_block_path: Path,
    artifact_dir: Path,
    chain_name: ChainName | str,
    family: ModelFamily | str,
    max_delay_seconds: int,
    *,
    device: str | None = None,
) -> TrainingRunReport:
    config = _coerce_config(config_or_path)
    spec = build_training_spec(
        config,
        chain_name=chain_name,
        family=family,
        max_delay_seconds=max_delay_seconds,
        device=device,
    )
    result = run_training(history_block_path, spec=spec)
    manifest = build_training_artifact_manifest(result.prepared, spec=spec)
    write_training_artifact(artifact_dir, manifest=manifest, model=result.model)
    report = build_training_run_report(
        result,
        spec=spec,
        manifest=manifest,
        prepared=result.prepared,
        artifact_dir=artifact_dir,
        history_block_path=history_block_path,
        device_requested=spec.training.device,
    )
    write_json_report(artifact_dir / TRAIN_REPORT_FILENAME, report)
    return report


def run_simulation_workflow(
    config_or_path: ExperimentConfig | Path,
    artifact_dir: Path,
    history_block_path: Path,
    evaluation_block_path: Path,
    *,
    device: str | None = None,
) -> SimulationReport:
    config = _coerce_config(config_or_path)
    loaded_artifact = load_training_artifact(artifact_dir)
    geometry = derive_dataset_geometry(
        lookback_seconds=loaded_artifact.manifest.lookback_seconds,
        max_delay_seconds=loaded_artifact.manifest.max_delay_seconds,
        block_time_seconds=loaded_artifact.manifest.chain.block_time_seconds,
    )
    history_blocks = load_block_records(history_block_path)
    evaluation_blocks = load_block_records(evaluation_block_path)
    prepared = prepare_inference_dataset(
        history_blocks,
        evaluation_blocks,
        geometry=geometry,
        scaler=loaded_artifact.manifest.scaler,
    )
    device_name = config.training.device if device is None else device
    predicted_offsets = predict_class_offsets(
        loaded_artifact.model,
        store=prepared.store,
        sample_indices=prepared.sample_indices,
        lookback_steps=prepared.geometry.lookback_steps,
        effective_batch_size=config.training.effective_batch_size,
        device=device_name,
    )
    simulation_spec = SimulationSpec.from_config(config.simulation)
    simulation = run_temporal_simulation(
        prepared.store,
        predicted_offsets,
        sample_indices=prepared.sample_indices,
        window_seconds=simulation_spec.window_seconds,
        arrival_rate_per_second=simulation_spec.arrival_rate_per_second,
        repetitions=simulation_spec.repetitions,
        seed=simulation_spec.seed,
    )
    report = build_simulation_report(
        loaded_artifact,
        artifact_dir=artifact_dir,
        history_block_path=history_block_path,
        evaluation_block_path=evaluation_block_path,
        prepared=prepared,
        simulation=simulation,
        spec=simulation_spec,
    )
    write_json_report(artifact_dir / SIMULATION_REPORT_FILENAME, report)
    return report


def _plan_block_pulls(config_or_path: ExperimentConfig | Path) -> list[CryoCommandPlan]:
    config = _coerce_config(config_or_path)
    return build_pull_plan(config)


def _enrich_blocks(
    config_or_path: ExperimentConfig | Path,
    chain_name: ChainName | str,
    input_path: Path,
    output_path: Path,
    *,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> list[Path]:
    config = _coerce_config(config_or_path)
    chain = _require_chain(config, chain_name)
    load_project_env()
    client = AlchemyRpcClient(resolve_rpc_url(chain.name))
    return enrich_path(
        input_path,
        output_path,
        fetch_gas_limits=client.get_block_gas_limits,
        batch_size=batch_size,
        max_methods_per_second=max_methods_per_second,
    )


def _pull_blocks(
    config_or_path: ExperimentConfig | Path,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
    *,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = False,
) -> BlockPullResult:
    config = _coerce_config(config_or_path)
    chain = _require_chain(config, chain_name)
    segment_name = BlockSegment(segment)
    if dry_run and validate_on_success:
        raise ValueError("Cannot use validate_on_success with dry-run pulls")

    load_project_env()
    output_dir, timestamps = _resolve_pull_target(config, chain, segment_name)
    process = run_cryo(
        chain,
        config.pull,
        output_dir,
        timestamps,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    validation = None
    if validate_on_success:
        validation = _validate_block_pull(config, chain.name, segment_name)
    return BlockPullResult(output_dir=output_dir, process=process, validation=validation)


def _validate_block_pull(
    config_or_path: ExperimentConfig | Path,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
) -> RawPullValidationReport:
    config = _coerce_config(config_or_path)
    chain = _require_chain(config, chain_name)
    segment_name = BlockSegment(segment)
    output_dir, timestamps = _resolve_pull_target(config, chain, segment_name)
    return validate_raw_pull(
        output_dir,
        expected_chain_name=chain.name,
        expected_chain_id=chain.chain_id,
        expected_start_timestamp=timestamps.start,
        expected_end_timestamp=timestamps.end,
    )


def _coerce_config(config_or_path: ExperimentConfig | Path) -> ExperimentConfig:
    if isinstance(config_or_path, ExperimentConfig):
        return config_or_path
    return load_config(config_or_path)


def _require_chain(config: ExperimentConfig, chain_name: ChainName | str) -> ChainConfig:
    resolved_name = ChainName(chain_name)
    chain = next((item for item in config.chains if item.name is resolved_name), None)
    if chain is None:
        raise ValueError(f"Unknown chain: {resolved_name}")
    return chain


def _resolve_pull_target(
    config: ExperimentConfig,
    chain: ChainConfig,
    segment: BlockSegment,
) -> tuple[Path, TimestampRange]:
    output_dir = config.output_root / "raw" / chain.name / segment.value
    timestamps = (
        history_range_for_chain(chain)
        if segment is BlockSegment.HISTORY
        else evaluation_range()
    )
    return output_dir, timestamps
