"""Supported high-level Python API for SPICE workflows."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .acquisition.cryo import (
    CryoCommandPlan,
    TimestampRange,
    build_pull_plan,
    evaluation_range,
    history_range_for_chain,
    run_cryo,
)
from .acquisition.enrich import enrich_path
from .acquisition.provenance import (
    source_manifest_path_for,
    update_source_manifest_for_promotion,
    write_enrichment_manifest,
    write_source_manifest,
)
from .acquisition.raw_validation import RawPullValidationReport, validate_raw_pull
from .acquisition.rpc import JsonRpcClient
from .acquisition.rpc_providers import (
    RpcProvider,
    RpcProviderName,
    resolve_acquisition_providers,
    resolve_rpc_provider,
)
from .core.config import (
    BlockSegment,
    ChainConfig,
    ChainName,
    ExperimentConfig,
    ModelConfig,
    ModelFamily,
)
from .core.console import NullReporter, Reporter
from .core.constants import SIMULATION_REPORT_FILENAME, TRAIN_REPORT_FILENAME
from .data.datasets import derive_dataset_geometry
from .data.io import load_enriched_block_frame
from .modeling.artifacts import (
    LoadedTrainingArtifact,
    build_training_artifact_manifest,
    load_training_artifact,
    write_training_artifact,
)
from .modeling.inference import predict_class_offsets
from .modeling.pipeline import TrainingSpec, prepare_inference_dataset, run_training
from .modeling.reporting import (
    SimulationReport,
    TrainingRunReport,
    build_simulation_report,
    build_training_run_report,
    write_json_report,
)
from .modeling.simulation import run_temporal_simulation

__all__ = [
    "ExperimentConfig",
    "SimulationReport",
    "TrainingRunReport",
    "acquire_blocks",
    "build_training_spec",
    "enrich_blocks",
    "load_artifact",
    "load_config",
    "plan_block_pulls",
    "promote_blocks",
    "pull_blocks",
    "run_simulation_workflow",
    "run_training_workflow",
    "stage_blocks",
    "validate_blocks",
]


@dataclass(slots=True)
class BlockPullResult:
    output_dir: Path
    validation: RawPullValidationReport | None
    source_manifest_path: Path | None
    command: str
    completed_chunks: int
    expected_chunks: int | None


@dataclass(slots=True)
class BlockPromotionResult:
    source_dir: Path
    output_dir: Path
    validation: RawPullValidationReport
    source_manifest_path: Path


@dataclass(slots=True)
class BlockAcquireResult:
    raw: BlockPullResult
    enriched_output_dir: Path
    enriched_source_manifest_path: Path
    enriched_file_count: int


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.load(path)


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
    chain = config.resolve_chain(chain_name)
    model = ModelConfig(family=ModelFamily(family))
    training = config.training if device is None else config.training.model_copy(
        update={"device": device}
    )
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
    config: ExperimentConfig,
    history_block_path: Path,
    artifact_dir: Path,
    chain_name: ChainName | str,
    family: ModelFamily | str,
    max_delay_seconds: int,
    *,
    device: str | None = None,
    reporter: Reporter | None = None,
) -> TrainingRunReport:
    reporter = reporter or NullReporter()
    spec = build_training_spec(
        config,
        chain_name=chain_name,
        family=family,
        max_delay_seconds=max_delay_seconds,
        device=device,
    )
    result = run_training(history_block_path, spec=spec, reporter=reporter)
    manifest = build_training_artifact_manifest(result.prepared, spec=spec)
    write_training_artifact(artifact_dir, manifest=manifest, model=result.model)
    report = build_training_run_report(
        result,
        target_anchor_count=spec.target_anchor_count,
        max_delay_seconds=spec.max_delay_seconds,
        lookback_seconds=spec.lookback_seconds,
        chain_name=spec.chain.name.value,
        family=spec.model.family.value,
        block_time_seconds=spec.chain.block_time_seconds,
        manifest=manifest,
        prepared=result.prepared,
        artifact_dir=artifact_dir,
        history_block_path=history_block_path,
        device_requested=spec.training.device,
    )
    write_json_report(artifact_dir / TRAIN_REPORT_FILENAME, report)
    return report


def run_simulation_workflow(
    config: ExperimentConfig,
    artifact_dir: Path,
    history_block_path: Path,
    evaluation_block_path: Path,
    *,
    device: str | None = None,
) -> SimulationReport:
    loaded_artifact = load_training_artifact(artifact_dir)
    geometry = derive_dataset_geometry(
        lookback_seconds=loaded_artifact.manifest.lookback_seconds,
        max_delay_seconds=loaded_artifact.manifest.max_delay_seconds,
        block_time_seconds=loaded_artifact.manifest.chain.block_time_seconds,
    )
    history_blocks = load_enriched_block_frame(history_block_path)
    evaluation_blocks = load_enriched_block_frame(evaluation_block_path)
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
    simulation = run_temporal_simulation(
        prepared.store,
        predicted_offsets,
        sample_indices=prepared.sample_indices,
        window_seconds=config.simulation.window_seconds,
        arrival_rate_per_second=config.simulation.arrival_rate_per_second,
        repetitions=config.simulation.repetitions,
        seed=config.simulation.seed,
    )
    report = build_simulation_report(
        loaded_artifact,
        artifact_dir=artifact_dir,
        history_block_path=history_block_path,
        evaluation_block_path=evaluation_block_path,
        prepared=prepared,
        simulation=simulation,
        window_seconds=config.simulation.window_seconds,
        arrival_rate_per_second=config.simulation.arrival_rate_per_second,
        repetitions=config.simulation.repetitions,
    )
    write_json_report(artifact_dir / SIMULATION_REPORT_FILENAME, report)
    return report


def plan_block_pulls(
    config: ExperimentConfig,
    *,
    rpc_provider: RpcProviderName | None = None,
) -> list[CryoCommandPlan]:
    provider = resolve_rpc_provider(rpc_provider, chains=(chain.name for chain in config.chains))
    return build_pull_plan(config, provider=provider)


def enrich_blocks(
    config: ExperimentConfig,
    chain_name: ChainName | str,
    input_path: Path,
    output_path: Path,
    *,
    rpc_provider: RpcProviderName | None = None,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
    reporter: Reporter | None = None,
) -> list[Path]:
    chain = config.resolve_chain(chain_name)
    provider = resolve_rpc_provider(rpc_provider, chains=(chain.name,))
    return _enrich_blocks_to_path(
        chain=chain,
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        batch_size=batch_size,
        max_methods_per_second=max_methods_per_second,
        reporter=reporter,
    )


def _staging_output_root(output_root: Path, provider_name: str) -> Path:
    return output_root.parent / "staging" / provider_name


def _resolve_dataset_output_dir(
    output_root: Path,
    *,
    dataset_kind: str,
    chain: ChainConfig,
    segment: BlockSegment,
) -> Path:
    return output_root / dataset_kind / chain.name.value / segment.value


def _resolve_pull_target(
    config: ExperimentConfig,
    chain: ChainConfig,
    segment: BlockSegment,
    *,
    output_root: Path,
) -> tuple[Path, TimestampRange]:
    timestamps = (
        history_range_for_chain(chain)
        if segment is BlockSegment.HISTORY
        else evaluation_range()
    )
    return output_root / "raw" / chain.name.value / segment.value, timestamps


def _validate_dataset_path(
    dataset_path: Path,
    *,
    chain: ChainConfig,
    timestamps: TimestampRange,
) -> RawPullValidationReport:
    return validate_raw_pull(
        dataset_path,
        expected_chain_name=chain.name.value,
        expected_chain_id=chain.chain_id,
        expected_start_timestamp=timestamps.start,
        expected_end_timestamp=timestamps.end,
    )


def _execute_block_pull(
    *,
    config: ExperimentConfig,
    config_path: Path | None,
    chain: ChainConfig,
    segment: BlockSegment,
    provider: RpcProvider,
    output_root: Path,
    dry_run: bool,
    overwrite: bool,
    validate_on_success: bool,
    reporter: Reporter | None = None,
) -> BlockPullResult:
    output_dir, timestamps = _resolve_pull_target(config, chain, segment, output_root=output_root)
    cryo_result = run_cryo(
        chain,
        config.pull,
        output_dir,
        timestamps,
        provider=provider,
        overwrite=overwrite,
        dry_run=dry_run,
        reporter=reporter,
    )
    validation = None
    if validate_on_success and not dry_run:
        validation = _validate_dataset_path(output_dir, chain=chain, timestamps=timestamps)
    source_manifest_path = None
    if not dry_run:
        source_manifest_path = write_source_manifest(
            output_dir,
            config_path=config_path,
            chain=chain,
            segment=segment,
            timestamps=timestamps,
            provider=provider,
            pull=config.pull,
            overwrite=overwrite,
            validation=validation,
        )
    return BlockPullResult(
        output_dir=output_dir,
        validation=validation,
        source_manifest_path=source_manifest_path,
        command=cryo_result.command,
        completed_chunks=cryo_result.completed_chunks,
        expected_chunks=cryo_result.expected_chunks,
    )


def _enrich_blocks_to_path(
    *,
    chain: ChainConfig,
    input_path: Path,
    output_path: Path,
    provider: RpcProvider,
    batch_size: int,
    max_methods_per_second: float,
    reporter: Reporter | None = None,
) -> list[Path]:
    with JsonRpcClient(provider.url_for(chain.name)) as client:
        return enrich_path(
            input_path,
            output_path,
            fetch_gas_limits=client.get_block_gas_limits,
            batch_size=batch_size,
            max_methods_per_second=max_methods_per_second,
            reporter=reporter,
        )


def pull_blocks(
    config: ExperimentConfig,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
    *,
    rpc_provider: RpcProviderName | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = False,
    reporter: Reporter | None = None,
    config_path: Path | None = None,
) -> BlockPullResult:
    chain = config.resolve_chain(chain_name)
    segment_name = BlockSegment(segment)
    if dry_run and validate_on_success:
        raise ValueError("Cannot use validate_on_success with dry-run pulls")
    provider = resolve_rpc_provider(rpc_provider, chains=(chain.name,))
    return _execute_block_pull(
        config=config,
        config_path=config_path,
        chain=chain,
        segment=segment_name,
        provider=provider,
        output_root=config.output_root,
        dry_run=dry_run,
        overwrite=overwrite,
        validate_on_success=validate_on_success,
        reporter=reporter,
    )


def stage_blocks(
    config: ExperimentConfig,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
    *,
    rpc_provider: RpcProviderName | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = False,
    reporter: Reporter | None = None,
    config_path: Path | None = None,
) -> BlockPullResult:
    chain = config.resolve_chain(chain_name)
    segment_name = BlockSegment(segment)
    if dry_run and validate_on_success:
        raise ValueError("Cannot use validate_on_success with dry-run pulls")
    provider = resolve_rpc_provider(rpc_provider, chains=(chain.name,))
    return _execute_block_pull(
        config=config,
        config_path=config_path,
        chain=chain,
        segment=segment_name,
        provider=provider,
        output_root=_staging_output_root(config.output_root, provider.name.value),
        dry_run=dry_run,
        overwrite=overwrite,
        validate_on_success=validate_on_success,
        reporter=reporter,
    )


def acquire_blocks(
    config: ExperimentConfig,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
    *,
    rpc_provider: RpcProviderName | None = None,
    pull_rpc_provider: RpcProviderName | None = None,
    enrich_rpc_provider: RpcProviderName | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = True,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
    reporter: Reporter | None = None,
    config_path: Path | None = None,
) -> BlockAcquireResult:
    chain = config.resolve_chain(chain_name)
    segment_name = BlockSegment(segment)
    if dry_run and validate_on_success:
        raise ValueError("Cannot use validate_on_success with dry-run pulls")
    providers = resolve_acquisition_providers(
        rpc_provider,
        pull_provider_name=pull_rpc_provider,
        enrich_provider_name=enrich_rpc_provider,
        chains=(chain.name,),
    )
    staging_root = _staging_output_root(config.output_root, providers.pull.name.value)
    raw_result = _execute_block_pull(
        config=config,
        config_path=config_path,
        chain=chain,
        segment=segment_name,
        provider=providers.pull,
        output_root=staging_root,
        dry_run=dry_run,
        overwrite=overwrite,
        validate_on_success=validate_on_success,
        reporter=reporter,
    )
    if raw_result.validation is not None and raw_result.validation.status == "error":
        raise ValueError("Cannot enrich dataset with raw validation errors")
    enriched_output_dir = _resolve_dataset_output_dir(
        staging_root,
        dataset_kind="enriched",
        chain=chain,
        segment=segment_name,
    )
    enriched_source_manifest_path = source_manifest_path_for(enriched_output_dir)
    enriched_file_count = 0
    if not dry_run:
        written = _enrich_blocks_to_path(
            chain=chain,
            input_path=raw_result.output_dir,
            output_path=enriched_output_dir,
            provider=providers.enrich,
            batch_size=batch_size,
            max_methods_per_second=max_methods_per_second,
            reporter=reporter,
        )
        enriched_file_count = len(written)
        enriched_source_manifest_path = write_enrichment_manifest(
            enriched_output_dir,
            config_path=config_path,
            input_path=raw_result.output_dir,
            chain=chain,
            segment=segment_name,
            provider=providers.enrich,
            batch_size=batch_size,
            max_methods_per_second=max_methods_per_second,
        )
    return BlockAcquireResult(
        raw=raw_result,
        enriched_output_dir=enriched_output_dir,
        enriched_source_manifest_path=enriched_source_manifest_path,
        enriched_file_count=enriched_file_count,
    )


def validate_blocks(
    config: ExperimentConfig,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
) -> RawPullValidationReport:
    chain = config.resolve_chain(chain_name)
    segment_name = BlockSegment(segment)
    output_dir, timestamps = _resolve_pull_target(
        config,
        chain,
        segment_name,
        output_root=config.output_root,
    )
    return _validate_dataset_path(output_dir, chain=chain, timestamps=timestamps)


def promote_blocks(
    config: ExperimentConfig,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
    source_dir: Path,
    *,
    overwrite: bool = False,
    allow_warnings: bool = False,
) -> BlockPromotionResult:
    chain = config.resolve_chain(chain_name)
    segment_name = BlockSegment(segment)
    output_dir, timestamps = _resolve_pull_target(
        config,
        chain,
        segment_name,
        output_root=config.output_root,
    )
    source_dir = source_dir.resolve()
    destination_dir = output_dir.resolve()
    validation = _validate_dataset_path(source_dir, chain=chain, timestamps=timestamps)
    if validation.status == "error":
        raise ValueError("Cannot promote a raw dataset with validation errors")
    if validation.status == "warning" and not allow_warnings:
        raise ValueError("Cannot promote a raw dataset with validation warnings")
    if destination_dir.exists():
        if not overwrite:
            raise ValueError(f"Destination already exists: {destination_dir}")
        shutil.rmtree(destination_dir)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(destination_dir))
    source_manifest_path = update_source_manifest_for_promotion(
        destination_dir,
        promoted_from=source_dir,
        validation=validation,
    )
    return BlockPromotionResult(
        source_dir=source_dir,
        output_dir=destination_dir,
        validation=validation,
        source_manifest_path=source_manifest_path,
    )
