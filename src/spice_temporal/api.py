"""Supported high-level Python API for the SPICE temporal baseline."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from subprocess import CompletedProcess

from spice_temporal._rpc import JsonRpcClient
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
from spice_temporal.env import load_project_env
from spice_temporal.inference import predict_class_offsets
from spice_temporal.io import load_block_records
from spice_temporal.pipeline import prepare_inference_dataset, run_training
from spice_temporal.provenance import (
    source_manifest_path_for,
    update_source_manifest_for_promotion,
    write_enrichment_manifest,
    write_source_manifest,
)
from spice_temporal.raw_validation import RawPullValidationReport, validate_raw_pull
from spice_temporal.reporting import (
    SimulationReport,
    TrainingRunReport,
    build_simulation_report,
    build_training_run_report,
    write_json_report,
)
from spice_temporal.rpc_providers import (
    RpcProvider,
    RpcProviderName,
    resolve_acquisition_providers,
    resolve_rpc_provider,
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
    source_manifest_path: Path | None


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


def _plan_block_pulls(
    config_or_path: ExperimentConfig | Path,
    *,
    rpc_provider: RpcProviderName | None = None,
) -> list[CryoCommandPlan]:
    config = _coerce_config(config_or_path)
    load_project_env()
    provider = resolve_rpc_provider(rpc_provider, chains=(chain.name for chain in config.chains))
    return build_pull_plan(config, provider=provider)


def _enrich_blocks(
    config_or_path: ExperimentConfig | Path,
    chain_name: ChainName | str,
    input_path: Path,
    output_path: Path,
    *,
    rpc_provider: RpcProviderName | None = None,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> list[Path]:
    config = _coerce_config(config_or_path)
    chain = _require_chain(config, chain_name)
    load_project_env()
    provider = resolve_rpc_provider(rpc_provider, chains=(chain.name,))
    return _enrich_blocks_to_path(
        chain=chain,
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        batch_size=batch_size,
        max_methods_per_second=max_methods_per_second,
    )


def _acquire_blocks(
    config_or_path: ExperimentConfig | Path,
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
) -> BlockAcquireResult:
    config_path = config_or_path if isinstance(config_or_path, Path) else None
    config = _coerce_config(config_or_path)
    chain = _require_chain(config, chain_name)
    segment_name = BlockSegment(segment)
    if dry_run and validate_on_success:
        raise ValueError("Cannot use validate_on_success with dry-run pulls")

    load_project_env()
    providers = resolve_acquisition_providers(
        rpc_provider,
        pull_provider_name=pull_rpc_provider,
        enrich_provider_name=enrich_rpc_provider,
        chains=(chain.name,),
    )
    staging_root = _staging_output_root(config.output_root, providers.pull.name)
    raw_result = _execute_block_pull(
        config_path=config_path,
        config=config,
        chain=chain,
        segment=segment_name,
        provider=providers.pull,
        output_root=staging_root,
        dry_run=dry_run,
        overwrite=overwrite,
        validate_on_success=validate_on_success,
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


def _pull_blocks(
    config_or_path: ExperimentConfig | Path,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
    *,
    rpc_provider: RpcProviderName | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = False,
) -> BlockPullResult:
    config_path = config_or_path if isinstance(config_or_path, Path) else None
    config = _coerce_config(config_or_path)
    chain = _require_chain(config, chain_name)
    segment_name = BlockSegment(segment)
    if dry_run and validate_on_success:
        raise ValueError("Cannot use validate_on_success with dry-run pulls")

    load_project_env()
    provider = resolve_rpc_provider(rpc_provider, chains=(chain.name,))
    return _execute_block_pull(
        config_path=config_path,
        config=config,
        chain=chain,
        segment=segment_name,
        provider=provider,
        output_root=config.output_root,
        dry_run=dry_run,
        overwrite=overwrite,
        validate_on_success=validate_on_success,
    )


def _stage_block_pull(
    config_or_path: ExperimentConfig | Path,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
    *,
    rpc_provider: RpcProviderName | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = False,
) -> BlockPullResult:
    config_path = config_or_path if isinstance(config_or_path, Path) else None
    config = _coerce_config(config_or_path)
    chain = _require_chain(config, chain_name)
    segment_name = BlockSegment(segment)
    if dry_run and validate_on_success:
        raise ValueError("Cannot use validate_on_success with dry-run pulls")

    load_project_env()
    provider = resolve_rpc_provider(rpc_provider, chains=(chain.name,))
    return _execute_block_pull(
        config_path=config_path,
        config=config,
        chain=chain,
        segment=segment_name,
        provider=provider,
        output_root=_staging_output_root(config.output_root, provider.name),
        dry_run=dry_run,
        overwrite=overwrite,
        validate_on_success=validate_on_success,
    )


def _validate_block_pull(
    config_or_path: ExperimentConfig | Path,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
) -> RawPullValidationReport:
    config = _coerce_config(config_or_path)
    chain = _require_chain(config, chain_name)
    segment_name = BlockSegment(segment)
    output_dir, timestamps = _resolve_pull_target(config, chain, segment_name)
    return _validate_dataset_path(output_dir, chain=chain, timestamps=timestamps)


def _promote_block_pull(
    config_or_path: ExperimentConfig | Path,
    chain_name: ChainName | str,
    segment: BlockSegment | str,
    source_dir: Path,
    *,
    overwrite: bool = False,
    allow_warnings: bool = False,
) -> BlockPromotionResult:
    config = _coerce_config(config_or_path)
    chain = _require_chain(config, chain_name)
    segment_name = BlockSegment(segment)
    output_dir, timestamps = _resolve_pull_target(config, chain, segment_name)
    source_dir = source_dir.resolve()
    destination_dir = output_dir.resolve()

    if not source_dir.is_dir():
        raise ValueError(f"Source dataset path does not exist or is not a directory: {source_dir}")
    if source_dir == destination_dir:
        raise ValueError("Source dataset path must differ from the canonical baseline path")

    validation = _validate_dataset_path(source_dir, chain=chain, timestamps=timestamps)
    if validation.status == "error":
        raise ValueError("Cannot promote dataset with validation errors")
    if validation.status == "warning" and not allow_warnings:
        raise ValueError("Cannot promote dataset with validation warnings unless allowed")

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = destination_dir.with_name(f"{destination_dir.name}.backup")
    if backup_dir.exists():
        raise ValueError(f"Backup path already exists: {backup_dir}")
    if destination_dir.exists():
        if not overwrite:
            raise ValueError(
                "Canonical baseline dataset already exists; rerun with overwrite to replace it"
            )
        destination_dir.rename(backup_dir)

    try:
        shutil.move(str(source_dir), str(destination_dir))
        source_manifest_path = update_source_manifest_for_promotion(
            destination_dir,
            promoted_from=source_dir,
            validation=validation,
        )
    except Exception:
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        if backup_dir.exists():
            backup_dir.rename(destination_dir)
        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    return BlockPromotionResult(
        source_dir=source_dir,
        output_dir=destination_dir,
        validation=validation,
        source_manifest_path=source_manifest_path,
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
    *,
    output_root: Path | None = None,
) -> tuple[Path, TimestampRange]:
    root = config.output_root if output_root is None else output_root
    output_dir = _resolve_dataset_output_dir(root, dataset_kind="raw", chain=chain, segment=segment)
    timestamps = (
        history_range_for_chain(chain)
        if segment is BlockSegment.HISTORY
        else evaluation_range()
    )
    return output_dir, timestamps


def _staging_output_root(output_root: Path, provider_name: str) -> Path:
    return output_root.parent / "staging" / provider_name


def _resolve_dataset_output_dir(
    output_root: Path,
    *,
    dataset_kind: str,
    chain: ChainConfig,
    segment: BlockSegment,
) -> Path:
    return output_root / dataset_kind / chain.name / segment.value


def _execute_block_pull(
    *,
    config_path: Path | None,
    config: ExperimentConfig,
    chain: ChainConfig,
    segment: BlockSegment,
    provider: RpcProvider,
    output_root: Path,
    dry_run: bool,
    overwrite: bool,
    validate_on_success: bool,
) -> BlockPullResult:
    output_dir, timestamps = _resolve_pull_target(
        config,
        chain,
        segment,
        output_root=output_root,
    )
    process = run_cryo(
        chain,
        config.pull,
        output_dir,
        timestamps,
        provider=provider,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    validation = None
    if validate_on_success:
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
        process=process,
        validation=validation,
        source_manifest_path=source_manifest_path,
    )


def _enrich_blocks_to_path(
    *,
    chain: ChainConfig,
    input_path: Path,
    output_path: Path,
    provider: RpcProvider,
    batch_size: int,
    max_methods_per_second: float,
) -> list[Path]:
    client = JsonRpcClient(provider.url_for(chain.name))
    return enrich_path(
        input_path,
        output_path,
        fetch_gas_limits=client.get_block_gas_limits,
        batch_size=batch_size,
        max_methods_per_second=max_methods_per_second,
    )


def _validate_dataset_path(
    dataset_path: Path,
    *,
    chain: ChainConfig,
    timestamps: TimestampRange,
) -> RawPullValidationReport:
    return validate_raw_pull(
        dataset_path,
        expected_chain_name=chain.name,
        expected_chain_id=chain.chain_id,
        expected_start_timestamp=timestamps.start,
        expected_end_timestamp=timestamps.end,
    )
