"""Typed dataset metadata helpers for acquisition workflows."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from ..core.config import ExperimentConfig
from ..data.io import iter_block_files
from ..data.validation import BlockDatasetValidationReport
from .rpc import AcquisitionRuntimeSnapshot


class MetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetIdentity(MetadataModel):
    id: str


class ChainMetadata(MetadataModel):
    name: str
    chain_id: int


class ProviderMetadata(MetadataModel):
    name: str
    reference: str
    endpoint_fingerprint: str


class DatasetPathsMetadata(MetadataModel):
    history: str
    evaluation: str


class DatasetWindowMetadata(MetadataModel):
    start_timestamp: int
    end_timestamp: int


class DatasetRequestMetadata(MetadataModel):
    history: DatasetWindowMetadata
    evaluation: DatasetWindowMetadata


class DatasetCoverageMetadata(MetadataModel):
    history: DatasetWindowMetadata
    evaluation: DatasetWindowMetadata


class DatasetSamplingSettings(MetadataModel):
    sample_count: int


class DatasetTemporalSettings(MetadataModel):
    lookback_seconds: int
    max_delay_seconds: int


class DatasetAcquisitionSettings(MetadataModel):
    history_sample_budget: int
    chunk_size: int
    rpc_batch_size: int
    rpc_concurrency: int
    rpc_min_batch_size: int
    rpc_concurrency_rungs: list[int]


class DatasetSettingsMetadata(MetadataModel):
    sampling: DatasetSamplingSettings
    temporal: DatasetTemporalSettings
    acquisition: DatasetAcquisitionSettings


class BlockRangeMetadata(MetadataModel):
    first: int | None
    last: int | None


class TimestampRangeMetadata(MetadataModel):
    first: int | None
    last: int | None


class CompactValidationReport(MetadataModel):
    status: str
    rows: int
    block_range: BlockRangeMetadata
    timestamp_range: TimestampRangeMetadata
    issues: dict[str, object] | None = None


class DatasetValidationMetadata(MetadataModel):
    history: CompactValidationReport
    evaluation: CompactValidationReport


class DatasetAcquisitionRuntimeMetadata(MetadataModel):
    configured_batch_size: int
    final_batch_size: int
    min_batch_size: int
    configured_concurrency: int
    final_concurrency: int
    concurrency_rungs: list[int]
    oversize_error_count: int
    transient_error_count: int
    oversize_backoffs: int
    transient_backoffs: int
    concurrency_recoveries: int


class DatasetRuntimeMetadata(MetadataModel):
    acquisition: DatasetAcquisitionRuntimeMetadata


class DatasetMetadata(MetadataModel):
    dataset: DatasetIdentity
    chain: ChainMetadata
    providers: list[ProviderMetadata]
    paths: DatasetPathsMetadata
    request: DatasetRequestMetadata
    coverage: DatasetCoverageMetadata
    settings: DatasetSettingsMetadata
    validation: DatasetValidationMetadata
    runtime: DatasetRuntimeMetadata


def has_block_files(path: Path) -> bool:
    try:
        return bool(iter_block_files(path))
    except ValueError:
        return False


def load_dataset_metadata(path: Path) -> DatasetMetadata | None:
    if not path.is_file():
        return None
    try:
        return DatasetMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        return None


def provider_metadata(config: ExperimentConfig) -> ProviderMetadata:
    endpoint = config.provider.endpoint_for(config.chain.name)
    return ProviderMetadata(
        name=config.provider.name.value,
        reference=config.provider.reference_for(config.chain.name),
        endpoint_fingerprint=sha256(endpoint.encode("utf-8")).hexdigest()[:16],
    )


def merge_providers(
    existing: list[ProviderMetadata] | None,
    current: ProviderMetadata,
) -> list[ProviderMetadata]:
    providers = list(existing or [])
    if current not in providers:
        providers.append(current)
    return providers


def compact_validation_report(report: BlockDatasetValidationReport) -> CompactValidationReport:
    issue_counts: dict[str, int] = {}
    for name in (
        "gap_count",
        "duplicate_count",
        "below_start_count",
        "above_end_count",
    ):
        value = getattr(report, name, 0)
        if value:
            issue_counts[name.removesuffix("_count")] = int(value)

    issues: dict[str, object] | None = None
    if issue_counts or report.errors:
        issues = {
            **issue_counts,
            **({"errors": report.errors} if report.errors else {}),
        }
    return CompactValidationReport(
        status=report.status,
        rows=report.row_count,
        block_range=BlockRangeMetadata(
            first=report.first_block_number,
            last=report.last_block_number,
        ),
        timestamp_range=TimestampRangeMetadata(
            first=report.first_timestamp,
            last=report.last_timestamp,
        ),
        issues=issues,
    )


def _coverage_window(report: BlockDatasetValidationReport) -> DatasetWindowMetadata:
    if report.first_timestamp is None or report.last_timestamp is None:
        raise ValueError("Clean dataset validation is required before building coverage metadata")
    return DatasetWindowMetadata(
        start_timestamp=report.first_timestamp,
        end_timestamp=report.last_timestamp,
    )


def build_dataset_metadata(
    *,
    config: ExperimentConfig,
    history_dir: Path,
    evaluation_dir: Path,
    history_request_start_timestamp: int,
    history_request_end_timestamp: int,
    evaluation_request_start_timestamp: int,
    evaluation_request_end_timestamp: int,
    providers: list[ProviderMetadata],
    history_validation: BlockDatasetValidationReport,
    evaluation_validation: BlockDatasetValidationReport,
    acquisition_runtime: AcquisitionRuntimeSnapshot,
) -> DatasetMetadata:
    return DatasetMetadata(
        dataset=DatasetIdentity(id=config.dataset.id),
        chain=ChainMetadata(
            name=config.chain.name.value,
            chain_id=config.chain.chain_id,
        ),
        providers=list(providers),
        paths=DatasetPathsMetadata(
            history=history_dir.as_posix(),
            evaluation=evaluation_dir.as_posix(),
        ),
        request=DatasetRequestMetadata(
            history=DatasetWindowMetadata(
                start_timestamp=history_request_start_timestamp,
                end_timestamp=history_request_end_timestamp,
            ),
            evaluation=DatasetWindowMetadata(
                start_timestamp=evaluation_request_start_timestamp,
                end_timestamp=evaluation_request_end_timestamp,
            ),
        ),
        coverage=DatasetCoverageMetadata(
            history=_coverage_window(history_validation),
            evaluation=_coverage_window(evaluation_validation),
        ),
        settings=DatasetSettingsMetadata(
            sampling=DatasetSamplingSettings(
                sample_count=config.dataset.sampling.sample_count,
            ),
            temporal=DatasetTemporalSettings(
                lookback_seconds=config.dataset.temporal.lookback_seconds,
                max_delay_seconds=config.dataset.temporal.max_delay_seconds,
            ),
            acquisition=DatasetAcquisitionSettings(
                history_sample_budget=config.effective_history_sample_budget,
                chunk_size=config.acquisition.chunk_size,
                rpc_batch_size=config.acquisition.rpc_batch_size,
                rpc_concurrency=config.acquisition.rpc_concurrency,
                rpc_min_batch_size=config.acquisition.rpc_min_batch_size,
                rpc_concurrency_rungs=list(config.acquisition.rpc_concurrency_rungs),
            ),
        ),
        validation=DatasetValidationMetadata(
            history=compact_validation_report(history_validation),
            evaluation=compact_validation_report(evaluation_validation),
        ),
        runtime=DatasetRuntimeMetadata(
            acquisition=DatasetAcquisitionRuntimeMetadata(
                configured_batch_size=acquisition_runtime.configured_batch_size,
                final_batch_size=acquisition_runtime.final_batch_size,
                min_batch_size=acquisition_runtime.min_batch_size,
                configured_concurrency=acquisition_runtime.configured_concurrency,
                final_concurrency=acquisition_runtime.final_concurrency,
                concurrency_rungs=list(acquisition_runtime.concurrency_rungs),
                oversize_error_count=acquisition_runtime.oversize_error_count,
                transient_error_count=acquisition_runtime.transient_error_count,
                oversize_backoffs=acquisition_runtime.oversize_backoffs,
                transient_backoffs=acquisition_runtime.transient_backoffs,
                concurrency_recoveries=acquisition_runtime.concurrency_recoveries,
            )
        ),
    )
