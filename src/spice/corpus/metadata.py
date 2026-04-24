# pyright: strict

"""Typed acquisition metadata builders."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ..acquisition.rpc import AcquisitionRuntimeSnapshot
from ..config.models import AcquireConfig
from ..corpus.io import iter_block_files
from ..corpus.validation import BlockDatasetValidationReport
from ..storage.layout import resolve_workflow_paths


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ChainMetadata:
    name: str
    chain_id: int


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    name: str
    reference: str
    endpoint_fingerprint: str


@dataclass(frozen=True, slots=True)
class DatasetWindowMetadata:
    start_timestamp: int
    end_timestamp: int


@dataclass(frozen=True, slots=True)
class DatasetRequestMetadata:
    history: DatasetWindowMetadata
    evaluation: DatasetWindowMetadata


@dataclass(frozen=True, slots=True)
class DatasetCoverageMetadata:
    history: DatasetWindowMetadata
    evaluation: DatasetWindowMetadata


@dataclass(frozen=True, slots=True)
class BlockRangeMetadata:
    first: int | None
    last: int | None


@dataclass(frozen=True, slots=True)
class TimestampRangeMetadata:
    first: int | None
    last: int | None


@dataclass(frozen=True, slots=True)
class CompactValidationReport:
    status: str
    rows: int
    block_range: BlockRangeMetadata
    timestamp_range: TimestampRangeMetadata
    issues: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class DatasetValidationMetadata:
    history: CompactValidationReport
    evaluation: CompactValidationReport


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset: DatasetIdentity
    chain: ChainMetadata
    request: DatasetRequestMetadata
    coverage: DatasetCoverageMetadata
    validation: DatasetValidationMetadata


@dataclass(frozen=True, slots=True)
class AcquisitionConfigSnapshot:
    chunk_size: int
    rpc_batch_size: int
    rpc_concurrency: int
    rpc_min_batch_size: int
    rpc_concurrency_rungs: list[int]


@dataclass(frozen=True, slots=True)
class AcquireRunFacts:
    requested_history_window_seconds: int
    resolved_capability_samples: int


@dataclass(frozen=True, slots=True)
class DatasetAcquisitionRuntimeMetadata:
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


@dataclass(frozen=True, slots=True)
class AcquireRunRecord:
    provider: ProviderMetadata
    settings: AcquisitionConfigSnapshot
    runtime: DatasetAcquisitionRuntimeMetadata
    facts: AcquireRunFacts


def has_block_files(path: Path) -> bool:
    try:
        return bool(iter_block_files(path))
    except ValueError:
        return False


def provider_metadata(config: AcquireConfig) -> ProviderMetadata:
    return ProviderMetadata(
        name=config.rpc_endpoint.provider_name,
        reference=config.rpc_endpoint.reference,
        endpoint_fingerprint=sha256(config.rpc_endpoint.url.encode("utf-8")).hexdigest()[:16],
    )


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


def acquisition_settings(config: AcquireConfig) -> AcquisitionConfigSnapshot:
    return AcquisitionConfigSnapshot(
        chunk_size=config.acquisition.chunk_size,
        rpc_batch_size=config.acquisition.rpc.batch_size,
        rpc_concurrency=config.acquisition.rpc.concurrency,
        rpc_min_batch_size=config.acquisition.rpc.min_batch_size,
        rpc_concurrency_rungs=list(config.acquisition.rpc.concurrency_rungs),
    )


def acquisition_runtime_metadata(
    runtime: AcquisitionRuntimeSnapshot,
) -> DatasetAcquisitionRuntimeMetadata:
    return DatasetAcquisitionRuntimeMetadata(
        configured_batch_size=runtime.configured_batch_size,
        final_batch_size=runtime.final_batch_size,
        min_batch_size=runtime.min_batch_size,
        configured_concurrency=runtime.configured_concurrency,
        final_concurrency=runtime.final_concurrency,
        concurrency_rungs=list(runtime.concurrency_rungs),
        oversize_error_count=runtime.oversize_error_count,
        transient_error_count=runtime.transient_error_count,
        oversize_backoffs=runtime.oversize_backoffs,
        transient_backoffs=runtime.transient_backoffs,
        concurrency_recoveries=runtime.concurrency_recoveries,
    )


def build_dataset_manifest(
    *,
    config: AcquireConfig,
    history_request_start_timestamp: int,
    history_request_end_timestamp: int,
    evaluation_request_start_timestamp: int,
    evaluation_request_end_timestamp: int,
    history_validation: BlockDatasetValidationReport,
    evaluation_validation: BlockDatasetValidationReport,
) -> DatasetManifest:
    paths = resolve_workflow_paths(config)
    return DatasetManifest(
        dataset=DatasetIdentity(
            id=paths.corpus_id,
            name=config.dataset.name,
        ),
        chain=ChainMetadata(
            name=config.chain.name,
            chain_id=config.chain.runtime.chain_id,
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
        validation=DatasetValidationMetadata(
            history=compact_validation_report(history_validation),
            evaluation=compact_validation_report(evaluation_validation),
        ),
    )


def build_acquire_run_record(
    *,
    config: AcquireConfig,
    provider: ProviderMetadata,
    acquisition_runtime: AcquisitionRuntimeSnapshot,
    requested_history_window_seconds: int,
    resolved_capability_samples: int,
) -> AcquireRunRecord:
    return AcquireRunRecord(
        provider=provider,
        settings=acquisition_settings(config),
        runtime=acquisition_runtime_metadata(acquisition_runtime),
        facts=AcquireRunFacts(
            requested_history_window_seconds=requested_history_window_seconds,
            resolved_capability_samples=resolved_capability_samples,
        ),
    )
