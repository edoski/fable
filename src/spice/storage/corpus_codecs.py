# pyright: strict

"""Strict payload codecs for corpus-root manifests and acquire runs."""

from __future__ import annotations

from dataclasses import asdict

from ..config.models import ChainRuntimeSpec
from ..corpus.metadata import (
    AcquireRunFacts,
    AcquireRunRecord,
    AcquisitionConfigSnapshot,
    BlockRangeMetadata,
    ChainMetadata,
    CompactValidationReport,
    CorpusAcquisitionSourceRequirements,
    CorpusSplitManifest,
    CorpusSplitManifests,
    DatasetAcquisitionRuntimeMetadata,
    DatasetIdentity,
    DatasetManifest,
    ProviderMetadata,
    SplitCoverageMetadata,
    SplitMaterializationMetadata,
    SplitRequestMetadata,
    TimestampRangeMetadata,
)
from .payloads import PayloadModel, decode_payload_model, model_payload


class DatasetIdentityPayload(PayloadModel):
    id: str
    name: str

    @classmethod
    def from_identity(cls, identity: DatasetIdentity) -> DatasetIdentityPayload:
        return cls(id=identity.id, name=identity.name)

    def to_identity(self) -> DatasetIdentity:
        return DatasetIdentity(id=self.id, name=self.name)


class ChainMetadataPayload(PayloadModel):
    name: str
    runtime: dict[str, object]

    @classmethod
    def from_metadata(cls, metadata: ChainMetadata) -> ChainMetadataPayload:
        return cls(
            name=metadata.name,
            runtime=metadata.runtime.model_dump(mode="json"),
        )

    def to_metadata(self) -> ChainMetadata:
        return ChainMetadata(
            name=self.name,
            runtime=ChainRuntimeSpec.model_validate(self.runtime, strict=True),
        )


class SplitRequestPayload(PayloadModel):
    start_timestamp: int
    end_timestamp: int
    start_block: int
    end_block: int

    @classmethod
    def from_request(cls, request: SplitRequestMetadata) -> SplitRequestPayload:
        return cls(
            start_timestamp=request.start_timestamp,
            end_timestamp=request.end_timestamp,
            start_block=request.start_block,
            end_block=request.end_block,
        )

    def to_request(self) -> SplitRequestMetadata:
        return SplitRequestMetadata(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            start_block=self.start_block,
            end_block=self.end_block,
        )


class SplitCoveragePayload(PayloadModel):
    first_timestamp: int | None
    last_timestamp: int | None
    first_block: int | None
    last_block: int | None
    rows: int

    @classmethod
    def from_coverage(cls, coverage: SplitCoverageMetadata) -> SplitCoveragePayload:
        return cls(
            first_timestamp=coverage.first_timestamp,
            last_timestamp=coverage.last_timestamp,
            first_block=coverage.first_block,
            last_block=coverage.last_block,
            rows=coverage.rows,
        )

    def to_coverage(self) -> SplitCoverageMetadata:
        return SplitCoverageMetadata(
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
            first_block=self.first_block,
            last_block=self.last_block,
            rows=self.rows,
        )


class BlockRangePayload(PayloadModel):
    first: int | None
    last: int | None

    @classmethod
    def from_range(cls, block_range: BlockRangeMetadata) -> BlockRangePayload:
        return cls(first=block_range.first, last=block_range.last)

    def to_range(self) -> BlockRangeMetadata:
        return BlockRangeMetadata(first=self.first, last=self.last)


class TimestampRangePayload(PayloadModel):
    first: int | None
    last: int | None

    @classmethod
    def from_range(
        cls,
        timestamp_range: TimestampRangeMetadata,
    ) -> TimestampRangePayload:
        return cls(first=timestamp_range.first, last=timestamp_range.last)

    def to_range(self) -> TimestampRangeMetadata:
        return TimestampRangeMetadata(first=self.first, last=self.last)


class CompactValidationReportPayload(PayloadModel):
    status: str
    rows: int
    block_range: BlockRangePayload
    timestamp_range: TimestampRangePayload
    issues: dict[str, object] | None = None

    @classmethod
    def from_report(
        cls,
        report: CompactValidationReport,
    ) -> CompactValidationReportPayload:
        return cls(
            status=report.status,
            rows=report.rows,
            block_range=BlockRangePayload.from_range(report.block_range),
            timestamp_range=TimestampRangePayload.from_range(report.timestamp_range),
            issues=report.issues,
        )

    def to_report(self) -> CompactValidationReport:
        return CompactValidationReport(
            status=self.status,
            rows=self.rows,
            block_range=self.block_range.to_range(),
            timestamp_range=self.timestamp_range.to_range(),
            issues=self.issues,
        )


class SplitMaterializationPayload(PayloadModel):
    outcome: str
    file_count: int

    @classmethod
    def from_materialization(
        cls,
        materialization: SplitMaterializationMetadata,
    ) -> SplitMaterializationPayload:
        return cls(outcome=materialization.outcome, file_count=materialization.file_count)

    def to_materialization(self) -> SplitMaterializationMetadata:
        return SplitMaterializationMetadata(
            outcome=self.outcome,
            file_count=self.file_count,
        )


class CorpusSplitManifestPayload(PayloadModel):
    kind: str
    request: SplitRequestPayload
    coverage: SplitCoveragePayload
    validation: CompactValidationReportPayload
    materialization: SplitMaterializationPayload

    @classmethod
    def from_split(cls, split: CorpusSplitManifest) -> CorpusSplitManifestPayload:
        return cls(
            kind=split.kind,
            request=SplitRequestPayload.from_request(split.request),
            coverage=SplitCoveragePayload.from_coverage(split.coverage),
            validation=CompactValidationReportPayload.from_report(split.validation),
            materialization=SplitMaterializationPayload.from_materialization(
                split.materialization
            ),
        )

    def to_split(self) -> CorpusSplitManifest:
        return CorpusSplitManifest(
            kind=self.kind,
            request=self.request.to_request(),
            coverage=self.coverage.to_coverage(),
            validation=self.validation.to_report(),
            materialization=self.materialization.to_materialization(),
        )


class CorpusSplitManifestsPayload(PayloadModel):
    history: CorpusSplitManifestPayload
    evaluation: CorpusSplitManifestPayload

    @classmethod
    def from_splits(cls, splits: CorpusSplitManifests) -> CorpusSplitManifestsPayload:
        return cls(
            history=CorpusSplitManifestPayload.from_split(splits.history),
            evaluation=CorpusSplitManifestPayload.from_split(splits.evaluation),
        )

    def to_splits(self) -> CorpusSplitManifests:
        return CorpusSplitManifests(
            history=self.history.to_split(),
            evaluation=self.evaluation.to_split(),
        )


class CorpusAcquisitionSourceRequirementsPayload(PayloadModel):
    required_columns: list[str]
    optional_enrichments: list[str]
    temporal_unit: str
    ordering_key: str
    partition_key: str | None

    @classmethod
    def from_requirements(
        cls,
        requirements: CorpusAcquisitionSourceRequirements,
    ) -> CorpusAcquisitionSourceRequirementsPayload:
        return cls(
            required_columns=sorted(requirements.required_columns),
            optional_enrichments=sorted(requirements.optional_enrichments),
            temporal_unit=requirements.temporal_unit,
            ordering_key=requirements.ordering_key,
            partition_key=requirements.partition_key,
        )

    def to_requirements(self) -> CorpusAcquisitionSourceRequirements:
        return CorpusAcquisitionSourceRequirements(
            required_columns=frozenset(self.required_columns),
            optional_enrichments=frozenset(self.optional_enrichments),
            temporal_unit=self.temporal_unit,
            ordering_key=self.ordering_key,
            partition_key=self.partition_key,
        )


class DatasetManifestPayload(PayloadModel):
    dataset: DatasetIdentityPayload
    chain: ChainMetadataPayload
    splits: CorpusSplitManifestsPayload
    source_requirements: CorpusAcquisitionSourceRequirementsPayload

    @classmethod
    def from_manifest(cls, manifest: DatasetManifest) -> DatasetManifestPayload:
        return cls(
            dataset=DatasetIdentityPayload.from_identity(manifest.dataset),
            chain=ChainMetadataPayload.from_metadata(manifest.chain),
            splits=CorpusSplitManifestsPayload.from_splits(manifest.splits),
            source_requirements=(
                CorpusAcquisitionSourceRequirementsPayload.from_requirements(
                    manifest.source_requirements
                )
            ),
        )

    def to_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            dataset=self.dataset.to_identity(),
            chain=self.chain.to_metadata(),
            splits=self.splits.to_splits(),
            source_requirements=self.source_requirements.to_requirements(),
        )


class ProviderPayload(PayloadModel):
    name: str
    reference: str
    endpoint_fingerprint: str

    @classmethod
    def from_provider(cls, provider: ProviderMetadata) -> ProviderPayload:
        return cls(
            name=provider.name,
            reference=provider.reference,
            endpoint_fingerprint=provider.endpoint_fingerprint,
        )

    def to_provider(self) -> ProviderMetadata:
        return ProviderMetadata(
            name=self.name,
            reference=self.reference,
            endpoint_fingerprint=self.endpoint_fingerprint,
        )


class AcquireRunFactsPayload(PayloadModel):
    requested_history_window_seconds: int
    resolved_capability_samples: int

    @classmethod
    def from_facts(cls, facts: AcquireRunFacts) -> AcquireRunFactsPayload:
        return cls(
            requested_history_window_seconds=facts.requested_history_window_seconds,
            resolved_capability_samples=facts.resolved_capability_samples,
        )

    def to_facts(self) -> AcquireRunFacts:
        return AcquireRunFacts(
            requested_history_window_seconds=self.requested_history_window_seconds,
            resolved_capability_samples=self.resolved_capability_samples,
        )


class AcquisitionConfigSnapshotPayload(PayloadModel):
    chunk_size: int
    rpc_batch_size: int
    rpc_concurrency: int
    rpc_min_batch_size: int
    rpc_concurrency_rungs: list[int]

    @classmethod
    def from_settings(
        cls,
        settings: AcquisitionConfigSnapshot,
    ) -> AcquisitionConfigSnapshotPayload:
        return cls(**asdict(settings))

    def to_settings(self) -> AcquisitionConfigSnapshot:
        return AcquisitionConfigSnapshot(
            chunk_size=self.chunk_size,
            rpc_batch_size=self.rpc_batch_size,
            rpc_concurrency=self.rpc_concurrency,
            rpc_min_batch_size=self.rpc_min_batch_size,
            rpc_concurrency_rungs=list(self.rpc_concurrency_rungs),
        )


class DatasetAcquisitionRuntimePayload(PayloadModel):
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

    @classmethod
    def from_runtime(
        cls,
        runtime: DatasetAcquisitionRuntimeMetadata,
    ) -> DatasetAcquisitionRuntimePayload:
        return cls(**asdict(runtime))

    def to_runtime(self) -> DatasetAcquisitionRuntimeMetadata:
        return DatasetAcquisitionRuntimeMetadata(
            configured_batch_size=self.configured_batch_size,
            final_batch_size=self.final_batch_size,
            min_batch_size=self.min_batch_size,
            configured_concurrency=self.configured_concurrency,
            final_concurrency=self.final_concurrency,
            concurrency_rungs=list(self.concurrency_rungs),
            oversize_error_count=self.oversize_error_count,
            transient_error_count=self.transient_error_count,
            oversize_backoffs=self.oversize_backoffs,
            transient_backoffs=self.transient_backoffs,
            concurrency_recoveries=self.concurrency_recoveries,
        )


class AcquireRunPayload(PayloadModel):
    provider: ProviderPayload
    facts: AcquireRunFactsPayload
    settings: AcquisitionConfigSnapshotPayload
    runtime: DatasetAcquisitionRuntimePayload

    @classmethod
    def from_record(cls, record: AcquireRunRecord) -> AcquireRunPayload:
        return cls(
            provider=ProviderPayload.from_provider(record.provider),
            facts=AcquireRunFactsPayload.from_facts(record.facts),
            settings=AcquisitionConfigSnapshotPayload.from_settings(record.settings),
            runtime=DatasetAcquisitionRuntimePayload.from_runtime(record.runtime),
        )

    def to_record(self) -> AcquireRunRecord:
        return AcquireRunRecord(
            provider=self.provider.to_provider(),
            facts=self.facts.to_facts(),
            settings=self.settings.to_settings(),
            runtime=self.runtime.to_runtime(),
        )


def dataset_manifest_payload(manifest: DatasetManifest) -> dict[str, object]:
    return model_payload(
        DatasetManifestPayload.from_manifest(manifest),
        label="dataset manifest",
    )


def dataset_manifest_from_payload(payload: dict[str, object]) -> DatasetManifest:
    return decode_payload_model(
        "dataset manifest",
        DatasetManifestPayload,
        payload,
        lambda model: model.to_manifest(),
    )


def acquire_run_payload(run: AcquireRunRecord) -> dict[str, object]:
    return model_payload(AcquireRunPayload.from_record(run), label="acquire run")


def acquire_run_from_payload(payload: dict[str, object]) -> AcquireRunRecord:
    return decode_payload_model(
        "acquire run",
        AcquireRunPayload,
        payload,
        lambda model: model.to_record(),
    )
