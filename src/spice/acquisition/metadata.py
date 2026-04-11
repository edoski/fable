"""Typed dataset metadata helpers for acquisition workflows."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..core.config import ExperimentConfig
from ..data.io import iter_block_files
from ..data.validation import BlockDatasetValidationReport
from .raw_validation import RawPullValidationReport


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


class DatasetPathPair(MetadataModel):
    history: str
    evaluation: str


class DatasetPathsMetadata(MetadataModel):
    raw: DatasetPathPair
    enriched: DatasetPathPair


class DatasetWindowMetadata(MetadataModel):
    start_timestamp: int
    end_timestamp: int


class DatasetWindowsMetadata(MetadataModel):
    history: DatasetWindowMetadata
    evaluation: DatasetWindowMetadata


class DatasetSamplingSettings(MetadataModel):
    anchor_count: int
    history_anchor_count: int


class DatasetTemporalSettings(MetadataModel):
    lookback_seconds: int
    max_delay_seconds: int


class DatasetAcquisitionSettings(MetadataModel):
    chunk_size: int
    enrich_batch_size: int
    max_methods_per_second: float


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
    files: int | None = None
    issues: dict[str, object] | None = None


class DatasetValidationSection(MetadataModel):
    history: CompactValidationReport
    evaluation: CompactValidationReport


class DatasetValidationMetadata(MetadataModel):
    raw: DatasetValidationSection
    enriched: DatasetValidationSection


class DatasetMetadata(MetadataModel):
    dataset: DatasetIdentity
    chain: ChainMetadata
    provider: ProviderMetadata
    paths: DatasetPathsMetadata
    windows: DatasetWindowsMetadata
    settings: DatasetSettingsMetadata
    validation: DatasetValidationMetadata


def has_block_files(path: Path) -> bool:
    try:
        return bool(iter_block_files(path))
    except ValueError:
        return False


def load_dataset_metadata(path: Path) -> DatasetMetadata | None:
    if not path.is_file():
        return None
    return DatasetMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def provider_metadata(config: ExperimentConfig) -> ProviderMetadata:
    endpoint = config.provider.endpoint_for(config.chain.name)
    return ProviderMetadata(
        name=config.provider.name.value,
        reference=config.provider.reference_for(config.chain.name),
        endpoint_fingerprint=sha256(endpoint.encode("utf-8")).hexdigest()[:16],
    )


def check_existing_dataset_metadata(
    *,
    config: ExperimentConfig,
    metadata_path: Path,
    overwrite: bool,
) -> DatasetMetadata | None:
    metadata = load_dataset_metadata(metadata_path)
    if metadata is None:
        if not overwrite and _metadata_has_dataset_files(config):
            raise ValueError(
                f"Dataset files exist without canonical metadata at {metadata_path}; "
                "rerun with acquisition.overwrite=true to replace them."
            )
        return None

    if overwrite:
        return metadata

    expected = {
        "dataset_id": config.dataset.id,
        "chain_name": config.chain.name.value,
        "chain_id": config.chain.chain_id,
        "provider": provider_metadata(config).model_dump(mode="json"),
        "evaluation_window": {
            "start_timestamp": config.dataset.window.start_timestamp,
            "end_timestamp": config.dataset.window.end_timestamp,
        },
    }
    actual = {
        "dataset_id": metadata.dataset.id,
        "chain_name": metadata.chain.name,
        "chain_id": metadata.chain.chain_id,
        "provider": metadata.provider.model_dump(mode="json"),
        "evaluation_window": metadata.windows.evaluation.model_dump(mode="json"),
    }
    if actual != expected:
        raise ValueError(
            "Existing dataset metadata does not match the requested dataset window/provider. "
            f"Expected {expected}, got {actual}. Use acquisition.overwrite=true to replace it."
        )
    return metadata


def compact_validation_report(
    report: RawPullValidationReport | BlockDatasetValidationReport,
) -> CompactValidationReport:
    issue_counts: dict[str, int] = {}
    for name in (
        "gap_count",
        "overlap_count",
        "duplicate_count",
        "chain_id_mismatch_count",
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
        files=report.file_count if isinstance(report, RawPullValidationReport) else None,
        issues=issues,
    )


def build_dataset_metadata(
    *,
    config: ExperimentConfig,
    raw_history_dir: Path,
    raw_evaluation_dir: Path,
    enriched_history_dir: Path,
    enriched_evaluation_dir: Path,
    history_window_start: int,
    history_window_end: int,
    evaluation_window_start: int,
    evaluation_window_end: int,
    history_validation: RawPullValidationReport,
    evaluation_validation: RawPullValidationReport,
    history_enriched: BlockDatasetValidationReport,
    evaluation_enriched: BlockDatasetValidationReport,
) -> DatasetMetadata:
    return DatasetMetadata(
        dataset=DatasetIdentity(id=config.dataset.id),
        chain=ChainMetadata(
            name=config.chain.name.value,
            chain_id=config.chain.chain_id,
        ),
        provider=provider_metadata(config),
        paths=DatasetPathsMetadata(
            raw=DatasetPathPair(
                history=raw_history_dir.as_posix(),
                evaluation=raw_evaluation_dir.as_posix(),
            ),
            enriched=DatasetPathPair(
                history=enriched_history_dir.as_posix(),
                evaluation=enriched_evaluation_dir.as_posix(),
            ),
        ),
        windows=DatasetWindowsMetadata(
            history=DatasetWindowMetadata(
                start_timestamp=history_window_start,
                end_timestamp=history_window_end,
            ),
            evaluation=DatasetWindowMetadata(
                start_timestamp=evaluation_window_start,
                end_timestamp=evaluation_window_end,
            ),
        ),
        settings=DatasetSettingsMetadata(
            sampling=DatasetSamplingSettings(
                anchor_count=config.dataset.sampling.anchor_count,
                history_anchor_count=config.dataset.sampling.effective_history_anchor_count,
            ),
            temporal=DatasetTemporalSettings(
                lookback_seconds=config.dataset.temporal.lookback_seconds,
                max_delay_seconds=config.dataset.temporal.max_delay_seconds,
            ),
            acquisition=DatasetAcquisitionSettings(
                chunk_size=config.acquisition.chunk_size,
                enrich_batch_size=config.acquisition.enrich_batch_size,
                max_methods_per_second=config.acquisition.max_methods_per_second,
            ),
        ),
        validation=DatasetValidationMetadata(
            raw=DatasetValidationSection(
                history=compact_validation_report(history_validation),
                evaluation=compact_validation_report(evaluation_validation),
            ),
            enriched=DatasetValidationSection(
                history=compact_validation_report(history_enriched),
                evaluation=compact_validation_report(evaluation_enriched),
            ),
        ),
    )


def _metadata_has_dataset_files(config: ExperimentConfig) -> bool:
    for candidate in (
        Path(config.paths.raw_history_dir),
        Path(config.paths.raw_evaluation_dir),
        Path(config.paths.enriched_history_dir),
        Path(config.paths.enriched_evaluation_dir),
    ):
        if has_block_files(candidate):
            return True
    return False
