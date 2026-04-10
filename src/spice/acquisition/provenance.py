"""Dataset-level provenance manifests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ..core.config import BlockSegment, ChainConfig, PullConfig
from ..core.constants import SOURCE_MANIFEST_DIRNAME, SOURCE_MANIFEST_FILENAME
from .cryo import TimestampRange, build_cryo_command
from .raw_validation import RawPullValidationReport
from .rpc_providers import RpcProvider


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationSummary(ManifestModel):
    status: str
    file_count: int
    row_count: int
    first_block_number: int | None
    last_block_number: int | None
    first_timestamp: int | None
    last_timestamp: int | None
    gap_count: int
    overlap_count: int
    duplicate_count: int
    chain_id_mismatch_count: int
    below_start_count: int
    above_end_count: int
    warnings: list[str]
    errors: list[str]


class TimestampWindow(ManifestModel):
    start: int
    end: int


class PromotionSummary(ManifestModel):
    promoted_at_utc: str
    promoted_from: str


class RawSourceManifest(ManifestModel):
    kind: Literal["raw_block_dataset_source"] = "raw_block_dataset_source"
    written_at_utc: str
    config_path: str | None
    output_dir: str
    chain: str
    chain_id: int
    segment: str
    provider: str
    provider_reference: str
    expected_timestamp_range: TimestampWindow
    overwrite: bool
    command: str
    validation: ValidationSummary | None = None
    promotion: PromotionSummary | None = None


class EnrichedSourceManifest(ManifestModel):
    kind: Literal["enriched_block_dataset_source"] = "enriched_block_dataset_source"
    written_at_utc: str
    config_path: str | None
    output_dir: str
    input_path: str
    input_source_manifest_path: str | None
    chain: str
    chain_id: int
    segment: str
    provider: str
    provider_reference: str
    batch_size: int
    max_methods_per_second: float


def source_manifest_path_for(dataset_dir: Path) -> Path:
    return dataset_dir / SOURCE_MANIFEST_DIRNAME / SOURCE_MANIFEST_FILENAME


def _write_manifest(path: Path, manifest: ManifestModel) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def _now_utc_isoformat() -> str:
    return datetime.now(UTC).isoformat()


def _serialize_validation(report: RawPullValidationReport | None) -> ValidationSummary | None:
    if report is None:
        return None
    return ValidationSummary(
        status=report.status,
        file_count=report.file_count,
        row_count=report.row_count,
        first_block_number=report.first_block_number,
        last_block_number=report.last_block_number,
        first_timestamp=report.first_timestamp,
        last_timestamp=report.last_timestamp,
        gap_count=report.gap_count,
        overlap_count=report.overlap_count,
        duplicate_count=report.duplicate_count,
        chain_id_mismatch_count=report.chain_id_mismatch_count,
        below_start_count=report.below_start_count,
        above_end_count=report.above_end_count,
        warnings=report.warnings,
        errors=report.errors,
    )


def load_source_manifest(dataset_dir: Path) -> RawSourceManifest | EnrichedSourceManifest | None:
    manifest_path = source_manifest_path_for(dataset_dir)
    if not manifest_path.is_file():
        return None
    raw = manifest_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid source manifest JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid source manifest payload: {manifest_path}")

    kind = payload.get("kind")
    try:
        if kind == "raw_block_dataset_source":
            return RawSourceManifest.model_validate(payload)
        if kind == "enriched_block_dataset_source":
            return EnrichedSourceManifest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid source manifest: {manifest_path}") from exc
    raise ValueError(f"Unsupported source manifest kind in {manifest_path}: {kind!r}")


def write_source_manifest(
    output_dir: Path,
    *,
    config_path: Path | None,
    chain: ChainConfig,
    segment: BlockSegment,
    timestamps: TimestampRange,
    provider: RpcProvider,
    pull: PullConfig,
    overwrite: bool,
    validation: RawPullValidationReport | None,
) -> Path:
    manifest = RawSourceManifest(
        written_at_utc=_now_utc_isoformat(),
        config_path=str(config_path.resolve()) if config_path is not None else None,
        output_dir=str(output_dir.resolve()),
        chain=chain.name.value,
        chain_id=chain.chain_id,
        segment=segment.value,
        provider=provider.name.value,
        provider_reference=provider.reference_for(chain.name),
        expected_timestamp_range=TimestampWindow(start=timestamps.start, end=timestamps.end),
        overwrite=overwrite,
        command=build_cryo_command(
            chain,
            pull,
            output_dir,
            timestamps,
            provider=provider,
            overwrite=overwrite,
        ),
        validation=_serialize_validation(validation),
    )
    return _write_manifest(source_manifest_path_for(output_dir), manifest)


def write_enrichment_manifest(
    output_dir: Path,
    *,
    config_path: Path | None,
    input_path: Path,
    chain: ChainConfig,
    segment: BlockSegment,
    provider: RpcProvider,
    batch_size: int,
    max_methods_per_second: float,
) -> Path:
    input_manifest_path = source_manifest_path_for(input_path)
    manifest = EnrichedSourceManifest(
        written_at_utc=_now_utc_isoformat(),
        config_path=str(config_path.resolve()) if config_path is not None else None,
        output_dir=str(output_dir.resolve()),
        input_path=str(input_path.resolve()),
        input_source_manifest_path=(
            str(input_manifest_path.resolve()) if input_manifest_path.is_file() else None
        ),
        chain=chain.name.value,
        chain_id=chain.chain_id,
        segment=segment.value,
        provider=provider.name.value,
        provider_reference=provider.reference_for(chain.name),
        batch_size=batch_size,
        max_methods_per_second=max_methods_per_second,
    )
    return _write_manifest(source_manifest_path_for(output_dir), manifest)


def update_source_manifest_for_promotion(
    output_dir: Path,
    *,
    promoted_from: Path,
    validation: RawPullValidationReport,
) -> Path:
    loaded = load_source_manifest(output_dir)
    if not isinstance(loaded, RawSourceManifest):
        raise ValueError(
            "Raw dataset promotion requires an existing raw source manifest at "
            f"{source_manifest_path_for(output_dir)}"
        )
    manifest = loaded.model_copy(
        update={
            "written_at_utc": _now_utc_isoformat(),
            "output_dir": str(output_dir.resolve()),
            "validation": _serialize_validation(validation),
            "promotion": PromotionSummary(
                promoted_at_utc=_now_utc_isoformat(),
                promoted_from=str(promoted_from.resolve()),
            ),
        }
    )
    return _write_manifest(source_manifest_path_for(output_dir), manifest)
