"""Dataset-level provenance manifests for acquired block datasets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spice_temporal.config import BlockSegment, ChainConfig, PullConfig
from spice_temporal.cryo import TimestampRange, build_cryo_command
from spice_temporal.raw_validation import RawPullValidationReport
from spice_temporal.rpc_providers import RpcProvider

SOURCE_MANIFEST_FILENAME = "source.json"
SOURCE_MANIFEST_DIRNAME = ".spice"


def source_manifest_path_for(dataset_dir: Path) -> Path:
    return dataset_dir / SOURCE_MANIFEST_DIRNAME / SOURCE_MANIFEST_FILENAME


def _serialize_validation(report: RawPullValidationReport | None) -> dict[str, object] | None:
    if report is None:
        return None
    return {
        "status": report.status,
        "file_count": report.file_count,
        "row_count": report.row_count,
        "first_block_number": report.first_block_number,
        "last_block_number": report.last_block_number,
        "first_timestamp": report.first_timestamp,
        "last_timestamp": report.last_timestamp,
        "gap_count": report.gap_count,
        "overlap_count": report.overlap_count,
        "duplicate_count": report.duplicate_count,
        "chain_id_mismatch_count": report.chain_id_mismatch_count,
        "below_start_count": report.below_start_count,
        "above_end_count": report.above_end_count,
        "warnings": report.warnings,
        "errors": report.errors,
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _now_utc_isoformat() -> str:
    return datetime.now(UTC).isoformat()


def load_source_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    manifest_path = source_manifest_path_for(dataset_dir)
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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
    manifest_path = source_manifest_path_for(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_cryo_command(
        chain,
        pull,
        output_dir,
        timestamps,
        provider=provider,
        overwrite=overwrite,
    )
    payload = {
        "kind": "raw_block_dataset_source",
        "written_at_utc": _now_utc_isoformat(),
        "config_path": str(config_path.resolve()) if config_path is not None else None,
        "output_dir": str(output_dir.resolve()),
        "chain": chain.name,
        "chain_id": chain.chain_id,
        "segment": segment.value,
        "provider": provider.name,
        "provider_reference": provider.reference_for(chain.name),
        "expected_timestamp_range": {
            "start": timestamps.start,
            "end": timestamps.end,
        },
        "overwrite": overwrite,
        "command": command,
        "validation": _serialize_validation(validation),
    }
    return _write_manifest(manifest_path, payload)


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
    manifest_path = source_manifest_path_for(output_dir)
    input_manifest_path = source_manifest_path_for(input_path)
    payload = {
        "kind": "enriched_block_dataset_source",
        "written_at_utc": _now_utc_isoformat(),
        "config_path": str(config_path.resolve()) if config_path is not None else None,
        "output_dir": str(output_dir.resolve()),
        "input_path": str(input_path.resolve()),
        "input_source_manifest_path": (
            str(input_manifest_path.resolve()) if input_manifest_path.is_file() else None
        ),
        "chain": chain.name,
        "chain_id": chain.chain_id,
        "segment": segment.value,
        "provider": provider.name,
        "provider_reference": provider.reference_for(chain.name),
        "batch_size": batch_size,
        "max_methods_per_second": max_methods_per_second,
    }
    return _write_manifest(manifest_path, payload)


def update_source_manifest_for_promotion(
    output_dir: Path,
    *,
    promoted_from: Path,
    validation: RawPullValidationReport,
) -> Path:
    manifest_path = source_manifest_path_for(output_dir)
    payload = load_source_manifest(output_dir) or {"kind": "raw_block_dataset_source"}
    payload["written_at_utc"] = _now_utc_isoformat()
    payload["output_dir"] = str(output_dir.resolve())
    payload["validation"] = _serialize_validation(validation)
    payload["promotion"] = {
        "promoted_at_utc": _now_utc_isoformat(),
        "promoted_from": str(promoted_from.resolve()),
    }
    return _write_manifest(manifest_path, payload)
