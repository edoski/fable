# pyright: strict

"""Corpus-root SQLite persistence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from ..core.errors import MissingStateError
from ..corpus.metadata import (
    AcquireRunFacts,
    AcquireRunRecord,
    AcquisitionConfigSnapshot,
    BlockRangeMetadata,
    ChainMetadata,
    CompactValidationReport,
    DatasetAcquisitionRuntimeMetadata,
    DatasetCoverageMetadata,
    DatasetIdentity,
    DatasetManifest,
    DatasetRequestMetadata,
    DatasetValidationMetadata,
    DatasetWindowMetadata,
    ProviderMetadata,
    TimestampRangeMetadata,
)
from ..modeling.result_codecs import corpus_semantics_from_payload, corpus_semantics_payload
from .engine import DATASET_ROOT_KIND, create_state_engine, ensure_state_db, touch_meta
from .payloads import PayloadCodec, SequencePayloadStore, SingletonPayloadStore, mapping_payload
from .schema import DATASET_TABLES, acquire_runs, dataset_manifest

_DATASET_MANIFEST_STORE = SingletonPayloadStore(
    table=dataset_manifest,
    codec=PayloadCodec(
        encode=lambda value: _dataset_manifest_payload(value),
        decode=lambda payload: _dataset_manifest_from_payload(payload),
    ),
)
_ACQUIRE_RUN_STORE = SequencePayloadStore(
    table=acquire_runs,
    codec=PayloadCodec(
        encode=lambda value: _acquire_run_payload(value),
        decode=lambda payload: _acquire_run_from_payload(payload),
    ),
)


def write_dataset_state(
    db_path: Path,
    *,
    manifest: DatasetManifest,
    acquire_run: AcquireRunRecord,
) -> None:
    """Persist the dataset manifest once and append one acquire-run delta record."""

    ensure_state_db(db_path, root_kind=DATASET_ROOT_KIND, tables=DATASET_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            _DATASET_MANIFEST_STORE.upsert(conn, manifest)
            _ACQUIRE_RUN_STORE.append(conn, acquire_run, recorded_at=_now_timestamp())
            touch_meta(conn, root_kind=DATASET_ROOT_KIND)
    finally:
        engine.dispose()


def load_dataset_manifest(db_path: Path) -> DatasetManifest:
    """Load the canonical dataset manifest that owns corpus provenance."""

    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            manifest = _DATASET_MANIFEST_STORE.load(conn)
        if manifest is None:
            raise MissingStateError(f"Missing dataset manifest: {db_path}")
        return manifest
    finally:
        engine.dispose()


def list_acquire_runs(db_path: Path) -> list[AcquireRunRecord]:
    """List acquire-run deltas newest first without duplicating manifest provenance."""

    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            return _ACQUIRE_RUN_STORE.list(conn, order_by=acquire_runs.c.run_id.desc())
    finally:
        engine.dispose()


def _dataset_manifest_payload(manifest: DatasetManifest) -> dict[str, object]:
    return {
        "dataset": {"id": manifest.dataset.id, "name": manifest.dataset.name},
        "chain": {"name": manifest.chain.name, "chain_id": manifest.chain.chain_id},
        "request": {
            "history": asdict(manifest.request.history),
            "evaluation": asdict(manifest.request.evaluation),
        },
        "coverage": {
            "history": asdict(manifest.coverage.history),
            "evaluation": asdict(manifest.coverage.evaluation),
        },
        "validation": {
            "history": _validation_payload(manifest.validation.history),
            "evaluation": _validation_payload(manifest.validation.evaluation),
        },
        "semantics": corpus_semantics_payload(manifest.semantics),
    }


def _dataset_manifest_from_payload(payload: dict[str, object]) -> DatasetManifest:
    dataset = mapping_payload(payload["dataset"], label="dataset")
    chain = mapping_payload(payload["chain"], label="chain")
    request = mapping_payload(payload["request"], label="request")
    coverage = mapping_payload(payload["coverage"], label="coverage")
    validation = mapping_payload(payload["validation"], label="validation")
    semantics = corpus_semantics_from_payload(
        mapping_payload(payload["semantics"], label="semantics")
    )
    return DatasetManifest(
        dataset=DatasetIdentity(
            id=str(dataset["id"]),
            name=str(dataset["name"]),
        ),
        chain=ChainMetadata(
            name=str(chain["name"]),
            chain_id=_int_value(chain["chain_id"]),
        ),
        request=DatasetRequestMetadata(
            history=_window_from_payload(request["history"]),
            evaluation=_window_from_payload(request["evaluation"]),
        ),
        coverage=DatasetCoverageMetadata(
            history=_window_from_payload(coverage["history"]),
            evaluation=_window_from_payload(coverage["evaluation"]),
        ),
        validation=DatasetValidationMetadata(
            history=_validation_from_payload(validation["history"]),
            evaluation=_validation_from_payload(validation["evaluation"]),
        ),
        semantics=semantics,
    )


def _window_from_payload(payload: object) -> DatasetWindowMetadata:
    mapping = mapping_payload(payload, label="window")
    return DatasetWindowMetadata(
        start_timestamp=_int_value(mapping["start_timestamp"]),
        end_timestamp=_int_value(mapping["end_timestamp"]),
    )


def _acquire_run_payload(run: AcquireRunRecord) -> dict[str, object]:
    return {
        "provider": {
            "name": run.provider.name,
            "reference": run.provider.reference,
            "endpoint_fingerprint": run.provider.endpoint_fingerprint,
        },
        "facts": {
            "required_history_seconds": run.facts.required_history_seconds,
            "acquired_history_window_seconds": run.facts.acquired_history_window_seconds,
            "valid_anchor_samples": run.facts.valid_anchor_samples,
        },
        "settings": asdict(run.settings),
        "runtime": asdict(run.runtime),
    }


def _acquire_run_from_payload(payload: dict[str, object]) -> AcquireRunRecord:
    provider = mapping_payload(payload["provider"], label="provider")
    facts = mapping_payload(payload["facts"], label="facts")
    settings = mapping_payload(payload["settings"], label="settings")
    runtime = mapping_payload(payload["runtime"], label="runtime")
    return AcquireRunRecord(
        provider=ProviderMetadata(
            name=str(provider["name"]),
            reference=str(provider["reference"]),
            endpoint_fingerprint=str(provider["endpoint_fingerprint"]),
        ),
        facts=AcquireRunFacts(
            required_history_seconds=_int_value(facts["required_history_seconds"]),
            acquired_history_window_seconds=_int_value(facts["acquired_history_window_seconds"]),
            valid_anchor_samples=_int_value(facts["valid_anchor_samples"]),
        ),
        settings=AcquisitionConfigSnapshot(
            chunk_size=_int_value(settings["chunk_size"]),
            rpc_batch_size=_int_value(settings["rpc_batch_size"]),
            rpc_concurrency=_int_value(settings["rpc_concurrency"]),
            rpc_min_batch_size=_int_value(settings["rpc_min_batch_size"]),
            rpc_concurrency_rungs=_int_list_value(settings["rpc_concurrency_rungs"]),
        ),
        runtime=DatasetAcquisitionRuntimeMetadata(
            configured_batch_size=_int_value(runtime["configured_batch_size"]),
            final_batch_size=_int_value(runtime["final_batch_size"]),
            min_batch_size=_int_value(runtime["min_batch_size"]),
            configured_concurrency=_int_value(runtime["configured_concurrency"]),
            final_concurrency=_int_value(runtime["final_concurrency"]),
            concurrency_rungs=_int_list_value(runtime["concurrency_rungs"]),
            oversize_error_count=_int_value(runtime["oversize_error_count"]),
            transient_error_count=_int_value(runtime["transient_error_count"]),
            oversize_backoffs=_int_value(runtime["oversize_backoffs"]),
            transient_backoffs=_int_value(runtime["transient_backoffs"]),
            concurrency_recoveries=_int_value(runtime["concurrency_recoveries"]),
        ),
    )


def _validation_payload(report: CompactValidationReport) -> dict[str, object]:
    return asdict(report)


def _validation_from_payload(payload: object) -> CompactValidationReport:
    mapping = mapping_payload(payload, label="validation")
    block_range = mapping_payload(mapping["block_range"], label="validation.block_range")
    timestamp_range = mapping_payload(
        mapping["timestamp_range"], label="validation.timestamp_range"
    )
    return CompactValidationReport(
        status=str(mapping["status"]),
        rows=_int_value(mapping["rows"]),
        block_range=BlockRangeMetadata(
            first=_optional_int(block_range.get("first")),
            last=_optional_int(block_range.get("last")),
        ),
        timestamp_range=TimestampRangeMetadata(
            first=_optional_int(timestamp_range.get("first")),
            last=_optional_int(timestamp_range.get("last")),
        ),
        issues=_issues_payload(mapping.get("issues")),
    )


def _issues_payload(payload: object) -> dict[str, object] | None:
    if payload is None:
        return None
    return mapping_payload(payload, label="validation.issues")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _int_value(value)


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("Expected integer-like payload")
    if isinstance(value, (int, float, str, bytes)):
        return int(value)
    raise TypeError("Expected integer-like payload")


def _int_list_value(values: object) -> list[int]:
    return [_int_value(value) for value in _sequence_payload(values)]


def _sequence_payload(values: object) -> Sequence[object]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("Expected list payload")
    return cast(Sequence[object], values)


def _now_timestamp() -> int:
    from time import time

    return int(time())
