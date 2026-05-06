from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from spice.config.models import ChainRuntimeSpec
from spice.core.errors import StateLayoutError
from spice.corpus.metadata import (
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
from spice.storage.artifact import load_artifact_manifest
from spice.storage.catalog.index import (
    list_artifact_records,
    list_dataset_records,
    reindex_catalog_root,
    upsert_catalog_record,
)
from spice.storage.corpus import write_dataset_state
from spice.storage.engine import DATASET_ROOT_KIND, ensure_state_db, state_db_path
from spice.storage.layout import catalog_db_path
from spice.storage.schema import DATASET_TABLES
from spice.storage.selectors import ArtifactSelector, DatasetSelector
from tests.catalog_helpers import artifact_record, dataset_record


def _dataset_timestamps(path: Path, dataset_id: str) -> tuple[int, int]:
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "select created_at, updated_at from dataset_index where dataset_id = ?",
            (dataset_id,),
        ).fetchone()
    assert row is not None
    return int(row[0]), int(row[1])


def _dataset_manifest(*, dataset_id: str) -> DatasetManifest:
    split = CorpusSplitManifest(
        kind="history",
        request=SplitRequestMetadata(
            start_timestamp=1,
            end_timestamp=2,
            start_block=1,
            end_block=2,
        ),
        coverage=SplitCoverageMetadata(
            first_timestamp=1,
            last_timestamp=2,
            first_block=1,
            last_block=1,
            rows=1,
        ),
        validation=CompactValidationReport(
            status="clean",
            rows=1,
            block_range=BlockRangeMetadata(first=1, last=1),
            timestamp_range=TimestampRangeMetadata(first=1, last=2),
        ),
        materialization=SplitMaterializationMetadata(outcome="reused", file_count=1),
    )
    return DatasetManifest(
        dataset=DatasetIdentity(id=dataset_id, name="dataset"),
        chain=ChainMetadata(
            name="ethereum",
            runtime=ChainRuntimeSpec(
                chain_id=1,
                uses_poa_extra_data=False,
                nominal_block_time_seconds=12.0,
            ),
        ),
        splits=CorpusSplitManifests(history=split, evaluation=split),
        source_requirements=CorpusAcquisitionSourceRequirements(
            required_columns=frozenset(
                {"block_number", "timestamp", "chain_id", "base_fee_per_gas"}
            ),
            optional_enrichments=frozenset(),
            temporal_unit="block",
            ordering_key="block_number",
            partition_key="chain_id",
        ),
    )


def _acquire_run() -> AcquireRunRecord:
    return AcquireRunRecord(
        provider=ProviderMetadata(
            name="publicnode",
            reference="ethereum",
            endpoint_fingerprint="abcdef0123456789",
        ),
        facts=AcquireRunFacts(
            requested_history_window_seconds=86400,
            resolved_capability_samples=1000,
        ),
        settings=AcquisitionConfigSnapshot(
            chunk_size=5000,
            rpc_batch_size=100,
            rpc_concurrency=8,
            rpc_min_batch_size=10,
            rpc_concurrency_rungs=[8, 4, 2],
        ),
        runtime=DatasetAcquisitionRuntimeMetadata(
            configured_batch_size=100,
            final_batch_size=50,
            min_batch_size=10,
            configured_concurrency=8,
            final_concurrency=4,
            concurrency_rungs=[8, 4, 2],
            oversize_error_count=0,
            transient_error_count=0,
            oversize_backoffs=0,
            transient_backoffs=0,
            concurrency_recoveries=0,
        ),
    )


def test_catalog_upsert_keeps_created_at_stable(tmp_path: Path, monkeypatch) -> None:
    from spice.storage.catalog import store

    storage_root = tmp_path / "outputs"
    catalog_path = catalog_db_path(storage_root)
    timestamps = iter([100, 200])
    monkeypatch.setattr(store, "_now_timestamp", lambda: next(timestamps))

    dataset_root = storage_root / "corpora" / "ethereum" / "dataset-1"
    record = dataset_record(
        dataset_root,
        dataset_id="dataset-1",
        dataset_name="old",
        chain_name="ethereum",
    )
    upsert_catalog_record(storage_root, record)
    upsert_catalog_record(
        storage_root,
        dataset_record(
            dataset_root,
            dataset_id=record.dataset_id,
            dataset_name="new",
            chain_name=record.chain_name,
        ),
    )

    assert _dataset_timestamps(catalog_path, "dataset-1") == (100, 200)


def test_artifact_reader_rejects_corpus_root_kind(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "corpora" / "ethereum" / "dataset-1"
    db_path = state_db_path(root)
    ensure_state_db(db_path, root_kind=DATASET_ROOT_KIND, tables=DATASET_TABLES)

    with pytest.raises(StateLayoutError, match="root kind mismatch"):
        load_artifact_manifest(db_path)


def test_catalog_filters_by_exact_root_ids(tmp_path: Path) -> None:
    storage_root = tmp_path / "outputs"
    dataset_root = storage_root / "corpora" / "ethereum" / "dataset-1"
    artifact_root = storage_root / "artifacts" / "ethereum" / "artifact-1"
    upsert_catalog_record(
        storage_root,
        dataset_record(
            dataset_root,
            dataset_id="dataset-1",
            dataset_name="daily",
            chain_name="ethereum",
        ),
    )
    upsert_catalog_record(
        storage_root,
        artifact_record(
            artifact_root,
            artifact_id="artifact-1",
            dataset_id="dataset-1",
            dataset_name="daily",
            chain_name="ethereum",
            features_id="features",
            prediction_id="prediction",
            model_id="model",
            problem_id="problem",
            variant="baseline",
            study_id=None,
            study_name=None,
        ),
    )

    dataset_records = list_dataset_records(
        storage_root,
        selector=DatasetSelector(dataset_id="dataset-1"),
    )
    assert [record.dataset_id for record in dataset_records] == ["dataset-1"]
    assert [
        record.artifact_id
        for record in list_artifact_records(
            storage_root,
            selector=ArtifactSelector(dataset_id="dataset-1"),
        )
    ] == ["artifact-1"]


def test_catalog_reindex_rejects_root_path_that_disagrees_with_manifest_identity(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "outputs"
    root = storage_root / "corpora" / "ethereum" / "wrong-dataset"
    db_path = state_db_path(root)
    manifest = _dataset_manifest(dataset_id="dataset-1")

    write_dataset_state(db_path, manifest=manifest, acquire_run=_acquire_run())

    with pytest.raises(StateLayoutError, match="manifest identity"):
        reindex_catalog_root(storage_root, root_path=root)
