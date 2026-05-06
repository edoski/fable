from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from spice.core.errors import StateLayoutError
from spice.storage.artifact import load_artifact_manifest
from spice.storage.catalog.records import CatalogArtifactRecord, CatalogDatasetRecord
from spice.storage.catalog.index import (
    list_artifact_records,
    list_dataset_records,
    upsert_catalog_record,
)
from spice.storage.engine import DATASET_ROOT_KIND, ensure_state_db, state_db_path
from spice.storage.layout import catalog_db_path
from spice.storage.schema import DATASET_TABLES
from spice.storage.selectors import ArtifactSelector, DatasetSelector


def _dataset_timestamps(path: Path, dataset_id: str) -> tuple[int, int]:
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "select created_at, updated_at from dataset_index where dataset_id = ?",
            (dataset_id,),
        ).fetchone()
    assert row is not None
    return int(row[0]), int(row[1])


def test_catalog_upsert_keeps_created_at_stable(tmp_path: Path, monkeypatch) -> None:
    from spice.storage.catalog import store

    storage_root = tmp_path / "outputs"
    catalog_path = catalog_db_path(storage_root)
    timestamps = iter([100, 200])
    monkeypatch.setattr(store, "_now_timestamp", lambda: next(timestamps))

    dataset_root = storage_root / "corpora" / "ethereum" / "dataset-1"
    record = CatalogDatasetRecord(
        dataset_id="dataset-1",
        dataset_name="old",
        chain_name="ethereum",
        root_path=dataset_root,
        state_db_path=state_db_path(dataset_root),
    )
    upsert_catalog_record(storage_root, record)
    upsert_catalog_record(
        storage_root,
        CatalogDatasetRecord(
            dataset_id=record.dataset_id,
            dataset_name="new",
            chain_name=record.chain_name,
            root_path=record.root_path,
            state_db_path=record.state_db_path,
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
        CatalogDatasetRecord(
            dataset_id="dataset-1",
            dataset_name="daily",
            chain_name="ethereum",
            root_path=dataset_root,
            state_db_path=state_db_path(dataset_root),
        ),
    )
    upsert_catalog_record(
        storage_root,
        CatalogArtifactRecord(
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
            root_path=artifact_root,
            state_db_path=state_db_path(artifact_root),
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
