from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from spice.core.errors import StateConflictError, StateLayoutError
from spice.storage.catalog import CatalogArtifactRecord, CatalogDatasetRecord
from spice.storage.catalog.index import ReindexedCatalogRoot
from spice.storage.engine import RootKind, ensure_state_db, state_db_path
from spice.storage.lifecycle import (
    delete_catalog_artifact_root,
    prepare_root_stage,
    promote_root_stage,
)
from spice.storage.schema import ARTIFACT_TABLES
from spice.storage.transactions import FullRootTransaction, PartialRootTransaction


def test_partial_root_commit_promotes_selected_paths_and_reindexes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    root_path = storage_root / "corpora" / "ethereum" / "dataset-1"
    source_dir = tmp_path / "stage" / "history"
    source_dir.mkdir(parents=True)
    (source_dir / "blocks.parquet").write_text("payload", encoding="utf-8")
    captured: dict[str, Path] = {}

    record = CatalogDatasetRecord(
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )

    def fake_reindex_catalog_root(storage_root: Path, *, root_path: Path) -> ReindexedCatalogRoot:
        captured.update({"storage_root": storage_root, "root_path": root_path})
        return ReindexedCatalogRoot(root_kind=RootKind.CORPUS, record=record)

    monkeypatch.setattr(
        "spice.storage.transactions.reindex_catalog_root",
        fake_reindex_catalog_root,
    )

    commit = PartialRootTransaction(storage_root=storage_root, root_path=root_path)
    commit.add(root_path / "history", source_dir)

    assert commit.commit() == ReindexedCatalogRoot(root_kind=RootKind.CORPUS, record=record)
    assert (root_path / "history" / "blocks.parquet").read_text(encoding="utf-8") == "payload"
    assert captured == {"storage_root": storage_root, "root_path": root_path}


def test_full_root_transaction_delegates_stage_policy(tmp_path: Path, monkeypatch) -> None:
    storage_root = tmp_path / "outputs"
    destination = storage_root / "artifacts" / "ethereum" / "artifact-1"
    stage = object()
    calls: list[dict[str, object]] = []

    @contextmanager
    def fake_staged_root(**kwargs):
        calls.append(kwargs)
        yield stage

    monkeypatch.setattr("spice.storage.transactions.staged_root", fake_staged_root)

    transaction = FullRootTransaction(
        storage_root=storage_root,
        destination_root=destination,
        expected_root_kind=RootKind.ARTIFACT,
        purpose="training",
        prune_stop_at=destination.parent.parent,
    )
    with transaction.open() as active_stage:
        assert active_stage is stage

    assert calls == [
        {
            "storage_root": storage_root,
            "destination_root": destination,
            "expected_root_kind": RootKind.ARTIFACT,
            "replace": True,
            "purpose": "training",
            "prune_stop_at": destination.parent.parent,
        }
    ]


def test_prepare_root_stage_rejects_existing_destination_without_replace(tmp_path: Path) -> None:
    destination = tmp_path / "outputs" / "artifacts" / "ethereum" / "artifact-1"
    destination.mkdir(parents=True)

    with pytest.raises(StateConflictError, match="Destination already exists"):
        prepare_root_stage(destination_root=destination, replace=False)


def test_promote_root_stage_rejects_destination_outside_expected_subtree(tmp_path: Path) -> None:
    storage_root = tmp_path / "outputs"
    staged_root = tmp_path / "stage"
    ensure_state_db(
        state_db_path(staged_root),
        root_kind=RootKind.ARTIFACT,
        tables=ARTIFACT_TABLES,
    )

    with pytest.raises(StateLayoutError, match="outside the artifact storage subtree"):
        promote_root_stage(
            storage_root=storage_root,
            destination_root=storage_root / "corpora" / "ethereum" / "artifact-1",
            staged_root=staged_root,
            expected_root_kind=RootKind.ARTIFACT,
            replace=True,
        )


def test_promote_root_stage_rejects_noncanonical_destination_layout(tmp_path: Path) -> None:
    storage_root = tmp_path / "outputs"
    staged_root = tmp_path / "stage"
    ensure_state_db(
        state_db_path(staged_root),
        root_kind=RootKind.ARTIFACT,
        tables=ARTIFACT_TABLES,
    )

    with pytest.raises(StateLayoutError, match="canonical <chain>/<root-id>"):
        promote_root_stage(
            storage_root=storage_root,
            destination_root=storage_root / "artifacts" / "artifact-1",
            staged_root=staged_root,
            expected_root_kind=RootKind.ARTIFACT,
            replace=True,
        )


def test_delete_catalog_root_rejects_noncanonical_record_path(tmp_path: Path) -> None:
    storage_root = tmp_path / "outputs"
    root_path = storage_root / "artifacts" / "artifact-1"
    ensure_state_db(
        state_db_path(root_path),
        root_kind=RootKind.ARTIFACT,
        tables=ARTIFACT_TABLES,
    )
    record = CatalogArtifactRecord(
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id="model",
        problem_id="problem",
        variant="baseline",
        study_id=None,
        study_name=None,
        root_path=root_path,
        state_db_path=state_db_path(root_path),
    )

    with pytest.raises(StateLayoutError, match="canonical <chain>/<root-id>"):
        delete_catalog_artifact_root(storage_root, record)

    assert root_path.exists()
