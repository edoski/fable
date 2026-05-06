from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from spice.config.models import ArtifactVariant
from spice.core.errors import StateConflictError, StateLayoutError
from spice.storage.catalog.index import ReindexedCatalogRoot
from spice.storage.catalog.records import CatalogArtifactRecord, CatalogDatasetRecord
from spice.storage.engine import (
    ARTIFACT_ROOT_KIND,
    DATASET_ROOT_KIND,
    STUDY_ROOT_KIND,
    RootKind,
    ensure_state_db,
    state_db_path,
)
from spice.storage.lifecycle import (
    delete_catalog_artifact_root,
    prepare_root_stage,
    promote_root_stage,
)
from spice.storage.schema import ARTIFACT_TABLES
from spice.storage.transactions import (
    FullRootTransaction,
    PartialRootTransaction,
    RootMutation,
    artifact_full_root_transaction,
    record_study_root_mutation,
)
from spice.storage.workflow_roots import ArtifactRootHandle, StudyRootHandle


def test_partial_root_transaction_promotes_selected_paths_and_reindexes(
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

    transaction = PartialRootTransaction(storage_root=storage_root, root_path=root_path)
    transaction.add(root_path / "history", source_dir)

    assert transaction.commit() == ReindexedCatalogRoot(
        root_kind=RootKind.CORPUS,
        record=record,
    )
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


def test_artifact_full_root_transaction_derives_storage_policy_from_handle(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "outputs"
    root_path = storage_root / "artifacts" / "ethereum" / "artifact-1"
    artifact = ArtifactRootHandle(
        storage_root=storage_root,
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
        variant=ArtifactVariant.BASELINE,
    )

    transaction = artifact_full_root_transaction(artifact, purpose="training")

    assert transaction.storage_root == storage_root
    assert transaction.destination_root == root_path
    assert transaction.expected_root_kind is ARTIFACT_ROOT_KIND
    assert transaction.prune_stop_at == storage_root / "artifacts"
    assert transaction.purpose == "training"


def test_full_root_transaction_commit_promotes_after_writer_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    destination = storage_root / "artifacts" / "ethereum" / "artifact-1"
    promoted: list[bool] = []

    class FakeStage:
        staged_root = tmp_path / "stage"

        def promote(self):
            promoted.append(True)
            return "reindexed"

    @contextmanager
    def fake_open(_self):
        yield FakeStage()

    monkeypatch.setattr(FullRootTransaction, "open", fake_open)

    transaction = FullRootTransaction(
        storage_root=storage_root,
        destination_root=destination,
        expected_root_kind=RootKind.ARTIFACT,
    )
    committed = transaction.commit(lambda staged_root: staged_root / "done")

    assert committed.result == tmp_path / "stage" / "done"
    assert committed.reindexed == "reindexed"
    assert promoted == [True]


def test_full_root_transaction_commit_does_not_promote_after_writer_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    destination = storage_root / "artifacts" / "ethereum" / "artifact-1"

    class FakeStage:
        staged_root = tmp_path / "stage"

        def promote(self):
            raise AssertionError("failed writer must not promote")

    @contextmanager
    def fake_open(_self):
        yield FakeStage()

    monkeypatch.setattr(FullRootTransaction, "open", fake_open)

    transaction = FullRootTransaction(
        storage_root=storage_root,
        destination_root=destination,
        expected_root_kind=RootKind.ARTIFACT,
    )
    def fail_writer(_staged_root: Path) -> None:
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        transaction.commit(fail_writer)


def test_reindex_root_state_validates_expected_kind_and_canonical_path(tmp_path: Path) -> None:
    from spice.storage.transactions import reindex_root_state

    storage_root = tmp_path / "outputs"
    artifact_root = storage_root / "artifacts" / "ethereum" / "artifact-1"
    ensure_state_db(
        state_db_path(artifact_root),
        root_kind=ARTIFACT_ROOT_KIND,
        tables=ARTIFACT_TABLES,
    )

    with pytest.raises(StateLayoutError, match="outside the corpus storage subtree"):
        reindex_root_state(
            storage_root,
            root_path=artifact_root,
            expected_root_kind=DATASET_ROOT_KIND,
        )
    with pytest.raises(StateLayoutError, match="outside the artifact storage subtree"):
        reindex_root_state(
            storage_root,
            root_path=tmp_path / "other" / "ethereum" / "artifact-1",
            expected_root_kind=ARTIFACT_ROOT_KIND,
        )


def test_record_mutated_root_reindexes_after_successful_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from spice.storage.transactions import record_mutated_root

    storage_root = tmp_path / "outputs"
    artifact_root = storage_root / "artifacts" / "ethereum" / "artifact-1"
    ensure_state_db(
        state_db_path(artifact_root),
        root_kind=ARTIFACT_ROOT_KIND,
        tables=ARTIFACT_TABLES,
    )
    calls: list[str] = []

    def fake_reindex(storage_root_arg, *, root_path):
        calls.append(f"reindex:{storage_root_arg == storage_root}:{root_path == artifact_root}")
        return "reindexed"

    monkeypatch.setattr("spice.storage.transactions.reindex_catalog_root", fake_reindex)

    mutation = record_mutated_root(
        storage_root,
        root_path=artifact_root,
        expected_root_kind=ARTIFACT_ROOT_KIND,
        mutation=lambda: calls.append("mutate") or "result",
    )

    assert mutation.result == "result"
    assert mutation.reindexed == "reindexed"
    assert calls == ["mutate", "reindex:True:True"]


def test_record_study_root_mutation_derives_storage_policy_from_handle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    root_path = storage_root / "studies" / "ethereum" / "study-1"
    study = StudyRootHandle(
        storage_root=storage_root,
        study_id="study-1",
        study_name="study",
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )
    seen: dict[str, object] = {}

    def fake_record_mutated_root(storage_root_arg, *, root_path, expected_root_kind, mutation):
        seen.update(
            {
                "storage_root": storage_root_arg,
                "root_path": root_path,
                "expected_root_kind": expected_root_kind,
            }
        )
        return RootMutation(result=mutation(), reindexed="reindexed")

    monkeypatch.setattr("spice.storage.transactions.record_mutated_root", fake_record_mutated_root)

    committed = record_study_root_mutation(study, mutation=lambda: "result")

    assert committed.result == "result"
    assert seen == {
        "storage_root": storage_root,
        "root_path": root_path,
        "expected_root_kind": STUDY_ROOT_KIND,
    }


def test_record_mutated_root_does_not_reindex_after_mutation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from spice.storage.transactions import record_mutated_root

    storage_root = tmp_path / "outputs"
    artifact_root = storage_root / "artifacts" / "ethereum" / "artifact-1"
    ensure_state_db(
        state_db_path(artifact_root),
        root_kind=ARTIFACT_ROOT_KIND,
        tables=ARTIFACT_TABLES,
    )
    monkeypatch.setattr(
        "spice.storage.transactions.reindex_catalog_root",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("failed mutation must not reindex")
        ),
    )

    with pytest.raises(RuntimeError, match="mutation failed"):
        record_mutated_root(
            storage_root,
            root_path=artifact_root,
            expected_root_kind=ARTIFACT_ROOT_KIND,
            mutation=lambda: (_ for _ in ()).throw(RuntimeError("mutation failed")),
        )


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
