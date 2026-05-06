from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from spice.core.errors import StateConflictError, StateLayoutError
from spice.storage.catalog.index import ReindexedCatalogRoot
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
    commit_artifact_root,
    commit_corpus_acquisition,
    record_artifact_evaluation_state,
    record_study_root_mutation,
)
from spice.storage.workflow_roots import CorpusRootHandle
from tests.catalog_helpers import artifact_record, dataset_record
from tests.root_handle_helpers import artifact_handle, corpus_handle, study_handle


def test_commit_corpus_acquisition_replaces_declared_paths_and_reindexes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    root_path = storage_root / "corpora" / "ethereum" / "dataset-1"
    source_dir = tmp_path / "stage" / "history"
    source_dir.mkdir(parents=True)
    (source_dir / "blocks.parquet").write_text("payload", encoding="utf-8")
    source_state = tmp_path / "stage" / "state.sqlite"
    ensure_state_db(source_state, root_kind=DATASET_ROOT_KIND, tables=())
    captured: dict[str, Path] = {}

    record = dataset_record(root_path)

    def fake_reindex_catalog_root(storage_root: Path, *, root_path: Path) -> ReindexedCatalogRoot:
        captured.update({"storage_root": storage_root, "root_path": root_path})
        return ReindexedCatalogRoot(root_kind=RootKind.CORPUS, record=record)

    monkeypatch.setattr(
        "spice.storage.transactions.reindex_catalog_root",
        fake_reindex_catalog_root,
    )

    corpus = corpus_handle(storage_root, dataset_id="dataset-1")
    assert commit_corpus_acquisition(
        corpus,
        history_dir=source_dir,
        evaluation_dir=None,
        state_db=source_state,
    ) == ReindexedCatalogRoot(
        root_kind=RootKind.CORPUS,
        record=record,
    )
    assert (root_path / "history" / "blocks.parquet").read_text(encoding="utf-8") == "payload"
    assert captured == {"storage_root": storage_root, "root_path": root_path}


def test_commit_artifact_root_delegates_stage_policy(tmp_path: Path, monkeypatch) -> None:
    storage_root = tmp_path / "outputs"
    corpus = corpus_handle(storage_root, dataset_id="dataset-1")
    artifact = artifact_handle(storage_root, corpus=corpus, artifact_id="artifact-1")
    calls: list[dict[str, object]] = []

    class FakeStage:
        staged_root = tmp_path / "stage"

        def promote(self):
            return "reindexed"

    @contextmanager
    def fake_staged_root(**kwargs):
        calls.append(kwargs)
        yield FakeStage()

    monkeypatch.setattr("spice.storage.transactions.staged_root", fake_staged_root)

    committed = commit_artifact_root(artifact, writer=lambda staged_root: staged_root)

    assert committed.result == tmp_path / "stage"
    assert calls == [
        {
            "storage_root": storage_root,
            "destination_root": artifact.root_path,
            "expected_root_kind": RootKind.ARTIFACT,
            "replace": True,
            "purpose": "staging",
            "prune_stop_at": storage_root / "artifacts",
        }
    ]


def test_commit_artifact_root_derives_storage_policy_from_handle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    corpus = corpus_handle(storage_root, dataset_id="dataset-1")
    artifact = artifact_handle(storage_root, corpus=corpus, artifact_id="artifact-1")
    seen: dict[str, object] = {}

    class FakeStage:
        staged_root = tmp_path / "stage"

        def promote(self):
            seen["promoted"] = True
            return "reindexed"

    @contextmanager
    def fake_staged_root(**kwargs):
        seen.update(kwargs)
        yield FakeStage()

    monkeypatch.setattr("spice.storage.transactions.staged_root", fake_staged_root)

    committed = commit_artifact_root(artifact, writer=lambda staged_root: staged_root)

    assert committed.result == tmp_path / "stage"
    assert committed.reindexed == "reindexed"
    assert seen["storage_root"] == storage_root
    assert seen["destination_root"] == artifact.root_path
    assert seen["expected_root_kind"] is ARTIFACT_ROOT_KIND
    assert seen["prune_stop_at"] == storage_root / "artifacts"
    assert seen["promoted"] is True


def test_commit_artifact_root_promotes_after_writer_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    promoted: list[bool] = []

    class FakeStage:
        staged_root = tmp_path / "stage"

        def promote(self):
            promoted.append(True)
            return "reindexed"

    @contextmanager
    def fake_staged_root(**_kwargs):
        yield FakeStage()

    artifact = artifact_handle(
        storage_root,
        corpus=corpus_handle(storage_root, dataset_id="dataset-1"),
        artifact_id="artifact-1",
    )
    monkeypatch.setattr("spice.storage.transactions.staged_root", fake_staged_root)

    committed = commit_artifact_root(artifact, writer=lambda staged_root: staged_root / "done")

    assert committed.result == tmp_path / "stage" / "done"
    assert committed.reindexed == "reindexed"
    assert promoted == [True]


def test_commit_artifact_root_does_not_promote_after_writer_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"

    class FakeStage:
        staged_root = tmp_path / "stage"

        def promote(self):
            raise AssertionError("failed writer must not promote")

    @contextmanager
    def fake_staged_root(**_kwargs):
        yield FakeStage()

    artifact = artifact_handle(
        storage_root,
        corpus=corpus_handle(storage_root, dataset_id="dataset-1"),
        artifact_id="artifact-1",
    )
    monkeypatch.setattr("spice.storage.transactions.staged_root", fake_staged_root)

    def fail_writer(_staged_root: Path) -> None:
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        commit_artifact_root(artifact, writer=fail_writer)


def test_commit_corpus_acquisition_validates_handle_root_layout(tmp_path: Path) -> None:
    storage_root = tmp_path / "outputs"
    corpus = CorpusRootHandle(
        storage_root=storage_root,
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        root_path=storage_root / "corpora" / "dataset-1",
        state_db_path=storage_root / "corpora" / "dataset-1" / ".spice" / "state.sqlite",
        history_dir=storage_root / "corpora" / "dataset-1" / "history",
        evaluation_dir=storage_root / "corpora" / "dataset-1" / "evaluation",
    )
    ensure_state_db(corpus.state_db_path, root_kind=DATASET_ROOT_KIND, tables=())

    with pytest.raises(StateLayoutError, match="canonical <chain>/<root-id>"):
        commit_corpus_acquisition(
            corpus,
            history_dir=None,
            evaluation_dir=None,
            state_db=corpus.state_db_path,
        )


def test_record_study_root_mutation_reindexes_after_successful_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    study = study_handle(storage_root, corpus=corpus_handle(storage_root), study_id="study-1")
    ensure_state_db(
        study.state_db_path,
        root_kind=STUDY_ROOT_KIND,
        tables=(),
    )
    calls: list[str] = []

    def fake_reindex(storage_root_arg, *, root_path):
        calls.append(f"reindex:{storage_root_arg == storage_root}:{root_path == study.root_path}")
        return "reindexed"

    monkeypatch.setattr("spice.storage.transactions.reindex_catalog_root", fake_reindex)

    mutation = record_study_root_mutation(
        study,
        mutation=lambda: calls.append("mutate") or "result",
    )

    assert mutation.result == "result"
    assert mutation.reindexed == "reindexed"
    assert calls == ["mutate", "reindex:True:True"]


def test_record_artifact_evaluation_state_validates_root_without_reindex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    artifact = artifact_handle(
        storage_root,
        corpus=corpus_handle(storage_root, dataset_id="dataset-1"),
        artifact_id="artifact-1",
    )
    ensure_state_db(
        artifact.state_db_path,
        root_kind=ARTIFACT_ROOT_KIND,
        tables=(),
    )
    calls: list[str] = []
    expected_summary = object()

    monkeypatch.setattr(
        "spice.storage.transactions.record_evaluation_state",
        lambda db_path, *, summary: calls.append(
            f"record:{db_path == artifact.state_db_path}:{summary is expected_summary}"
        )
        or "loaded",
    )
    monkeypatch.setattr(
        "spice.storage.transactions.reindex_catalog_root",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("evaluation state should not reindex")
        ),
    )

    assert record_artifact_evaluation_state(artifact, summary=expected_summary) == "loaded"
    assert calls == ["record:True:True"]


def test_record_study_root_mutation_does_not_reindex_after_mutation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    study = study_handle(storage_root, corpus=corpus_handle(storage_root), study_id="study-1")
    monkeypatch.setattr(
        "spice.storage.transactions.reindex_catalog_root",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("failed mutation must not reindex")
        ),
    )

    with pytest.raises(RuntimeError, match="mutation failed"):
        record_study_root_mutation(
            study,
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


def test_delete_catalog_root_materializes_canonical_record_path(tmp_path: Path) -> None:
    storage_root = tmp_path / "outputs"
    root_path = storage_root / "artifacts" / "artifact-1"
    ensure_state_db(
        state_db_path(root_path),
        root_kind=RootKind.ARTIFACT,
        tables=ARTIFACT_TABLES,
    )
    record = artifact_record(
        root_path,
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
    )

    with pytest.raises(StateLayoutError, match="missing state DB"):
        delete_catalog_artifact_root(storage_root, record)

    assert root_path.exists()
