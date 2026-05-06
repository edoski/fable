from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from spice.core.errors import StateConflictError
from spice.execution.session import ExecutionSession
from spice.execution.transfer_transaction import StorageTransferTransaction
from spice.storage.artifact import write_artifact_manifest
from spice.storage.catalog.records import CatalogArtifactRecord, CatalogDatasetRecord
from spice.storage.catalog.codecs import encode_remote_catalog_record
from spice.storage.catalog.index import ReindexedCatalogRoot
from spice.storage.engine import RootKind
from tests.artifact_helpers import manifest


class _FakeSession:
    def __init__(
        self,
        remote_storage_root: Path,
        *,
        record_json: str = "",
        fail_on: str | None = None,
        fail_cleanup: bool = False,
    ) -> None:
        self.target = SimpleNamespace(
            spec=SimpleNamespace(paths=SimpleNamespace(storage_root=remote_storage_root))
        )
        self.record_json = record_json
        self.fail_on = fail_on
        self.fail_cleanup = fail_cleanup
        self.execution_calls: list[str] = []
        self.captured: dict[str, object] = {}

    def run_module(
        self,
        module: str,
        args: list[str],
        *,
        capture_output: bool = True,
        check_action: str | None = None,
    ):
        del capture_output, check_action
        command = args[0]
        self.execution_calls.append(command)
        if command == self.fail_on:
            raise RuntimeError(f"{command} failed")
        if command == "cleanup-stage" and self.fail_cleanup:
            raise RuntimeError("cleanup failed")
        if command == "prepare-stage":
            self.captured.update(
                {
                    "destination_root": Path(args[args.index("--destination-root") + 1]),
                    "staged_root": Path(args[args.index("--staged-root") + 1]),
                    "replace": "--replace" in args,
                }
            )
        if command == "finalize-stage":
            self.captured["root_kind"] = args[args.index("--root-kind") + 1]
        return subprocess.CompletedProcess(
            args=[module, *args],
            returncode=0,
            stdout=self.record_json,
            stderr="",
        )

    def rsync_to(self, *, source_root: Path, destination_root: Path) -> None:
        del source_root, destination_root
        self.execution_calls.append("rsync-to")

    def rsync_from(self, *, source_root: Path, destination_root: Path) -> None:
        self.execution_calls.append("rsync-from")
        shutil.copytree(source_root, destination_root, dirs_exist_ok=True)


def _dataset_record(root_path: Path) -> CatalogDatasetRecord:
    return CatalogDatasetRecord(
        dataset_id="dataset-1",
        dataset_name="current_row_fee_dynamics",
        chain_name="ethereum",
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )


def _artifact_record(root_path: Path) -> CatalogArtifactRecord:
    return CatalogArtifactRecord(
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        dataset_name="current_row_fee_dynamics",
        chain_name="ethereum",
        features_id="current_row_fee_dynamics",
        prediction_id="icdcs_2026",
        model_id="lstm",
        problem_id="current_row_fee_dynamics",
        variant="baseline",
        study_id=None,
        study_name=None,
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )


def test_storage_transfer_transaction_pushes_dataset_to_canonical_corpus_destination(
    tmp_path,
    monkeypatch,
) -> None:
    local_storage_root = tmp_path / "outputs"
    record = _dataset_record(local_storage_root / "corpora" / "ethereum" / "dataset-1")
    record.root_path.mkdir(parents=True)
    remote_storage_root = tmp_path / "remote-storage"
    session = _FakeSession(remote_storage_root)
    monkeypatch.setattr(
        "spice.execution.transfer_transaction.resolve_dataset_record",
        lambda _root, *, selector: record,
    )
    transaction = StorageTransferTransaction(
        local_storage_root=local_storage_root,
        session=cast(ExecutionSession, session),
    )

    pushed = transaction.push_dataset(record.dataset_id, replace=False)

    assert pushed == record
    assert session.captured["destination_root"] == (
        remote_storage_root / "corpora" / record.chain_name / record.dataset_id
    )
    staged_root = session.captured["staged_root"]
    assert isinstance(staged_root, Path)
    assert staged_root.parent == remote_storage_root / "corpora" / record.chain_name
    assert "incoming" in staged_root.name
    assert session.captured["replace"] is False
    assert session.captured["root_kind"] == RootKind.CORPUS.value
    assert session.execution_calls == ["prepare-stage", "rsync-to", "finalize-stage"]


def test_storage_transfer_transaction_pulls_artifact_and_returns_local_record(
    tmp_path,
    monkeypatch,
) -> None:
    remote_root = tmp_path / "remote-storage" / "artifacts" / "ethereum" / "artifact-1"
    remote_root.mkdir(parents=True)
    (remote_root / "payload.txt").write_text("artifact payload", encoding="utf-8")
    record = _artifact_record(remote_root)
    local_storage_root = tmp_path / "local-outputs"
    captured: dict[str, object] = {}
    session = _FakeSession(
        tmp_path / "remote-storage",
        record_json=encode_remote_catalog_record(record),
    )

    def fake_promote_root_stage(
        *,
        storage_root,
        destination_root,
        staged_root,
        expected_root_kind,
        replace,
    ):
        del replace
        shutil.move(staged_root, destination_root)
        captured.update(
            {
                "storage_root": storage_root,
                "root_path": destination_root,
                "root_kind": expected_root_kind,
            }
        )
        return ReindexedCatalogRoot(
            root_kind=RootKind.ARTIFACT,
            record=_artifact_record(destination_root),
        )

    monkeypatch.setattr(
        "spice.execution.transfer_transaction.promote_root_stage",
        fake_promote_root_stage,
    )
    transaction = StorageTransferTransaction(
        local_storage_root=local_storage_root,
        session=cast(ExecutionSession, session),
    )

    pulled = transaction.pull_artifact(record.artifact_id, replace=False)

    destination_root = local_storage_root / "artifacts" / record.chain_name / record.artifact_id
    assert pulled.source_record == record
    assert pulled.local_record == _artifact_record(destination_root)
    assert pulled.destination_root == destination_root
    assert pulled.dataset_present is False
    assert (destination_root / "payload.txt").read_text(encoding="utf-8") == "artifact payload"
    assert captured == {
        "storage_root": local_storage_root,
        "root_path": destination_root,
        "root_kind": RootKind.ARTIFACT,
    }


def test_storage_transfer_transaction_uses_promoted_catalog_record(tmp_path) -> None:
    remote_storage_root = tmp_path / "remote-storage"
    remote_root = remote_storage_root / "artifacts" / "ethereum" / "artifact-1"
    write_artifact_manifest(remote_root / ".spice" / "state.sqlite", manifest=manifest())
    record = _artifact_record(remote_root)
    local_storage_root = tmp_path / "local-outputs"
    session = _FakeSession(
        remote_storage_root,
        record_json=encode_remote_catalog_record(record),
    )

    transaction = StorageTransferTransaction(
        local_storage_root=local_storage_root,
        session=cast(ExecutionSession, session),
    )
    pulled = transaction.pull_artifact(record.artifact_id, replace=False)

    destination_root = local_storage_root / "artifacts" / record.chain_name / record.artifact_id
    assert pulled.local_record.root_path == destination_root
    assert pulled.local_record.state_db_path == destination_root / ".spice" / "state.sqlite"
    assert pulled.local_record.artifact_id == manifest().artifact_id


def test_storage_transfer_transaction_rejects_existing_destination(tmp_path) -> None:
    record = _artifact_record(tmp_path / "remote-storage" / "artifacts" / "ethereum" / "artifact-1")
    destination_root = tmp_path / "outputs" / "artifacts" / record.chain_name / record.artifact_id
    destination_root.mkdir(parents=True)
    session = _FakeSession(
        tmp_path / "remote-storage",
        record_json=encode_remote_catalog_record(record),
    )

    with pytest.raises(StateConflictError, match="Destination already exists"):
        StorageTransferTransaction(
            local_storage_root=tmp_path / "outputs",
            session=cast(ExecutionSession, session),
        ).pull_artifact(
            record.artifact_id,
            replace=False,
        )


def test_storage_transfer_transaction_cleanup_failure_preserves_primary_exception(
    tmp_path,
    monkeypatch,
) -> None:
    record = _dataset_record(tmp_path / "outputs" / "corpora" / "ethereum" / "dataset-1")
    record.root_path.mkdir(parents=True)
    session = _FakeSession(
        tmp_path / "remote-storage",
        fail_on="finalize-stage",
        fail_cleanup=True,
    )
    monkeypatch.setattr(
        "spice.execution.transfer_transaction.resolve_dataset_record",
        lambda _root, *, selector: record,
    )
    transaction = StorageTransferTransaction(
        local_storage_root=tmp_path / "outputs",
        session=cast(ExecutionSession, session),
    )

    with pytest.raises(RuntimeError, match="finalize-stage failed") as exc_info:
        transaction.push_dataset(record.dataset_id, replace=True)

    assert "cleanup failed" in "\n".join(exc_info.value.__notes__)


def test_storage_transfer_transaction_pull_cleanup_failure_preserves_primary_exception(
    tmp_path,
    monkeypatch,
) -> None:
    remote_storage_root = tmp_path / "remote-storage"
    remote_root = remote_storage_root / "artifacts" / "ethereum" / "artifact-1"
    remote_root.mkdir(parents=True)
    (remote_root / "payload.txt").write_text("artifact payload", encoding="utf-8")
    record = _artifact_record(remote_root)
    session = _FakeSession(
        remote_storage_root,
        record_json=encode_remote_catalog_record(record),
    )
    transaction = StorageTransferTransaction(
        local_storage_root=tmp_path / "outputs",
        session=cast(ExecutionSession, session),
    )
    monkeypatch.setattr(
        "spice.execution.transfer_transaction.promote_root_stage",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("promote failed")),
    )
    monkeypatch.setattr(
        "spice.execution.transfer_transaction.cleanup_root_stage",
        lambda _path: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    with pytest.raises(RuntimeError, match="promote failed") as exc_info:
        transaction.pull_artifact(record.artifact_id, replace=True)

    assert "cleanup failed" in "\n".join(exc_info.value.__notes__)
