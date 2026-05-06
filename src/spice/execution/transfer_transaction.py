"""Storage transfer transaction orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..core.errors import SpiceOperatorError
from ..storage.catalog.codecs import decode_remote_catalog_record
from ..storage.catalog.index import ReindexedCatalogRoot, resolve_dataset_record
from ..storage.catalog.materialization import materialize_catalog_root
from ..storage.catalog.records import CatalogArtifactRecord, CatalogDatasetRecord, CatalogRecord
from ..storage.engine import RootKind
from ..storage.lifecycle import (
    cleanup_root_stage,
    prepare_root_stage,
    promote_root_stage,
    staged_root_path,
)
from ..storage.selectors import DatasetSelector
from .session import ExecutionSession, open_execution_session


@dataclass(frozen=True, slots=True)
class TransferredArtifactRoot:
    source_record: CatalogArtifactRecord
    local_record: CatalogArtifactRecord
    destination_root: Path
    dataset_present: bool


class TransferAdapter(Protocol):
    def prepare_stage(
        self,
        *,
        destination_root: Path,
        replace: bool,
    ) -> Path: ...

    def promote_stage(
        self,
        *,
        destination_root: Path,
        staged_root: Path,
        root_kind: RootKind,
        replace: bool,
    ) -> ReindexedCatalogRoot | None: ...

    def cleanup_stage(self, staged_root: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class LocalStorageTransferAdapter:
    storage_root: Path

    def prepare_stage(
        self,
        *,
        destination_root: Path,
        replace: bool,
    ) -> Path:
        return prepare_root_stage(
            destination_root=destination_root,
            replace=replace,
            purpose="incoming",
        )

    def promote_stage(
        self,
        *,
        destination_root: Path,
        staged_root: Path,
        root_kind: RootKind,
        replace: bool,
    ) -> ReindexedCatalogRoot:
        return promote_root_stage(
            storage_root=self.storage_root,
            destination_root=destination_root,
            staged_root=staged_root,
            expected_root_kind=root_kind,
            replace=replace,
        )

    def cleanup_stage(self, staged_root: Path) -> None:
        cleanup_root_stage(staged_root)


@dataclass(frozen=True, slots=True)
class RemoteStorageTransferAdapter:
    session: ExecutionSession

    @property
    def storage_root(self) -> Path:
        return self.session.target.spec.paths.storage_root

    def prepare_stage(
        self,
        *,
        destination_root: Path,
        replace: bool,
    ) -> Path:
        staged_root = staged_root_path(destination_root, purpose="incoming")
        self.session.run_module(
            "spice.storage.sync_cli",
            [
                "prepare-stage",
                "--destination-root",
                str(destination_root),
                "--staged-root",
                str(staged_root),
                *(["--replace"] if replace else []),
            ],
            check_action=f"prepare stage {destination_root}",
        )
        return staged_root

    def promote_stage(
        self,
        *,
        destination_root: Path,
        staged_root: Path,
        root_kind: RootKind,
        replace: bool,
    ) -> None:
        self.session.run_module(
            "spice.storage.sync_cli",
            [
                "finalize-stage",
                "--storage-root",
                str(self.storage_root),
                "--destination-root",
                str(destination_root),
                "--staged-root",
                str(staged_root),
                "--root-kind",
                root_kind.value,
                *(["--replace"] if replace else []),
            ],
            check_action=f"finalize transfer {destination_root}",
        )

    def cleanup_stage(self, staged_root: Path) -> None:
        self.session.run_module(
            "spice.storage.sync_cli",
            ["cleanup-stage", "--staged-root", str(staged_root)],
            check_action=f"cleanup stage {staged_root}",
        )

    def resolve_artifact_record(self, artifact_id: str) -> CatalogArtifactRecord:
        record = self._resolve_record(
            root_kind=RootKind.ARTIFACT,
            action_label=f"artifact {artifact_id}",
            root_id=artifact_id,
        )
        if not isinstance(record, CatalogArtifactRecord):
            raise SpiceOperatorError("resolved remote record is not an artifact")
        return record

    def _resolve_record(
        self,
        *,
        root_kind: RootKind,
        action_label: str,
        root_id: str,
    ) -> CatalogRecord:
        result = self.session.run_module(
            "spice.storage.sync_cli",
            [
                "resolve-record",
                "--storage-root",
                str(self.storage_root),
                "--root-kind",
                root_kind.value,
                "--root-id",
                root_id,
            ],
            check_action=f"resolve {action_label}",
        )
        return decode_remote_catalog_record(
            result.stdout,
            expected_root_kind=root_kind,
        )


@dataclass(slots=True)
class StorageTransferTransaction:
    local_storage_root: Path
    session: ExecutionSession
    _remote_adapter: RemoteStorageTransferAdapter = field(init=False)
    _local_adapter: LocalStorageTransferAdapter = field(init=False)
    _artifact_cache: dict[str, TransferredArtifactRoot] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        self._remote_adapter = RemoteStorageTransferAdapter(self.session)
        self._local_adapter = LocalStorageTransferAdapter(self.local_storage_root)

    def push_dataset(
        self,
        dataset_id: str,
        *,
        replace: bool = False,
    ) -> CatalogDatasetRecord:
        record = resolve_dataset_record(
            self.local_storage_root,
            selector=DatasetSelector(dataset_id=dataset_id),
        )
        source = materialize_catalog_root(self.local_storage_root, record)
        destination = materialize_catalog_root(
            self._remote_adapter.storage_root,
            record,
        )
        _execute_transfer(
            adapter=self._remote_adapter,
            destination_root=destination.root_path,
            root_kind=RootKind.CORPUS,
            replace=replace,
            rsync=lambda staged_root: self.session.rsync_to(
                source_root=source.root_path,
                destination_root=staged_root,
            ),
        )
        return record

    def pull_artifact(
        self,
        artifact_id: str,
        *,
        replace: bool = True,
    ) -> TransferredArtifactRoot:
        cached = self._artifact_cache.get(artifact_id)
        if cached is not None:
            return cached
        source_record = self._remote_adapter.resolve_artifact_record(artifact_id)
        source = materialize_catalog_root(self._remote_adapter.storage_root, source_record)
        destination = materialize_catalog_root(self.local_storage_root, source_record)
        promoted = _execute_transfer(
            adapter=self._local_adapter,
            destination_root=destination.root_path,
            root_kind=RootKind.ARTIFACT,
            replace=replace,
            rsync=lambda staged_root: self.session.rsync_from(
                source_root=source.root_path,
                destination_root=staged_root,
            ),
        )
        if promoted is None or not isinstance(promoted.record, CatalogArtifactRecord):
            raise SpiceOperatorError("promoted local record is not an artifact")
        transferred = TransferredArtifactRoot(
            source_record=source_record,
            local_record=promoted.record,
            destination_root=destination.root_path,
            dataset_present=materialize_catalog_root(
                self.local_storage_root,
                CatalogDatasetRecord(
                    dataset_id=source_record.dataset_id,
                    dataset_name=source_record.dataset_name,
                    chain_name=source_record.chain_name,
                ),
            ).root_path.exists(),
        )
        self._artifact_cache[artifact_id] = transferred
        return transferred


def open_storage_transfer_transaction(
    target_name: str,
    *,
    local_storage_root: Path,
) -> StorageTransferTransaction:
    return StorageTransferTransaction(
        local_storage_root=local_storage_root,
        session=open_execution_session(target_name),
    )


def _execute_transfer(
    *,
    adapter: TransferAdapter,
    destination_root: Path,
    root_kind: RootKind,
    replace: bool,
    rsync: Callable[[Path], None],
) -> ReindexedCatalogRoot | None:
    staged_root = adapter.prepare_stage(
        destination_root=destination_root,
        replace=replace,
    )
    try:
        rsync(staged_root)
        return adapter.promote_stage(
            destination_root=destination_root,
            staged_root=staged_root,
            root_kind=root_kind,
            replace=replace,
        )
    except Exception as exc:
        _cleanup_preserving_primary(adapter, staged_root, exc)
        raise


def _cleanup_preserving_primary(
    adapter: TransferAdapter,
    staged_root: Path,
    primary: BaseException,
) -> None:
    try:
        adapter.cleanup_stage(staged_root)
    except Exception as cleanup_error:
        primary.add_note(f"cleanup failed: {cleanup_error}")
