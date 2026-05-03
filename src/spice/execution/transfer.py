"""Remote storage transfer orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.errors import SpiceOperatorError
from ..storage.catalog import CatalogArtifactRecord, CatalogDatasetRecord
from ..storage.catalog.codecs import decode_remote_catalog_record
from ..storage.catalog.index import ReindexedCatalogRoot, resolve_dataset_record
from ..storage.catalog.records import CatalogRecord
from ..storage.catalog.registry import spec_for_root_kind
from ..storage.engine import RootKind
from ..storage.layout import corpus_root_path
from ..storage.lifecycle import (
    cleanup_root_stage,
    prepare_root_stage,
    promote_root_stage,
    staged_root_path,
)
from ..storage.selectors import DatasetSelector
from .session import ExecutionSession


@dataclass(frozen=True, slots=True)
class PulledArtifactRoot:
    source_record: CatalogArtifactRecord
    local_record: CatalogArtifactRecord
    destination_root: Path
    dataset_present: bool


def push_dataset_to_cluster(
    *,
    storage_root: Path,
    session: ExecutionSession,
    dataset_id: str,
    replace: bool,
) -> CatalogDatasetRecord:
    record = resolve_dataset_record(
        storage_root,
        selector=DatasetSelector(dataset_id=dataset_id),
    )
    _push_root_to_cluster(
        session=session,
        local_root=record.root_path,
        destination_root=spec_for_root_kind(RootKind.CORPUS).root_path(
            session.target.spec.paths.storage_root,
            record,
        ),
        root_kind=RootKind.CORPUS,
        replace=replace,
    )
    return record


def pull_artifact_from_cluster(
    *,
    storage_root: Path,
    session: ExecutionSession,
    artifact_id: str,
    replace: bool,
) -> PulledArtifactRoot:
    record = _resolve_cluster_artifact_record(session, artifact_id=artifact_id)
    destination_root = spec_for_root_kind(RootKind.ARTIFACT).root_path(storage_root, record)
    promoted = _pull_root_from_cluster(
        session=session,
        cluster_root=record.root_path,
        local_storage_root=storage_root,
        destination_root=destination_root,
        root_kind=RootKind.ARTIFACT,
        replace=replace,
    )
    if not isinstance(promoted.record, CatalogArtifactRecord):
        raise SpiceOperatorError("promoted local record is not an artifact")
    dataset_present = _local_dataset_root(storage_root, record).exists()
    return PulledArtifactRoot(
        source_record=record,
        local_record=promoted.record,
        destination_root=destination_root,
        dataset_present=dataset_present,
    )


def _push_root_to_cluster(
    *,
    session: ExecutionSession,
    local_root: Path,
    destination_root: Path,
    root_kind: RootKind,
    replace: bool,
) -> None:
    staged_root = staged_root_path(destination_root, purpose="incoming")
    try:
        _prepare_cluster_stage(
            session,
            destination_root=destination_root,
            staged_root=staged_root,
            replace=replace,
        )
        session.rsync_to(source_root=local_root, destination_root=staged_root)
        _finalize_cluster_stage(
            session,
            cluster_storage_root=session.target.spec.paths.storage_root,
            destination_root=destination_root,
            staged_root=staged_root,
            root_kind=root_kind,
            replace=replace,
        )
    except Exception as exc:
        _cleanup_cluster_path_preserving_primary(session, staged_root, exc)
        raise


def _pull_root_from_cluster(
    *,
    session: ExecutionSession,
    cluster_root: Path,
    local_storage_root: Path,
    destination_root: Path,
    root_kind: RootKind,
    replace: bool,
) -> ReindexedCatalogRoot:
    staged_root = prepare_root_stage(
        destination_root=destination_root,
        replace=replace,
        purpose="incoming",
    )
    try:
        session.rsync_from(source_root=cluster_root, destination_root=staged_root)
        return promote_root_stage(
            storage_root=local_storage_root,
            destination_root=destination_root,
            staged_root=staged_root,
            expected_root_kind=root_kind,
            replace=replace,
        )
    except Exception as exc:
        _cleanup_local_path_preserving_primary(staged_root, exc)
        raise


def _prepare_cluster_stage(
    session: ExecutionSession,
    *,
    destination_root: Path,
    staged_root: Path,
    replace: bool,
) -> None:
    session.run_module(
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


def _finalize_cluster_stage(
    session: ExecutionSession,
    *,
    cluster_storage_root: Path,
    destination_root: Path,
    staged_root: Path,
    root_kind: RootKind,
    replace: bool,
) -> None:
    session.run_module(
        "spice.storage.sync_cli",
        [
            "finalize-stage",
            "--storage-root",
            str(cluster_storage_root),
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


def _cleanup_cluster_path(session: ExecutionSession, path: Path) -> None:
    session.run_module(
        "spice.storage.sync_cli",
        ["cleanup-stage", "--staged-root", str(path)],
        check_action=f"cleanup stage {path}",
    )


def _resolve_cluster_artifact_record(
    session: ExecutionSession,
    *,
    artifact_id: str,
) -> CatalogArtifactRecord:
    record = _resolve_cluster_record(
        session,
        root_kind=RootKind.ARTIFACT,
        action_label=f"artifact {artifact_id}",
        root_id=artifact_id,
    )
    if not isinstance(record, CatalogArtifactRecord):
        raise SpiceOperatorError("resolved remote record is not an artifact")
    return record


def _resolve_cluster_record(
    session: ExecutionSession,
    *,
    root_kind: RootKind,
    action_label: str,
    root_id: str,
) -> CatalogRecord:
    result = session.run_module(
        "spice.storage.sync_cli",
        [
            "resolve-record",
            "--storage-root",
            str(session.target.spec.paths.storage_root),
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


def _cleanup_cluster_path_preserving_primary(
    session: ExecutionSession,
    path: Path,
    primary: BaseException,
) -> None:
    try:
        _cleanup_cluster_path(session, path)
    except Exception as cleanup_error:
        primary.add_note(f"cleanup failed: {cleanup_error}")


def _cleanup_local_path_preserving_primary(path: Path, primary: BaseException) -> None:
    try:
        cleanup_root_stage(path)
    except Exception as cleanup_error:
        primary.add_note(f"cleanup failed: {cleanup_error}")


def _local_dataset_root(storage_root: Path, record: CatalogArtifactRecord) -> Path:
    return corpus_root_path(storage_root, chain_name=record.chain_name, corpus_id=record.dataset_id)
