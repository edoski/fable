"""Remote storage transfer helpers."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

from ..core.errors import SpiceOperatorError, StateConflictError
from ..core.files import promote_paths_atomic, remove_path
from ..storage.catalog import CatalogArtifactRecord, CatalogDatasetRecord, CatalogStudyRecord
from ..storage.roots import (
    ArtifactSelector,
    DatasetSelector,
    StudySelector,
    reindex_root,
    resolve_dataset_record,
    resolve_study_record,
)
from .shell import (
    RemoteExecutionTarget,
    ensure_remote_success,
    resolve_remote_target,
    run_remote_command,
    run_remote_module,
    run_rsync_from_remote,
    run_rsync_to_remote,
)

RecordT = TypeVar("RecordT", CatalogDatasetRecord, CatalogStudyRecord)
RemoteRecordT = TypeVar("RemoteRecordT", CatalogStudyRecord, CatalogArtifactRecord)


def push_dataset_to_remote(
    *,
    storage_root: Path,
    selector: DatasetSelector,
    replace: bool,
) -> CatalogDatasetRecord:
    return _push_record_to_remote(
        storage_root=storage_root,
        selector=selector,
        resolve_record=lambda root, selected: resolve_dataset_record(
            root,
            selector=cast(DatasetSelector, selected),
        ),
        destination_root=_remote_dataset_root,
        replace=replace,
    )


def push_study_to_remote(
    *,
    storage_root: Path,
    selector: StudySelector,
    replace: bool,
) -> CatalogStudyRecord:
    return _push_record_to_remote(
        storage_root=storage_root,
        selector=selector,
        resolve_record=lambda root, selected: resolve_study_record(
            root,
            selector=cast(StudySelector, selected),
        ),
        destination_root=_remote_study_root,
        replace=replace,
    )


def pull_artifact_from_remote(
    *,
    storage_root: Path,
    selector: ArtifactSelector,
    replace: bool,
) -> tuple[CatalogArtifactRecord, bool]:
    target = resolve_remote_target()
    record = _resolve_remote_artifact_record(target, selector=selector)
    destination_root = _local_artifact_root(storage_root, record)
    _pull_root_from_remote(
        target=target,
        remote_root=record.root_path,
        local_storage_root=storage_root,
        destination_root=destination_root,
        replace=replace,
    )
    dataset_present = _local_dataset_root(storage_root, record).exists()
    return record, dataset_present


def pull_study_from_remote(
    *,
    storage_root: Path,
    selector: StudySelector,
    replace: bool,
) -> CatalogStudyRecord:
    target = resolve_remote_target()
    record = _resolve_remote_study_record(target, selector=selector)
    _pull_root_from_remote(
        target=target,
        remote_root=record.root_path,
        local_storage_root=storage_root,
        destination_root=_local_study_root(storage_root, record),
        replace=replace,
    )
    return record


def _push_root_to_remote(
    *,
    local_root: Path,
    remote_storage_root: Path,
    destination_root: Path,
    replace: bool,
    target: RemoteExecutionTarget,
) -> None:
    staged_root = destination_root.parent / f".{destination_root.name}.incoming.{uuid4().hex}"
    try:
        _prepare_remote_stage(
            target,
            destination_root=destination_root,
            staged_root=staged_root,
            replace=replace,
        )
        run_rsync_to_remote(target, source_root=local_root, destination_root=staged_root)
        _finalize_remote_stage(
            target,
            remote_storage_root=remote_storage_root,
            destination_root=destination_root,
            staged_root=staged_root,
            replace=replace,
        )
    except Exception:
        _cleanup_remote_path(target, staged_root)
        raise


def _push_record_to_remote(
    *,
    storage_root: Path,
    selector: object,
    resolve_record: Callable[[Path, object], RecordT],
    destination_root: Callable[[Path, RecordT], Path],
    replace: bool,
) -> RecordT:
    target = resolve_remote_target()
    record = resolve_record(storage_root, selector)
    _push_root_to_remote(
        local_root=record.root_path,
        remote_storage_root=target.spec.paths.storage_root,
        destination_root=destination_root(target.spec.paths.storage_root, record),
        replace=replace,
        target=target,
    )
    return record


def _pull_root_from_remote(
    *,
    target: RemoteExecutionTarget,
    remote_root: Path,
    local_storage_root: Path,
    destination_root: Path,
    replace: bool,
) -> None:
    staged_root = destination_root.parent / f".{destination_root.name}.incoming.{uuid4().hex}"
    if destination_root.exists() and not replace:
        raise StateConflictError(f"Destination already exists: {destination_root}")
    staged_root.parent.mkdir(parents=True, exist_ok=True)
    remove_path(staged_root)
    staged_root.mkdir(parents=True, exist_ok=True)
    try:
        run_rsync_from_remote(target, source_root=remote_root, destination_root=staged_root)
        promote_paths_atomic([(destination_root, staged_root)])
        reindex_root(local_storage_root, root_path=destination_root)
    except Exception:
        remove_path(staged_root)
        raise


def _prepare_remote_stage(
    target: RemoteExecutionTarget,
    *,
    destination_root: Path,
    staged_root: Path,
    replace: bool,
) -> None:
    ensure_remote_success(
        run_remote_module(
            target,
            "spice.remote.actions",
            [
                "prepare-stage",
                "--destination-root",
                str(destination_root),
                "--staged-root",
                str(staged_root),
                *(["--replace"] if replace else []),
            ],
        ),
        action=f"prepare remote stage {destination_root}",
    )


def _finalize_remote_stage(
    target: RemoteExecutionTarget,
    *,
    remote_storage_root: Path,
    destination_root: Path,
    staged_root: Path,
    replace: bool,
) -> None:
    ensure_remote_success(
        run_remote_module(
            target,
            "spice.remote.actions",
            [
                "finalize-stage",
                "--storage-root",
                str(remote_storage_root),
                "--destination-root",
                str(destination_root),
                "--staged-root",
                str(staged_root),
                *(["--replace"] if replace else []),
            ],
        ),
        action=f"finalize remote transfer {destination_root}",
    )


def _cleanup_remote_path(target: RemoteExecutionTarget, path: Path) -> None:
    run_remote_command(target, f"rm -rf {shlex.quote(path.as_posix())}")


def _resolve_remote_study_record(
    target: RemoteExecutionTarget,
    *,
    selector: StudySelector,
) -> CatalogStudyRecord:
    return _resolve_remote_record(
        target,
        command="resolve-study-record",
        action_label="StudySelector",
        selector_payload=_selector_payload(selector),
        record_type=CatalogStudyRecord,
    )


def _resolve_remote_artifact_record(
    target: RemoteExecutionTarget,
    *,
    selector: ArtifactSelector,
) -> CatalogArtifactRecord:
    return _resolve_remote_record(
        target,
        command="resolve-artifact-record",
        action_label="ArtifactSelector",
        selector_payload=_selector_payload(selector),
        record_type=CatalogArtifactRecord,
        nullable_fields=frozenset({"study_id", "study_name"}),
    )


def _resolve_remote_record(
    target: RemoteExecutionTarget,
    *,
    command: str,
    action_label: str,
    selector_payload: dict[str, object | None],
    record_type: type[RemoteRecordT],
    nullable_fields: frozenset[str] = frozenset(),
) -> RemoteRecordT:
    payload = _resolve_remote_record_payload(
        target,
        command=command,
        action_label=action_label,
        selector_payload=selector_payload,
    )
    record_payload: dict[str, object] = {}
    for field in fields(record_type):
        value = payload[field.name]
        if field.name in {"root_path", "state_db_path"}:
            record_payload[field.name] = Path(str(value))
        elif field.name in nullable_fields and value is None:
            record_payload[field.name] = None
        else:
            record_payload[field.name] = str(value)
    return cast(RemoteRecordT, record_type(**cast(Any, record_payload)))


def _resolve_remote_record_payload(
    target: RemoteExecutionTarget,
    *,
    command: str,
    action_label: str,
    selector_payload: dict[str, object | None],
) -> dict[str, object | None]:
    result = ensure_remote_success(
        run_remote_module(
            target,
            "spice.remote.actions",
            [
                command,
                "--storage-root",
                str(target.spec.paths.storage_root),
                "--selector-json",
                json.dumps(selector_payload),
            ],
        ),
        action=f"resolve remote {action_label}",
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise SpiceOperatorError("remote selector payload must be a mapping")
    return payload


def _selector_payload(selector: StudySelector | ArtifactSelector) -> dict[str, object | None]:
    return {
        field.name: cast(object | None, getattr(selector, field.name))
        for field in fields(selector)
    }


def _remote_dataset_root(remote_storage_root: Path, record: CatalogDatasetRecord) -> Path:
    return remote_storage_root / "corpora" / record.chain_name / record.dataset_id


def _remote_study_root(remote_storage_root: Path, record: CatalogStudyRecord) -> Path:
    return remote_storage_root / "studies" / record.chain_name / record.study_id


def _local_study_root(storage_root: Path, record: CatalogStudyRecord) -> Path:
    return storage_root / "studies" / record.chain_name / record.study_id


def _local_artifact_root(storage_root: Path, record: CatalogArtifactRecord) -> Path:
    return storage_root / "artifacts" / record.chain_name / record.artifact_id


def _local_dataset_root(storage_root: Path, record: CatalogArtifactRecord) -> Path:
    return storage_root / "corpora" / record.chain_name / record.dataset_id
