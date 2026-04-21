"""Storage sync helpers."""

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
from ..execution.slurm_ssh import (
    ExecutionTarget,
    ensure_execution_success,
    load_execution_target,
    run_execution_command,
    run_execution_module,
    run_rsync_from_execution_target,
    run_rsync_to_execution_target,
)
from .catalog import CatalogArtifactRecord, CatalogDatasetRecord, CatalogStudyRecord
from .roots import (
    ArtifactSelector,
    DatasetSelector,
    StudySelector,
    reindex_root,
    resolve_artifact_record,
    resolve_dataset_record,
    resolve_study_record,
)

RecordT = TypeVar("RecordT", CatalogDatasetRecord, CatalogStudyRecord)
RemoteRecordT = TypeVar("RemoteRecordT", CatalogStudyRecord, CatalogArtifactRecord)


def push_dataset_to_cluster(
    *,
    storage_root: Path,
    selector: DatasetSelector,
    replace: bool,
) -> CatalogDatasetRecord:
    return _push_record_to_cluster(
        storage_root=storage_root,
        selector=selector,
        resolve_record=lambda root, selected: resolve_dataset_record(
            root,
            selector=cast(DatasetSelector, selected),
        ),
        destination_root=_cluster_dataset_root,
        replace=replace,
    )


def push_study_to_cluster(
    *,
    storage_root: Path,
    selector: StudySelector,
    replace: bool,
) -> CatalogStudyRecord:
    return _push_record_to_cluster(
        storage_root=storage_root,
        selector=selector,
        resolve_record=lambda root, selected: resolve_study_record(
            root,
            selector=cast(StudySelector, selected),
        ),
        destination_root=_cluster_study_root,
        replace=replace,
    )


def pull_artifact_from_cluster(
    *,
    storage_root: Path,
    selector: ArtifactSelector,
    replace: bool,
) -> tuple[CatalogArtifactRecord, bool]:
    target = load_execution_target()
    record = _resolve_cluster_artifact_record(target, selector=selector)
    destination_root = _local_artifact_root(storage_root, record)
    _pull_root_from_cluster(
        target=target,
        cluster_root=record.root_path,
        local_storage_root=storage_root,
        destination_root=destination_root,
        replace=replace,
    )
    dataset_present = _local_dataset_root(storage_root, record).exists()
    return record, dataset_present


def pull_study_from_cluster(
    *,
    storage_root: Path,
    selector: StudySelector,
    replace: bool,
) -> CatalogStudyRecord:
    target = load_execution_target()
    record = _resolve_cluster_study_record(target, selector=selector)
    _pull_root_from_cluster(
        target=target,
        cluster_root=record.root_path,
        local_storage_root=storage_root,
        destination_root=_local_study_root(storage_root, record),
        replace=replace,
    )
    return record


def _push_root_to_cluster(
    *,
    local_root: Path,
    cluster_storage_root: Path,
    destination_root: Path,
    replace: bool,
    target: ExecutionTarget,
) -> None:
    staged_root = destination_root.parent / f".{destination_root.name}.incoming.{uuid4().hex}"
    try:
        _prepare_cluster_stage(
            target,
            destination_root=destination_root,
            staged_root=staged_root,
            replace=replace,
        )
        run_rsync_to_execution_target(
            target,
            source_root=local_root,
            destination_root=staged_root,
        )
        _finalize_cluster_stage(
            target,
            cluster_storage_root=cluster_storage_root,
            destination_root=destination_root,
            staged_root=staged_root,
            replace=replace,
        )
    except Exception:
        _cleanup_cluster_path(target, staged_root)
        raise


def _push_record_to_cluster(
    *,
    storage_root: Path,
    selector: object,
    resolve_record: Callable[[Path, object], RecordT],
    destination_root: Callable[[Path, RecordT], Path],
    replace: bool,
) -> RecordT:
    target = load_execution_target()
    record = resolve_record(storage_root, selector)
    _push_root_to_cluster(
        local_root=record.root_path,
        cluster_storage_root=target.spec.paths.storage_root,
        destination_root=destination_root(target.spec.paths.storage_root, record),
        replace=replace,
        target=target,
    )
    return record


def _pull_root_from_cluster(
    *,
    target: ExecutionTarget,
    cluster_root: Path,
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
        run_rsync_from_execution_target(
            target,
            source_root=cluster_root,
            destination_root=staged_root,
        )
        promote_paths_atomic([(destination_root, staged_root)])
        reindex_root(local_storage_root, root_path=destination_root)
    except Exception:
        remove_path(staged_root)
        raise


def _prepare_cluster_stage(
    target: ExecutionTarget,
    *,
    destination_root: Path,
    staged_root: Path,
    replace: bool,
) -> None:
    ensure_execution_success(
        run_execution_module(
            target,
            "spice.storage.sync_cli",
            [
                "prepare-stage",
                "--destination-root",
                str(destination_root),
                "--staged-root",
                str(staged_root),
                *(["--replace"] if replace else []),
            ],
        ),
        action=f"prepare stage {destination_root}",
    )


def _finalize_cluster_stage(
    target: ExecutionTarget,
    *,
    cluster_storage_root: Path,
    destination_root: Path,
    staged_root: Path,
    replace: bool,
) -> None:
    ensure_execution_success(
        run_execution_module(
            target,
            "spice.storage.sync_cli",
            [
                "finalize-stage",
                "--storage-root",
                str(cluster_storage_root),
                "--destination-root",
                str(destination_root),
                "--staged-root",
                str(staged_root),
                *(["--replace"] if replace else []),
            ],
        ),
        action=f"finalize transfer {destination_root}",
    )


def _cleanup_cluster_path(target: ExecutionTarget, path: Path) -> None:
    run_execution_command(target, f"rm -rf {shlex.quote(path.as_posix())}")


def _resolve_cluster_study_record(
    target: ExecutionTarget,
    *,
    selector: StudySelector,
) -> CatalogStudyRecord:
    return _resolve_cluster_record(
        target,
        command="resolve-study-record",
        action_label="StudySelector",
        selector_payload=_selector_payload(selector),
        record_type=CatalogStudyRecord,
    )


def _resolve_cluster_artifact_record(
    target: ExecutionTarget,
    *,
    selector: ArtifactSelector,
) -> CatalogArtifactRecord:
    return _resolve_cluster_record(
        target,
        command="resolve-artifact-record",
        action_label="ArtifactSelector",
        selector_payload=_selector_payload(selector),
        record_type=CatalogArtifactRecord,
        nullable_fields=frozenset({"study_id", "study_name"}),
    )


def _resolve_cluster_record(
    target: ExecutionTarget,
    *,
    command: str,
    action_label: str,
    selector_payload: dict[str, object | None],
    record_type: type[RemoteRecordT],
    nullable_fields: frozenset[str] = frozenset(),
) -> RemoteRecordT:
    payload = _resolve_cluster_record_payload(
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
    return cast(RemoteRecordT, record_type(**cast(dict[str, Any], record_payload)))


def _resolve_cluster_record_payload(
    target: ExecutionTarget,
    *,
    command: str,
    action_label: str,
    selector_payload: dict[str, object | None],
) -> dict[str, object | None]:
    result = ensure_execution_success(
        run_execution_module(
            target,
            "spice.storage.sync_cli",
            [
                command,
                "--storage-root",
                str(target.spec.paths.storage_root),
                "--selector-json",
                json.dumps(selector_payload),
            ],
        ),
        action=f"resolve {action_label}",
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise SpiceOperatorError("selector payload must be a mapping")
    return payload


def _selector_payload(selector: StudySelector | ArtifactSelector) -> dict[str, object | None]:
    return {
        field.name: cast(object | None, getattr(selector, field.name)) for field in fields(selector)
    }


def _cluster_dataset_root(cluster_storage_root: Path, record: CatalogDatasetRecord) -> Path:
    return cluster_storage_root / "corpora" / record.chain_name / record.dataset_id


def _cluster_study_root(cluster_storage_root: Path, record: CatalogStudyRecord) -> Path:
    return cluster_storage_root / "studies" / record.chain_name / record.study_id


def _local_study_root(storage_root: Path, record: CatalogStudyRecord) -> Path:
    return storage_root / "studies" / record.chain_name / record.study_id


def _local_artifact_root(storage_root: Path, record: CatalogArtifactRecord) -> Path:
    return storage_root / "artifacts" / record.chain_name / record.artifact_id


def _local_dataset_root(storage_root: Path, record: CatalogArtifactRecord) -> Path:
    return storage_root / "corpora" / record.chain_name / record.dataset_id
