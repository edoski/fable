"""Storage root lifecycle operations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from ..core.errors import DeleteBlockedError, StateConflictError, StateLayoutError
from ..core.files import (
    prune_empty_directories,
    remove_path,
    replace_path_atomic,
    replace_paths_atomic,
)
from .catalog import CatalogArtifactRecord, CatalogDatasetRecord, CatalogStudyRecord
from .catalog.index import (
    ReindexedCatalogRoot,
    list_artifacts_for_dataset,
    list_artifacts_for_study,
    list_studies_for_dataset,
    reindex_catalog_root,
    resolve_artifact_record,
    resolve_dataset_record,
    resolve_study_record,
)
from .catalog.store import (
    delete_artifact_record as _delete_artifact_catalog_record,
)
from .catalog.store import (
    delete_dataset_record as _delete_dataset_catalog_record,
)
from .catalog.store import (
    delete_study_record as _delete_study_catalog_record,
)
from .engine import RootKind, detect_root_kind, require_root_kind, state_db_path
from .layout import ARTIFACTS_ROOT_NAME, CORPORA_ROOT_NAME, STUDIES_ROOT_NAME, catalog_db_path
from .selectors import ArtifactSelector, DatasetSelector, StudySelector


@dataclass(slots=True)
class RootStage:
    storage_root: Path
    destination_root: Path
    staged_root: Path
    expected_root_kind: RootKind
    replace: bool
    _promoted: bool = False

    def promote(self) -> ReindexedCatalogRoot:
        reindexed = promote_root_stage(
            storage_root=self.storage_root,
            destination_root=self.destination_root,
            staged_root=self.staged_root,
            expected_root_kind=self.expected_root_kind,
            replace=self.replace,
        )
        self._promoted = True
        return reindexed


@dataclass(slots=True)
class PartialRootCommit:
    storage_root: Path
    root_path: Path
    promotions: list[tuple[Path, Path]] = field(default_factory=list)

    def add(self, target: Path, source: Path | None) -> None:
        if source is not None:
            self.promotions.append((target, source))

    def commit(self) -> ReindexedCatalogRoot:
        if self.promotions:
            replace_paths_atomic(self.promotions, replace=True)
        return reindex_catalog_root(self.storage_root, root_path=self.root_path)


def staged_root_path(destination_root: Path, *, purpose: str = "staging") -> Path:
    return destination_root.parent / f".{destination_root.name}.{purpose}.{uuid4().hex}"


def prepare_root_stage(
    *,
    destination_root: Path,
    staged_root: Path | None = None,
    replace: bool,
    purpose: str = "staging",
) -> Path:
    if destination_root.exists() and not replace:
        raise StateConflictError(f"Destination already exists: {destination_root}")
    resolved = staged_root or staged_root_path(destination_root, purpose=purpose)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    remove_path(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def cleanup_root_stage(staged_root: Path, *, prune_stop_at: Path | None = None) -> None:
    remove_path(staged_root)
    prune_empty_directories(staged_root.parent, stop_at=prune_stop_at)


def promote_root_stage(
    *,
    storage_root: Path,
    destination_root: Path,
    staged_root: Path,
    expected_root_kind: RootKind,
    replace: bool,
) -> ReindexedCatalogRoot:
    validate_root_destination_path(
        storage_root,
        destination_root=destination_root,
        expected_root_kind=expected_root_kind,
    )
    require_root_kind(state_db_path(staged_root), expected_root_kind)
    replace_path_atomic(destination_root, staged_root, replace=replace)
    return reindex_catalog_root(storage_root, root_path=destination_root)


def validate_root_destination_path(
    storage_root: Path,
    *,
    destination_root: Path,
    expected_root_kind: RootKind,
) -> Path:
    expected_parent = (storage_root / _root_kind_parent_name(expected_root_kind)).resolve()
    resolved_destination = destination_root.resolve()
    try:
        relative = resolved_destination.relative_to(expected_parent)
    except ValueError as exc:
        raise StateLayoutError(
            f"Root destination is outside the {expected_root_kind.value} storage subtree: "
            f"{destination_root}"
        ) from exc
    if len(relative.parts) != 2:
        raise StateLayoutError(
            f"Root destination must use canonical <chain>/<root-id> layout under "
            f"{expected_parent}: {destination_root}"
        )
    return resolved_destination


def _root_kind_parent_name(root_kind: RootKind) -> str:
    if root_kind is RootKind.CORPUS:
        return CORPORA_ROOT_NAME
    if root_kind is RootKind.STUDY:
        return STUDIES_ROOT_NAME
    if root_kind is RootKind.ARTIFACT:
        return ARTIFACTS_ROOT_NAME
    raise ValueError(f"Unsupported root kind: {root_kind}")


@contextmanager
def staged_root(
    *,
    storage_root: Path,
    destination_root: Path,
    expected_root_kind: RootKind,
    replace: bool = True,
    purpose: str = "staging",
    prune_stop_at: Path | None = None,
) -> Iterator[RootStage]:
    stage_path = prepare_root_stage(
        destination_root=destination_root,
        replace=replace,
        purpose=purpose,
    )
    stage = RootStage(
        storage_root=storage_root,
        destination_root=destination_root,
        staged_root=stage_path,
        expected_root_kind=expected_root_kind,
        replace=replace,
    )
    try:
        yield stage
    finally:
        if not stage._promoted:
            cleanup_root_stage(stage_path, prune_stop_at=prune_stop_at)


def delete_dataset_record(
    storage_root: Path,
    *,
    selector: DatasetSelector | None = None,
    record: CatalogDatasetRecord | None = None,
    cascade: bool = False,
) -> CatalogDatasetRecord:
    record = record or resolve_dataset_record(storage_root, selector=selector)
    dependent_artifacts = list_artifacts_for_dataset(storage_root, dataset_id=record.dataset_id)
    dependent_studies = list_studies_for_dataset(storage_root, dataset_id=record.dataset_id)
    if (dependent_artifacts or dependent_studies) and not cascade:
        raise DeleteBlockedError(
            message="Dataset has dependent studies or artifacts. Re-run with --cascade.",
            artifact_records=dependent_artifacts,
            study_records=dependent_studies,
        )
    for artifact_record in dependent_artifacts:
        delete_catalog_artifact_root(storage_root, artifact_record)
    for study_record in dependent_studies:
        delete_catalog_study_root(storage_root, study_record)
    delete_catalog_dataset_root(storage_root, record)
    return record


def delete_study_record(
    storage_root: Path,
    *,
    selector: StudySelector | None = None,
    record: CatalogStudyRecord | None = None,
    cascade: bool = False,
) -> CatalogStudyRecord:
    record = record or resolve_study_record(storage_root, selector=selector)
    dependent_artifacts = list_artifacts_for_study(storage_root, study_id=record.study_id)
    if dependent_artifacts and not cascade:
        raise DeleteBlockedError(
            message="Study has dependent artifacts. Re-run with --cascade.",
            artifact_records=dependent_artifacts,
        )
    for artifact_record in dependent_artifacts:
        delete_catalog_artifact_root(storage_root, artifact_record)
    delete_catalog_study_root(storage_root, record)
    return record


def delete_artifact_record(
    storage_root: Path,
    *,
    selector: ArtifactSelector | None = None,
    record: CatalogArtifactRecord | None = None,
) -> CatalogArtifactRecord:
    record = record or resolve_artifact_record(storage_root, selector=selector)
    delete_catalog_artifact_root(storage_root, record)
    return record


def delete_catalog_artifact_root(storage_root: Path, record: CatalogArtifactRecord) -> None:
    _delete_catalog_root_record(
        storage_root,
        root_path=record.root_path,
        catalog_delete=lambda catalog_path: _delete_artifact_catalog_record(
            catalog_path,
            artifact_id=record.artifact_id,
        ),
        stop_dir_name=ARTIFACTS_ROOT_NAME,
        expected_root_kind=RootKind.ARTIFACT,
    )


def delete_catalog_study_root(storage_root: Path, record: CatalogStudyRecord) -> None:
    _delete_catalog_root_record(
        storage_root,
        root_path=record.root_path,
        catalog_delete=lambda catalog_path: _delete_study_catalog_record(
            catalog_path,
            study_id=record.study_id,
        ),
        stop_dir_name=STUDIES_ROOT_NAME,
        expected_root_kind=RootKind.STUDY,
    )


def delete_catalog_dataset_root(storage_root: Path, record: CatalogDatasetRecord) -> None:
    _delete_catalog_root_record(
        storage_root,
        root_path=record.root_path,
        catalog_delete=lambda catalog_path: _delete_dataset_catalog_record(
            catalog_path,
            dataset_id=record.dataset_id,
        ),
        stop_dir_name=CORPORA_ROOT_NAME,
        expected_root_kind=RootKind.CORPUS,
    )


def validated_catalog_root_path(
    storage_root: Path,
    *,
    root_path: Path,
    expected_root_kind: RootKind,
) -> Path:
    resolved_root = validate_root_destination_path(
        storage_root,
        destination_root=root_path,
        expected_root_kind=expected_root_kind,
    )

    db_path = state_db_path(resolved_root)
    if not db_path.is_file():
        raise StateLayoutError(f"Catalog root is missing state DB: {db_path}")
    actual_root_kind = detect_root_kind(db_path)
    if actual_root_kind is not expected_root_kind:
        raise StateLayoutError(
            "Catalog root kind mismatch: "
            f"expected {expected_root_kind}, got {actual_root_kind}"
        )
    return resolved_root


def _delete_catalog_root_record(
    storage_root: Path,
    *,
    root_path: Path,
    catalog_delete: Callable[[Path], None],
    stop_dir_name: str,
    expected_root_kind: RootKind,
) -> None:
    root_path = validated_catalog_root_path(
        storage_root,
        root_path=root_path,
        expected_root_kind=expected_root_kind,
    )
    remove_path(root_path)
    catalog_delete(catalog_db_path(storage_root))
    prune_empty_directories(root_path.parent, stop_at=storage_root / stop_dir_name)
