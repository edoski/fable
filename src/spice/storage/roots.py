"""Root-level catalog query, delete, and refresh helpers."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from ..core.errors import DeleteBlockedError, SelectorResolutionError, StateLayoutError
from ..core.files import prune_empty_directories, remove_path
from .artifact import load_artifact_manifest
from .catalog import (
    CatalogArtifactRecord,
    CatalogDatasetRecord,
    CatalogStudyRecord,
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
from .catalog.store import (
    ensure_catalog_db,
    upsert_artifact_record,
    upsert_dataset_record,
    upsert_study_record,
)
from .catalog.store import (
    list_artifact_records as _list_artifact_catalog_records,
)
from .catalog.store import (
    list_artifacts_for_dataset as _list_artifacts_for_dataset_catalog_records,
)
from .catalog.store import (
    list_artifacts_for_study as _list_artifacts_for_study_catalog_records,
)
from .catalog.store import (
    list_dataset_records as _list_dataset_catalog_records,
)
from .catalog.store import (
    list_studies_for_dataset as _list_studies_for_dataset_catalog_records,
)
from .catalog.store import (
    list_study_records as _list_study_catalog_records,
)
from .corpus import load_dataset_manifest
from .engine import RootKind, detect_root_kind, state_db_path
from .layout import (
    ARTIFACTS_ROOT_NAME,
    CORPORA_ROOT_NAME,
    STUDIES_ROOT_NAME,
    catalog_db_path,
)
from .study_manifest import load_study_manifest

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DatasetSelector:
    chain_name: str | None = None
    dataset_name: str | None = None


@dataclass(frozen=True, slots=True)
class StudySelector:
    study_id: str | None = None
    chain_name: str | None = None
    dataset_name: str | None = None
    features_id: str | None = None
    prediction_id: str | None = None
    model_id: str | None = None
    problem_id: str | None = None
    study_name: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactSelector:
    artifact_id: str | None = None
    chain_name: str | None = None
    dataset_name: str | None = None
    features_id: str | None = None
    prediction_id: str | None = None
    model_id: str | None = None
    problem_id: str | None = None
    variant: str | None = None
    study_name: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogRefreshSummary:
    dataset_roots: int = 0
    study_roots: int = 0
    artifact_roots: int = 0


SelectorT = TypeVar("SelectorT", DatasetSelector, StudySelector, ArtifactSelector)


def list_dataset_records(
    storage_root: Path,
    *,
    selector: DatasetSelector | None = None,
) -> list[CatalogDatasetRecord]:
    return _list_catalog_records(
        storage_root,
        selector=selector or DatasetSelector(),
        list_records=_list_dataset_catalog_records,
    )


def list_study_records(
    storage_root: Path,
    *,
    selector: StudySelector | None = None,
) -> list[CatalogStudyRecord]:
    return _list_catalog_records(
        storage_root,
        selector=selector or StudySelector(),
        list_records=_list_study_catalog_records,
    )


def list_artifact_records(
    storage_root: Path,
    *,
    selector: ArtifactSelector | None = None,
) -> list[CatalogArtifactRecord]:
    return _list_catalog_records(
        storage_root,
        selector=selector or ArtifactSelector(),
        list_records=_list_artifact_catalog_records,
    )


def list_studies_for_dataset(
    storage_root: Path,
    *,
    dataset_id: str,
) -> list[CatalogStudyRecord]:
    return _list_studies_for_dataset_catalog_records(
        catalog_db_path(storage_root),
        dataset_id=dataset_id,
    )


def list_artifacts_for_dataset(
    storage_root: Path,
    *,
    dataset_id: str,
) -> list[CatalogArtifactRecord]:
    return _list_artifacts_for_dataset_catalog_records(
        catalog_db_path(storage_root),
        dataset_id=dataset_id,
    )


def list_artifacts_for_study(
    storage_root: Path,
    *,
    study_id: str,
) -> list[CatalogArtifactRecord]:
    return _list_artifacts_for_study_catalog_records(
        catalog_db_path(storage_root),
        study_id=study_id,
    )


def resolve_dataset_record(
    storage_root: Path,
    *,
    selector: DatasetSelector | None = None,
) -> CatalogDatasetRecord:
    return _resolve_catalog_record(
        "dataset",
        storage_root,
        selector=selector or DatasetSelector(),
        list_records=_list_dataset_catalog_records,
    )


def resolve_study_record(
    storage_root: Path,
    *,
    selector: StudySelector | None = None,
) -> CatalogStudyRecord:
    return _resolve_catalog_record(
        "study",
        storage_root,
        selector=selector or StudySelector(),
        list_records=_list_study_catalog_records,
    )


def resolve_artifact_record(
    storage_root: Path,
    *,
    selector: ArtifactSelector | None = None,
) -> CatalogArtifactRecord:
    return _resolve_catalog_record(
        "artifact",
        storage_root,
        selector=selector or ArtifactSelector(),
        list_records=_list_artifact_catalog_records,
    )


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
        _delete_artifact_record(storage_root, artifact_record)
    for study_record in dependent_studies:
        _delete_study_record(storage_root, study_record)
    _delete_dataset_record(storage_root, record)
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
        _delete_artifact_record(storage_root, artifact_record)
    _delete_study_record(storage_root, record)
    return record


def delete_artifact_record(
    storage_root: Path,
    *,
    selector: ArtifactSelector | None = None,
    record: CatalogArtifactRecord | None = None,
) -> CatalogArtifactRecord:
    record = record or resolve_artifact_record(storage_root, selector=selector)
    _delete_artifact_record(storage_root, record)
    return record


def reindex_root(storage_root: Path, *, root_path: Path) -> RootKind:
    return _reindex_catalog_root(catalog_db_path(storage_root), root_path=root_path)


def refresh_catalog(storage_root: Path) -> CatalogRefreshSummary:
    catalog_path = catalog_db_path(storage_root)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temp_catalog_path = catalog_path.parent / f".{catalog_path.name}.rebuild.{uuid4().hex}.tmp"
    if temp_catalog_path.exists():
        temp_catalog_path.unlink()
    try:
        ensure_catalog_db(temp_catalog_path)
        counts = {"dataset_roots": 0, "study_roots": 0, "artifact_roots": 0}
        for parent_name, count_key in (
            (CORPORA_ROOT_NAME, "dataset_roots"),
            (STUDIES_ROOT_NAME, "study_roots"),
            (ARTIFACTS_ROOT_NAME, "artifact_roots"),
        ):
            for root_path in _roots_under(storage_root / parent_name):
                _reindex_catalog_root(temp_catalog_path, root_path=root_path)
                counts[count_key] += 1
        os.replace(temp_catalog_path, catalog_path)
        return CatalogRefreshSummary(**counts)
    except Exception:
        temp_catalog_path.unlink(missing_ok=True)
        raise


def _reindex_catalog_root(catalog_path: Path, *, root_path: Path) -> RootKind:
    db_path = state_db_path(root_path)
    root_kind = detect_root_kind(db_path)
    _ROOT_UPSERT_HANDLERS[root_kind](catalog_path, root_path=root_path, db_path=db_path)
    return root_kind


def _delete_artifact_record(storage_root: Path, record: CatalogArtifactRecord) -> None:
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


def _delete_study_record(storage_root: Path, record: CatalogStudyRecord) -> None:
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


def _delete_dataset_record(storage_root: Path, record: CatalogDatasetRecord) -> None:
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


def _delete_catalog_root_record(
    storage_root: Path,
    *,
    root_path: Path,
    catalog_delete: Callable[[Path], None],
    stop_dir_name: str,
    expected_root_kind: RootKind,
) -> None:
    root_path = _validated_catalog_root_path(
        storage_root,
        root_path=root_path,
        stop_dir_name=stop_dir_name,
        expected_root_kind=expected_root_kind,
    )
    remove_path(root_path)
    catalog_delete(catalog_db_path(storage_root))
    prune_empty_directories(root_path.parent, stop_at=storage_root / stop_dir_name)


def _validated_catalog_root_path(
    storage_root: Path,
    *,
    root_path: Path,
    stop_dir_name: str,
    expected_root_kind: RootKind,
) -> Path:
    storage_root = storage_root.resolve()
    expected_parent = (storage_root / stop_dir_name).resolve()
    resolved_root = root_path.resolve()
    try:
        resolved_root.relative_to(expected_parent)
    except ValueError as exc:
        raise StateLayoutError(
            f"Catalog root path is outside storage {stop_dir_name} root: {root_path}"
        ) from exc

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


def _roots_under(parent: Path) -> list[Path]:
    if not parent.exists():
        return []
    roots: list[Path] = []
    chain_dirs = sorted(
        path for path in parent.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    for chain_dir in chain_dirs:
        for root_dir in sorted(
            path for path in chain_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        ):
            if state_db_path(root_dir).is_file():
                roots.append(root_dir)
    return roots


def _resolve_one(label: str, records: list[T]) -> T:
    if len(records) == 1:
        return records[0]
    raise SelectorResolutionError(kind=label, records=records)


def _selector_kwargs(selector: SelectorT) -> dict[str, str]:
    return {
        field.name: value
        for field in fields(selector)
        if (value := getattr(selector, field.name)) is not None
    }


def _list_catalog_records(
    storage_root: Path,
    *,
    selector: SelectorT,
    list_records: Callable[..., list[T]],
) -> list[T]:
    return list_records(catalog_db_path(storage_root), **_selector_kwargs(selector))


def _resolve_catalog_record(
    label: str,
    storage_root: Path,
    *,
    selector: SelectorT,
    list_records: Callable[..., list[T]],
) -> T:
    return _resolve_one(
        label,
        _list_catalog_records(storage_root, selector=selector, list_records=list_records),
    )


def _upsert_corpus_root(catalog_path: Path, *, root_path: Path, db_path: Path) -> None:
    manifest = load_dataset_manifest(db_path)
    upsert_dataset_record(
        catalog_path,
        dataset_id=manifest.dataset.id,
        dataset_name=manifest.dataset.name,
        chain_name=manifest.chain.name,
        root_path=root_path,
        state_db_path=db_path,
    )


def _upsert_study_root(catalog_path: Path, *, root_path: Path, db_path: Path) -> None:
    manifest = load_study_manifest(db_path)
    upsert_study_record(
        catalog_path,
        study_id=manifest.study_id,
        study_name=manifest.study_name,
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        chain_name=manifest.chain_name,
        features_id=manifest.features.id,
        prediction_id=manifest.prediction.id,
        model_id=manifest.model.id,
        problem_id=manifest.problem.id,
        root_path=root_path,
        state_db_path=db_path,
    )


def _upsert_artifact_root(catalog_path: Path, *, root_path: Path, db_path: Path) -> None:
    manifest = load_artifact_manifest(db_path)
    upsert_artifact_record(
        catalog_path,
        artifact_id=manifest.artifact_id,
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        chain_name=manifest.chain_name,
        features_id=manifest.features_id,
        prediction_id=manifest.prediction_id,
        model_id=manifest.model.id,
        problem_id=manifest.problem_id,
        variant=manifest.variant.value,
        study_id=manifest.study_id,
        study_name=None if manifest.study is None else manifest.study.name,
        root_path=root_path,
        state_db_path=db_path,
    )


_ROOT_UPSERT_HANDLERS: dict[RootKind, Callable[..., None]] = {
    RootKind.CORPUS: _upsert_corpus_root,
    RootKind.STUDY: _upsert_study_root,
    RootKind.ARTIFACT: _upsert_artifact_root,
}
