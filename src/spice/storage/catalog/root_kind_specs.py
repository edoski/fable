"""Catalog root-kind specs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Table

from ...core.errors import SpiceOperatorError
from ..artifact import load_artifact_manifest
from ..corpus import load_dataset_manifest
from ..engine import RootKind
from ..layout import (
    artifact_root_path,
    corpus_root_path,
    study_root_path,
)
from ..selectors import ArtifactSelector, DatasetSelector, StudySelector
from ..study_manifest import load_study_manifest
from .records import CatalogArtifactRecord, CatalogDatasetRecord, CatalogRecord, CatalogStudyRecord
from .schema import artifact_index, dataset_index, study_index
from .store import (
    upsert_artifact_record,
    upsert_dataset_record,
    upsert_study_record,
)


@dataclass(frozen=True, slots=True)
class CatalogRootKindSpec:
    root_kind: RootKind
    record_type: type[CatalogRecord]
    table: Table
    key_field: str
    root_path: Callable[[Path, CatalogRecord], Path]
    resolve_record: Callable[[Path, str], CatalogRecord]
    build_record: Callable[[Path, Path], CatalogRecord]
    upsert_record: Callable[[Path, CatalogRecord], None]
    path_fields: frozenset[str] = frozenset({"root_path", "state_db_path"})
    nullable_fields: frozenset[str] = frozenset()

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self.record_type))

    def to_record_payload(self, record: CatalogRecord) -> dict[str, object | None]:
        if not isinstance(record, self.record_type):
            raise TypeError(
                f"codec for {self.root_kind} cannot encode {type(record).__name__}"
            )
        return {
            field_name: str(value) if isinstance(value, Path) else value
            for field_name in self.field_names
            for value in (getattr(record, field_name),)
        }

    def from_record_payload(self, payload: dict[str, object | None]) -> CatalogRecord:
        fields_set = set(self.field_names)
        payload_keys = set(payload)
        missing = sorted(fields_set - payload_keys)
        extra = sorted(payload_keys - fields_set)
        if missing:
            raise SpiceOperatorError(
                f"remote catalog record is missing fields: {', '.join(missing)}"
            )
        if extra:
            raise SpiceOperatorError(f"remote catalog record has extra fields: {', '.join(extra)}")
        values: dict[str, object | None] = {}
        for name in self.field_names:
            value = payload[name]
            if value is None:
                if name not in self.nullable_fields:
                    raise SpiceOperatorError(f"remote catalog record field {name} cannot be null")
                values[name] = None
                continue
            if name in self.path_fields:
                values[name] = Path(_require_string(name, value))
            else:
                values[name] = _require_string(name, value)
        return cast(CatalogRecord, self.record_type(**cast(Any, values)))


def _build_dataset_record(root_path: Path, db_path: Path) -> CatalogDatasetRecord:
    manifest = load_dataset_manifest(db_path)
    return CatalogDatasetRecord(
        dataset_id=manifest.dataset.id,
        dataset_name=manifest.dataset.name,
        chain_name=manifest.chain.name,
        root_path=root_path,
        state_db_path=db_path,
    )


def _build_study_record(root_path: Path, db_path: Path) -> CatalogStudyRecord:
    manifest = load_study_manifest(db_path)
    return CatalogStudyRecord(
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


def _build_artifact_record(root_path: Path, db_path: Path) -> CatalogArtifactRecord:
    manifest = load_artifact_manifest(db_path)
    return CatalogArtifactRecord(
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


def _upsert_dataset(catalog_path: Path, record: CatalogRecord) -> None:
    assert isinstance(record, CatalogDatasetRecord)
    upsert_dataset_record(
        catalog_path,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        chain_name=record.chain_name,
        root_path=record.root_path,
        state_db_path=record.state_db_path,
    )


def _upsert_study(catalog_path: Path, record: CatalogRecord) -> None:
    assert isinstance(record, CatalogStudyRecord)
    upsert_study_record(
        catalog_path,
        study_id=record.study_id,
        study_name=record.study_name,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        chain_name=record.chain_name,
        features_id=record.features_id,
        prediction_id=record.prediction_id,
        model_id=record.model_id,
        problem_id=record.problem_id,
        root_path=record.root_path,
        state_db_path=record.state_db_path,
    )


def _upsert_artifact(catalog_path: Path, record: CatalogRecord) -> None:
    assert isinstance(record, CatalogArtifactRecord)
    upsert_artifact_record(
        catalog_path,
        artifact_id=record.artifact_id,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        chain_name=record.chain_name,
        features_id=record.features_id,
        prediction_id=record.prediction_id,
        model_id=record.model_id,
        problem_id=record.problem_id,
        variant=record.variant,
        study_id=record.study_id,
        study_name=record.study_name,
        root_path=record.root_path,
        state_db_path=record.state_db_path,
    )


def _dataset_root_path(storage_root: Path, record: CatalogRecord) -> Path:
    assert isinstance(record, CatalogDatasetRecord)
    return corpus_root_path(
        storage_root,
        chain_name=record.chain_name,
        corpus_id=record.dataset_id,
    )


def _study_root_path(storage_root: Path, record: CatalogRecord) -> Path:
    assert isinstance(record, CatalogStudyRecord)
    return study_root_path(
        storage_root,
        chain_name=record.chain_name,
        study_id=record.study_id,
    )


def _artifact_root_path(storage_root: Path, record: CatalogRecord) -> Path:
    assert isinstance(record, CatalogArtifactRecord)
    return artifact_root_path(
        storage_root,
        chain_name=record.chain_name,
        artifact_id=record.artifact_id,
    )


def _resolve_dataset(storage_root: Path, root_id: str) -> CatalogDatasetRecord:
    from .index import resolve_dataset_record

    return resolve_dataset_record(storage_root, selector=DatasetSelector(dataset_id=root_id))


def _resolve_study(storage_root: Path, root_id: str) -> CatalogStudyRecord:
    from .index import resolve_study_record

    return resolve_study_record(storage_root, selector=StudySelector(study_id=root_id))


def _resolve_artifact(storage_root: Path, root_id: str) -> CatalogArtifactRecord:
    from .index import resolve_artifact_record

    return resolve_artifact_record(storage_root, selector=ArtifactSelector(artifact_id=root_id))


DATASET_ROOT_SPEC = CatalogRootKindSpec(
    root_kind=RootKind.CORPUS,
    record_type=CatalogDatasetRecord,
    table=dataset_index,
    key_field="dataset_id",
    root_path=_dataset_root_path,
    resolve_record=_resolve_dataset,
    build_record=_build_dataset_record,
    upsert_record=_upsert_dataset,
)
STUDY_ROOT_SPEC = CatalogRootKindSpec(
    root_kind=RootKind.STUDY,
    record_type=CatalogStudyRecord,
    table=study_index,
    key_field="study_id",
    root_path=_study_root_path,
    resolve_record=_resolve_study,
    build_record=_build_study_record,
    upsert_record=_upsert_study,
)
ARTIFACT_ROOT_SPEC = CatalogRootKindSpec(
    root_kind=RootKind.ARTIFACT,
    record_type=CatalogArtifactRecord,
    table=artifact_index,
    key_field="artifact_id",
    root_path=_artifact_root_path,
    resolve_record=_resolve_artifact,
    build_record=_build_artifact_record,
    upsert_record=_upsert_artifact,
    nullable_fields=frozenset({"study_id", "study_name"}),
)

_SPECS_BY_ROOT_KIND = {
    RootKind.CORPUS: DATASET_ROOT_SPEC,
    RootKind.STUDY: STUDY_ROOT_SPEC,
    RootKind.ARTIFACT: ARTIFACT_ROOT_SPEC,
}
_ROOT_KIND_BY_RECORD_TYPE = {
    CatalogDatasetRecord: RootKind.CORPUS,
    CatalogStudyRecord: RootKind.STUDY,
    CatalogArtifactRecord: RootKind.ARTIFACT,
}


def spec_for_root_kind(root_kind: RootKind) -> CatalogRootKindSpec:
    return _SPECS_BY_ROOT_KIND[root_kind]


def spec_for_record(record: CatalogRecord) -> CatalogRootKindSpec:
    return spec_for_root_kind(_ROOT_KIND_BY_RECORD_TYPE[type(record)])


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise SpiceOperatorError(f"remote catalog record field {name} must be a string")
    return value
