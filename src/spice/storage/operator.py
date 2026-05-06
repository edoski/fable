"""Storage operator queries and commands."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias, cast

from ..core.errors import SpiceOperatorError
from .catalog.index import (
    CatalogRefreshSummary,
    list_artifact_records,
    list_dataset_records,
    list_study_records,
    refresh_catalog,
)
from .catalog.materialization import materialize_catalog_root
from .catalog.records import (
    CatalogArtifactRecord,
    CatalogDatasetRecord,
    CatalogRecord,
    CatalogStudyRecord,
)
from .errors import DeleteBlockedError
from .inspect import describe_root, sectioned_summary
from .inspect_artifact import artifact_list_sections
from .inspect_dataset import dataset_list_sections
from .inspect_study import study_list_sections
from .lifecycle import delete_artifact_record, delete_dataset_record, delete_study_record
from .selectors import ArtifactSelector, DatasetSelector, StudySelector

StorageRootKind: TypeAlias = Literal["dataset", "study", "artifact"]
StorageSelector: TypeAlias = DatasetSelector | StudySelector | ArtifactSelector
StorageShowOutcome: TypeAlias = "StorageShowRendered | StorageShowFailure"
StorageDeleteOutcome: TypeAlias = "StorageDeleteCompleted | StorageDeleteFailure"
SectionRows: TypeAlias = list[tuple[str, list[tuple[str, str]]]]


class DatasetInspectionDetail(StrEnum):
    RUNS = "runs"


class StudyInspectionDetail(StrEnum):
    TRIALS = "trials"
    CONFIG = "config"


class ArtifactInspectionDetail(StrEnum):
    EPOCHS = "epochs"
    RUNS = "runs"


@dataclass(frozen=True, slots=True)
class RenderableSections:
    title: str
    sections: SectionRows


@dataclass(frozen=True, slots=True)
class StorageShowQuery:
    storage_root: Path
    kind: StorageRootKind
    selector: StorageSelector
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class StorageDeleteCommand:
    storage_root: Path
    kind: StorageRootKind
    selector: StorageSelector
    cascade: bool = False


@dataclass(frozen=True, slots=True)
class StorageShowRendered:
    renderable: RenderableSections


@dataclass(frozen=True, slots=True)
class StorageShowFailure:
    message: str
    diagnostics: tuple[RenderableSections, ...] = ()
    narrowing_attributes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StorageDeleteCompleted:
    record: CatalogRecord


@dataclass(frozen=True, slots=True)
class StorageDeleteFailure:
    message: str
    diagnostics: tuple[RenderableSections, ...] = ()
    narrowing_attributes: tuple[str, ...] = ()
    cascade_available: bool = False


@dataclass(frozen=True, slots=True)
class _OperatorSpec:
    kind: StorageRootKind
    valid_details: frozenset[str]
    narrowing_attributes: tuple[str, ...]
    list_sections: Callable[[list], SectionRows]


_SPECS: dict[StorageRootKind, _OperatorSpec] = {
    "dataset": _OperatorSpec(
        kind="dataset",
        valid_details=frozenset(detail.value for detail in DatasetInspectionDetail),
        narrowing_attributes=("chain_name", "dataset_name"),
        list_sections=dataset_list_sections,
    ),
    "study": _OperatorSpec(
        kind="study",
        valid_details=frozenset(detail.value for detail in StudyInspectionDetail),
        narrowing_attributes=(
            "chain_name",
            "dataset_name",
            "features_id",
            "prediction_id",
            "model_id",
            "problem_id",
            "study_name",
        ),
        list_sections=study_list_sections,
    ),
    "artifact": _OperatorSpec(
        kind="artifact",
        valid_details=frozenset(detail.value for detail in ArtifactInspectionDetail),
        narrowing_attributes=(
            "chain_name",
            "dataset_name",
            "features_id",
            "prediction_id",
            "model_id",
            "problem_id",
            "variant",
            "study_name",
        ),
        list_sections=artifact_list_sections,
    ),
}


def show_storage(query: StorageShowQuery) -> StorageShowOutcome:
    spec = _SPECS[query.kind]
    if query.detail is not None and query.detail not in spec.valid_details:
        return StorageShowFailure(
            message=f"Unsupported {query.kind} detail: {query.detail}",
        )
    records = _list_records(
        query.storage_root,
        kind=query.kind,
        selector=query.selector,
    )
    if not records:
        return StorageShowFailure(message=f"No {query.kind} matches found")
    if query.detail is not None and len(records) != 1:
        return StorageShowFailure(
            message=f"--detail requires exactly one {query.kind} match",
            diagnostics=(_matches_renderable(spec, records),),
            narrowing_attributes=_narrowing_attributes(spec, records, query.selector),
        )
    if query.detail is not None:
        return _describe_record(query.storage_root, records[0], detail=query.detail)
    if not _selector_has_filters(query.selector) or len(records) != 1:
        return StorageShowRendered(_list_renderable(spec, records))
    return _describe_record(query.storage_root, records[0], detail=None)


def delete_storage(command: StorageDeleteCommand) -> StorageDeleteOutcome:
    spec = _SPECS[command.kind]
    records = _list_records(
        command.storage_root,
        kind=command.kind,
        selector=command.selector,
    )
    if len(records) != 1:
        return StorageDeleteFailure(
            message=(
                f"Expected exactly one {command.kind} match"
                if records
                else f"No {command.kind} matches found"
            ),
            diagnostics=() if not records else (_matches_renderable(spec, records),),
            narrowing_attributes=_narrowing_attributes(spec, records, command.selector),
        )
    record = records[0]
    try:
        if command.kind == "dataset":
            deleted = delete_dataset_record(
                command.storage_root,
                record=_dataset_record(record),
                cascade=command.cascade,
            )
        elif command.kind == "study":
            deleted = delete_study_record(
                command.storage_root,
                record=_study_record(record),
                cascade=command.cascade,
            )
        else:
            deleted = delete_artifact_record(
                command.storage_root,
                record=_artifact_record(record),
            )
    except DeleteBlockedError as error:
        return StorageDeleteFailure(
            message=str(error),
            diagnostics=_delete_blocked_diagnostics(error),
            cascade_available=True,
        )
    return StorageDeleteCompleted(record=deleted)


def refresh_storage_catalog(storage_root: Path) -> CatalogRefreshSummary:
    return refresh_catalog(storage_root)


def render_catalog_refresh(summary: CatalogRefreshSummary) -> str:
    return " ".join(
        [
            "catalog refreshed",
            f"datasets={summary.dataset_roots}",
            f"studies={summary.study_roots}",
            f"artifacts={summary.artifact_roots}",
        ]
    )


def _list_records(
    storage_root: Path,
    *,
    kind: StorageRootKind,
    selector: StorageSelector,
) -> list[CatalogRecord]:
    if kind == "dataset":
        return cast(
            list[CatalogRecord],
            list_dataset_records(storage_root, selector=_dataset_selector(selector)),
        )
    if kind == "study":
        return cast(
            list[CatalogRecord],
            list_study_records(storage_root, selector=_study_selector(selector)),
        )
    return cast(
        list[CatalogRecord],
        list_artifact_records(storage_root, selector=_artifact_selector(selector)),
    )


def _selector_has_filters(selector: StorageSelector) -> bool:
    return any(getattr(selector, field.name) is not None for field in fields(selector))


def _describe_record(
    storage_root: Path,
    record: CatalogRecord,
    *,
    detail: str | None,
) -> StorageShowRendered:
    description = describe_root(
        materialize_catalog_root(storage_root, record).root_path,
        detail=detail,
    )
    title, sections = sectioned_summary(description)
    return StorageShowRendered(RenderableSections(title=title, sections=sections))


def _list_renderable(
    spec: _OperatorSpec,
    records: Sequence[CatalogRecord],
) -> RenderableSections:
    return RenderableSections(
        title=f"{spec.kind} list",
        sections=spec.list_sections(list(records)),
    )


def _matches_renderable(
    spec: _OperatorSpec,
    records: Sequence[CatalogRecord],
) -> RenderableSections:
    return RenderableSections(
        title=f"{spec.kind} matches",
        sections=spec.list_sections(list(records)),
    )


def _delete_blocked_diagnostics(error: DeleteBlockedError) -> tuple[RenderableSections, ...]:
    diagnostics: list[RenderableSections] = []
    if error.artifact_records:
        diagnostics.append(
            RenderableSections(
                title="artifact matches",
                sections=artifact_list_sections(list(error.artifact_records)),
            )
        )
    if error.study_records:
        diagnostics.append(
            RenderableSections(
                title="study matches",
                sections=study_list_sections(list(error.study_records)),
            )
        )
    return tuple(diagnostics)


def _narrowing_attributes(
    spec: _OperatorSpec,
    records: Sequence[CatalogRecord],
    selector: StorageSelector,
) -> tuple[str, ...]:
    if len(records) <= 1:
        return ()
    return tuple(
        attribute
        for attribute in spec.narrowing_attributes
        if getattr(selector, attribute, None) is None
        and len({getattr(record, attribute, None) for record in records}) > 1
    )


def _dataset_selector(selector: StorageSelector) -> DatasetSelector:
    if isinstance(selector, DatasetSelector):
        return selector
    raise SpiceOperatorError("Dataset command requires a dataset selector")


def _study_selector(selector: StorageSelector) -> StudySelector:
    if isinstance(selector, StudySelector):
        return selector
    raise SpiceOperatorError("Study command requires a study selector")


def _artifact_selector(selector: StorageSelector) -> ArtifactSelector:
    if isinstance(selector, ArtifactSelector):
        return selector
    raise SpiceOperatorError("Artifact command requires an artifact selector")


def _dataset_record(record: CatalogRecord) -> CatalogDatasetRecord:
    if isinstance(record, CatalogDatasetRecord):
        return record
    raise SpiceOperatorError("Dataset delete requires a dataset record")


def _study_record(record: CatalogRecord) -> CatalogStudyRecord:
    if isinstance(record, CatalogStudyRecord):
        return record
    raise SpiceOperatorError("Study delete requires a study record")


def _artifact_record(record: CatalogRecord) -> CatalogArtifactRecord:
    if isinstance(record, CatalogArtifactRecord):
        return record
    raise SpiceOperatorError("Artifact delete requires an artifact record")
