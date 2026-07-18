"""Storage operator queries and commands."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, Literal, TypeAlias, TypeVar

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
    CatalogCorpusRecord,
    CatalogRecord,
    CatalogStudyRecord,
)
from .errors import DeleteBlockedError
from .inspect import describe_root, sectioned_summary
from .inspect_artifact import artifact_list_sections
from .inspect_dataset import dataset_list_sections
from .inspect_study import study_list_sections
from .lifecycle import delete_artifact_record, delete_dataset_record, delete_study_record
from .selectors import ArtifactSelector, CorpusSelector, StudySelector

StorageRootKind: TypeAlias = Literal["corpus", "study", "artifact"]
StorageSelector: TypeAlias = CorpusSelector | StudySelector | ArtifactSelector
StorageShowOutcome: TypeAlias = "StorageShowRendered | StorageShowFailure"
StorageDeleteOutcome: TypeAlias = "StorageDeleteCompleted | StorageDeleteFailure"
SectionRows: TypeAlias = list[tuple[str, list[tuple[str, str]]]]
SelectorT = TypeVar("SelectorT", bound=StorageSelector)
RecordT = TypeVar("RecordT", bound=CatalogRecord)


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
class _OperatorSpec(Generic[SelectorT, RecordT]):
    kind: StorageRootKind
    selector_type: type[SelectorT]
    record_type: type[RecordT]
    valid_details: frozenset[str]
    narrowing_attributes: tuple[str, ...]
    list_catalog_records: Callable[[Path, SelectorT], list[RecordT]]
    delete_catalog_record: Callable[[Path, RecordT, bool], RecordT]
    render_list_sections: Callable[[list[RecordT]], SectionRows]

    def list_records(
        self,
        storage_root: Path,
        selector: StorageSelector,
    ) -> list[CatalogRecord]:
        return list(self.list_catalog_records(storage_root, self._selector(selector)))

    def delete_record(
        self,
        storage_root: Path,
        record: CatalogRecord,
        *,
        cascade: bool,
    ) -> CatalogRecord:
        return self.delete_catalog_record(storage_root, self._record(record), cascade)

    def list_sections(self, records: Sequence[CatalogRecord]) -> SectionRows:
        return self.render_list_sections([self._record(record) for record in records])

    def _selector(self, selector: StorageSelector) -> SelectorT:
        if isinstance(selector, self.selector_type):
            return selector
        raise SpiceOperatorError(f"{self.kind.title()} command requires a {self.kind} selector")

    def _record(self, record: CatalogRecord) -> RecordT:
        if isinstance(record, self.record_type):
            return record
        raise SpiceOperatorError(f"{self.kind.title()} delete requires a {self.kind} record")


_SPECS: dict[StorageRootKind, _OperatorSpec[Any, Any]] = {
    "corpus": _OperatorSpec(
        kind="corpus",
        selector_type=CorpusSelector,
        record_type=CatalogCorpusRecord,
        valid_details=frozenset(),
        narrowing_attributes=("chain_name", "corpus_name"),
        list_catalog_records=lambda root, selector: list_dataset_records(
            root,
            selector=selector,
        ),
        delete_catalog_record=lambda root, record, cascade: delete_dataset_record(
            root,
            record=record,
            cascade=cascade,
        ),
        render_list_sections=dataset_list_sections,
    ),
    "study": _OperatorSpec(
        kind="study",
        selector_type=StudySelector,
        record_type=CatalogStudyRecord,
        valid_details=frozenset(detail.value for detail in StudyInspectionDetail),
        narrowing_attributes=(
            "chain_name",
            "corpus_name",
            "features_id",
            "prediction_id",
            "model_id",
            "problem_id",
            "study_name",
        ),
        list_catalog_records=lambda root, selector: list_study_records(
            root,
            selector=selector,
        ),
        delete_catalog_record=lambda root, record, cascade: delete_study_record(
            root,
            record=record,
            cascade=cascade,
        ),
        render_list_sections=study_list_sections,
    ),
    "artifact": _OperatorSpec(
        kind="artifact",
        selector_type=ArtifactSelector,
        record_type=CatalogArtifactRecord,
        valid_details=frozenset(detail.value for detail in ArtifactInspectionDetail),
        narrowing_attributes=(
            "chain_name",
            "corpus_name",
            "features_id",
            "prediction_id",
            "model_id",
            "problem_id",
            "variant",
            "study_name",
        ),
        list_catalog_records=lambda root, selector: list_artifact_records(
            root,
            selector=selector,
        ),
        delete_catalog_record=lambda root, record, _cascade: delete_artifact_record(
            root,
            record=record,
        ),
        render_list_sections=artifact_list_sections,
    ),
}


def show_storage(query: StorageShowQuery) -> StorageShowOutcome:
    spec = _SPECS[query.kind]
    if query.detail is not None and query.detail not in spec.valid_details:
        return StorageShowFailure(
            message=f"Unsupported {query.kind} detail: {query.detail}",
        )
    records = spec.list_records(query.storage_root, query.selector)
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
    records = spec.list_records(command.storage_root, command.selector)
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
        deleted = spec.delete_record(
            command.storage_root,
            record,
            cascade=command.cascade,
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
    spec: _OperatorSpec[Any, Any],
    records: Sequence[CatalogRecord],
) -> RenderableSections:
    return RenderableSections(
        title=f"{spec.kind} list",
        sections=spec.list_sections(records),
    )


def _matches_renderable(
    spec: _OperatorSpec[Any, Any],
    records: Sequence[CatalogRecord],
) -> RenderableSections:
    return RenderableSections(
        title=f"{spec.kind} matches",
        sections=spec.list_sections(records),
    )


def _delete_blocked_diagnostics(error: DeleteBlockedError) -> tuple[RenderableSections, ...]:
    diagnostics: list[RenderableSections] = []
    if error.artifact_records:
        diagnostics.append(_matches_renderable(_SPECS["artifact"], error.artifact_records))
    if error.study_records:
        diagnostics.append(_matches_renderable(_SPECS["study"], error.study_records))
    return tuple(diagnostics)


def _narrowing_attributes(
    spec: _OperatorSpec[Any, Any],
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
