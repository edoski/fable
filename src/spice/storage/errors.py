"""Storage-owned operator error types."""

from __future__ import annotations

from ..core.errors import SpiceOperatorError
from .catalog.records import CatalogArtifactRecord, CatalogStudyRecord


class DeleteBlockedError(SpiceOperatorError):
    """Raised when delete would orphan dependent state without explicit cascade."""

    def __init__(
        self,
        *,
        message: str,
        artifact_records: list[CatalogArtifactRecord] | None = None,
        study_records: list[CatalogStudyRecord] | None = None,
    ) -> None:
        self.artifact_records = tuple(artifact_records or ())
        self.study_records = tuple(study_records or ())
        super().__init__(message)
