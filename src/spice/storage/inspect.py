"""Tiny dispatcher for selector-based `spice show` commands."""

from __future__ import annotations

from pathlib import Path

from ..core.errors import StateLayoutError
from .engine import (
    ARTIFACT_ROOT_KIND,
    DATASET_ROOT_KIND,
    STUDY_ROOT_KIND,
    detect_root_kind,
    state_db_path,
)
from .inspect_artifact import ArtifactRootDescription, artifact_sections, describe_artifact_root
from .inspect_dataset import DatasetRootDescription, dataset_sections, describe_dataset_root
from .inspect_study import StudyRootDescription, describe_study_root, study_sections

RootDescription = DatasetRootDescription | ArtifactRootDescription | StudyRootDescription


def describe_root(root: Path, *, detail: str | None = None) -> RootDescription:
    db_path = state_db_path(root)
    root_kind = detect_root_kind(db_path)
    if root_kind == DATASET_ROOT_KIND:
        return describe_dataset_root(db_path, detail=detail)
    if root_kind == ARTIFACT_ROOT_KIND:
        return describe_artifact_root(db_path, detail=detail)
    if root_kind == STUDY_ROOT_KIND:
        return describe_study_root(db_path, detail=detail)
    raise StateLayoutError(f"Unsupported root kind: {root_kind}")


def sectioned_summary(
    description: RootDescription,
) -> tuple[str, list[tuple[str, list[tuple[str, str]]]]]:
    if isinstance(description, DatasetRootDescription):
        return "dataset summary", dataset_sections(description)
    if isinstance(description, ArtifactRootDescription):
        return "artifact summary", artifact_sections(description)
    if isinstance(description, StudyRootDescription):
        return "study summary", study_sections(description)
    raise StateLayoutError(f"Unsupported root description: {type(description).__name__}")
