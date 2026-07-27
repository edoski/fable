"""Human-readable groupings of canonical experiment records."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self
from uuid import UUID

from pydantic import UUID4, Field, model_validator

from .records import StrictFrozenRecord


class ExperimentKind(StrEnum):
    FEATURE_ABLATION = "feature_ablation"
    C_STUDY = "c_study"
    HPO = "hpo"
    K_STUDY = "k_study"
    HELD_OUT = "held_out"


class ExperimentEntry(StrictFrozenRecord):
    cell: Annotated[str, Field(min_length=1)]
    artifact_id: UUID4 | None = None
    study_id: UUID4 | None = None
    evaluation_id: UUID4 | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.artifact_id is None and self.study_id is None and self.evaluation_id is None:
            raise ValueError("entry must reference a canonical record")
        return self

    def require_artifact_id(self) -> UUID:
        if self.artifact_id is None:
            raise ValueError("experiment entry must reference an artifact")
        return self.artifact_id

    def require_study_id(self) -> UUID:
        if self.study_id is None:
            raise ValueError("experiment entry must reference a Study")
        return self.study_id

    def require_evaluation_id(self) -> UUID:
        if self.evaluation_id is None:
            raise ValueError("experiment entry must reference an evaluation")
        return self.evaluation_id


class ExperimentManifest(StrictFrozenRecord):
    experiment_id: UUID4
    entries: Annotated[tuple[ExperimentEntry, ...], Field(min_length=1)]


def experiment_manifest_path(
    storage_root: Path,
    kind: ExperimentKind,
    experiment_id: UUID,
) -> Path:
    return storage_root / "experiments" / kind / f"{experiment_id}.json"


def write_experiment_manifest(
    storage_root: Path,
    kind: ExperimentKind,
    manifest: ExperimentManifest,
) -> None:
    path = experiment_manifest_path(storage_root, kind, manifest.experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        destination.write(manifest.model_dump_json(exclude_none=True))


def load_experiment_manifest(
    storage_root: Path,
    kind: ExperimentKind,
    experiment_id: UUID,
) -> ExperimentManifest:
    manifest = ExperimentManifest.model_validate_json(
        experiment_manifest_path(storage_root, kind, experiment_id).read_bytes(),
        strict=True,
    )
    if manifest.experiment_id != experiment_id:
        raise ValueError("manifest ID does not match the requested experiment")
    return manifest
