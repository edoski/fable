"""Human-readable groupings of canonical experiment records."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated
from uuid import UUID

from pydantic import UUID4, Field

from .records import StrictFrozenRecord


class ExperimentKind(StrEnum):
    FEATURE_ABLATION = "feature_ablation"
    C_STUDY = "c_study"
    HPO = "hpo"
    K_STUDY = "k_study"
    HELD_OUT = "held_out"


class ExperimentEntry(StrictFrozenRecord):
    cell: Annotated[str, Field(min_length=1)]
    record_id: UUID4


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
        destination.write(manifest.model_dump_json())


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
