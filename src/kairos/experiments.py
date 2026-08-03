"""Human-readable groupings of canonical experiment records."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated
from uuid import UUID

from pydantic import UUID4, ConfigDict, Field, RootModel


class ExperimentKind(StrEnum):
    FEATURE_ABLATION = "feature_ablation"
    C_STUDY = "c_study"
    HPO = "hpo"
    K_STUDY = "k_study"
    HELD_OUT = "held_out"


class ExperimentManifest(
    RootModel[Annotated[dict[Annotated[str, Field(min_length=1)], UUID4], Field(min_length=1)]]
):
    model_config = ConfigDict(frozen=True, strict=True)


def experiment_directory(storage_root: Path, kind: ExperimentKind, experiment_id: UUID) -> Path:
    return storage_root / "experiments" / kind / str(experiment_id)


def experiment_manifest_path(storage_root: Path, kind: ExperimentKind, experiment_id: UUID) -> Path:
    return experiment_directory(storage_root, kind, experiment_id) / "manifest.json"


def load_experiment_manifest(
    storage_root: Path, kind: ExperimentKind, experiment_id: UUID
) -> dict[str, UUID4]:
    return ExperimentManifest.model_validate_json(
        experiment_manifest_path(storage_root, kind, experiment_id).read_bytes(), strict=True
    ).root
