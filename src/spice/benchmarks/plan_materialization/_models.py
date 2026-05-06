# pyright: strict

"""Public benchmark planning models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from ...config.models import ArtifactVariant, WorkflowTask
from ...config.workflow_snapshots import ResolvedWorkflowConfig
from ...core.config_model import ConfigModel
from ...core.errors import ConfigResolutionError

BenchmarkRootRole = Literal["consumed", "produced", "source"]
BenchmarkRootKind = Literal["dataset", "study", "artifact"]


@dataclass(frozen=True, slots=True)
class BenchmarkDependencyLedger:
    local_run_ids: tuple[str, ...]
    external_slurm_dependencies: tuple[str, ...]
    artifact_from_run_id: str | None


class BenchmarkSelectionLedger(ConfigModel):
    surface: str | None = None
    chain: str | None = None
    problem: str | None = None
    features: str | None = None
    model: str | None = None
    tuning_space: str | None = None
    objective: str | None = None
    evaluation: str | None = None
    training: str | None = None
    split: str | None = None
    tuning: str | None = None
    study: str | None = None
    variant: ArtifactVariant | None = None
    trial_count: int | None = Field(default=None, gt=0)
    delay_seconds: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    storage_root: Path | None = None


class BenchmarkRootLedgerEntry(ConfigModel):
    run_id: str
    workflow: WorkflowTask
    role: BenchmarkRootRole
    root_kind: BenchmarkRootKind
    root_id: str
    source_run_id: str | None = None
    dataset_id: str | None = None
    study_id: str | None = None
    artifact_id: str | None = None


class BenchmarkRootLedger(ConfigModel):
    entries: tuple[BenchmarkRootLedgerEntry, ...] = ()

    def root_id(
        self,
        *,
        role: BenchmarkRootRole,
        root_kind: BenchmarkRootKind,
    ) -> str | None:
        matches = [
            entry.root_id
            for entry in self.entries
            if entry.role == role and entry.root_kind == root_kind
        ]
        if len(matches) > 1:
            raise ConfigResolutionError(
                f"benchmark root ledger has multiple {role} {root_kind} roots"
            )
        return matches[0] if matches else None

    def consumed_dataset_id(self) -> str | None:
        return self.root_id(role="consumed", root_kind="dataset")

    def consumed_artifact_id(self) -> str | None:
        return self.root_id(role="consumed", root_kind="artifact")

    def produced_study_id(self) -> str | None:
        return self.root_id(role="produced", root_kind="study")

    def produced_artifact_id(self) -> str | None:
        return self.root_id(role="produced", root_kind="artifact")

    def artifact_source_dataset_id(self) -> str | None:
        return self.root_id(role="source", root_kind="dataset")


@dataclass(frozen=True, slots=True)
class BenchmarkPlanEntry:
    run_id: str
    case_id: str
    step_id: str
    workflow: WorkflowTask
    dependencies: BenchmarkDependencyLedger
    dimension_labels: dict[str, str]
    selection: BenchmarkSelectionLedger
    root_ledger: BenchmarkRootLedger
    config: ResolvedWorkflowConfig
