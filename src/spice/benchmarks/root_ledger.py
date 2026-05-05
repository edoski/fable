# pyright: strict

"""Benchmark root ledger materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config.models import ArtifactVariant, TrainConfig, TuneConfig, WorkflowTask
from ..config.selections import (
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    WorkflowSelection,
)
from ..config.workflow_snapshots import ResolvedWorkflowConfig
from ..core.config_model import ConfigModel
from ..core.errors import ConfigResolutionError, SelectorResolutionError
from ..storage.catalog.index import resolve_study_record
from ..storage.selectors import StudySelector
from ..workflows.preparation import produced_artifact_id, produced_study_id
from .dependency_ledger import BenchmarkDependencyLedger

BenchmarkRootRole = Literal["consumed", "produced", "source"]
BenchmarkRootKind = Literal["dataset", "study", "artifact"]


class BenchmarkConsumedRoots(ConfigModel):
    dataset_id: str | None = None
    study_id: str | None = None
    artifact_id: str | None = None


class BenchmarkProducedRoots(ConfigModel):
    study_id: str | None = None
    artifact_id: str | None = None


class BenchmarkMaterializedRoot(ConfigModel):
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
    entries: tuple[BenchmarkMaterializedRoot, ...] = ()


@dataclass(frozen=True, slots=True)
class _MaterializedArtifactRoot:
    artifact_id: str
    dataset_id: str


@dataclass(frozen=True, slots=True)
class _PreparedRootSelection:
    selection: WorkflowSelection
    artifact_from_run_id: str | None
    artifact_source_dataset_id: str | None


@dataclass(slots=True)
class BenchmarkRootMaterializer:
    root_entries: list[BenchmarkMaterializedRoot]

    @classmethod
    def create(cls) -> BenchmarkRootMaterializer:
        return cls(root_entries=[])

    def prepare_selection(
        self,
        workflow_selection: WorkflowSelection,
        dependencies: BenchmarkDependencyLedger,
    ) -> _PreparedRootSelection:
        if (
            isinstance(workflow_selection, TrainWorkflowSelection)
            and workflow_selection.variant == ArtifactVariant.TUNED.value
            and workflow_selection.study_id is None
        ):
            study_id = self._dependency_study_id(dependencies.local_run_ids)
            selection = workflow_selection.model_copy(
                update={"study_id": study_id, "dataset_id": None}
            )
            return _PreparedRootSelection(
                selection=selection,
                artifact_from_run_id=dependencies.artifact_from_run_id,
                artifact_source_dataset_id=None,
            )
        if (
            isinstance(workflow_selection, EvaluateWorkflowSelection)
            and dependencies.artifact_from_run_id is not None
        ):
            materialized = self._dependency_artifact_root(dependencies.artifact_from_run_id)
            updates: dict[str, object] = {"artifact_id": materialized.artifact_id}
            dataset_id = workflow_selection.dataset_id
            if dataset_id is None:
                dataset_id = materialized.dataset_id
                updates["dataset_id"] = dataset_id
            selection = workflow_selection.model_copy(update=updates)
            return _PreparedRootSelection(
                selection=selection,
                artifact_from_run_id=dependencies.artifact_from_run_id,
                artifact_source_dataset_id=materialized.dataset_id,
            )
        return _PreparedRootSelection(
            selection=workflow_selection,
            artifact_from_run_id=dependencies.artifact_from_run_id,
            artifact_source_dataset_id=None,
        )

    def finalize_ledger(
        self,
        *,
        run_id: str,
        workflow: WorkflowTask,
        config: ResolvedWorkflowConfig,
        prepared: _PreparedRootSelection,
    ) -> BenchmarkRootLedger:
        consumed = self._consumed_roots(config)
        produced = self._produced_roots(config)
        entries: list[BenchmarkMaterializedRoot] = []
        if consumed.dataset_id is not None:
            entries.append(
                BenchmarkMaterializedRoot(
                    run_id=run_id,
                    workflow=workflow,
                    role="consumed",
                    root_kind="dataset",
                    root_id=consumed.dataset_id,
                    dataset_id=consumed.dataset_id,
                )
            )
        if consumed.study_id is not None:
            entries.append(
                BenchmarkMaterializedRoot(
                    run_id=run_id,
                    workflow=workflow,
                    role="consumed",
                    root_kind="study",
                    root_id=consumed.study_id,
                    study_id=consumed.study_id,
                    dataset_id=consumed.dataset_id,
                )
            )
        if consumed.artifact_id is not None:
            entries.append(
                BenchmarkMaterializedRoot(
                    run_id=run_id,
                    workflow=workflow,
                    role="consumed",
                    root_kind="artifact",
                    root_id=consumed.artifact_id,
                    artifact_id=consumed.artifact_id,
                    dataset_id=consumed.dataset_id,
                    source_run_id=prepared.artifact_from_run_id,
                )
            )
        if produced.study_id is not None:
            entries.append(
                BenchmarkMaterializedRoot(
                    run_id=run_id,
                    workflow=workflow,
                    role="produced",
                    root_kind="study",
                    root_id=produced.study_id,
                    study_id=produced.study_id,
                    dataset_id=config.dataset_id if isinstance(config, TuneConfig) else None,
                )
            )
        if produced.artifact_id is not None:
            entries.append(
                BenchmarkMaterializedRoot(
                    run_id=run_id,
                    workflow=workflow,
                    role="produced",
                    root_kind="artifact",
                    root_id=produced.artifact_id,
                    artifact_id=produced.artifact_id,
                    dataset_id=self._train_dataset_id(config)
                    if isinstance(config, TrainConfig)
                    else None,
                )
            )
        if prepared.artifact_source_dataset_id is not None:
            entries.append(
                BenchmarkMaterializedRoot(
                    run_id=run_id,
                    workflow=workflow,
                    role="source",
                    root_kind="dataset",
                    root_id=prepared.artifact_source_dataset_id,
                    dataset_id=prepared.artifact_source_dataset_id,
                    source_run_id=prepared.artifact_from_run_id,
                )
            )
        return BenchmarkRootLedger(entries=tuple(entries))

    def record_config(
        self,
        run_id: str,
        config: ResolvedWorkflowConfig,
        ledger: BenchmarkRootLedger,
    ) -> None:
        del run_id, config
        self.root_entries.extend(ledger.entries)

    def _consumed_roots(self, config: ResolvedWorkflowConfig) -> BenchmarkConsumedRoots:
        if isinstance(config, TuneConfig):
            return BenchmarkConsumedRoots(dataset_id=config.dataset_id)
        if isinstance(config, TrainConfig):
            return BenchmarkConsumedRoots(
                dataset_id=config.dataset_id,
                study_id=config.study_id,
            )
        return BenchmarkConsumedRoots(
            dataset_id=config.dataset_id,
            artifact_id=config.artifact_id,
        )

    def _produced_roots(self, config: ResolvedWorkflowConfig) -> BenchmarkProducedRoots:
        if isinstance(config, TuneConfig):
            return BenchmarkProducedRoots(study_id=produced_study_id(config))
        if isinstance(config, TrainConfig):
            return BenchmarkProducedRoots(
                artifact_id=produced_artifact_id(
                    config,
                    dataset_id=self._train_dataset_id(config),
                )
            )
        return BenchmarkProducedRoots()

    def _dependency_study_id(self, depends_on: tuple[str, ...]) -> str:
        for run_id in depends_on:
            entry = self._produced_root_for_run(run_id, root_kind="study")
            if entry is not None:
                return entry.root_id
        raise ConfigResolutionError("tuned train requires a tune dependency or explicit study_id")

    def _dependency_artifact_root(self, artifact_from_run_id: str) -> _MaterializedArtifactRoot:
        entry = self._produced_root_for_run(artifact_from_run_id, root_kind="artifact")
        if entry is None:
            raise ConfigResolutionError("artifact_from may reference train steps only")
        if entry.dataset_id is None or entry.artifact_id is None:
            raise ConfigResolutionError("artifact_from train step has incomplete root ledger")
        return _MaterializedArtifactRoot(
            artifact_id=entry.artifact_id,
            dataset_id=entry.dataset_id,
        )

    def _train_dataset_id(self, config: TrainConfig) -> str:
        if config.dataset_id is not None:
            return config.dataset_id
        if config.study_id is None:
            raise ConfigResolutionError(
                "train artifact source did not declare dataset_id or study_id"
            )
        for entry in self.root_entries:
            if (
                entry.role == "produced"
                and entry.root_kind == "study"
                and entry.study_id == config.study_id
                and entry.dataset_id is not None
            ):
                return entry.dataset_id
        try:
            study = resolve_study_record(
                config.storage.root,
                selector=StudySelector(study_id=config.study_id),
            )
        except SelectorResolutionError as exc:
            raise ConfigResolutionError(str(exc)) from exc
        return study.dataset_id

    def _produced_root_for_run(
        self,
        run_id: str,
        *,
        root_kind: BenchmarkRootKind,
    ) -> BenchmarkMaterializedRoot | None:
        matches = [
            entry
            for entry in self.root_entries
            if entry.run_id == run_id
            and entry.role == "produced"
            and entry.root_kind == root_kind
        ]
        if len(matches) > 1:
            raise ConfigResolutionError(
                f"benchmark run {run_id} produced multiple {root_kind} roots"
            )
        return matches[0] if matches else None


def root_id(
    ledger: BenchmarkRootLedger,
    *,
    role: BenchmarkRootRole,
    root_kind: BenchmarkRootKind,
) -> str | None:
    matches = [
        entry.root_id
        for entry in ledger.entries
        if entry.role == role and entry.root_kind == root_kind
    ]
    if len(matches) > 1:
        raise ConfigResolutionError(
            f"benchmark root ledger has multiple {role} {root_kind} roots"
        )
    return matches[0] if matches else None


def consumed_dataset_id(ledger: BenchmarkRootLedger) -> str | None:
    return root_id(ledger, role="consumed", root_kind="dataset")


def consumed_artifact_id(ledger: BenchmarkRootLedger) -> str | None:
    return root_id(ledger, role="consumed", root_kind="artifact")


def produced_study_root_id(ledger: BenchmarkRootLedger) -> str | None:
    return root_id(ledger, role="produced", root_kind="study")


def produced_artifact_root_id(ledger: BenchmarkRootLedger) -> str | None:
    return root_id(ledger, role="produced", root_kind="artifact")


def artifact_source_dataset_id(ledger: BenchmarkRootLedger) -> str | None:
    return root_id(ledger, role="source", root_kind="dataset")
