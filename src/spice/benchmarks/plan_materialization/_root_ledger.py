# pyright: strict

"""Benchmark-owned root ledger policy."""

from __future__ import annotations

from dataclasses import dataclass

from ...config.models import ArtifactVariant, WorkflowTask
from ...config.selections import (
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    WorkflowSelection,
)
from ...config.workflow_snapshots import ResolvedWorkflowConfig
from ...core.errors import ConfigResolutionError
from ...storage.workflow_root_materialization import materialize_workflow_root_facts
from ._models import (
    BenchmarkDependencyLedger,
    BenchmarkRootFacts,
    BenchmarkRootKind,
    BenchmarkRootLedger,
    BenchmarkRootLedgerEntry,
)


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactSource:
    artifact_id: str
    dataset_id: str


@dataclass(frozen=True, slots=True)
class PreparedBenchmarkRootSelection:
    selection: WorkflowSelection
    artifact_from_run_id: str | None
    artifact_source_dataset_id: str | None


@dataclass(frozen=True, slots=True)
class FinalizedBenchmarkRoots:
    facts: BenchmarkRootFacts
    ledger: BenchmarkRootLedger


@dataclass(slots=True)
class BenchmarkProducedRootIndex:
    root_entries: list[BenchmarkRootLedgerEntry]

    @classmethod
    def create(cls) -> BenchmarkProducedRootIndex:
        return cls(root_entries=[])

    def record_ledger(self, ledger: BenchmarkRootLedger) -> None:
        self.root_entries.extend(ledger.entries)

    def dependency_study_id(self, depends_on: tuple[str, ...]) -> str:
        for run_id in depends_on:
            entry = self.produced_root_for_run(run_id, root_kind="study")
            if entry is not None:
                return entry.root_id
        raise ConfigResolutionError("tuned train requires a tune dependency or explicit study_id")

    def dependency_artifact_source(self, artifact_from_run_id: str) -> BenchmarkArtifactSource:
        entry = self.produced_root_for_run(artifact_from_run_id, root_kind="artifact")
        if entry is None:
            raise ConfigResolutionError("artifact_from may reference train steps only")
        if entry.dataset_id is None or entry.artifact_id is None:
            raise ConfigResolutionError("artifact_from train step has incomplete root ledger")
        return BenchmarkArtifactSource(
            artifact_id=entry.artifact_id,
            dataset_id=entry.dataset_id,
        )

    def produced_study_dataset_ids(self) -> dict[str, str]:
        return {
            entry.study_id: entry.dataset_id
            for entry in self.root_entries
            if entry.role == "produced"
            and entry.root_kind == "study"
            and entry.study_id is not None
            and entry.dataset_id is not None
        }

    def produced_artifact_dataset_ids(self) -> dict[str, str]:
        return {
            entry.artifact_id: entry.dataset_id
            for entry in self.root_entries
            if entry.role == "produced"
            and entry.root_kind == "artifact"
            and entry.artifact_id is not None
            and entry.dataset_id is not None
        }

    def produced_root_for_run(
        self,
        run_id: str,
        *,
        root_kind: BenchmarkRootKind,
    ) -> BenchmarkRootLedgerEntry | None:
        matches = [
            entry
            for entry in self.root_entries
            if entry.run_id == run_id and entry.role == "produced" and entry.root_kind == root_kind
        ]
        if len(matches) > 1:
            raise ConfigResolutionError(
                f"benchmark run {run_id} produced multiple {root_kind} roots"
            )
        return matches[0] if matches else None


@dataclass(slots=True)
class BenchmarkRootLedgerBuilder:
    produced_roots: BenchmarkProducedRootIndex

    @classmethod
    def create(cls) -> BenchmarkRootLedgerBuilder:
        return cls(
            produced_roots=BenchmarkProducedRootIndex.create(),
        )

    def prepare_selection(
        self,
        workflow_selection: WorkflowSelection,
        dependencies: BenchmarkDependencyLedger,
    ) -> PreparedBenchmarkRootSelection:
        if (
            isinstance(workflow_selection, TrainWorkflowSelection)
            and workflow_selection.variant == ArtifactVariant.TUNED.value
            and workflow_selection.study_id is None
        ):
            study_id = self.produced_roots.dependency_study_id(dependencies.local_run_ids)
            selection = workflow_selection.model_copy(
                update={"study_id": study_id, "dataset_id": None}
            )
            return PreparedBenchmarkRootSelection(
                selection=selection,
                artifact_from_run_id=dependencies.artifact_from_run_id,
                artifact_source_dataset_id=None,
            )
        if (
            isinstance(workflow_selection, EvaluateWorkflowSelection)
            and dependencies.artifact_from_run_id is not None
        ):
            source = self.produced_roots.dependency_artifact_source(
                dependencies.artifact_from_run_id
            )
            updates: dict[str, object] = {"artifact_id": source.artifact_id}
            dataset_id = workflow_selection.dataset_id
            if dataset_id is None:
                dataset_id = source.dataset_id
                updates["dataset_id"] = dataset_id
            selection = workflow_selection.model_copy(update=updates)
            return PreparedBenchmarkRootSelection(
                selection=selection,
                artifact_from_run_id=dependencies.artifact_from_run_id,
                artifact_source_dataset_id=source.dataset_id,
            )
        return PreparedBenchmarkRootSelection(
            selection=workflow_selection,
            artifact_from_run_id=dependencies.artifact_from_run_id,
            artifact_source_dataset_id=None,
        )

    def finalize_roots(
        self,
        *,
        run_id: str,
        workflow: WorkflowTask,
        config: ResolvedWorkflowConfig,
        prepared: PreparedBenchmarkRootSelection,
    ) -> FinalizedBenchmarkRoots:
        root_facts = materialize_workflow_root_facts(
            config,
            known_study_dataset_ids=self.produced_roots.produced_study_dataset_ids(),
            known_artifact_dataset_ids=self.produced_roots.produced_artifact_dataset_ids(),
            artifact_source_dataset_id=prepared.artifact_source_dataset_id,
        )
        consumed = root_facts.consumed
        produced = root_facts.produced
        facts = BenchmarkRootFacts(
            consumed_dataset_id=consumed.dataset_id,
            consumed_study_id=consumed.study_id,
            consumed_study_dataset_id=root_facts.consumed_study_dataset_id,
            consumed_artifact_id=consumed.artifact_id,
            consumed_artifact_dataset_id=root_facts.consumed_artifact_dataset_id,
            produced_study_id=produced.study_id,
            produced_study_dataset_id=root_facts.produced_study_dataset_id,
            produced_artifact_id=produced.artifact_id,
            produced_artifact_dataset_id=root_facts.produced_artifact_dataset_id,
            artifact_source_dataset_id=root_facts.source.artifact_dataset_id,
        )
        entries: list[BenchmarkRootLedgerEntry] = []
        if consumed.dataset_id is not None:
            entries.append(
                BenchmarkRootLedgerEntry(
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
                BenchmarkRootLedgerEntry(
                    run_id=run_id,
                    workflow=workflow,
                    role="consumed",
                    root_kind="study",
                    root_id=consumed.study_id,
                    study_id=consumed.study_id,
                    dataset_id=root_facts.consumed_study_dataset_id,
                )
            )
        if consumed.artifact_id is not None:
            entries.append(
                BenchmarkRootLedgerEntry(
                    run_id=run_id,
                    workflow=workflow,
                    role="consumed",
                    root_kind="artifact",
                    root_id=consumed.artifact_id,
                    artifact_id=consumed.artifact_id,
                    dataset_id=root_facts.consumed_artifact_dataset_id,
                    source_run_id=prepared.artifact_from_run_id,
                )
            )
        if produced.study_id is not None:
            entries.append(
                BenchmarkRootLedgerEntry(
                    run_id=run_id,
                    workflow=workflow,
                    role="produced",
                    root_kind="study",
                    root_id=produced.study_id,
                    study_id=produced.study_id,
                    dataset_id=root_facts.produced_study_dataset_id,
                )
            )
        if produced.artifact_id is not None:
            entries.append(
                BenchmarkRootLedgerEntry(
                    run_id=run_id,
                    workflow=workflow,
                    role="produced",
                    root_kind="artifact",
                    root_id=produced.artifact_id,
                    artifact_id=produced.artifact_id,
                    dataset_id=root_facts.produced_artifact_dataset_id,
                )
            )
        if root_facts.source.artifact_dataset_id is not None:
            entries.append(
                BenchmarkRootLedgerEntry(
                    run_id=run_id,
                    workflow=workflow,
                    role="source",
                    root_kind="dataset",
                    root_id=root_facts.source.artifact_dataset_id,
                    dataset_id=root_facts.source.artifact_dataset_id,
                    source_run_id=prepared.artifact_from_run_id,
                )
            )
        return FinalizedBenchmarkRoots(
            facts=facts,
            ledger=BenchmarkRootLedger(entries=tuple(entries)),
        )

    def record_ledger(self, ledger: BenchmarkRootLedger) -> None:
        self.produced_roots.record_ledger(ledger)
