# pyright: strict

"""Benchmark root ledger materialization."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from ..config.models import ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig
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
from ..storage.workflow_roots import produced_artifact_id, produced_study_id
from .dependency_ledger import BenchmarkDependencyLedger


class BenchmarkConsumedRoots(ConfigModel):
    dataset_id: str | None = None
    study_id: str | None = None
    artifact_id: str | None = None


class BenchmarkProducedRoots(ConfigModel):
    study_id: str | None = None
    artifact_id: str | None = None


class BenchmarkRootLedger(ConfigModel):
    consumed: BenchmarkConsumedRoots = Field(default_factory=BenchmarkConsumedRoots)
    produced: BenchmarkProducedRoots = Field(default_factory=BenchmarkProducedRoots)
    artifact_from_run_id: str | None = None
    artifact_source_dataset_id: str | None = None


@dataclass(frozen=True, slots=True)
class _MaterializedArtifactRoot:
    artifact_id: str
    dataset_id: str


@dataclass(frozen=True, slots=True)
class _PreparedRootSelection:
    selection: WorkflowSelection
    ledger: BenchmarkRootLedger


def _config_state() -> dict[str, ResolvedWorkflowConfig]:
    return {}


def _study_dataset_state() -> dict[str, str]:
    return {}


def _artifact_root_state() -> dict[str, _MaterializedArtifactRoot]:
    return {}


@dataclass(slots=True)
class BenchmarkRootMaterializer:
    configs_by_run_id: dict[str, ResolvedWorkflowConfig]
    study_dataset_by_study_id: dict[str, str]
    artifact_roots_by_run_id: dict[str, _MaterializedArtifactRoot]

    @classmethod
    def create(cls) -> BenchmarkRootMaterializer:
        return cls(
            configs_by_run_id=_config_state(),
            study_dataset_by_study_id=_study_dataset_state(),
            artifact_roots_by_run_id=_artifact_root_state(),
        )

    def prepare_selection(
        self,
        workflow_selection: WorkflowSelection,
        dependencies: BenchmarkDependencyLedger,
    ) -> _PreparedRootSelection:
        ledger = BenchmarkRootLedger(
            artifact_from_run_id=dependencies.artifact_from_run_id,
        )
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
                ledger=ledger.model_copy(
                    update={"consumed": BenchmarkConsumedRoots(study_id=study_id)}
                ),
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
                ledger=ledger.model_copy(
                    update={
                        "consumed": BenchmarkConsumedRoots(
                            artifact_id=materialized.artifact_id,
                            dataset_id=dataset_id,
                        ),
                        "artifact_source_dataset_id": materialized.dataset_id,
                    }
                ),
            )
        return _PreparedRootSelection(selection=workflow_selection, ledger=ledger)

    def finalize_ledger(
        self,
        config: ResolvedWorkflowConfig,
        ledger: BenchmarkRootLedger,
    ) -> BenchmarkRootLedger:
        consumed = self._consumed_roots(config, ledger.consumed)
        produced = self._produced_roots(config)
        return ledger.model_copy(update={"consumed": consumed, "produced": produced})

    def record_config(self, run_id: str, config: ResolvedWorkflowConfig) -> None:
        self.configs_by_run_id[run_id] = config
        if isinstance(config, TuneConfig):
            self.study_dataset_by_study_id[produced_study_id(config)] = config.dataset_id

    def _consumed_roots(
        self,
        config: ResolvedWorkflowConfig,
        fallback: BenchmarkConsumedRoots,
    ) -> BenchmarkConsumedRoots:
        if isinstance(config, TuneConfig):
            return BenchmarkConsumedRoots(dataset_id=config.dataset_id)
        if isinstance(config, TrainConfig):
            return BenchmarkConsumedRoots(
                dataset_id=config.dataset_id,
                study_id=config.study_id,
            )
        if isinstance(config, EvaluateConfig):
            return BenchmarkConsumedRoots(
                dataset_id=config.dataset_id,
                artifact_id=config.artifact_id,
            )
        return fallback

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
            config = self.configs_by_run_id[run_id]
            if isinstance(config, TuneConfig):
                return produced_study_id(config)
        raise ConfigResolutionError("tuned train requires a tune dependency or explicit study_id")

    def _dependency_artifact_root(self, artifact_from_run_id: str) -> _MaterializedArtifactRoot:
        cached = self.artifact_roots_by_run_id.get(artifact_from_run_id)
        if cached is not None:
            return cached
        source = self.configs_by_run_id[artifact_from_run_id]
        if not isinstance(source, TrainConfig):
            raise ConfigResolutionError("artifact_from may reference train steps only")
        dataset_id = self._train_dataset_id(source)
        root = _MaterializedArtifactRoot(
            artifact_id=produced_artifact_id(source, dataset_id=dataset_id),
            dataset_id=dataset_id,
        )
        self.artifact_roots_by_run_id[artifact_from_run_id] = root
        return root

    def _train_dataset_id(self, config: TrainConfig) -> str:
        if config.dataset_id is not None:
            return config.dataset_id
        if config.study_id is None:
            raise ConfigResolutionError(
                "train artifact source did not declare dataset_id or study_id"
            )
        cached = self.study_dataset_by_study_id.get(config.study_id)
        if cached is not None:
            return cached
        try:
            study = resolve_study_record(
                config.storage.root,
                selector=StudySelector(study_id=config.study_id),
            )
        except SelectorResolutionError as exc:
            raise ConfigResolutionError(str(exc)) from exc
        self.study_dataset_by_study_id[config.study_id] = study.dataset_id
        return study.dataset_id
