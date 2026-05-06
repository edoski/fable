# pyright: strict

"""Storage fact access for benchmark plan materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...config.workflow_snapshots import ResolvedWorkflowConfig
from ...core.errors import ConfigResolutionError, SelectorResolutionError
from ...storage.catalog.index import resolve_artifact_record, resolve_study_record
from ...storage.root_identity import (
    ConsumedRootFacts,
    ProducedRootFacts,
    consumed_root_facts,
    produced_root_facts,
)
from ...storage.selectors import ArtifactSelector, StudySelector


@dataclass(frozen=True, slots=True)
class BenchmarkRootStorageAdapter:
    def consumed_roots(self, config: ResolvedWorkflowConfig) -> ConsumedRootFacts:
        return consumed_root_facts(config)

    def produced_roots(
        self,
        config: ResolvedWorkflowConfig,
        *,
        dataset_id: str | None = None,
    ) -> ProducedRootFacts:
        return produced_root_facts(config, dataset_id=dataset_id)

    def study_dataset_id(
        self,
        storage_root: Path,
        *,
        study_id: str,
    ) -> str:
        try:
            study = resolve_study_record(
                storage_root,
                selector=StudySelector(study_id=study_id),
            )
        except SelectorResolutionError as exc:
            raise ConfigResolutionError(str(exc)) from exc
        return study.dataset_id

    def artifact_dataset_id(
        self,
        storage_root: Path,
        *,
        artifact_id: str,
    ) -> str:
        try:
            artifact = resolve_artifact_record(
                storage_root,
                selector=ArtifactSelector(artifact_id=artifact_id),
            )
        except SelectorResolutionError as exc:
            raise ConfigResolutionError(str(exc)) from exc
        return artifact.dataset_id
