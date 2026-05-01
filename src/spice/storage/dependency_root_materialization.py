"""Dependency-produced root id materialization for benchmark plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..config.models import AcquireConfig, EvaluateConfig, TrainConfig, TuneConfig
from ..core.errors import ConfigResolutionError, SelectorResolutionError
from .catalog.index import resolve_study_record
from .root_producer_handles import produced_artifact_id, produced_study_id
from .selectors import StudySelector

WorkflowConfig = AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig


@dataclass(frozen=True, slots=True)
class DependencyArtifactMaterialization:
    artifact_id: str
    dataset_id: str


def materialize_dependency_study_id(
    *,
    depends_on: Sequence[str],
    configs_by_run_id: Mapping[str, WorkflowConfig],
) -> str:
    for run_id in depends_on:
        config = configs_by_run_id[run_id]
        if isinstance(config, TuneConfig):
            return produced_study_id(config)
    raise ConfigResolutionError("tuned train requires a tune dependency or explicit study_id")


def materialize_dependency_artifact(
    *,
    artifact_from: str,
    configs_by_run_id: Mapping[str, WorkflowConfig],
) -> DependencyArtifactMaterialization:
    source = configs_by_run_id[artifact_from]
    if not isinstance(source, TrainConfig):
        raise ConfigResolutionError("artifact_from may reference train steps only")
    dataset_id = materialize_train_dataset_id(source, configs_by_run_id=configs_by_run_id)
    return DependencyArtifactMaterialization(
        artifact_id=produced_artifact_id(source, dataset_id=dataset_id),
        dataset_id=dataset_id,
    )


def materialize_train_dataset_id(
    config: TrainConfig,
    *,
    configs_by_run_id: Mapping[str, WorkflowConfig],
) -> str:
    if config.dataset_id is not None:
        return config.dataset_id
    if config.study_id is None:
        raise ConfigResolutionError("train artifact source did not declare dataset_id or study_id")
    for candidate in configs_by_run_id.values():
        if isinstance(candidate, TuneConfig) and produced_study_id(candidate) == config.study_id:
            return candidate.dataset_id
    try:
        study = resolve_study_record(
            config.storage.root,
            selector=StudySelector(study_id=config.study_id),
        )
    except SelectorResolutionError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    return study.dataset_id
