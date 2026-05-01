# pyright: strict

"""Benchmark Plan Materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from ..config.models import ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig, WorkflowTask
from ..config.resolution import WorkflowConfig, resolve_workflow_config
from ..config.selections import EvaluateWorkflowSelection, TrainWorkflowSelection, WorkflowSelection
from ..core.errors import ConfigResolutionError, SelectorResolutionError
from ..storage.catalog.index import resolve_study_record
from ..storage.selectors import StudySelector
from ..storage.workflow_roots import produced_artifact_id, produced_study_id
from .models import BenchmarkPlanEntry
from .planning import BenchmarkWorkflowSelection


@dataclass(frozen=True, slots=True)
class _MaterializedArtifactRoot:
    artifact_id: str
    dataset_id: str


def materialize_benchmark_plan(
    selections: Sequence[BenchmarkWorkflowSelection],
) -> list[BenchmarkPlanEntry]:
    entries: list[BenchmarkPlanEntry] = []
    configs_by_run_id: dict[str, WorkflowConfig] = {}
    for selection in selections:
        entry = _materialize_benchmark_selection(
            selection,
            configs_by_run_id=configs_by_run_id,
        )
        entries.append(entry)
        configs_by_run_id[entry.run_id] = entry.config
    return entries


def _materialize_benchmark_selection(
    selection: BenchmarkWorkflowSelection,
    *,
    configs_by_run_id: Mapping[str, WorkflowConfig],
) -> BenchmarkPlanEntry:
    try:
        workflow_selection = _materialized_selection(selection, configs_by_run_id)
        config = _resolved_benchmark_config(
            resolve_workflow_config(selection.workflow, workflow_selection)
        )
    except (ConfigResolutionError, ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(
            f"case {selection.case_id} step {selection.step_id}: {exc}"
        ) from exc
    return BenchmarkPlanEntry(
        run_id=selection.run_id,
        case_id=selection.case_id,
        step_id=selection.step_id,
        workflow=selection.workflow,
        depends_on=selection.depends_on,
        external_dependencies=selection.external_dependencies,
        dimension_labels=dict(selection.dimension_labels),
        selection=_materialized_selection_payload(selection, workflow_selection),
        artifact_from=selection.artifact_from,
        config=config,
    )


def _materialized_selection_payload(
    selection: BenchmarkWorkflowSelection,
    workflow_selection: WorkflowSelection,
) -> dict[str, object]:
    payload = dict(selection.selection_payload)
    if isinstance(workflow_selection, TrainWorkflowSelection):
        if workflow_selection.study_id is not None:
            payload["study_id"] = workflow_selection.study_id
    if isinstance(workflow_selection, EvaluateWorkflowSelection):
        if workflow_selection.artifact_id is not None:
            payload["artifact_id"] = workflow_selection.artifact_id
        if workflow_selection.dataset_id is not None:
            payload["dataset_id"] = workflow_selection.dataset_id
    return payload


def _materialized_selection(
    selection: BenchmarkWorkflowSelection,
    configs_by_run_id: Mapping[str, WorkflowConfig],
) -> WorkflowSelection:
    workflow_selection = selection.selection
    if (
        selection.workflow is WorkflowTask.TRAIN
        and isinstance(workflow_selection, TrainWorkflowSelection)
        and workflow_selection.variant == ArtifactVariant.TUNED.value
        and workflow_selection.study_id is None
    ):
        study_id = _dependency_study_id(selection.depends_on, configs_by_run_id)
        return workflow_selection.model_copy(update={"study_id": study_id, "dataset_id": None})
    if (
        selection.workflow is WorkflowTask.EVALUATE
        and isinstance(workflow_selection, EvaluateWorkflowSelection)
        and selection.artifact_from is not None
    ):
        materialized = _dependency_artifact_root(selection.artifact_from, configs_by_run_id)
        updates: dict[str, object] = {
            "artifact_id": materialized.artifact_id,
        }
        if workflow_selection.dataset_id is None:
            updates["dataset_id"] = materialized.dataset_id
        return workflow_selection.model_copy(update=updates)
    return workflow_selection


def _resolved_benchmark_config(config: WorkflowConfig) -> TrainConfig | TuneConfig | EvaluateConfig:
    if isinstance(config, (TrainConfig, TuneConfig, EvaluateConfig)):
        return config
    raise ConfigResolutionError("benchmark plans support train, tune, and evaluate workflows")


def _dependency_study_id(
    depends_on: Sequence[str],
    configs_by_run_id: Mapping[str, WorkflowConfig],
) -> str:
    for run_id in depends_on:
        config = configs_by_run_id[run_id]
        if isinstance(config, TuneConfig):
            return produced_study_id(config)
    raise ConfigResolutionError("tuned train requires a tune dependency or explicit study_id")


def _dependency_artifact_root(
    artifact_from: str,
    configs_by_run_id: Mapping[str, WorkflowConfig],
) -> _MaterializedArtifactRoot:
    source = configs_by_run_id[artifact_from]
    if not isinstance(source, TrainConfig):
        raise ConfigResolutionError("artifact_from may reference train steps only")
    dataset_id = _train_dataset_id(source, configs_by_run_id)
    return _MaterializedArtifactRoot(
        artifact_id=produced_artifact_id(source, dataset_id=dataset_id),
        dataset_id=dataset_id,
    )


def _train_dataset_id(
    config: TrainConfig,
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
