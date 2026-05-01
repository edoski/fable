# pyright: strict

"""Benchmark Plan Materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from ..config.models import ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig, WorkflowTask
from ..config.resolution import WorkflowConfig, resolve_workflow_config
from ..config.selections import EvaluateWorkflowSelection, TrainWorkflowSelection, WorkflowSelection
from ..core.errors import ConfigResolutionError
from ..storage.dependency_root_materialization import (
    materialize_dependency_artifact,
    materialize_dependency_study_id,
)
from .models import BenchmarkPlanEntry
from .planning import BenchmarkWorkflowSelection


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
        study_id = materialize_dependency_study_id(
            depends_on=selection.depends_on,
            configs_by_run_id=configs_by_run_id,
        )
        return workflow_selection.model_copy(update={"study_id": study_id, "dataset_id": None})
    if (
        selection.workflow is WorkflowTask.EVALUATE
        and isinstance(workflow_selection, EvaluateWorkflowSelection)
        and selection.artifact_from is not None
    ):
        materialized = materialize_dependency_artifact(
            artifact_from=selection.artifact_from,
            configs_by_run_id=configs_by_run_id,
        )
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
