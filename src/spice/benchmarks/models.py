# pyright: strict

"""Benchmark plan data models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..config.models import WorkflowTask
from ..config.workflow_snapshots import ResolvedWorkflowConfig


@dataclass(frozen=True, slots=True)
class BenchmarkPlanEntry:
    run_id: str
    case_id: str
    step_id: str
    workflow: WorkflowTask
    depends_on: tuple[str, ...]
    external_dependencies: tuple[str, ...]
    dimension_labels: Mapping[str, str]
    selection: Mapping[str, object]
    artifact_from: str | None
    config: ResolvedWorkflowConfig


@dataclass(frozen=True, slots=True)
class LoadedBenchmarkPlanEntry:
    run_id: str
    case_id: str
    step_id: str
    workflow: WorkflowTask
    depends_on: tuple[str, ...]
    external_dependencies: tuple[str, ...]
    dimension_labels: dict[str, str]
    selection: dict[str, object]
    artifact_from: str | None
    config: ResolvedWorkflowConfig
