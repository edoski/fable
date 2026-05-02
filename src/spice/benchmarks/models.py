# pyright: strict

"""Benchmark plan data models."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import WorkflowTask
from ..config.workflow_snapshots import ResolvedWorkflowConfig
from .dependency_ledger import BenchmarkDependencyLedger
from .root_ledger import BenchmarkRootLedger
from .selection_ledger import BenchmarkSelectionLedger


@dataclass(frozen=True, slots=True)
class BenchmarkPlanEntry:
    run_id: str
    case_id: str
    step_id: str
    workflow: WorkflowTask
    dependencies: BenchmarkDependencyLedger
    dimension_labels: dict[str, str]
    selection: BenchmarkSelectionLedger
    roots: BenchmarkRootLedger
    config: ResolvedWorkflowConfig
