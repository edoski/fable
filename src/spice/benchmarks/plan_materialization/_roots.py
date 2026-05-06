# pyright: strict

"""Benchmark root ledger materialization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...config.models import WorkflowTask
from ...config.selections import WorkflowSelection
from ...config.workflow_snapshots import ResolvedWorkflowConfig
from ._dependencies import (
    BenchmarkDependencyPlan,
    BenchmarkDependencyResolver,
    BenchmarkPlanSeed,
)
from ._models import (
    BenchmarkDependencyLedger,
    BenchmarkRootLedger,
)
from ._root_ledger import BenchmarkRootLedgerBuilder


@dataclass(frozen=True, slots=True)
class BenchmarkPlanLedgers:
    dependencies: BenchmarkDependencyLedger
    selection: WorkflowSelection
    config: ResolvedWorkflowConfig
    root_ledger: BenchmarkRootLedger


@dataclass(slots=True)
class BenchmarkPlanLedgerMaterializer:
    dependency_resolver: BenchmarkDependencyResolver
    root_ledger_builder: BenchmarkRootLedgerBuilder

    @classmethod
    def create(
        cls,
        seeds: Sequence[BenchmarkPlanSeed],
        run_ids: Sequence[str],
        *,
        dependency_plan: BenchmarkDependencyPlan,
    ) -> BenchmarkPlanLedgerMaterializer:
        return cls(
            dependency_resolver=BenchmarkDependencyResolver(
                seeds,
                run_ids,
                dependency_plan=dependency_plan,
            ),
            root_ledger_builder=BenchmarkRootLedgerBuilder.create(),
        )

    def materialize(
        self,
        *,
        seed: BenchmarkPlanSeed,
        run_id: str,
        workflow_selection: WorkflowSelection,
        resolve_config: Callable[[WorkflowTask, WorkflowSelection], ResolvedWorkflowConfig],
    ) -> BenchmarkPlanLedgers:
        dependencies = self.dependency_resolver.resolve(seed)
        prepared_roots = self.root_ledger_builder.prepare_selection(
            workflow_selection,
            dependencies,
        )
        config = resolve_config(seed.workflow, prepared_roots.selection)
        root_ledger = self.root_ledger_builder.finalize_ledger(
            run_id=run_id,
            workflow=seed.workflow,
            config=config,
            prepared=prepared_roots,
        )
        self.root_ledger_builder.record_ledger(root_ledger)
        return BenchmarkPlanLedgers(
            dependencies=dependencies,
            selection=prepared_roots.selection,
            config=config,
            root_ledger=root_ledger,
        )
