# pyright: strict

"""Benchmark planning orchestration."""

from __future__ import annotations

from pydantic import ValidationError

from ...config.groups import load_named_group_payload
from ...config.models import (
    EvaluateConfig,
    TrainConfig,
    TuneConfig,
    WorkflowTask,
)
from ...config.resolution import resolve_workflow_config
from ...config.resolved_workflows import ResolvedWorkflowConfig
from ...config.selections import (
    WorkflowSelection,
    workflow_selection_field_set,
    workflow_selection_type,
)
from ...core.errors import ConfigResolutionError
from ..schema import BenchmarkCase, BenchmarkSpec
from ._dependencies import BenchmarkDependencyPlan
from ._expansion import PlanSeed, expand_case, run_ids
from ._models import BenchmarkPlanEntry
from ._roots import BenchmarkPlanLedgerMaterializer
from ._selection import materialize_selection_ledger


def materialize_benchmark_plan(name: str) -> list[BenchmarkPlanEntry]:
    spec = _load_benchmark_spec(name)
    return _materialize_benchmark_spec(name, spec)


def _materialize_benchmark_spec(name: str, spec: BenchmarkSpec) -> list[BenchmarkPlanEntry]:
    entries: list[BenchmarkPlanEntry] = []
    errors: list[str] = []
    for case_index, case in enumerate(spec.cases):
        try:
            entries.extend(_materialize_benchmark_case(case))
        except ConfigResolutionError as exc:
            errors.append(
                _format_benchmark_error(
                    name,
                    case_index=case_index,
                    error=exc,
                )
            )
    if errors:
        raise ConfigResolutionError("\n".join(errors))
    return entries


def _load_benchmark_spec(name: str) -> BenchmarkSpec:
    try:
        return BenchmarkSpec.model_validate(load_named_group_payload(name, "benchmark"))
    except ValidationError as exc:
        raise ConfigResolutionError(str(exc)) from exc


def _format_benchmark_error(
    benchmark: str,
    *,
    case_index: int,
    error: ConfigResolutionError,
) -> str:
    return f"benchmark {benchmark} case {case_index}: {error.message}"


def _materialize_benchmark_case(case: BenchmarkCase) -> list[BenchmarkPlanEntry]:
    dependency_plan = BenchmarkDependencyPlan.from_steps(case.steps)
    seeds = expand_case(case)
    resolved_run_ids = run_ids(seeds)
    ledger_materializer = BenchmarkPlanLedgerMaterializer.create(
        seeds,
        resolved_run_ids,
        dependency_plan=dependency_plan,
    )
    entries: list[BenchmarkPlanEntry] = []
    for index, seed in enumerate(seeds):
        try:
            workflow_selection = _selection_for_seed(seed)
            ledgers = ledger_materializer.materialize(
                seed=seed,
                run_id=resolved_run_ids[index],
                workflow_selection=workflow_selection,
                resolve_config=_resolve_benchmark_config,
            )
            entry = BenchmarkPlanEntry(
                run_id=resolved_run_ids[index],
                case_id=seed.case_id,
                step_id=seed.step_id,
                workflow=seed.workflow,
                dependencies=ledgers.dependencies,
                dimension_labels=dict(seed.dimension_labels),
                selection=materialize_selection_ledger(
                    source_row=seed.row,
                    workflow_selection=ledgers.selection,
                ),
                root_facts=ledgers.root_facts,
                root_ledger=ledgers.root_ledger,
                config=ledgers.config,
            )
            entries.append(entry)
        except ConfigResolutionError as exc:
            raise ConfigResolutionError(
                f"case {case.id} step {seed.step_id}: {exc.message}"
            ) from exc
        except (ValidationError, ValueError, TypeError) as exc:
            raise ConfigResolutionError(
                f"case {case.id} step {seed.step_id}: {exc}"
            ) from exc
    return entries


def _resolve_benchmark_config(
    selection: WorkflowSelection,
) -> ResolvedWorkflowConfig:
    config = resolve_workflow_config(selection)
    if isinstance(config, (TrainConfig, TuneConfig, EvaluateConfig)):
        return config
    raise ConfigResolutionError("benchmark plans support train, tune, and evaluate workflows")


def _selection_for_seed(seed: PlanSeed) -> WorkflowSelection:
    try:
        fields = workflow_selection_field_set(seed.workflow)
        payload = {
            key: value
            for key, value in seed.row.items()
            if key in fields and value is not None
        }
        selection = workflow_selection_type(seed.workflow).model_validate(payload)
        if (
            seed.workflow is not WorkflowTask.EVALUATE
            and getattr(selection, "surface", None) is None
        ):
            raise ConfigResolutionError("surface is required")
        return selection
    except (ConfigResolutionError, ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc
