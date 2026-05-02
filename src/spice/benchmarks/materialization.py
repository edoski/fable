# pyright: strict

"""Benchmark Plan Materialization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import cast

from pydantic import ValidationError

from ..config.models import (
    EvaluateConfig,
    TrainConfig,
    TuneConfig,
    WorkflowTask,
    coerce_problem_spec,
)
from ..config.registry import load_named_group_payload, load_problem_spec
from ..config.resolution import resolve_workflow_config
from ..config.selections import (
    WorkflowSelection,
    workflow_selection_from_values,
)
from ..config.workflow_snapshots import ResolvedWorkflowConfig
from ..core.errors import ConfigResolutionError
from .dependency_ledger import (
    BenchmarkDependencyResolver,
    external_after_dependencies,
    local_after_steps,
    validate_step_graph,
)
from .models import BenchmarkPlanEntry
from .root_ledger import BenchmarkRootMaterializer
from .schema import (
    BenchmarkCase,
    BenchmarkSpec,
    ProblemDimensionEntry,
    SetDimensionEntry,
)
from .selection_ledger import materialize_selection_ledger


@dataclass(frozen=True, slots=True)
class _PlanSeed:
    case_id: str
    step_id: str
    workflow: WorkflowTask
    dimension_labels: Mapping[str, str]
    depends_on_steps: tuple[str, ...]
    external_dependencies: tuple[str, ...]
    artifact_from_step: str | None
    row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _DimensionVariant:
    dimension: str
    label: str
    patch: Mapping[str, object]


def plan_benchmark(name: str) -> list[BenchmarkPlanEntry]:
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
    validate_step_graph(case.steps)
    seeds = _expand_case(case)
    run_ids = _run_ids(seeds)
    dependency_resolver = BenchmarkDependencyResolver(seeds, run_ids)
    root_materializer = BenchmarkRootMaterializer.create()
    entries: list[BenchmarkPlanEntry] = []
    for index, seed in enumerate(seeds):
        try:
            dependencies = dependency_resolver.resolve(seed)
            workflow_selection = _selection_for_seed(seed)
            prepared_roots = root_materializer.prepare_selection(
                workflow_selection,
                dependencies,
            )
            config = _resolve_benchmark_config(seed.workflow, prepared_roots.selection)
            roots = root_materializer.finalize_ledger(config, prepared_roots.ledger)
            entry = BenchmarkPlanEntry(
                run_id=run_ids[index],
                case_id=seed.case_id,
                step_id=seed.step_id,
                workflow=seed.workflow,
                dependencies=dependencies,
                dimension_labels=dict(seed.dimension_labels),
                selection=materialize_selection_ledger(
                    source_row=seed.row,
                    workflow_selection=prepared_roots.selection,
                ),
                roots=roots,
                config=config,
            )
            entries.append(entry)
            root_materializer.record_config(entry.run_id, config)
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
    workflow: WorkflowTask,
    selection: WorkflowSelection,
) -> ResolvedWorkflowConfig:
    config = resolve_workflow_config(workflow, selection)
    if isinstance(config, (TrainConfig, TuneConfig, EvaluateConfig)):
        return config
    raise ConfigResolutionError("benchmark plans support train, tune, and evaluate workflows")


def _expand_case(case: BenchmarkCase) -> list[_PlanSeed]:
    global_combos = _dimension_combinations(_expand_dimensions(case.dimensions))
    seeds: list[_PlanSeed] = []
    for global_combo in global_combos:
        global_patch = _merge_patches([variant.patch for variant in global_combo])
        global_labels = {variant.dimension: variant.label for variant in global_combo}
        for step in case.steps:
            step_combos = _dimension_combinations(_expand_step_dimensions(step.dimensions))
            for step_combo in step_combos:
                step_patch = _merge_patches([variant.patch for variant in step_combo])
                step_labels = {
                    **global_labels,
                    **{variant.dimension: variant.label for variant in step_combo},
                }
                row = {
                    **case.base,
                    **global_patch,
                    **step.set,
                    **step_patch,
                }
                seeds.append(
                    _PlanSeed(
                        case_id=case.id,
                        step_id=step.id,
                        workflow=step.workflow,
                        dimension_labels=step_labels,
                        depends_on_steps=local_after_steps(step.after),
                        external_dependencies=external_after_dependencies(step.after),
                        artifact_from_step=step.artifact_from,
                        row=row,
                    )
                )
    return seeds


def _selection_for_seed(seed: _PlanSeed) -> WorkflowSelection:
    try:
        selection = workflow_selection_from_values(seed.workflow, seed.row)
        if (
            seed.workflow is not WorkflowTask.EVALUATE
            and getattr(selection, "surface", None) is None
        ):
            raise ConfigResolutionError("surface is required")
        return selection
    except (ConfigResolutionError, ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc


def _expand_dimensions(
    dimensions: Mapping[str, list[SetDimensionEntry] | list[ProblemDimensionEntry]],
) -> list[list[_DimensionVariant]]:
    expanded: list[list[_DimensionVariant]] = []
    for name, entries in dimensions.items():
        if name == "problems":
            variants: list[_DimensionVariant] = []
            for entry in cast(list[ProblemDimensionEntry], entries):
                variants.extend(_expand_problem_entry(entry))
            expanded.append(variants)
            continue
        expanded.append(
            [
                _DimensionVariant(
                    dimension=name,
                    label=_label_for_patch(entry.set),
                    patch=entry.set,
                )
                for entry in cast(list[SetDimensionEntry], entries)
            ]
        )
    return expanded


def _expand_step_dimensions(
    dimensions: Mapping[str, list[SetDimensionEntry]],
) -> list[list[_DimensionVariant]]:
    return [
        [
            _DimensionVariant(
                dimension=name,
                label=_label_for_patch(entry.set),
                patch=entry.set,
            )
            for entry in entries
        ]
        for name, entries in dimensions.items()
    ]


def _expand_problem_entry(entry: ProblemDimensionEntry) -> list[_DimensionVariant]:
    if entry.ref is not None:
        return [
            _DimensionVariant(
                dimension="problems",
                label=entry.ref,
                patch={"problem": entry.ref},
            )
        ]
    grid = entry.grid
    if grid is None:
        raise ConfigResolutionError("problem dimension entry is empty")
    base_problem = load_problem_spec(grid.base)
    field_names = tuple(grid.fields)
    variants: list[_DimensionVariant] = []
    for values in product(*(grid.fields[field] for field in field_names)):
        updates = dict(zip(field_names, values, strict=True))
        problem_id = _problem_grid_id(grid.base, updates)
        problem = coerce_problem_spec(
            {
                **base_problem.model_dump(mode="json"),
                **updates,
                "id": problem_id,
            }
        )
        variants.append(
            _DimensionVariant(
                dimension="problems",
                label=problem_id,
                patch={"problem": problem},
            )
        )
    return variants


def _dimension_combinations(
    dimensions: Sequence[Sequence[_DimensionVariant]],
) -> list[tuple[_DimensionVariant, ...]]:
    if not dimensions:
        return [()]
    return list(product(*dimensions))


def _merge_patches(patches: Iterable[Mapping[str, object]]) -> dict[str, object]:
    merged: dict[str, object] = {}
    owners: dict[str, object] = {}
    for patch in patches:
        for key, value in patch.items():
            previous = owners.get(key)
            if previous is not None and merged[key] != value:
                raise ConfigResolutionError(f"field {key} is set by multiple dimensions")
            owners[key] = value
            merged[key] = value
    return merged


def _run_ids(seeds: Sequence[_PlanSeed]) -> list[str]:
    run_ids: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        parts = [seed.case_id]
        parts.extend(f"{name}-{label}" for name, label in seed.dimension_labels.items())
        parts.append(seed.step_id)
        run_id = ".".join(parts)
        if run_id in seen:
            raise ConfigResolutionError(f"duplicate benchmark run_id: {run_id}")
        seen.add(run_id)
        run_ids.append(run_id)
    return run_ids


def _label_for_patch(patch: Mapping[str, object]) -> str:
    return "__".join(f"{key}-{_label_value(value)}" for key, value in patch.items())


def _label_value(value: object) -> str:
    return str(value).replace(".", "_")


def _problem_grid_id(base: str, updates: Mapping[str, int]) -> str:
    suffix = "__".join(f"{field}-{value}" for field, value in updates.items())
    return f"{base}__{suffix}"
