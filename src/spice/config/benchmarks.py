# pyright: strict

"""Benchmark matrix planning."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..core.errors import ConfigResolutionError
from .models import ProblemSpec, WorkflowTask, coerce_problem_spec
from .registry import load_named_group
from .resolution import (
    WorkflowConfig,
    WorkflowConfigRequest,
    resolve_workflow_config,
    resolve_workflow_frame_config,
    workflow_request_fields,
    workflow_request_type,
)
from .surfaces import SurfaceFrame, apply_request_overrides, load_surface_frame

_MODEL_WORKFLOWS = frozenset({WorkflowTask.TRAIN, WorkflowTask.TUNE, WorkflowTask.EVALUATE})
_STANDARD_DIMENSIONS = frozenset(
    {
        "data",
        "features",
        "models",
        "problems",
        "scoring",
        "runtime",
    }
)
_DIMENSION_FIELDS = {
    "data": frozenset({"surface", "chain"}),
    "features": frozenset({"features", "surface"}),
    "models": frozenset({"model", "tuning_space"}),
    "scoring": frozenset({"objective", "evaluation"}),
    "runtime": frozenset(
        {
            "training",
            "split",
            "tuning",
            "study",
            "trial_count",
            "variant",
            "delay_seconds",
        }
    ),
}
_PROBLEM_GRID_FIELDS = frozenset({"lookback_seconds", "sample_count", "max_delay_seconds"})
_BASE_FIELDS: frozenset[str] = frozenset(
    field for workflow in _MODEL_WORKFLOWS for field in workflow_request_fields(workflow)
)


class SlurmAfterDependency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slurm: str


AfterDependency: TypeAlias = str | SlurmAfterDependency


class SetDimensionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    set: dict[str, object]

    @model_validator(mode="after")
    def validate_set(self) -> SetDimensionEntry:
        if not self.set:
            raise ValueError("dimension set cannot be empty")
        _reject_lists(self.set, label="dimension set")
        return self


class ProblemGrid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: str
    fields: dict[str, list[int]]

    @model_validator(mode="after")
    def validate_grid(self) -> ProblemGrid:
        if not self.fields:
            raise ValueError("problem grid fields cannot be empty")
        unknown = sorted(set(self.fields) - _PROBLEM_GRID_FIELDS)
        if unknown:
            raise ValueError("unknown problem grid fields: " + ", ".join(unknown))
        for field, values in self.fields.items():
            if not values:
                raise ValueError(f"problem grid field {field} cannot be empty")
            if any(value <= 0 for value in values):
                raise ValueError(f"problem grid field {field} values must be positive")
        return self


class ProblemDimensionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str | None = None
    grid: ProblemGrid | None = None

    @model_validator(mode="after")
    def validate_entry(self) -> ProblemDimensionEntry:
        if (self.ref is None) == (self.grid is None):
            raise ValueError("problem dimension entries must declare exactly one of ref or grid")
        return self


class BenchmarkStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    workflow: WorkflowTask
    set: dict[str, object] = Field(default_factory=dict)
    dimensions: dict[str, list[SetDimensionEntry]] = Field(default_factory=dict)
    after: list[AfterDependency] = Field(default_factory=lambda: [])

    @model_validator(mode="after")
    def validate_step(self) -> BenchmarkStep:
        if self.workflow not in _MODEL_WORKFLOWS:
            raise ValueError(f"benchmark step workflow {self.workflow.value} is not supported")
        _reject_lists(self.set, label=f"step {self.id} set")
        unknown_dimensions = sorted(set(self.dimensions) - _STANDARD_DIMENSIONS)
        if unknown_dimensions:
            raise ValueError("unknown step dimensions: " + ", ".join(unknown_dimensions))
        if "problems" in self.dimensions:
            raise ValueError("step dimensions do not support problems")
        for name, entries in self.dimensions.items():
            if not entries:
                raise ValueError(f"step dimension {name} cannot be empty")
            _validate_dimension_set_fields(name, [entry.set for entry in entries])
        return self


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    base: dict[str, object]
    dimensions: dict[str, list[SetDimensionEntry] | list[ProblemDimensionEntry]] = Field(
        default_factory=dict
    )
    steps: list[BenchmarkStep] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case(self) -> BenchmarkCase:
        if not self.base:
            raise ValueError("benchmark case base cannot be empty")
        _reject_lists(self.base, label=f"case {self.id} base")
        unknown_base = sorted(set(self.base) - _BASE_FIELDS)
        if unknown_base:
            raise ValueError("unknown benchmark base fields: " + ", ".join(unknown_base))
        unknown_dimensions = sorted(set(self.dimensions) - _STANDARD_DIMENSIONS)
        if unknown_dimensions:
            raise ValueError("unknown benchmark dimensions: " + ", ".join(unknown_dimensions))
        step_ids = [step.id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("benchmark step ids must be unique")
        for name, entries in self.dimensions.items():
            if not entries:
                raise ValueError(f"benchmark dimension {name} cannot be empty")
            if name == "problems":
                for entry in entries:
                    if not isinstance(entry, ProblemDimensionEntry):
                        raise ValueError("problems dimension entries must use ref or grid")
                continue
            for entry in entries:
                if not isinstance(entry, SetDimensionEntry):
                    raise ValueError(f"dimension {name} entries must use set")
            _validate_dimension_set_fields(
                name,
                [cast(SetDimensionEntry, entry).set for entry in entries],
            )
        _validate_step_graph(self.steps)
        return self


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[BenchmarkCase] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _DimensionVariant:
    dimension: str
    label: str
    patch: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ExpandedStep:
    case_id: str
    step_id: str
    workflow: WorkflowTask
    dimension_labels: Mapping[str, str]
    depends_on_steps: tuple[str, ...]
    external_dependencies: tuple[str, ...]
    row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BenchmarkPlanEntry:
    run_id: str
    case_id: str
    step_id: str
    workflow: WorkflowTask
    depends_on: tuple[str, ...]
    external_dependencies: tuple[str, ...]
    selection: Mapping[str, object]
    config: WorkflowConfig

    def to_json_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "step_id": self.step_id,
            "workflow": self.workflow.value,
            "depends_on": list(self.depends_on),
            "external_dependencies": list(self.external_dependencies),
            "selection": dict(self.selection),
            "config": self.config.model_dump(mode="json", exclude_none=True),
        }


def plan_benchmark(name: str) -> list[BenchmarkPlanEntry]:
    spec = _load_benchmark_spec(name)
    entries: list[BenchmarkPlanEntry] = []
    errors: list[str] = []
    for case_index, case in enumerate(spec.cases):
        try:
            expanded_steps = _expand_case(case)
            entries.extend(_resolve_case_plan(case, expanded_steps))
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
        return BenchmarkSpec.model_validate(load_named_group(name, "benchmark"))
    except ValidationError as exc:
        raise ConfigResolutionError(str(exc)) from exc


def _expand_case(case: BenchmarkCase) -> list[_ExpandedStep]:
    global_combos = _dimension_combinations(_expand_dimensions(case.dimensions))
    rows: list[_ExpandedStep] = []
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
                rows.append(
                    _ExpandedStep(
                        case_id=case.id,
                        step_id=step.id,
                        workflow=step.workflow,
                        dimension_labels=step_labels,
                        depends_on_steps=_local_after_steps(step.after),
                        external_dependencies=_external_after_dependencies(step.after),
                        row=row,
                    )
                )
    return rows


def _resolve_case_plan(
    case: BenchmarkCase,
    expanded_steps: list[_ExpandedStep],
) -> list[BenchmarkPlanEntry]:
    run_ids = _run_ids(expanded_steps)
    by_step = _entries_by_step(expanded_steps, run_ids)
    entries: list[BenchmarkPlanEntry] = []
    for index, expanded in enumerate(expanded_steps):
        try:
            depends_on = _resolve_dependencies(expanded, by_step)
            config = _resolve_expanded_step(expanded)
            entries.append(
                BenchmarkPlanEntry(
                    run_id=run_ids[index],
                    case_id=expanded.case_id,
                    step_id=expanded.step_id,
                    workflow=expanded.workflow,
                    depends_on=depends_on,
                    external_dependencies=expanded.external_dependencies,
                    selection=_selection_payload(expanded.row),
                    config=config,
                )
            )
        except ConfigResolutionError as exc:
            raise ConfigResolutionError(
                f"case {case.id} step {expanded.step_id}: {exc.message}"
            ) from exc
    return entries


def _resolve_expanded_step(expanded: _ExpandedStep) -> WorkflowConfig:
    workflow = expanded.workflow
    row = expanded.row
    workflow_fields = frozenset(workflow_request_fields(workflow))
    unknown = sorted(set(row) - workflow_fields)
    if unknown:
        raise ConfigResolutionError(
            f"{workflow.value} benchmark step does not support fields: " + ", ".join(unknown)
        )
    request_payload = {
        key: value
        for key, value in row.items()
        if key in workflow_fields and not isinstance(value, ProblemSpec)
    }
    try:
        request = cast(
            WorkflowConfigRequest,
            workflow_request_type(workflow).model_validate(request_payload),
        )
        if request.surface is None:
            raise ConfigResolutionError("surface is required")
        frame = apply_request_overrides(
            load_surface_frame(request.surface),
            chain=request.chain,
            problem=request.problem,
            features=request.features,
            objective=getattr(request, "objective", None),
            evaluation=getattr(request, "evaluation", None),
            model=getattr(request, "model", None),
            tuning_space=getattr(request, "tuning_space", None),
            provider=getattr(request, "provider", None),
            training=getattr(request, "training", None),
            split=getattr(request, "split", None),
            tuning=getattr(request, "tuning", None),
            study=getattr(request, "study", None),
            variant=getattr(request, "variant", None),
            delay_seconds=getattr(request, "delay_seconds", None),
            storage_root=request.storage_root,
        )
        problem = row.get("problem")
        if isinstance(problem, ProblemSpec):
            frame = frame.model_copy(update={"problem": problem})
        if not isinstance(problem, ProblemSpec):
            return resolve_workflow_config(workflow, request)
        return _resolve_frame(workflow, request, frame)
    except (ConfigResolutionError, ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc


def _resolve_frame(
    workflow: WorkflowTask,
    request: WorkflowConfigRequest,
    frame: SurfaceFrame,
) -> WorkflowConfig:
    return resolve_workflow_frame_config(workflow, frame, request=request)


def _selection_payload(row: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, ProblemSpec):
            payload[key] = value.id
            continue
        payload[key] = value
    return payload


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
    expanded: list[list[_DimensionVariant]] = []
    for name, entries in dimensions.items():
        expanded.append(
            [
                _DimensionVariant(
                    dimension=name,
                    label=_label_for_patch(entry.set),
                    patch=entry.set,
                )
                for entry in entries
            ]
        )
    return expanded


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
    base_problem = coerce_problem_spec(load_named_group(grid.base, "problem"))
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


def _run_ids(entries: Sequence[_ExpandedStep]) -> list[str]:
    run_ids: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        parts = [entry.case_id]
        parts.extend(f"{name}-{label}" for name, label in entry.dimension_labels.items())
        parts.append(entry.step_id)
        run_id = ".".join(parts)
        if run_id in seen:
            raise ConfigResolutionError(f"duplicate benchmark run_id: {run_id}")
        seen.add(run_id)
        run_ids.append(run_id)
    return run_ids


def _entries_by_step(
    entries: Sequence[_ExpandedStep],
    run_ids: Sequence[str],
) -> Mapping[str, list[tuple[_ExpandedStep, str]]]:
    by_step: dict[str, list[tuple[_ExpandedStep, str]]] = defaultdict(list)
    for entry, run_id in zip(entries, run_ids, strict=True):
        by_step[entry.step_id].append((entry, run_id))
    return by_step


def _resolve_dependencies(
    entry: _ExpandedStep,
    by_step: Mapping[str, list[tuple[_ExpandedStep, str]]],
) -> tuple[str, ...]:
    run_ids: list[str] = []
    for step_id in entry.depends_on_steps:
        candidates = [
            run_id
            for candidate, run_id in by_step[step_id]
            if _labels_match(candidate.dimension_labels, entry.dimension_labels)
        ]
        if not candidates:
            raise ConfigResolutionError(f"dependency {step_id} has no matching expanded row")
        if len(candidates) > 1:
            raise ConfigResolutionError(f"dependency {step_id} is ambiguous")
        run_ids.append(candidates[0])
    return tuple(run_ids)


def _labels_match(upstream: Mapping[str, str], downstream: Mapping[str, str]) -> bool:
    return all(downstream.get(name) == label for name, label in upstream.items())


def _local_after_steps(after: Sequence[AfterDependency]) -> tuple[str, ...]:
    return tuple(value for value in after if isinstance(value, str))


def _external_after_dependencies(
    after: Sequence[AfterDependency],
) -> tuple[str, ...]:
    return tuple(value.slurm for value in after if isinstance(value, SlurmAfterDependency))


def _validate_step_graph(steps: Sequence[BenchmarkStep]) -> None:
    step_ids = {step.id for step in steps}
    positions = {step.id: index for index, step in enumerate(steps)}
    edges: dict[str, set[str]] = {step.id: set() for step in steps}
    for step in steps:
        for dependency in _local_after_steps(step.after):
            if dependency not in step_ids:
                raise ValueError(f"step {step.id} depends on unknown step {dependency}")
            if dependency == step.id:
                raise ValueError(f"step {step.id} cannot depend on itself")
            if positions[dependency] > positions[step.id]:
                raise ValueError(f"step {step.id} depends on future step {dependency}")
            edges[dependency].add(step.id)
    indegree = {step.id: 0 for step in steps}
    for dependents in edges.values():
        for dependent in dependents:
            indegree[dependent] += 1
    queue = deque(step_id for step_id, count in indegree.items() if count == 0)
    visited = 0
    while queue:
        step_id = queue.popleft()
        visited += 1
        for dependent in edges[step_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    if visited != len(steps):
        raise ValueError("benchmark step dependencies contain a cycle")


def _validate_dimension_set_fields(name: str, patches: Sequence[Mapping[str, object]]) -> None:
    allowed = _DIMENSION_FIELDS.get(name)
    if allowed is None:
        return
    for patch in patches:
        unknown = sorted(set(patch) - allowed)
        if unknown:
            raise ValueError(f"dimension {name} does not support fields: " + ", ".join(unknown))
        _reject_lists(patch, label=f"dimension {name}")


def _reject_lists(payload: Mapping[str, object], *, label: str) -> None:
    for key, value in payload.items():
        if isinstance(value, list):
            raise ValueError(f"{label} field {key} cannot be a list")


def _label_for_patch(patch: Mapping[str, object]) -> str:
    return "__".join(f"{key}-{_label_value(value)}" for key, value in patch.items())


def _label_value(value: object) -> str:
    return str(value).replace(".", "_")


def _problem_grid_id(base: str, updates: Mapping[str, int]) -> str:
    suffix = "__".join(f"{field}-{value}" for field, value in updates.items())
    return f"{base}__{suffix}"


def _format_benchmark_error(
    benchmark: str,
    *,
    case_index: int,
    error: ConfigResolutionError,
) -> str:
    return f"benchmark {benchmark} case {case_index}: {error.message}"
