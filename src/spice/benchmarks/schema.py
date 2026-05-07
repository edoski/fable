# pyright: strict

"""Benchmark YAML schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config.models import WorkflowTask
from ..config.selections import (
    benchmark_base_fields,
    benchmark_dimension_field_names,
    benchmark_dimension_fields,
    benchmark_workflows,
    workflow_selection_field_set,
)

_MODEL_WORKFLOWS = benchmark_workflows()
_STANDARD_DIMENSIONS = benchmark_dimension_field_names() | {"problems"}
_PROBLEM_GRID_FIELDS = frozenset({"lookback_seconds", "sample_count", "max_delay_seconds"})
_BASE_FIELDS = benchmark_base_fields()


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
    artifact_from: str | None = None

    @model_validator(mode="after")
    def validate_step(self) -> BenchmarkStep:
        if self.workflow not in _MODEL_WORKFLOWS:
            raise ValueError(f"benchmark step workflow {self.workflow.value} is not supported")
        _reject_lists(self.set, label=f"step {self.id} set")
        _validate_workflow_fields(
            self.workflow,
            self.set,
            label=f"step {self.id} set",
        )
        unknown_dimensions = sorted(set(self.dimensions) - _STANDARD_DIMENSIONS)
        if unknown_dimensions:
            raise ValueError("unknown step dimensions: " + ", ".join(unknown_dimensions))
        if "problems" in self.dimensions:
            raise ValueError("step dimensions do not support problems")
        if self.artifact_from is not None and self.workflow is not WorkflowTask.EVALUATE:
            raise ValueError("artifact_from is only valid on evaluate steps")
        for name, entries in self.dimensions.items():
            if not entries:
                raise ValueError(f"step dimension {name} cannot be empty")
            _validate_dimension_set_fields(name, [entry.set for entry in entries])
            for entry in entries:
                _validate_workflow_fields(
                    self.workflow,
                    entry.set,
                    label=f"step dimension {name}",
                )
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
        return self


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[BenchmarkCase] = Field(min_length=1)


def _validate_dimension_set_fields(name: str, patches: Sequence[Mapping[str, object]]) -> None:
    allowed = benchmark_dimension_fields(name)
    if allowed is None:
        return
    for patch in patches:
        unknown = sorted(set(patch) - allowed)
        if unknown:
            raise ValueError(f"dimension {name} does not support fields: " + ", ".join(unknown))
        _reject_lists(patch, label=f"dimension {name}")


def _validate_workflow_fields(
    workflow: WorkflowTask,
    patch: Mapping[str, object],
    *,
    label: str,
) -> None:
    unknown = sorted(set(patch) - workflow_selection_field_set(workflow))
    if unknown:
        raise ValueError(
            f"{label} for {workflow.value} does not support fields: " + ", ".join(unknown)
        )


def _reject_lists(payload: Mapping[str, object], *, label: str) -> None:
    for key, value in payload.items():
        if isinstance(value, list):
            raise ValueError(f"{label} field {key} cannot be a list")
