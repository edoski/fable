# pyright: strict

"""Benchmark-local problem-grid materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from ...config import typed_groups as typed
from ...config.models import ProblemSpec, coerce_problem_spec
from ...core.errors import ConfigResolutionError
from ..schema import ProblemDimensionEntry


@dataclass(frozen=True, slots=True)
class MaterializedProblemVariant:
    label: str
    problem: str | ProblemSpec


def materialize_problem_variants(
    entry: ProblemDimensionEntry,
) -> list[MaterializedProblemVariant]:
    if entry.ref is not None:
        return [MaterializedProblemVariant(label=entry.ref, problem=entry.ref)]
    grid = entry.grid
    if grid is None:
        raise ConfigResolutionError("problem dimension entry is empty")
    base_problem = typed.load(typed.PROBLEM, grid.base)
    field_names = tuple(grid.fields)
    variants: list[MaterializedProblemVariant] = []
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
        variants.append(MaterializedProblemVariant(label=problem_id, problem=problem))
    return variants


def _problem_grid_id(base: str, updates: Mapping[str, int]) -> str:
    suffix = "__".join(f"{field}-{value}" for field, value in updates.items())
    return f"{base}__{suffix}"
