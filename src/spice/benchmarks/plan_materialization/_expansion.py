# pyright: strict

"""Benchmark case and dimension expansion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import cast

from ...config.models import WorkflowTask
from ...core.errors import ConfigResolutionError
from ..schema import (
    BenchmarkCase,
    ProblemDimensionEntry,
    SetDimensionEntry,
)
from ._problem_grid import materialize_problem_variants


@dataclass(frozen=True, slots=True)
class PlanSeed:
    case_id: str
    step_id: str
    workflow: WorkflowTask
    dimension_labels: Mapping[str, str]
    row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _DimensionVariant:
    dimension: str
    label: str
    patch: Mapping[str, object]


def expand_case(case: BenchmarkCase) -> list[PlanSeed]:
    global_combos = _dimension_combinations(_expand_dimensions(case.dimensions))
    seeds: list[PlanSeed] = []
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
                    PlanSeed(
                        case_id=case.id,
                        step_id=step.id,
                        workflow=step.workflow,
                        dimension_labels=step_labels,
                        row=row,
                    )
                )
    return seeds


def run_ids(seeds: Sequence[PlanSeed]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        parts = [seed.case_id]
        parts.extend(f"{name}-{label}" for name, label in seed.dimension_labels.items())
        parts.append(seed.step_id)
        run_id = ".".join(parts)
        if run_id in seen:
            raise ConfigResolutionError(f"duplicate benchmark run_id: {run_id}")
        seen.add(run_id)
        resolved.append(run_id)
    return resolved


def _expand_dimensions(
    dimensions: Mapping[str, list[SetDimensionEntry] | list[ProblemDimensionEntry]],
) -> list[list[_DimensionVariant]]:
    expanded: list[list[_DimensionVariant]] = []
    for name, entries in dimensions.items():
        if name == "problems":
            variants: list[_DimensionVariant] = []
            for entry in cast(list[ProblemDimensionEntry], entries):
                for variant in materialize_problem_variants(entry):
                    variants.append(
                        _DimensionVariant(
                            dimension="problems",
                            label=variant.label,
                            patch={"problem": variant.problem},
                        )
                    )
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


def _label_for_patch(patch: Mapping[str, object]) -> str:
    return "__".join(f"{key}-{_label_value(value)}" for key, value in patch.items())


def _label_value(value: object) -> str:
    return str(value).replace(".", "_")
