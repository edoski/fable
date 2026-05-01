# pyright: strict

"""Benchmark workflow-selection compilation."""

from __future__ import annotations

from pydantic import ValidationError

from ..config.registry import load_named_group
from ..core.errors import ConfigResolutionError
from .materialization import materialize_benchmark_plan
from .models import BenchmarkPlanEntry
from .planning import plan_benchmark_workflow_selections
from .schema import BenchmarkSpec


def plan_benchmark(name: str) -> list[BenchmarkPlanEntry]:
    spec = _load_benchmark_spec(name)
    entries: list[BenchmarkPlanEntry] = []
    errors: list[str] = []
    for case_index, case in enumerate(spec.cases):
        try:
            selections = plan_benchmark_workflow_selections(BenchmarkSpec(cases=[case]))
            entries.extend(materialize_benchmark_plan(selections))
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


def _format_benchmark_error(
    benchmark: str,
    *,
    case_index: int,
    error: ConfigResolutionError,
) -> str:
    return f"benchmark {benchmark} case {case_index}: {error.message}"
