# pyright: strict

"""Benchmark dependency ledger materialization."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..config.models import WorkflowTask
from ..core.errors import ConfigResolutionError
from .schema import AfterDependency, BenchmarkStep, SlurmAfterDependency


class BenchmarkPlanSeed(Protocol):
    step_id: str
    workflow: WorkflowTask
    dimension_labels: Mapping[str, str]
    depends_on_steps: tuple[str, ...]
    external_dependencies: tuple[str, ...]
    artifact_from_step: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkDependencyLedger:
    local_run_ids: tuple[str, ...]
    external_slurm_dependencies: tuple[str, ...]
    artifact_from_run_id: str | None


class BenchmarkDependencyResolver:
    def __init__(
        self,
        seeds: Sequence[BenchmarkPlanSeed],
        run_ids: Sequence[str],
    ) -> None:
        self._by_step: dict[str, list[tuple[BenchmarkPlanSeed, str]]] = defaultdict(list)
        for seed, run_id in zip(seeds, run_ids, strict=True):
            self._by_step[seed.step_id].append((seed, run_id))

    def resolve(self, seed: BenchmarkPlanSeed) -> BenchmarkDependencyLedger:
        artifact_from_run_id = self._resolve_artifact_from_run_id(seed)
        local_run_ids = self._resolve_local_run_ids(
            seed,
            artifact_from_run_id=artifact_from_run_id,
        )
        return BenchmarkDependencyLedger(
            local_run_ids=local_run_ids,
            external_slurm_dependencies=seed.external_dependencies,
            artifact_from_run_id=artifact_from_run_id,
        )

    def _resolve_local_run_ids(
        self,
        seed: BenchmarkPlanSeed,
        *,
        artifact_from_run_id: str | None,
    ) -> tuple[str, ...]:
        run_ids: list[str] = []
        for step_id in seed.depends_on_steps:
            candidates = [
                run_id
                for candidate, run_id in self._by_step[step_id]
                if _labels_match(candidate.dimension_labels, seed.dimension_labels)
            ]
            if not candidates:
                raise ConfigResolutionError(f"dependency {step_id} has no matching plan row")
            if len(candidates) > 1:
                raise ConfigResolutionError(f"dependency {step_id} is ambiguous")
            run_ids.append(candidates[0])
        if artifact_from_run_id is not None and artifact_from_run_id not in run_ids:
            run_ids.append(artifact_from_run_id)
        return tuple(run_ids)

    def _resolve_artifact_from_run_id(self, seed: BenchmarkPlanSeed) -> str | None:
        step_id = seed.artifact_from_step
        if step_id is None:
            return None
        candidates = [
            (candidate, run_id)
            for candidate, run_id in self._by_step[step_id]
            if _labels_match(candidate.dimension_labels, seed.dimension_labels)
        ]
        if not candidates:
            raise ConfigResolutionError(f"artifact_from {step_id} has no matching plan row")
        if len(candidates) > 1:
            raise ConfigResolutionError(f"artifact_from {step_id} is ambiguous")
        candidate, run_id = candidates[0]
        if candidate.workflow is not WorkflowTask.TRAIN:
            raise ConfigResolutionError("artifact_from may reference train steps only")
        return run_id


def local_after_steps(after: Sequence[AfterDependency]) -> tuple[str, ...]:
    return tuple(value for value in after if isinstance(value, str))


def external_after_dependencies(
    after: Sequence[AfterDependency],
) -> tuple[str, ...]:
    return tuple(value.slurm for value in after if isinstance(value, SlurmAfterDependency))


def validate_step_graph(steps: Sequence[BenchmarkStep]) -> None:
    step_ids = [step.id for step in steps]
    if len(set(step_ids)) != len(step_ids):
        raise ConfigResolutionError("benchmark step ids must be unique")
    step_id_set = set(step_ids)
    positions = {step.id: index for index, step in enumerate(steps)}
    edges: dict[str, set[str]] = {step.id: set() for step in steps}
    for step in steps:
        for dependency in _local_dependency_steps(step):
            if dependency not in step_id_set:
                raise ConfigResolutionError(f"step {step.id} depends on unknown step {dependency}")
            if dependency == step.id:
                raise ConfigResolutionError(f"step {step.id} cannot depend on itself")
            if positions[dependency] > positions[step.id]:
                raise ConfigResolutionError(f"step {step.id} depends on future step {dependency}")
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
        raise ConfigResolutionError("benchmark step dependencies contain a cycle")


def _local_dependency_steps(step: BenchmarkStep) -> tuple[str, ...]:
    dependencies = list(local_after_steps(step.after))
    if step.artifact_from is not None and step.artifact_from not in dependencies:
        dependencies.append(step.artifact_from)
    return tuple(dependencies)


def _labels_match(upstream: Mapping[str, str], downstream: Mapping[str, str]) -> bool:
    return all(downstream.get(name) == label for name, label in upstream.items())
