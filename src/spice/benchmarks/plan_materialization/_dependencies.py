# pyright: strict

"""Benchmark dependency ledger materialization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Protocol

from ...config.models import WorkflowTask
from ...core.errors import ConfigResolutionError
from ..schema import AfterDependency, BenchmarkStep, SlurmAfterDependency
from ._models import BenchmarkDependencyLedger


class BenchmarkPlanSeed(Protocol):
    @property
    def step_id(self) -> str: ...

    @property
    def workflow(self) -> WorkflowTask: ...

    @property
    def dimension_labels(self) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkStepDependencies:
    local_step_ids: tuple[str, ...]
    external_slurm_dependencies: tuple[str, ...]
    artifact_from_step: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkDependencyPlan:
    by_step_id: Mapping[str, BenchmarkStepDependencies]

    @classmethod
    def from_steps(cls, steps: Sequence[BenchmarkStep]) -> BenchmarkDependencyPlan:
        _validate_step_graph(steps)
        return cls(
            by_step_id={
                step.id: BenchmarkStepDependencies(
                    local_step_ids=_local_after_steps(step.after),
                    external_slurm_dependencies=_external_after_dependencies(step.after),
                    artifact_from_step=step.artifact_from,
                )
                for step in steps
            }
        )

    def for_step(self, step_id: str) -> BenchmarkStepDependencies:
        try:
            return self.by_step_id[step_id]
        except KeyError as exc:
            raise ConfigResolutionError(f"unknown benchmark step: {step_id}") from exc


class BenchmarkDependencyResolver:
    def __init__(
        self,
        seeds: Sequence[BenchmarkPlanSeed],
        run_ids: Sequence[str],
        *,
        dependency_plan: BenchmarkDependencyPlan,
    ) -> None:
        self._dependency_plan = dependency_plan
        self._by_step: dict[str, list[tuple[BenchmarkPlanSeed, str]]] = defaultdict(list)
        for seed, run_id in zip(seeds, run_ids, strict=True):
            self._by_step[seed.step_id].append((seed, run_id))

    def resolve(self, seed: BenchmarkPlanSeed) -> BenchmarkDependencyLedger:
        dependencies = self._dependency_plan.for_step(seed.step_id)
        artifact_from_run_id = self._resolve_artifact_from_run_id(
            seed,
            step_id=dependencies.artifact_from_step,
        )
        local_run_ids = self._resolve_local_run_ids(
            seed,
            step_ids=dependencies.local_step_ids,
            artifact_from_run_id=artifact_from_run_id,
        )
        return BenchmarkDependencyLedger(
            local_run_ids=local_run_ids,
            external_slurm_dependencies=dependencies.external_slurm_dependencies,
            artifact_from_run_id=artifact_from_run_id,
        )

    def _resolve_local_run_ids(
        self,
        seed: BenchmarkPlanSeed,
        *,
        step_ids: tuple[str, ...],
        artifact_from_run_id: str | None,
    ) -> tuple[str, ...]:
        run_ids: list[str] = []
        for step_id in step_ids:
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

    def _resolve_artifact_from_run_id(
        self,
        seed: BenchmarkPlanSeed,
        *,
        step_id: str | None,
    ) -> str | None:
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


def _local_after_steps(after: Sequence[AfterDependency]) -> tuple[str, ...]:
    return tuple(value for value in after if isinstance(value, str))


def _external_after_dependencies(
    after: Sequence[AfterDependency],
) -> tuple[str, ...]:
    return tuple(value.slurm for value in after if isinstance(value, SlurmAfterDependency))


def _validate_step_graph(steps: Sequence[BenchmarkStep]) -> None:
    step_ids = [step.id for step in steps]
    if len(set(step_ids)) != len(step_ids):
        raise ConfigResolutionError("benchmark step ids must be unique")
    step_id_set = set(step_ids)
    positions = {step.id: index for index, step in enumerate(steps)}
    graph: dict[str, tuple[str, ...]] = {}
    for step in steps:
        dependencies = _local_dependency_steps(step)
        graph[step.id] = dependencies
        for dependency in dependencies:
            if dependency not in step_id_set:
                raise ConfigResolutionError(f"step {step.id} depends on unknown step {dependency}")
            if dependency == step.id:
                raise ConfigResolutionError(f"step {step.id} cannot depend on itself")
            if positions[dependency] > positions[step.id]:
                raise ConfigResolutionError(f"step {step.id} depends on future step {dependency}")
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise ConfigResolutionError("benchmark step dependencies contain a cycle") from exc


def _local_dependency_steps(step: BenchmarkStep) -> tuple[str, ...]:
    dependencies = list(_local_after_steps(step.after))
    if step.artifact_from is not None and step.artifact_from not in dependencies:
        dependencies.append(step.artifact_from)
    return tuple(dependencies)


def _labels_match(upstream: Mapping[str, str], downstream: Mapping[str, str]) -> bool:
    return all(downstream.get(name) == label for name, label in upstream.items())
