"""Reporter protocol definitions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .state import StageMetricDescriptor, StageMetricValue

ReporterTask = int


class Reporter(Protocol):
    def log(self, message: str, *, level: str = "info") -> None: ...

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> ReporterTask: ...

    def update_task(
        self,
        task_id: ReporterTask,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
    ) -> None: ...

    def finish_task(
        self,
        task_id: ReporterTask,
        *,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
        silent: bool = False,
    ) -> None: ...

    def configure_workflow(
        self,
        *,
        title: str,
        facts: Iterable[tuple[str, str]] = (),
    ) -> None: ...

    def stage_reporter(
        self,
        key: str,
        *,
        label: str,
        total: int | None = None,
        unit: str | None = None,
        status: str = "pending",
        running_status: str = "running",
        done_status: str = "done",
        metric_descriptors: Iterable[StageMetricDescriptor] = (),
    ) -> Reporter: ...

    def set_stage_state(
        self,
        key: str,
        *,
        label: str | None = None,
        status: str | None = None,
        total: int | None = None,
        unit: str | None = None,
        completed: int | None = None,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
        metric_descriptors: Iterable[StageMetricDescriptor] = (),
    ) -> None: ...

    def close(self) -> None: ...
