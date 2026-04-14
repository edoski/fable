"""Plain and shared reporter implementations."""

from __future__ import annotations

import time
from collections.abc import Iterable

from rich.console import Console

from .metrics import (
    _ACTIVE_STAGE_STATUSES,
    _FINAL_STAGE_STATUSES,
    _progress_bucket,
    _smooth_value,
)
from .protocol import Reporter, ReporterTask
from .state import StageMetricDescriptor, StageMetricValue, _StageState, _TaskBinding


class NullReporter:
    """Silent reporter used by library entrypoints and tests."""

    def log(self, message: str, *, level: str = "info") -> None:
        return None

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> ReporterTask:
        del name, total, unit
        return 0

    def update_task(
        self,
        task_id: ReporterTask,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
    ) -> None:
        del task_id, completed, advance, message, metrics
        return None

    def finish_task(
        self,
        task_id: ReporterTask,
        *,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
        silent: bool = False,
    ) -> None:
        del task_id, message, metrics, silent
        return None

    def configure_workflow(
        self,
        *,
        title: str,
        facts: Iterable[tuple[str, str]] = (),
    ) -> None:
        del title, facts
        return None

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
    ) -> Reporter:
        del key, label, total, unit, status, running_status, done_status, metric_descriptors
        return self

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
    ) -> None:
        del key, label, status, total, unit, completed, message, metrics, metric_descriptors
        return None

    def close(self) -> None:
        return None


class _BoundStageReporter(NullReporter):
    def __init__(
        self,
        owner: _BaseWorkflowReporter,
        *,
        key: str,
        label: str,
        running_status: str,
        done_status: str,
    ) -> None:
        self._owner = owner
        self._key = key
        self._label = label
        self._running_status = running_status
        self._done_status = done_status

    @property
    def console(self) -> Console:
        return self._owner.console

    def log(self, message: str, *, level: str = "info") -> None:
        self._owner.log(message, level=level)

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> ReporterTask:
        return self._owner._start_bound_task(
            self._key,
            label=self._label,
            task_name=name,
            total=total,
            unit=unit,
            running_status=self._running_status,
            done_status=self._done_status,
        )

    def update_task(
        self,
        task_id: ReporterTask,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
    ) -> None:
        self._owner.update_task(
            task_id,
            completed=completed,
            advance=advance,
            message=message,
            metrics=metrics,
        )

    def finish_task(
        self,
        task_id: ReporterTask,
        *,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
        silent: bool = False,
    ) -> None:
        self._owner.finish_task(task_id, message=message, metrics=metrics, silent=silent)

    def configure_workflow(
        self,
        *,
        title: str,
        facts: Iterable[tuple[str, str]] = (),
    ) -> None:
        self._owner.configure_workflow(title=title, facts=facts)

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
    ) -> Reporter:
        return self._owner.stage_reporter(
            key,
            label=label,
            total=total,
            unit=unit,
            status=status,
            running_status=running_status,
            done_status=done_status,
            metric_descriptors=metric_descriptors,
        )

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
    ) -> None:
        self._owner.set_stage_state(
            key,
            label=label,
            status=status,
            total=total,
            unit=unit,
            completed=completed,
            message=message,
            metrics=metrics,
            metric_descriptors=metric_descriptors,
        )

    def close(self) -> None:
        return None


class _BaseWorkflowReporter(NullReporter):
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._workflow_title: str | None = None
        self._workflow_facts: list[tuple[str, str]] = []
        self._stages: dict[str, _StageState] = {}
        self._next_task_id = 1
        self._task_bindings: dict[ReporterTask, _TaskBinding] = {}

    def log(self, message: str, *, level: str = "info") -> None:
        style = None
        prefix = ""
        if level == "warning":
            style = "yellow"
            prefix = "warning: "
        elif level == "error":
            style = "bold red"
            prefix = "error: "
        self.console.print(f"{prefix}{message}", style=style, markup=False)

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> ReporterTask:
        return self._start_bound_task(
            f"task-{self._next_task_id}",
            label=name,
            task_name=name,
            total=total,
            unit=unit,
            running_status="running",
            done_status="done",
        )

    def update_task(
        self,
        task_id: ReporterTask,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
    ) -> None:
        binding = self._task_bindings.get(task_id)
        if binding is None:
            return
        stage = self._stages.get(binding.stage_key)
        if stage is None:
            return
        if stage.started_at is None:
            stage.started_at = time.monotonic()
            stage.last_progress_at = stage.started_at
            stage.last_progress_completed = stage.completed
        stage.finished_at = None
        previous_completed = stage.completed
        if completed is not None:
            stage.completed = max(0, completed)
        if advance is not None:
            stage.completed = max(0, stage.completed + advance)
        self._update_stage_rate(stage, previous_completed=previous_completed)
        stage.metric_values = {metric.id: metric.value for metric in metrics}
        stage.detail = message
        self._on_stage_change(stage)

    def finish_task(
        self,
        task_id: ReporterTask,
        *,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
        silent: bool = False,
    ) -> None:
        binding = self._task_bindings.pop(task_id, None)
        if binding is None:
            return
        stage = self._stages.get(binding.stage_key)
        if stage is None:
            return
        if stage.started_at is None:
            stage.started_at = time.monotonic()
            stage.last_progress_at = stage.started_at
            stage.last_progress_completed = stage.completed
        stage.finished_at = time.monotonic()
        previous_completed = stage.completed
        if stage.total is not None:
            stage.completed = max(stage.completed, stage.total)
        self._update_stage_rate(stage, previous_completed=previous_completed)
        stage.status = binding.done_status
        if not silent:
            stage.metric_values = {metric.id: metric.value for metric in metrics}
            stage.detail = message
        self._on_stage_change(stage)

    def configure_workflow(
        self,
        *,
        title: str,
        facts: Iterable[tuple[str, str]] = (),
    ) -> None:
        self._workflow_title = title
        self._workflow_facts = list(facts)
        self._on_workflow_configured()

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
    ) -> Reporter:
        stage = self._ensure_stage(
            key,
            label=label,
            status=status,
            total=total,
            unit=unit,
            metric_descriptors=tuple(metric_descriptors),
        )
        self._on_stage_change(stage)
        return _BoundStageReporter(
            self,
            key=key,
            label=label,
            running_status=running_status,
            done_status=done_status,
        )

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
    ) -> None:
        stage = self._ensure_stage(
            key,
            label=label or key,
            metric_descriptors=tuple(metric_descriptors),
        )
        if label is not None:
            stage.label = label
        if metric_descriptors:
            stage.metric_descriptors = tuple(metric_descriptors)
        if total is not None:
            stage.total = total
        if unit is not None:
            stage.unit = unit
        if completed is not None:
            previous_completed = stage.completed
            stage.completed = max(0, completed)
            self._update_stage_rate(stage, previous_completed=previous_completed)
        metric_values = tuple(metrics)
        stage.metric_values = {metric.id: metric.value for metric in metric_values}
        if message is not None:
            stage.detail = message
        if status is not None:
            if status in _FINAL_STAGE_STATUSES and stage.finished_at is None:
                stage.finished_at = time.monotonic()
            if status == "pending":
                stage.started_at = None
                stage.finished_at = None
                stage.last_progress_at = None
                stage.last_progress_completed = stage.completed
                stage.smoothed_rate = None
                stage.metric_values = {}
            elif status in _ACTIVE_STAGE_STATUSES and stage.started_at is None:
                started_at = time.monotonic()
                stage.started_at = started_at
                stage.last_progress_at = started_at
                stage.last_progress_completed = stage.completed
            stage.status = status
        self._on_stage_change(stage)

    def close(self) -> None:
        return None

    def _start_bound_task(
        self,
        stage_key: str,
        *,
        label: str,
        task_name: str,
        total: int | None,
        unit: str | None,
        running_status: str,
        done_status: str,
    ) -> ReporterTask:
        stage = self._ensure_stage(stage_key, label=label, total=total, unit=unit)
        stage.status = running_status
        started_at = time.monotonic()
        stage.started_at = started_at
        stage.finished_at = None
        stage.total = total
        stage.unit = unit
        stage.completed = 0
        stage.metric_values = {}
        stage.last_progress_at = started_at
        stage.last_progress_completed = 0
        stage.smoothed_rate = None
        stage.detail = None
        task_id = self._next_task_id
        self._next_task_id += 1
        self._task_bindings[task_id] = _TaskBinding(
            stage_key=stage_key,
            task_name=task_name,
            done_status=done_status,
        )
        self._on_stage_change(stage)
        return task_id

    def _ensure_stage(
        self,
        key: str,
        *,
        label: str,
        status: str | None = None,
        total: int | None = None,
        unit: str | None = None,
        metric_descriptors: tuple[StageMetricDescriptor, ...] = (),
    ) -> _StageState:
        stage = self._stages.get(key)
        if stage is None:
            stage = _StageState(
                key=key,
                label=label,
                status=status or "pending",
                metric_descriptors=metric_descriptors,
                total=total,
                unit=unit,
            )
            self._stages[key] = stage
            return stage
        stage.label = label
        if metric_descriptors:
            stage.metric_descriptors = metric_descriptors
        if status is not None:
            stage.status = status
        if total is not None:
            stage.total = total
        if unit is not None:
            stage.unit = unit
        return stage

    def _on_workflow_configured(self) -> None:
        raise NotImplementedError

    def _on_stage_change(self, stage: _StageState) -> None:
        raise NotImplementedError

    def _update_stage_rate(self, stage: _StageState, *, previous_completed: int) -> None:
        if stage.started_at is None:
            return
        if stage.unit is None:
            return
        now = time.monotonic()
        elapsed = max(0.0, now - stage.started_at)
        if elapsed <= 0.0:
            return
        if stage.completed <= previous_completed:
            stage.last_progress_at = now
            stage.last_progress_completed = stage.completed
            return
        checkpoint_at = (
            stage.last_progress_at
            if stage.last_progress_at is not None
            else stage.started_at
        )
        delta_elapsed = max(0.0, now - checkpoint_at)
        delta_completed = stage.completed - stage.last_progress_completed
        if delta_elapsed > 0.0 and delta_completed > 0:
            recent_rate = delta_completed / delta_elapsed
            if recent_rate > 0.0:
                stage.smoothed_rate = _smooth_value(stage.smoothed_rate, recent_rate, alpha=0.22)
        stage.last_progress_at = now
        stage.last_progress_completed = stage.completed


class PlainReporter(_BaseWorkflowReporter):
    """Line-oriented reporter for non-interactive output."""

    def __init__(self, console: Console | None = None) -> None:
        super().__init__(console=console)

    def _on_workflow_configured(self) -> None:
        if self._workflow_title is None:
            return
        self.console.print(self._workflow_title, markup=False)
        for label, value in self._workflow_facts:
            self.console.print(f"{label}: {value}", markup=False)

    def _on_stage_change(self, stage: _StageState) -> None:
        if not self._should_emit(stage):
            return
        self.console.print(self._format_stage_line(stage), markup=False)
        stage.last_emitted_status = stage.status
        stage.last_emitted_detail = stage.detail
        stage.last_emitted_metric_values = dict(stage.metric_values)
        stage.last_emitted_completed = stage.completed
        stage.last_emitted_bucket = _progress_bucket(stage)

    def _should_emit(self, stage: _StageState) -> bool:
        if stage.status != stage.last_emitted_status:
            return True
        if stage.detail != stage.last_emitted_detail:
            return True
        if stage.metric_values != stage.last_emitted_metric_values:
            return True
        if stage.total is None:
            return stage.completed != stage.last_emitted_completed
        return (
            _progress_bucket(stage) != stage.last_emitted_bucket
            or stage.completed >= stage.total
        )

    def _format_stage_line(self, stage: _StageState) -> str:
        status = stage.status
        pieces = [f"{stage.label} [{status}]"]
        if stage.total is not None:
            suffix = "" if stage.unit is None else f" {stage.unit}"
            pieces.append(f"{stage.completed:,}/{stage.total:,}{suffix}")
        elif stage.completed > 0:
            suffix = "" if stage.unit is None else f" {stage.unit}"
            pieces.append(f"{stage.completed:,}{suffix}")
        metric_tokens = [
            f"{descriptor.label}={stage.metric_values[descriptor.id]}"
            for descriptor in stage.metric_descriptors
            if descriptor.id in stage.metric_values
        ]
        if metric_tokens:
            pieces.append(" ".join(metric_tokens))
        detail = stage.detail
        if detail:
            pieces.append(detail)
        return " - ".join(pieces)
