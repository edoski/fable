"""Console logging and progress reporting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ReporterTask = int


class Reporter(Protocol):
    def log(self, message: str, *, level: str = "info") -> None: ...
    def throttled_log(
        self,
        key: str,
        message: str,
        *,
        interval_seconds: float = 10.0,
        level: str = "info",
    ) -> None: ...
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
    ) -> None: ...
    def finish_task(self, task_id: ReporterTask, *, message: str | None = None) -> None: ...
    def close(self) -> None: ...


class NullReporter:
    """Silent reporter used by library entrypoints and tests."""

    def log(self, message: str, *, level: str = "info") -> None:
        return None

    def throttled_log(
        self,
        key: str,
        message: str,
        *,
        interval_seconds: float = 10.0,
        level: str = "info",
    ) -> None:
        return None

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> ReporterTask:
        return 0

    def update_task(
        self,
        task_id: ReporterTask,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
    ) -> None:
        return None

    def finish_task(self, task_id: ReporterTask, *, message: str | None = None) -> None:
        return None

    def close(self) -> None:
        return None


class _BaseConsoleReporter(NullReporter):
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._throttle_times: dict[str, float] = {}

    def throttled_log(
        self,
        key: str,
        message: str,
        *,
        interval_seconds: float = 10.0,
        level: str = "info",
    ) -> None:
        now = time.monotonic()
        last = self._throttle_times.get(key)
        if last is not None and now - last < interval_seconds:
            return
        self._throttle_times[key] = now
        self.log(message, level=level)

    def log(self, message: str, *, level: str = "info") -> None:
        style = None
        prefix = ""
        if level == "warning":
            style = "yellow"
            prefix = "warning: "
        elif level == "error":
            style = "bold red"
            prefix = "error: "
        self.console.print(f"{prefix}{message}", style=style)


@dataclass(slots=True)
class _PlainTaskState:
    name: str
    total: int | None
    unit: str | None
    completed: int
    message: str | None
    last_emitted_completed: int | None = None
    last_emitted_message: str | None = None
    last_emitted_bucket: int | None = None


class PlainReporter(_BaseConsoleReporter):
    """Line-oriented reporter for non-interactive output."""

    def __init__(self, console: Console | None = None) -> None:
        super().__init__(console=console)
        self._next_task_id = 1
        self._tasks: dict[ReporterTask, _PlainTaskState] = {}

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> ReporterTask:
        task_id = self._next_task_id
        self._next_task_id += 1
        self._tasks[task_id] = _PlainTaskState(
            name=name,
            total=total,
            unit=unit,
            completed=0,
            message=None,
        )
        self.log(self._format_start(name, total=total, unit=unit))
        return task_id

    def update_task(
        self,
        task_id: ReporterTask,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
    ) -> None:
        state = self._tasks.get(task_id)
        if state is None:
            return
        if completed is not None:
            state.completed = max(0, completed)
        if advance is not None:
            state.completed = max(0, state.completed + advance)
        if message is not None:
            state.message = message
        if not self._should_emit(state):
            return
        self.log(self._format_progress(state))
        state.last_emitted_completed = state.completed
        state.last_emitted_message = state.message
        state.last_emitted_bucket = self._progress_bucket(state)

    def finish_task(self, task_id: ReporterTask, *, message: str | None = None) -> None:
        state = self._tasks.pop(task_id, None)
        if state is None:
            return
        if message is not None:
            state.message = message
        self.log(self._format_finish(state))

    def _should_emit(self, state: _PlainTaskState) -> bool:
        if state.total is None:
            return (
                state.completed != state.last_emitted_completed
                or state.message != state.last_emitted_message
            )
        if state.total <= 100:
            return (
                state.completed != state.last_emitted_completed
                or state.message != state.last_emitted_message
            )
        bucket = self._progress_bucket(state)
        return bucket != state.last_emitted_bucket or state.completed >= state.total

    def _progress_bucket(self, state: _PlainTaskState) -> int:
        assert state.total is not None
        if state.total <= 0:
            return 20
        return min(20, (state.completed * 20) // state.total)

    def _format_start(self, name: str, *, total: int | None, unit: str | None) -> str:
        if total is None:
            return f"{name} started"
        suffix = "" if unit is None else f" {unit}"
        return f"{name} started ({total}{suffix})"

    def _format_progress(self, state: _PlainTaskState) -> str:
        if state.total is None:
            detail = state.message
            if detail is None and state.completed:
                detail = str(state.completed)
            return state.name if detail is None else f"{state.name}: {detail}"
        suffix = "" if state.unit is None else f" {state.unit}"
        percent = 100 if state.total == 0 else int((state.completed * 100) / state.total)
        progress = f"{state.name}: {state.completed}/{state.total}{suffix} ({percent}%)"
        return progress if state.message is None else f"{progress} - {state.message}"

    def _format_finish(self, state: _PlainTaskState) -> str:
        if state.message is None:
            return f"{state.name} finished"
        return f"{state.name} finished: {state.message}"


class RichReporter(_BaseConsoleReporter):
    """Interactive reporter for real terminals."""

    def __init__(self, console: Console | None = None) -> None:
        super().__init__(console=console)
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self.progress.start()
        self._task_names: dict[ReporterTask, str] = {}

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> ReporterTask:
        task_id = self.progress.add_task(name, total=total)
        self._task_names[task_id] = name
        return task_id

    def update_task(
        self,
        task_id: ReporterTask,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
    ) -> None:
        name = self._task_names.get(task_id)
        if name is None:
            return
        description = name if message is None else f"{name} | {message}"
        self.progress.update(
            TaskID(task_id),
            completed=completed,
            advance=advance,
            description=description,
        )

    def finish_task(self, task_id: ReporterTask, *, message: str | None = None) -> None:
        name = self._task_names.pop(task_id, None)
        if name is None:
            return
        self.progress.remove_task(TaskID(task_id))
        self.log(f"{name} finished" if message is None else f"{name} finished: {message}")

    def close(self) -> None:
        self.progress.stop()


def create_reporter(console: Console | None = None) -> Reporter:
    active_console = console or Console()
    if active_console.is_terminal:
        return RichReporter(console=active_console)
    return PlainReporter(console=active_console)
