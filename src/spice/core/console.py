"""Console runtime, stage reporting, and native tool log bridging."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Protocol

import optuna
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import (
    BatchesProcessedColumn,
    CustomBarColumn,
    CustomProgress,
    CustomTimeColumn,
    MetricsTextColumn,
    ProcessingSpeedColumn,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    import lightning.pytorch as pl

ReporterTask = int

_LIGHTNING_LOGGER_NAMES = (
    "lightning",
    "lightning.pytorch",
    "lightning.pytorch.utilities.rank_zero",
    "lightning.fabric",
    "lightning.fabric.utilities.rank_zero",
    "lightning_fabric",
)
_NATIVE_NOISE_PATTERNS = (
    re.compile(r"^Seed set to \d+$"),
    re.compile(r"^GPU available: .*"),
    re.compile(r"^TPU available: .*"),
    re.compile(r"litlogger", re.IGNORECASE),
    re.compile(r"`Trainer\.fit` stopped: `max_epochs=.*` reached\."),
    re.compile(r"GPU available but not used"),
    re.compile(r"`isinstance\(treespec, LeafSpec\)` is deprecated"),
    re.compile(r"The 'train_dataloader' does not have many workers"),
    re.compile(r"The 'val_dataloader' does not have many workers"),
)


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
    ) -> None: ...
    def finish_task(self, task_id: ReporterTask, *, message: str | None = None) -> None: ...
    def close(self) -> None: ...


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
        if state.total <= 10:
            return (
                state.completed != state.last_emitted_completed
                or state.message != state.last_emitted_message
            )
        bucket = self._progress_bucket(state)
        return bucket != state.last_emitted_bucket or state.completed >= state.total

    def _progress_bucket(self, state: _PlainTaskState) -> int:
        assert state.total is not None
        if state.total <= 0:
            return 10
        bucket_count = min(10, state.total)
        return min(bucket_count, (state.completed * bucket_count) // state.total)

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
    """Interactive reporter with task-local Rich progress."""

    def __init__(self, console: Console | None = None) -> None:
        super().__init__(console=console)
        self._progress: Progress | None = None
        self._task_names: dict[ReporterTask, str] = {}

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> ReporterTask:
        progress = self._ensure_progress()
        task_id = progress.add_task(name, total=total)
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
        if name is None or self._progress is None:
            return
        description = name if message is None else f"{name} | {message}"
        self._progress.update(
            TaskID(task_id),
            completed=completed,
            advance=advance,
            description=description,
        )

    def finish_task(self, task_id: ReporterTask, *, message: str | None = None) -> None:
        name = self._task_names.pop(task_id, None)
        if name is None or self._progress is None:
            return
        self._progress.remove_task(TaskID(task_id))
        if not self._task_names:
            self._progress.stop()
            self._progress = None
        self.log(f"{name} finished" if message is None else f"{name} finished: {message}")

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def _ensure_progress(self) -> Progress:
        if self._progress is None:
            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            )
            self._progress.start()
        return self._progress


@dataclass(slots=True)
class _LoggerState:
    handlers: list[logging.Handler]
    level: int
    propagate: bool


class _NativeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in _NATIVE_NOISE_PATTERNS)


class SharedConsoleRichProgressBar(RichProgressBar):
    """Lightning Rich progress bar bound to the workflow console."""

    def __init__(self, console: Console) -> None:
        super().__init__()
        self._shared_console = console

    def _init_progress(self, trainer: pl.Trainer) -> None:
        if not self.is_enabled or (self.progress is not None and not self._progress_stopped):
            return
        self._reset_progress_bar_ids()
        self._console = self._shared_console
        if hasattr(self._console, "_live_stack") and len(self._console._live_stack) > 0:
            self._console.clear_live()
        elif getattr(self._console, "_live", None) is not None:
            self._console.clear_live()
        self._metric_component = MetricsTextColumn(
            trainer,
            self.theme.metrics,
            self.theme.metrics_text_delimiter,
            self.theme.metrics_format,
        )
        self.progress = CustomProgress(
            *self.configure_columns(trainer),
            self._metric_component,
            auto_refresh=True,
            refresh_per_second=self.refresh_rate if self.is_enabled else 1,
            disable=self.is_disabled,
            console=self._console,
        )
        self.progress.start()
        self._progress_stopped = False

    def configure_columns(self, trainer: pl.Trainer) -> list[str | ProgressColumn]:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]


class ConsoleRuntime:
    """Workflow-scoped console owner with native log bridging."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        active_console = console or _console_from_reporter(reporter) or Console()
        self.console = active_console
        self.reporter = reporter or create_reporter(active_console)
        self._owns_reporter = reporter is None
        self._activation_depth = 0
        self._root_state: _LoggerState | None = None
        self._pywarnings_state: _LoggerState | None = None
        self._lightning_states: dict[str, _LoggerState] = {}
        self._root_handler: RichHandler | None = None

    @contextmanager
    def activate(self) -> Iterator[ConsoleRuntime]:
        first_entry = self._activation_depth == 0
        self._activation_depth += 1
        if first_entry:
            self._install_logging_bridge()
        try:
            yield self
        finally:
            self._activation_depth -= 1
            if first_entry and self._activation_depth == 0:
                self._restore_logging_bridge()

    @contextmanager
    def optuna_logging(self) -> Iterator[None]:
        optuna.logging.get_verbosity()
        logger = logging.getLogger("optuna")
        state = _LoggerState(list(logger.handlers), logger.level, logger.propagate)
        optuna.logging.disable_default_handler()
        optuna.logging.enable_propagation()
        optuna.logging.set_verbosity(optuna.logging.INFO)
        try:
            yield
        finally:
            logger.handlers.clear()
            for handler in state.handlers:
                logger.addHandler(handler)
            logger.setLevel(state.level)
            logger.propagate = state.propagate

    def lightning_progress_bar(self) -> SharedConsoleRichProgressBar | None:
        if not self.console.is_terminal:
            return None
        return SharedConsoleRichProgressBar(self.console)

    def log_summary(self, title: str, rows: list[tuple[str, str]]) -> None:
        if self.console.is_terminal:
            table = Table.grid(padding=(0, 1))
            table.add_column(style="bold cyan", justify="right")
            table.add_column()
            for label, value in rows:
                table.add_row(label, value)
            self.console.print(Panel(table, title=title, border_style="cyan"))
            return
        self.reporter.log(title)
        for label, value in rows:
            self.reporter.log(f"{label}: {value}")

    def close(self) -> None:
        if self._activation_depth > 0:
            self._activation_depth = 0
            self._restore_logging_bridge()
        if self._owns_reporter:
            self.reporter.close()

    def _install_logging_bridge(self) -> None:
        root_logger = logging.getLogger()
        self._root_state = _LoggerState(
            handlers=list(root_logger.handlers),
            level=root_logger.level,
            propagate=root_logger.propagate,
        )
        root_logger.handlers.clear()
        self._root_handler = RichHandler(
            console=self.console,
            show_time=False,
            show_path=False,
            markup=False,
        )
        self._root_handler.addFilter(_NativeLogFilter())
        root_logger.addHandler(self._root_handler)
        root_logger.setLevel(logging.INFO)

        pywarnings_logger = logging.getLogger("py.warnings")
        self._pywarnings_state = _LoggerState(
            handlers=list(pywarnings_logger.handlers),
            level=pywarnings_logger.level,
            propagate=pywarnings_logger.propagate,
        )
        pywarnings_logger.handlers.clear()
        pywarnings_logger.setLevel(logging.WARNING)
        pywarnings_logger.propagate = True
        logging.captureWarnings(True)

        self._lightning_states = {}
        for name in _LIGHTNING_LOGGER_NAMES:
            logger = logging.getLogger(name)
            self._lightning_states[name] = _LoggerState(
                handlers=list(logger.handlers),
                level=logger.level,
                propagate=logger.propagate,
            )
            logger.handlers.clear()
            logger.setLevel(logging.INFO)
            logger.propagate = True

    def _restore_logging_bridge(self) -> None:
        logging.captureWarnings(False)

        if self._pywarnings_state is not None:
            pywarnings_logger = logging.getLogger("py.warnings")
            pywarnings_logger.handlers.clear()
            for handler in self._pywarnings_state.handlers:
                pywarnings_logger.addHandler(handler)
            pywarnings_logger.setLevel(self._pywarnings_state.level)
            pywarnings_logger.propagate = self._pywarnings_state.propagate
            self._pywarnings_state = None

        for name, state in self._lightning_states.items():
            logger = logging.getLogger(name)
            logger.handlers.clear()
            for handler in state.handlers:
                logger.addHandler(handler)
            logger.setLevel(state.level)
            logger.propagate = state.propagate
        self._lightning_states = {}

        if self._root_state is not None:
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            for handler in self._root_state.handlers:
                root_logger.addHandler(handler)
            root_logger.setLevel(self._root_state.level)
            root_logger.propagate = self._root_state.propagate
            self._root_state = None
        self._root_handler = None


def create_reporter(console: Console | None = None) -> Reporter:
    active_console = console or Console()
    if active_console.is_terminal:
        return RichReporter(console=active_console)
    return PlainReporter(console=active_console)


def create_console_runtime(
    *,
    console: Console | None = None,
    reporter: Reporter | None = None,
) -> ConsoleRuntime:
    return ConsoleRuntime(console=console, reporter=reporter)


def _console_from_reporter(reporter: Reporter | None) -> Console | None:
    if reporter is None:
        return None
    candidate = getattr(reporter, "console", None)
    if isinstance(candidate, Console):
        return candidate
    return Console(file=StringIO(), force_terminal=False, width=120)
