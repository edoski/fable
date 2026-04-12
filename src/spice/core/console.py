"""Workflow-scoped console presentation and native log bridging."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from typing import Protocol

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

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
    re.compile(r"^Successfully disconnected from: .*"),
    re.compile(r"litlogger", re.IGNORECASE),
    re.compile(r"`Trainer\.fit` stopped: `max_epochs=.*` reached\."),
    re.compile(r"GPU available but not used"),
    re.compile(r"`isinstance\(treespec, LeafSpec\)` is deprecated"),
    re.compile(r"The 'train_dataloader' does not have many workers"),
    re.compile(r"The 'val_dataloader' does not have many workers"),
)
_FINAL_STAGE_STATUSES = frozenset(
    {"done", "failed", "reused", "extended", "rebuilt", "created"}
)
_STAGE_STATUS_STYLES = {
    "pending": "dim",
    "running": "cyan",
    "pulling": "cyan",
    "writing": "cyan",
    "done": "green",
    "reused": "green",
    "created": "green",
    "extended": "yellow",
    "rebuilt": "yellow",
    "failed": "bold red",
}


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

    def finish_task(
        self,
        task_id: ReporterTask,
        *,
        message: str | None = None,
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
    ) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _TaskBinding:
    stage_key: str
    task_name: str
    done_status: str


@dataclass(slots=True)
class _StageState:
    key: str
    label: str
    status: str = "pending"
    total: int | None = None
    unit: str | None = None
    completed: int = 0
    detail: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    last_emitted_status: str | None = None
    last_emitted_detail: str | None = None
    last_emitted_completed: int | None = None
    last_emitted_bucket: int | None = None


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
    ) -> None:
        del task_id, completed, advance, message
        return None

    def finish_task(
        self,
        task_id: ReporterTask,
        *,
        message: str | None = None,
        silent: bool = False,
    ) -> None:
        del task_id, message, silent
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
    ) -> Reporter:
        del key, label, total, unit, status, running_status, done_status
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
    ) -> None:
        del key, label, status, total, unit, completed, message
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
    ) -> None:
        self._owner.update_task(
            task_id,
            completed=completed,
            advance=advance,
            message=message,
        )

    def finish_task(
        self,
        task_id: ReporterTask,
        *,
        message: str | None = None,
        silent: bool = False,
    ) -> None:
        self._owner.finish_task(task_id, message=message, silent=silent)

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
    ) -> Reporter:
        return self._owner.stage_reporter(
            key,
            label=label,
            total=total,
            unit=unit,
            status=status,
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
    ) -> None:
        self._owner.set_stage_state(
            key,
            label=label,
            status=status,
            total=total,
            unit=unit,
            completed=completed,
            message=message,
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
    ) -> None:
        binding = self._task_bindings.get(task_id)
        if binding is None:
            return
        stage = self._stages.get(binding.stage_key)
        if stage is None:
            return
        if stage.started_at is None:
            stage.started_at = time.monotonic()
        stage.finished_at = None
        if completed is not None:
            stage.completed = max(0, completed)
        if advance is not None:
            stage.completed = max(0, stage.completed + advance)
        stage.detail = _format_stage_detail(stage.label, binding.task_name, message)
        self._on_stage_change(stage)

    def finish_task(
        self,
        task_id: ReporterTask,
        *,
        message: str | None = None,
        silent: bool = False,
    ) -> None:
        del silent
        binding = self._task_bindings.pop(task_id, None)
        if binding is None:
            return
        stage = self._stages.get(binding.stage_key)
        if stage is None:
            return
        if stage.started_at is None:
            stage.started_at = time.monotonic()
        stage.finished_at = time.monotonic()
        if stage.total is not None:
            stage.completed = max(stage.completed, stage.total)
        stage.status = binding.done_status
        stage.detail = _format_stage_detail(stage.label, binding.task_name, message)
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
    ) -> Reporter:
        self._ensure_stage(key, label=label, status=status, total=total, unit=unit)
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
    ) -> None:
        stage = self._ensure_stage(key, label=label or key)
        if label is not None:
            stage.label = label
        if total is not None:
            stage.total = total
        if unit is not None:
            stage.unit = unit
        if completed is not None:
            stage.completed = max(0, completed)
        if message is not None:
            stage.detail = message
        if status is not None:
            if status in _FINAL_STAGE_STATUSES and stage.finished_at is None:
                stage.finished_at = time.monotonic()
            if status not in _FINAL_STAGE_STATUSES and stage.started_at is None:
                stage.started_at = time.monotonic()
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
        stage.started_at = time.monotonic()
        stage.finished_at = None
        stage.total = total
        stage.unit = unit
        stage.completed = 0
        stage.detail = _format_stage_detail(stage.label, task_name, None)
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
    ) -> _StageState:
        stage = self._stages.get(key)
        if stage is None:
            stage = _StageState(
                key=key,
                label=label,
                status=status or "pending",
                total=total,
                unit=unit,
            )
            self._stages[key] = stage
            return stage
        stage.label = label
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


class PlainReporter(_BaseWorkflowReporter):
    """Line-oriented reporter for non-interactive output."""

    def __init__(self, console: Console | None = None) -> None:
        super().__init__(console=console)
        self._workflow_announced = False

    def _on_workflow_configured(self) -> None:
        if self._workflow_title is None:
            return
        self.console.print(self._workflow_title, markup=False)
        for label, value in self._workflow_facts:
            self.console.print(f"{label}: {value}", markup=False)
        self._workflow_announced = True

    def _on_stage_change(self, stage: _StageState) -> None:
        if not self._should_emit(stage):
            return
        self.console.print(self._format_stage_line(stage), markup=False)
        stage.last_emitted_status = stage.status
        stage.last_emitted_detail = stage.detail
        stage.last_emitted_completed = stage.completed
        stage.last_emitted_bucket = _progress_bucket(stage)

    def _should_emit(self, stage: _StageState) -> bool:
        if stage.status != stage.last_emitted_status:
            return True
        if stage.detail != stage.last_emitted_detail:
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
        if stage.detail:
            pieces.append(stage.detail)
        return " - ".join(pieces)


class RichReporter(_BaseWorkflowReporter):
    """Interactive reporter with shared workflow staging."""

    def __init__(self, console: Console | None = None) -> None:
        super().__init__(console=console)
        self._live: Live | None = None

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _on_workflow_configured(self) -> None:
        self._refresh_live()

    def _on_stage_change(self, stage: _StageState) -> None:
        del stage
        self._refresh_live()

    def _refresh_live(self) -> None:
        if self._live is None:
            self._live = Live(
                self._render_workflow(),
                console=self.console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
            return
        self._live.update(self._render_workflow(), refresh=True)

    def _render_workflow(self):
        elements: list[object] = []
        if self._workflow_title is not None:
            elements.append(Text(self._workflow_title, style="bold cyan"))
        if self._workflow_facts:
            facts = Table.grid(padding=(0, 2))
            facts.add_column(style="bold cyan", no_wrap=True)
            facts.add_column()
            for label, value in self._workflow_facts:
                facts.add_row(label, value)
            elements.append(facts)
        if self._stages:
            elements.append(self._render_stage_table())
        if not elements:
            elements.append(Text(""))
        return Group(*elements)

    def _render_stage_table(self) -> Table:
        table = Table(
            show_header=True,
            header_style="bold",
            expand=True,
            box=None,
            pad_edge=False,
        )
        table.add_column("stage", no_wrap=True, style="bold")
        table.add_column("status", no_wrap=True)
        table.add_column("progress", ratio=2)
        table.add_column("time", no_wrap=True)
        table.add_column("detail", ratio=3)
        for stage in self._stages.values():
            table.add_row(
                stage.label,
                Text(stage.status, style=_STAGE_STATUS_STYLES.get(stage.status, "")),
                self._render_progress(stage),
                _format_elapsed(stage),
                stage.detail or "",
            )
        return table

    def _render_progress(self, stage: _StageState):
        if stage.total is None:
            return Text("--" if stage.status == "pending" else f"{stage.completed:,}", style="dim")
        progress = Table.grid(padding=(0, 1))
        progress.add_column(width=24)
        progress.add_column(no_wrap=True)
        progress.add_row(
            ProgressBar(
                total=max(float(stage.total), 1.0),
                completed=float(min(stage.completed, stage.total)),
                width=24,
            ),
            _format_progress_count(stage),
        )
        return progress


@dataclass(slots=True)
class _LoggerState:
    handlers: list[logging.Handler]
    level: int
    propagate: bool


class _NativeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in _NATIVE_NOISE_PATTERNS)


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
        import optuna

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

    def configure_workflow(self, title: str, facts: Iterable[tuple[str, str]]) -> None:
        self.reporter.configure_workflow(title=title, facts=facts)

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
    ) -> Reporter:
        return self.reporter.stage_reporter(
            key,
            label=label,
            total=total,
            unit=unit,
            status=status,
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
    ) -> None:
        self.reporter.set_stage_state(
            key,
            label=label,
            status=status,
            total=total,
            unit=unit,
            completed=completed,
            message=message,
        )

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

    def log_sectioned_summary(
        self,
        title: str,
        sections: list[tuple[str, list[tuple[str, str]]]],
    ) -> None:
        if self.console.is_terminal:
            body = Table.grid(expand=True)
            body.add_column()
            for index, (section_title, rows) in enumerate(sections):
                if index > 0:
                    body.add_row("")
                section = Table.grid(padding=(0, 1))
                section.add_column(style="bold cyan", justify="right", no_wrap=True)
                section.add_column()
                for label, value in rows:
                    section.add_row(label, value)
                body.add_row(f"[bold]{section_title}[/bold]")
                body.add_row(section)
            self.console.print(Panel(body, title=title, border_style="cyan"))
            return

        self.reporter.log(title)
        for section_title, rows in sections:
            self.reporter.log(f"{section_title}:")
            for label, value in rows:
                self.reporter.log(f"  {label}: {value}")

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


def _format_stage_detail(label: str, task_name: str, message: str | None) -> str | None:
    prefix = None if task_name == label else task_name
    if message is None:
        return prefix
    if prefix is None:
        return message
    return f"{prefix}: {message}"


def _progress_bucket(stage: _StageState) -> int | None:
    if stage.total is None:
        return None
    if stage.total <= 0:
        return 10
    bucket_count = min(10, stage.total)
    return min(bucket_count, (stage.completed * bucket_count) // stage.total)


def _format_progress_count(stage: _StageState) -> str:
    suffix = "" if stage.unit is None else f" {stage.unit}"
    if stage.total is None:
        return f"{stage.completed:,}{suffix}" if stage.completed else "--"
    return f"{stage.completed:,}/{stage.total:,}{suffix}"


def _format_elapsed(stage: _StageState) -> str:
    if stage.started_at is None:
        return "--"
    end = stage.finished_at if stage.finished_at is not None else time.monotonic()
    elapsed = max(0.0, end - stage.started_at)
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    minutes, seconds = divmod(int(elapsed), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"
