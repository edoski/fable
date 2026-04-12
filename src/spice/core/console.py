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
from rich.rule import Rule
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
_ACTIVE_STAGE_STATUSES = frozenset({"planning", "running", "pulling", "writing"})
_STAGE_STATUS_STYLES = {
    "pending": "dim",
    "planning": "cyan",
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
_PROGRESS_BAR_STYLES = {
    "pending": ("grey23", "grey35", "grey50", "grey50"),
    "planning": ("grey23", "cyan", "cyan", "cyan"),
    "running": ("grey23", "cyan", "cyan", "cyan"),
    "pulling": ("grey23", "cyan", "cyan", "cyan"),
    "writing": ("grey23", "cyan", "cyan", "cyan"),
    "done": ("grey23", "green", "green", "green"),
    "reused": ("grey23", "green", "green", "green"),
    "created": ("grey23", "green", "green", "green"),
    "extended": ("grey23", "yellow", "yellow", "yellow"),
    "rebuilt": ("grey23", "yellow", "yellow", "yellow"),
    "failed": ("grey23", "red", "red", "red"),
}
_DETAIL_VALUE_LABELS = frozenset({"batch", "conc"})
_RATE_COLUMN_WIDTH = 11
_TIME_COLUMN_WIDTH = 7


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
    last_progress_at: float | None = None
    last_progress_completed: int = 0
    smoothed_rate: float | None = None
    last_emitted_status: str | None = None
    last_emitted_detail: str | None = None
    last_emitted_completed: int | None = None
    last_emitted_bucket: int | None = None


@dataclass(frozen=True, slots=True)
class _StageLayout:
    stage_width: int
    status_width: int
    progress_bar_width: int
    show_rate: bool
    show_eta: bool
    show_detail: bool

    @property
    def progress_width(self) -> int:
        return self.progress_bar_width + 5


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
            stage.last_progress_at = stage.started_at
            stage.last_progress_completed = stage.completed
        stage.finished_at = None
        previous_completed = stage.completed
        if completed is not None:
            stage.completed = max(0, completed)
        if advance is not None:
            stage.completed = max(0, stage.completed + advance)
        self._update_stage_rate(stage, previous_completed=previous_completed)
        stage.detail = _format_stage_detail(stage.label, binding.task_name, message)
        self._on_stage_change(stage)

    def finish_task(
        self,
        task_id: ReporterTask,
        *,
        message: str | None = None,
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
        stage = self._ensure_stage(key, label=label, status=status, total=total, unit=unit)
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
    ) -> None:
        stage = self._ensure_stage(key, label=label or key)
        if label is not None:
            stage.label = label
        if total is not None:
            stage.total = total
        if unit is not None:
            stage.unit = unit
        if completed is not None:
            previous_completed = stage.completed
            stage.completed = max(0, completed)
            self._update_stage_rate(stage, previous_completed=previous_completed)
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
        stage.last_progress_at = started_at
        stage.last_progress_completed = 0
        stage.smoothed_rate = None
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

    def _update_stage_rate(self, stage: _StageState, *, previous_completed: int) -> None:
        if stage.started_at is None:
            return
        if stage.unit is None:
            return
        if stage.completed <= previous_completed:
            return
        now = time.monotonic()
        elapsed = max(0.0, now - stage.started_at)
        if elapsed <= 0.0:
            return
        average_rate = stage.completed / elapsed
        if average_rate <= 0.0:
            return
        stage.smoothed_rate = _smooth_value(stage.smoothed_rate, average_rate, alpha=0.22)
        stage.last_progress_at = now
        stage.last_progress_completed = stage.completed


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
                auto_refresh=False,
                transient=False,
            )
            self._live.start()
            return
        self._live.update(self._render_workflow(), refresh=True)

    def _render_workflow(self):
        elements: list[object] = []
        if self._workflow_facts:
            elements.append(self._render_fact_grid())
        if self._stages:
            if elements:
                elements.append(Rule(style="grey35"))
            elements.append(self._render_stage_table())
        body = Group(*elements) if elements else Text("")
        return Panel(
            body,
            title=Text(self._workflow_title or "", style="bold cyan"),
            border_style="cyan",
            padding=(0, 1),
            expand=True,
        )

    def _render_fact_grid(self) -> Table:
        facts = Table.grid(padding=(0, 1), expand=False)
        facts.add_column(style="bold cyan", no_wrap=True)
        facts.add_column()
        for label, value in self._workflow_facts:
            facts.add_row(label, Text(value))
        return facts

    def _render_stage_table(self) -> Table:
        has_detail = any(stage.detail for stage in self._stages.values())
        layout = _stage_layout(_panel_body_width(self.console), has_detail=has_detail)
        table = Table(
            show_header=True,
            header_style="bold dim",
            expand=False,
            box=None,
            pad_edge=False,
            padding=(0, 1),
            collapse_padding=True,
        )
        table.add_column(
            "stage",
            width=layout.stage_width,
            no_wrap=True,
            overflow="ellipsis",
            style="bold",
        )
        table.add_column(
            "status",
            width=layout.status_width,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column("progress", width=layout.progress_width, no_wrap=True)
        if layout.show_rate:
            table.add_column("rate", width=_RATE_COLUMN_WIDTH, no_wrap=True, justify="right")
        table.add_column("elapsed", width=_TIME_COLUMN_WIDTH, no_wrap=True, justify="right")
        if layout.show_eta:
            table.add_column("eta", width=_TIME_COLUMN_WIDTH, no_wrap=True, justify="right")
        if layout.show_detail:
            table.add_column("detail", ratio=1, no_wrap=True, overflow="ellipsis")
        for stage in self._stages.values():
            row = [
                Text(stage.label, style="bold"),
                Text(stage.status, style=_STAGE_STATUS_STYLES.get(stage.status, "")),
                self._render_progress(stage, bar_width=layout.progress_bar_width),
                _render_elapsed(stage),
            ]
            if layout.show_rate:
                row.insert(3, _render_rate(stage))
            if layout.show_eta:
                row.append(_render_eta(stage))
            if layout.show_detail:
                row.append(_render_stage_detail(stage))
            table.add_row(*row)
        return table

    def _render_progress(self, stage: _StageState, *, bar_width: int):
        if stage.total is None:
            return Text("--", style="dim")
        progress = Table.grid(padding=(0, 1))
        progress.add_column(width=bar_width)
        progress.add_column(width=4, no_wrap=True, justify="right")
        style, complete_style, finished_style, pulse_style = _PROGRESS_BAR_STYLES.get(
            stage.status,
            ("grey23", "cyan", "cyan", "cyan"),
        )
        progress.add_row(
            ProgressBar(
                total=max(float(stage.total), 1.0),
                completed=float(min(stage.completed, stage.total)),
                width=bar_width,
                style=style,
                complete_style=complete_style,
                finished_style=finished_style,
                pulse_style=pulse_style,
            ),
            Text(_format_progress_percent(stage), style="dim"),
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
            self.reporter.close()
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
            self.reporter.close()
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
    del label, task_name
    return message


def _panel_body_width(console: Console) -> int:
    configured_width = getattr(console, "_width", None)
    if isinstance(configured_width, int) and configured_width > 0:
        return max(40, configured_width - 4)
    return max(40, console.size.width - 4)


def _stage_layout(available_width: int, *, has_detail: bool) -> _StageLayout:
    if has_detail:
        if available_width >= 132:
            return _StageLayout(10, 8, 18, True, True, True)
        if available_width >= 112:
            return _StageLayout(10, 8, 16, True, True, False)
        if available_width >= 92:
            return _StageLayout(9, 8, 14, True, False, False)
        return _StageLayout(8, 7, 12, False, False, False)

    if available_width >= 96:
        return _StageLayout(10, 8, 20, True, True, False)
    if available_width >= 78:
        return _StageLayout(9, 8, 16, True, False, False)
    return _StageLayout(8, 7, 12, False, False, False)


def _progress_bucket(stage: _StageState) -> int | None:
    if stage.total is None:
        return None
    if stage.total <= 0:
        return 10
    bucket_count = min(10, stage.total)
    return min(bucket_count, (stage.completed * bucket_count) // stage.total)


def _format_progress_count(stage: _StageState, *, include_unit: bool = True) -> str:
    suffix = ""
    if include_unit and stage.unit is not None:
        suffix = f" {stage.unit}"
    if stage.total is None:
        return f"{stage.completed:,}{suffix}" if stage.completed else "--"
    return f"{stage.completed:,}/{stage.total:,}{suffix}"


def _format_progress_percent(stage: _StageState) -> str:
    if stage.total is None or stage.total <= 0:
        return "--"
    percent = int((min(stage.completed, stage.total) * 100) / stage.total)
    return f"{percent:>3d}%"


def _unit_rate_suffix(unit: str | None) -> str:
    if unit is None:
        return "u/s"
    aliases = {
        "blocks": "blk/s",
        "block": "blk/s",
        "batches": "bat/s",
        "batch": "bat/s",
        "repetitions": "rep/s",
        "repetition": "rep/s",
        "trials": "trl/s",
        "trial": "trl/s",
    }
    return aliases.get(unit, f"{unit}/s")


def _elapsed_seconds(stage: _StageState) -> float | None:
    if stage.started_at is None:
        return None
    end = stage.finished_at if stage.finished_at is not None else time.monotonic()
    return max(0.0, end - stage.started_at)


def _remaining_seconds(stage: _StageState) -> float | None:
    if (
        stage.total is None
        or stage.total <= 0
        or stage.completed <= 0
        or stage.completed >= stage.total
    ):
        return None
    rate = stage.smoothed_rate
    if rate is None or rate <= 0:
        elapsed = _elapsed_seconds(stage)
        if elapsed is None or elapsed <= 0:
            return None
        rate = stage.completed / elapsed
    if rate <= 0:
        return None
    return (stage.total - stage.completed) / rate


def _format_clock(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    whole_seconds = max(0, int(seconds))
    hours, remainder = divmod(whole_seconds, 60 * 60)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _render_elapsed(stage: _StageState) -> Text:
    elapsed = _elapsed_seconds(stage)
    if elapsed is None:
        return Text("--", style="dim")
    style = "green" if stage.status in _FINAL_STAGE_STATUSES else "cyan"
    return Text(_format_clock(elapsed), style=style)


def _render_eta(stage: _StageState) -> Text:
    if stage.status in _FINAL_STAGE_STATUSES:
        return Text("done", style="green")
    remaining = _remaining_seconds(stage)
    if remaining is None:
        return Text("--", style="dim")
    return Text(_format_clock(remaining), style="magenta")


def _render_rate(stage: _StageState) -> Text:
    if stage.status not in _ACTIVE_STAGE_STATUSES:
        return Text("--", style="dim")
    if stage.smoothed_rate is None:
        return Text("--", style="dim")
    suffix = _unit_rate_suffix(stage.unit)
    value = _format_rate_value(stage.smoothed_rate)
    return Text(f"{value} {suffix}", style="bright_cyan")


def _render_stage_detail(stage: _StageState) -> Text:
    if not stage.detail:
        return Text("")

    detail = Text()
    prefix, separator, remainder = stage.detail.partition(": ")
    if separator:
        detail.append(prefix, style="white")
        detail.append(separator, style="dim")
        _append_detail_parts(detail, remainder)
        return detail

    _append_detail_parts(detail, stage.detail)
    return detail


def _append_detail_parts(detail: Text, raw_detail: str) -> None:
    for index, part in enumerate(raw_detail.split(" | ")):
        if index > 0:
            detail.append(" | ", style="dim")
        _append_detail_fragment(detail, part.strip())


def _append_detail_fragment(detail: Text, fragment: str) -> None:
    if fragment.lower().startswith("waiting"):
        detail.append(fragment, style="yellow")
        return
    if fragment.lower().startswith("resolving"):
        detail.append(fragment, style="cyan")
        return

    label, _, value = fragment.partition(" ")
    if label in _DETAIL_VALUE_LABELS and value:
        detail.append(label, style="dim")
        detail.append(" ", style="dim")
        detail.append(value, style="bright_cyan")
        return

    number_match = re.match(r"^(?P<number>\d[\d,]*) (?P<label>[A-Za-z].+)$", fragment)
    if number_match is not None:
        detail.append(number_match.group("number"), style="bright_white")
        detail.append(" ", style="dim")
        detail.append(number_match.group("label"), style="dim")
        return

    detail.append(fragment, style="dim")


def _smooth_value(previous: float | None, current: float, *, alpha: float) -> float:
    if previous is None:
        return current
    return previous + alpha * (current - previous)


def _format_rate_value(rate: float) -> str:
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.2f}M"
    if rate >= 100_000:
        return f"{rate / 1_000:.0f}k"
    if rate >= 10_000:
        return f"{rate / 1_000:.1f}k"
    if rate >= 1_000:
        return f"{rate / 1_000:.2f}k"
    if rate >= 10:
        return f"{rate:.1f}"
    if rate >= 1:
        return f"{rate:.2f}"
    return f"{rate:.3f}"
