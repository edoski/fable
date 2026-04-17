"""Workflow runtime and native logging bridge."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .reporting import (
    PlainReporter,
    Reporter,
    RichReporter,
    StageMetricDescriptor,
    StageMetricValue,
)
from .reporting.metrics import _with_top_terminal_spacer

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
        metric_descriptors: Iterable[StageMetricDescriptor] = (),
    ) -> Reporter:
        return self.reporter.stage_reporter(
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
        progress_finalized: bool | None = None,
        total: int | None = None,
        unit: str | None = None,
        completed: int | None = None,
        message: str | None = None,
        metrics: Iterable[StageMetricValue] = (),
        metric_descriptors: Iterable[StageMetricDescriptor] = (),
    ) -> None:
        self.reporter.set_stage_state(
            key,
            label=label,
            status=status,
            progress_finalized=progress_finalized,
            total=total,
            unit=unit,
            completed=completed,
            message=message,
            metrics=metrics,
            metric_descriptors=metric_descriptors,
        )

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
            self.console.print(
                _with_top_terminal_spacer(Panel(body, title=title, border_style="cyan"))
            )
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
