"""Workflow runtime and logging bridge."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOBase

from .reporting import (
    PlainReporter,
    Reporter,
    StageMetricDescriptor,
    StageMetricValue,
)

_NOISE_PATTERNS = (
    re.compile(r"^Seed set to \d+$"),
    re.compile(r"litlogger", re.IGNORECASE),
)


@dataclass(slots=True)
class _LoggerState:
    handlers: list[logging.Handler]
    level: int
    propagate: bool


class _NativeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in _NOISE_PATTERNS)


class _ReporterLogHandler(logging.Handler):
    def __init__(self, reporter: Reporter) -> None:
        super().__init__(level=logging.INFO)
        self._reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        level = (
            "error"
            if record.levelno >= logging.ERROR
            else "warning" if record.levelno >= logging.WARNING else "info"
        )
        self._reporter.log(message, level=level)


class WorkflowRuntime:
    """Workflow-scoped reporter owner with stdlib log bridging."""

    def __init__(
        self,
        *,
        reporter: Reporter | None = None,
        stream: TextIOBase | None = None,
    ) -> None:
        self.reporter = reporter or create_reporter(stream=stream)
        self._owns_reporter = reporter is None
        self._activation_depth = 0
        self._root_state: _LoggerState | None = None
        self._pywarnings_state: _LoggerState | None = None
        self._root_handler: _ReporterLogHandler | None = None

    @contextmanager
    def activate(self) -> Iterator[WorkflowRuntime]:
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
        self._root_handler = _ReporterLogHandler(self.reporter)
        self._root_handler.addFilter(_NativeLogFilter())
        root_logger.addHandler(self._root_handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

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

        if self._root_state is not None:
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            for handler in self._root_state.handlers:
                root_logger.addHandler(handler)
            root_logger.setLevel(self._root_state.level)
            root_logger.propagate = self._root_state.propagate
            self._root_state = None
        self._root_handler = None


def create_reporter(*, stream: TextIOBase | None = None) -> Reporter:
    return PlainReporter(stream=stream)


def create_workflow_runtime(
    *,
    reporter: Reporter | None = None,
    stream: TextIOBase | None = None,
) -> WorkflowRuntime:
    return WorkflowRuntime(reporter=reporter, stream=stream)
