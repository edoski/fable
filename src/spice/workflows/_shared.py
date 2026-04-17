"""Shared workflow helpers."""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ..core.reporting import Reporter
from ..core.runtime import ConsoleRuntime, create_console_runtime


@dataclass(slots=True)
class WorkflowSession:
    runtime: ConsoleRuntime
    reporter: Reporter


@dataclass(slots=True)
class _InterruptState:
    interrupted: bool = False


@contextmanager
def _capture_interrupts() -> Iterator[_InterruptState]:
    state = _InterruptState()
    signal_ids = [
        getattr(signal, name)
        for name in ("SIGINT", "SIGTERM")
        if hasattr(signal, name)
    ]
    if not signal_ids:
        yield state
        return
    previous_handlers: dict[int, object] = {
        signum: signal.getsignal(signum) for signum in signal_ids
    }

    def _handle_interrupt(signum, frame) -> None:
        state.interrupted = True
        previous_handler = previous_handlers.get(signum, signal.SIG_DFL)
        if previous_handler == signal.SIG_IGN:
            return
        if previous_handler == signal.SIG_DFL:
            raise KeyboardInterrupt
            return
        if callable(previous_handler):
            previous_handler(signum, frame)
            return
        raise KeyboardInterrupt

    registered: list[int] = []
    try:
        for signum in signal_ids:
            signal.signal(signum, _handle_interrupt)
            registered.append(signum)
    except ValueError:
        yield state
        return
    try:
        yield state
    finally:
        for signum in registered:
            try:
                signal.signal(signum, previous_handlers[signum])
            except ValueError:
                pass


def _cleanup_after_interrupt(
    reporter: Reporter,
    *,
    label: str,
    cleanup: Callable[[], None],
) -> None:
    cleanup()
    reporter.close()
    reporter.log(f"{label} cancelled; partial outputs removed", level="warning")


@contextmanager
def abort_cleanup(
    reporter: Reporter,
    *,
    label: str,
    cleanup: Callable[[], None],
) -> Iterator[None]:
    with _capture_interrupts() as interrupt_state:
        try:
            yield
        except BaseException as exc:
            if interrupt_state.interrupted or isinstance(exc, KeyboardInterrupt):
                _cleanup_after_interrupt(reporter, label=label, cleanup=cleanup)
            raise
        if interrupt_state.interrupted:
            _cleanup_after_interrupt(reporter, label=label, cleanup=cleanup)
            raise KeyboardInterrupt


@contextmanager
def managed_workflow(
    config: object,
    *,
    run_name: str,
    runtime: ConsoleRuntime | None = None,
    reporter: Reporter | None = None,
    default_runtime_factory: Callable[..., ConsoleRuntime] = create_console_runtime,
    nested: bool = False,
) -> Iterator[WorkflowSession]:
    del config, run_name, nested
    active_runtime = runtime or default_runtime_factory(reporter=reporter)
    owns_runtime = runtime is None
    try:
        with active_runtime.activate():
            yield WorkflowSession(
                runtime=active_runtime,
                reporter=active_runtime.reporter,
            )
    finally:
        if owns_runtime:
            active_runtime.close()
