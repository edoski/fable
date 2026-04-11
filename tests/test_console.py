from __future__ import annotations

import logging
import warnings
from io import StringIO

import optuna
from rich.console import Console

from spice.core.console import (
    PlainReporter,
    RichReporter,
    SharedConsoleRichProgressBar,
    create_console_runtime,
    create_reporter,
)


def test_create_reporter_uses_plain_for_non_terminal_console() -> None:
    reporter = create_reporter(Console(file=StringIO(), force_terminal=False, width=120))
    assert isinstance(reporter, PlainReporter)


def test_rich_reporter_logs_task_finish() -> None:
    stream = StringIO()
    reporter = RichReporter(console=Console(file=stream, force_terminal=True, width=120))

    task_id = reporter.start_task("train epochs", total=3, unit="epochs")
    reporter.update_task(task_id, completed=1, message="loss=1.0")
    reporter.finish_task(task_id, message="best_epoch=1")
    reporter.close()
    output = stream.getvalue()

    assert "train epochs" in output
    assert "best_epoch=1" in output


def test_console_runtime_bridges_python_logging_and_warnings() -> None:
    stream = StringIO()
    runtime = create_console_runtime(
        console=Console(file=stream, force_terminal=False, width=120),
    )

    with runtime.activate():
        logging.getLogger("demo").info("hello from logging")
        warnings.warn("hello from warnings", stacklevel=1)
    runtime.close()

    output = stream.getvalue()
    assert "hello from logging" in output
    assert "hello from warnings" in output


def test_console_runtime_filters_native_noise() -> None:
    stream = StringIO()
    runtime = create_console_runtime(
        console=Console(file=stream, force_terminal=False, width=120),
    )

    with runtime.activate():
        lightning_logger = logging.getLogger("lightning")
        lightning_logger.info("GPU available: True (mps), used: False")
        lightning_logger.info("Seed set to 7")
        lightning_logger.warning("real warning")
        warnings.warn(
            "The 'train_dataloader' does not have many workers which may be a bottleneck.",
            stacklevel=1,
        )
    runtime.close()

    output = stream.getvalue()
    assert "GPU available" not in output
    assert "Seed set to 7" not in output
    assert "train_dataloader" not in output
    assert "real warning" in output


def test_console_runtime_restores_logging_state() -> None:
    stream = StringIO()
    runtime = create_console_runtime(
        console=Console(file=stream, force_terminal=False, width=120),
    )
    root_logger = logging.getLogger()
    lightning_logger = logging.getLogger("lightning")
    root_handlers = list(root_logger.handlers)
    root_level = root_logger.level
    lightning_handlers = list(lightning_logger.handlers)
    lightning_propagate = lightning_logger.propagate
    lightning_level = lightning_logger.level

    with runtime.activate():
        assert list(root_logger.handlers) != root_handlers or root_logger.level != root_level
        assert lightning_logger.handlers == []
        assert lightning_logger.propagate is True

    runtime.close()

    assert list(root_logger.handlers) == root_handlers
    assert root_logger.level == root_level
    assert list(lightning_logger.handlers) == lightning_handlers
    assert lightning_logger.propagate == lightning_propagate
    assert lightning_logger.level == lightning_level


def test_console_runtime_bridges_optuna_logs() -> None:
    stream = StringIO()
    runtime = create_console_runtime(
        console=Console(file=stream, force_terminal=False, width=120),
    )

    with runtime.activate():
        with runtime.optuna_logging():
            study = optuna.create_study()
            study.optimize(
                lambda trial: trial.suggest_float("x", -1.0, 1.0) ** 2,
                n_trials=1,
            )
    runtime.close()

    output = stream.getvalue()
    assert "A new study created" in output
    assert "Trial 0 finished" in output


def test_console_runtime_exposes_shared_lightning_progress_bar() -> None:
    runtime = create_console_runtime(
        console=Console(file=StringIO(), force_terminal=True, width=120),
    )
    progress_bar = runtime.lightning_progress_bar()

    assert isinstance(progress_bar, SharedConsoleRichProgressBar)
    assert progress_bar._shared_console is runtime.console
