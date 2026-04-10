from __future__ import annotations

from io import StringIO

from rich.console import Console

from spice.core.console import RichReporter
from spice.modeling.training import EpochMetrics


def test_rich_reporter_throttles_pull_output(monkeypatch) -> None:
    stream = StringIO()
    reporter = RichReporter(console=Console(file=stream, force_terminal=False, width=120))
    times = iter([10.0, 15.0, 25.0])
    monkeypatch.setattr("spice.core.console.time.monotonic", lambda: next(times))

    reporter.update_pull(completed_chunks=1, total_chunks=10, latest_output="first line")
    reporter.update_pull(completed_chunks=2, total_chunks=10, latest_output="second line")
    reporter.update_pull(completed_chunks=3, total_chunks=10, latest_output="third line")
    reporter.close()
    output = stream.getvalue()

    assert "first line" in output
    assert "second line" not in output
    assert "third line" in output


def test_rich_reporter_logs_epoch_summary() -> None:
    stream = StringIO()
    reporter = RichReporter(console=Console(file=stream, force_terminal=False, width=120))
    metrics = EpochMetrics(
        total_loss=1.0,
        accuracy=0.75,
        mean_cost_over_optimum=0.1,
        mean_profit_over_baseline=0.2,
    )

    reporter.training_epoch(
        epoch=1,
        total_epochs=3,
        train_metrics=metrics,
        validation_metrics=metrics,
        best_epoch=1,
        patience_left=0,
    )
    reporter.close()

    assert "epoch 1/3" in stream.getvalue()


def test_rich_reporter_starts_known_pull_task_on_first_progress_update() -> None:
    reporter = RichReporter(console=Console(file=StringIO(), force_terminal=True, width=120))

    reporter.start_pull(label="pull test", total_chunks=10)
    assert reporter.progress.tasks[reporter._pull_task].started is False

    reporter.update_pull(completed_chunks=1, total_chunks=10)

    assert reporter.progress.tasks[reporter._pull_task].started is True
    reporter.close()
