from __future__ import annotations

from io import StringIO

from rich.console import Console

from spice.core.console import PlainReporter, RichReporter, create_reporter


def test_plain_reporter_throttles_logs(monkeypatch) -> None:
    stream = StringIO()
    reporter = PlainReporter(console=Console(file=stream, force_terminal=False, width=120))
    times = iter([10.0, 15.0, 25.0])
    monkeypatch.setattr("spice.core.console.time.monotonic", lambda: next(times))

    reporter.throttled_log("key", "first line")
    reporter.throttled_log("key", "second line")
    reporter.throttled_log("key", "third line")
    reporter.close()
    output = stream.getvalue()

    assert "first line" in output
    assert "second line" not in output
    assert "third line" in output


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
