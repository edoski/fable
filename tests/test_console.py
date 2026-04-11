from __future__ import annotations

from io import StringIO

from rich.console import Console

from spice.core.console import RichReporter


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
