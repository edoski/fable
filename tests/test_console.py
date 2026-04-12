from __future__ import annotations

from io import StringIO

from rich.console import Console

from spice.core.console import PlainReporter, RichReporter


def test_plain_reporter_renders_staged_workflow_lines() -> None:
    output = StringIO()
    reporter = PlainReporter(console=Console(file=output, force_terminal=False, width=120))

    reporter.configure_workflow(
        title="acquire",
        facts=[("dataset", "icdcs_2026"), ("chain", "Avalanche"), ("provider", "PublicNode")],
    )
    history = reporter.stage_reporter(
        "history",
        label="history",
        total=10,
        unit="blocks",
        status="pending",
        running_status="pulling",
    )
    reporter.set_stage_state(
        "evaluation",
        label="evaluation",
        status="pending",
        total=4,
        unit="blocks",
        message="waiting for history",
    )

    task_id = history.start_task("pull Avalanche blocks", total=10, unit="blocks")
    history.update_task(task_id, completed=5, message="batch 256 | conc 8")
    history.finish_task(task_id, message="5 files")
    reporter.set_stage_state(
        "history",
        status="extended",
        total=10,
        completed=10,
        unit="blocks",
        message="5 files",
    )

    rendered = output.getvalue()
    assert "acquire" in rendered
    assert "dataset: icdcs_2026" in rendered
    assert "provider: PublicNode" in rendered
    assert "evaluation [pending] - 0/4 blocks - waiting for history" in rendered
    assert "history [pulling] - 5/10 blocks - batch 256 | conc 8" in rendered
    assert "history [extended] - 10/10 blocks - 5 files" in rendered


def test_rich_reporter_smoke() -> None:
    output = StringIO()
    reporter = RichReporter(console=Console(file=output, force_terminal=True, width=200))

    reporter.configure_workflow(
        title="acquire",
        facts=[("dataset", "icdcs_2026"), ("chain", "polygon"), ("provider", "publicnode")],
    )
    history = reporter.stage_reporter(
        "history",
        label="history",
        total=10,
        unit="blocks",
        status="pending",
        running_status="pulling",
    )
    reporter.set_stage_state(
        "evaluation",
        label="evaluation",
        status="pending",
        total=4,
        unit="blocks",
    )
    task_id = history.start_task("pull polygon blocks", total=10, unit="blocks")
    history.update_task(task_id, completed=5)
    reporter.close()

    rendered = output.getvalue()
    assert "acquire" in rendered
    assert "history" in rendered
    assert "evaluation" in rendered
    assert "pulling" in rendered
