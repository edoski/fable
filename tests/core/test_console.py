from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from spice.core.reporting import (
    PlainReporter,
    RichReporter,
    StageMetricDescriptor,
    StageMetricValue,
)
from spice.core.reporting.metrics import _render_eta, _render_rate

_ACQUIRE_METRICS = (
    StageMetricDescriptor(id="batch", label="batch", width=7),
    StageMetricDescriptor(id="conc", label="conc", width=5),
)
_TRAIN_METRICS = (
    StageMetricDescriptor(id="epoch", label="epoch", width=7),
    StageMetricDescriptor(id="total_loss", label="loss", width=7),
    StageMetricDescriptor(id="profit_over_baseline", label="profit", width=8),
)


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
        metric_descriptors=_ACQUIRE_METRICS,
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
    history.update_task(
        task_id,
        completed=5,
        metrics=(
            StageMetricValue(id="batch", value="256"),
            StageMetricValue(id="conc", value="8"),
        ),
    )
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
    assert "history [pulling] - 5/10 blocks - batch=256 conc=8" in rendered
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
    lines = rendered.splitlines()
    assert lines[0].strip() == ""
    assert "acquire" in rendered
    assert "history" in rendered
    assert "eval" in rendered
    assert "pull" in rendered
def test_rich_reporter_renders_train_metrics_in_columns() -> None:
    output = StringIO()
    reporter = RichReporter(console=Console(file=output, force_terminal=False, width=240))
    reporter._refresh_live = lambda: None

    reporter.configure_workflow(
        title="train",
        facts=[("dataset", "icdcs_2026"), ("chain", "ethereum"), ("model", "lstm")],
    )
    fit = reporter.stage_reporter(
        "fit",
        label="fit",
        total=100,
        unit="batches",
        metric_descriptors=_TRAIN_METRICS,
    )
    task_id = fit.start_task("train epochs", total=100, unit="batches")
    fit.update_task(
        task_id,
        completed=10,
        message="batch 10/100",
        metrics=(
            StageMetricValue(id="epoch", value="1/50"),
            StageMetricValue(id="total_loss", value="1.31"),
            StageMetricValue(id="profit_over_baseline", value="0.081"),
        ),
    )
    reporter.console.print(reporter._render_stage_table())

    rendered = output.getvalue()
    assert "epoch" in rendered
    assert "loss" in rendered
    assert "profit" in rendered
    assert "batch 10/100" in rendered
    assert "epoch=1/50" not in rendered
    assert "loss=1.31" not in rendered


def test_rich_reporter_renders_acquire_metrics_in_columns_on_wide_terminals() -> None:
    output = StringIO()
    reporter = RichReporter(console=Console(file=output, force_terminal=False, width=220))
    reporter._refresh_live = lambda: None

    reporter.configure_workflow(
        title="acquire",
        facts=[("dataset", "icdcs_2026"), ("chain", "ethereum"), ("provider", "publicnode")],
    )
    history = reporter.stage_reporter(
        "history",
        label="history",
        total=64,
        unit="blocks",
        status="pending",
        running_status="pulling",
        metric_descriptors=_ACQUIRE_METRICS,
    )
    task_id = history.start_task("pull ethereum blocks", total=64, unit="blocks")
    history.update_task(
        task_id,
        completed=32,
        message="retrying",
        metrics=(
            StageMetricValue(id="batch", value="256"),
            StageMetricValue(id="conc", value="8"),
        ),
    )
    reporter.console.print(reporter._render_stage_table())

    rendered = output.getvalue()
    assert "blocks" in rendered
    assert "batch" in rendered
    assert "conc" in rendered
    assert "32/64" in rendered
    assert "256" in rendered
    assert "8" in rendered
    assert "retrying" in rendered
    assert "batch=256" not in rendered
    assert "conc=8" not in rendered


def test_rich_reporter_keeps_hidden_acquire_metrics_in_detail_on_narrow_terminals() -> None:
    output = StringIO()
    reporter = RichReporter(console=Console(file=output, force_terminal=False, width=100))
    reporter._refresh_live = lambda: None

    reporter.configure_workflow(
        title="acquire",
        facts=[("dataset", "icdcs_2026"), ("chain", "ethereum"), ("provider", "publicnode")],
    )
    history = reporter.stage_reporter(
        "history",
        label="history",
        total=64,
        unit="blocks",
        status="pending",
        running_status="pulling",
        metric_descriptors=_ACQUIRE_METRICS,
    )
    task_id = history.start_task("pull ethereum blocks", total=64, unit="blocks")
    history.update_task(
        task_id,
        completed=32,
        message="retrying",
        metrics=(
            StageMetricValue(id="batch", value="256"),
            StageMetricValue(id="conc", value="8"),
        ),
    )
    reporter.console.print(reporter._render_stage_table())

    rendered = output.getvalue()
    assert "batch=256" in rendered
    assert "conc=8" in rendered
    assert "retrying" in rendered


def test_reporter_rate_uses_recent_progress_while_eta_uses_stage_average(monkeypatch) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

    clock = _Clock()
    monkeypatch.setattr("spice.core.reporting.plain.time.monotonic", clock.monotonic)
    monkeypatch.setattr("spice.core.reporting.metrics.time.monotonic", clock.monotonic)

    reporter = RichReporter(console=Console(file=StringIO(), force_terminal=False, width=160))
    reporter._refresh_live = lambda: None

    fit = reporter.stage_reporter(
        "fit",
        label="fit",
        total=100,
        unit="batches",
        metric_descriptors=_TRAIN_METRICS,
    )
    task_id = fit.start_task("train epochs", total=100, unit="batches")

    clock.now = 1.0
    fit.update_task(task_id, advance=10)
    stage = reporter._stages["fit"]
    assert stage.smoothed_rate == pytest.approx(10.0)

    clock.now = 11.0
    fit.update_task(
        task_id,
        completed=10,
        message="validation",
        metrics=(StageMetricValue(id="profit_over_baseline", value="0.020"),),
    )
    assert stage.smoothed_rate == pytest.approx(10.0)

    clock.now = 12.0
    fit.update_task(task_id, advance=10)
    assert stage.smoothed_rate == pytest.approx(10.0)
    assert _render_rate(stage).plain == "10.0 bat/s"
    assert _render_eta(stage).plain == "0:48"


def test_reporter_clears_rate_when_stage_restarts(monkeypatch) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

    clock = _Clock()
    monkeypatch.setattr("spice.core.reporting.plain.time.monotonic", clock.monotonic)
    monkeypatch.setattr("spice.core.reporting.metrics.time.monotonic", clock.monotonic)

    reporter = RichReporter(console=Console(file=StringIO(), force_terminal=False, width=160))
    reporter._refresh_live = lambda: None

    history = reporter.stage_reporter(
        "history",
        label="history",
        total=100,
        unit="blocks",
        status="pending",
        running_status="pulling",
    )
    task_id = history.start_task("pull history", total=100, unit="blocks")

    clock.now = 1.0
    history.update_task(task_id, advance=10)
    history.finish_task(task_id, silent=True)

    reporter.set_stage_state(
        "history",
        status="planning",
        total=200,
        completed=10,
        unit="blocks",
        message="extending history window",
    )

    stage = reporter._stages["history"]
    assert stage.smoothed_rate is None
    assert _render_rate(stage).plain == "--"


def test_reporter_keeps_provisional_progress_below_complete() -> None:
    output = StringIO()
    reporter = RichReporter(console=Console(file=output, force_terminal=False, width=220))
    reporter._refresh_live = lambda: None

    reporter.configure_workflow(
        title="acquire",
        facts=[("dataset", "icdcs_2026"), ("chain", "ethereum"), ("provider", "publicnode")],
    )
    reporter.set_stage_state(
        "history",
        label="history",
        status="planning",
        progress_finalized=False,
        total=400_000,
        completed=400_000,
        unit="blocks",
        message="checking sufficiency",
    )
    reporter.console.print(reporter._render_stage_table())

    rendered = output.getvalue()
    assert "400k/400k" in rendered
    assert " 99%" in rendered
    assert "100%" not in rendered
