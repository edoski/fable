from __future__ import annotations

from io import StringIO

from spice.core.reporting import PlainReporter, StageMetricDescriptor, StageMetricValue
from spice.core.runtime import create_workflow_runtime

_TRAIN_METRICS = (
    StageMetricDescriptor(id="epoch", label="epoch"),
    StageMetricDescriptor(id="loss", label="loss"),
)


def test_plain_reporter_renders_structured_stage_lines() -> None:
    output = StringIO()
    reporter = PlainReporter(stream=output)

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
            StageMetricValue(id="loss", value="1.23"),
        ),
    )
    fit.finish_task(task_id, message="best epoch 1")

    rendered = output.getvalue()
    assert "train" in rendered
    assert "dataset: icdcs_2026" in rendered
    assert "fit [running] - 10/100 batches - epoch=1/50 loss=1.23 - batch 10/100" in rendered
    assert "fit [done] - 100/100 batches - best epoch 1" in rendered


def test_workflow_runtime_renders_plain_sectioned_summary() -> None:
    output = StringIO()
    runtime = create_workflow_runtime(stream=output)

    with runtime.activate():
        runtime.log_sectioned_summary(
            "training summary",
            [
                ("training", [("best epoch", "6"), ("device", "cuda")]),
                ("validation", [("profit", "0.0118")]),
            ],
        )

    rendered = output.getvalue()
    assert "training summary" in rendered
    assert "training:" in rendered
    assert "  best epoch: 6" in rendered
    assert "validation:" in rendered
