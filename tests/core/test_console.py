from __future__ import annotations

from io import StringIO

from spice.core.reporting import Reporter


def test_reporter_renders_header_milestone_and_result() -> None:
    output = StringIO()
    reporter = Reporter(stream=output)

    reporter.header(
        "train",
        [("dataset", "same_block_closed"), ("chain", "ethereum"), ("model", "lstm")],
    )
    reporter.milestone("fit epoch=1/10 objective.total_loss=1.2300 best_epoch=1")
    reporter.result(
        "train",
        [("artifact", "/tmp/artifact"), ("best_epoch", "6"), ("test.total_loss", "0.1200")],
    )

    rendered = output.getvalue()
    assert "train dataset=same_block_closed chain=ethereum model=lstm" in rendered
    assert "fit epoch=1/10 objective.total_loss=1.2300 best_epoch=1" in rendered
    assert "train complete artifact=/tmp/artifact best_epoch=6 test.total_loss=0.1200" in rendered


def test_reporter_renders_warnings_and_sections() -> None:
    output = StringIO()
    errors = StringIO()
    reporter = Reporter(stream=output, error_stream=errors)

    reporter.milestone("train cancelled; partial outputs removed", level="warning")
    reporter.sections(
        "artifact summary",
        [("training", [("best epoch", "6"), ("best objective", "0.0118")])],
    )
    reporter.diagnostic_sections(
        "artifact matches",
        [("artifacts", [("artifact", "art_1"), ("artifact", "art_2")])],
    )

    rendered = output.getvalue()
    assert "artifact summary" in rendered
    assert "training:" in rendered
    assert "  best epoch: 6" in rendered
    assert "artifact matches" not in rendered
    assert errors.getvalue() == (
        "warning: train cancelled; partial outputs removed\n"
        "artifact matches\n"
        "artifacts:\n"
        "  artifact: art_1\n"
        "  artifact: art_2\n"
    )
