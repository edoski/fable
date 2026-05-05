from __future__ import annotations

from pathlib import Path

from spice.config.selection_application import apply_surface_selection
from spice.config.selections import TrainWorkflowSelection


def test_surface_selection_applies_nested_overrides() -> None:
    applied = apply_surface_selection(
        TrainWorkflowSelection(
            surface="current_row_fee_dynamics",
            training="fast",
            split="rolling",
            tuning="bayes",
            tuning_space="wide",
            storage_root=Path("/tmp/spice"),
            study="ablation",
            variant="tuned",
        )
    )

    assert applied.surface_name == "current_row_fee_dynamics"
    assert applied.frame.training.id == "fast"
    assert applied.frame.training.split == "rolling"
    assert applied.frame.tuning.id == "bayes"
    assert applied.frame.tuning.space == "wide"
    assert applied.frame.storage is not None
    assert applied.frame.storage.root == Path("/tmp/spice")
    assert applied.frame.study is not None
    assert applied.frame.study.name == "ablation"
    assert applied.frame.artifact is not None
    assert applied.frame.artifact.variant.value == "tuned"
