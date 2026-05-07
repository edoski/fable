from __future__ import annotations

from pathlib import Path

from spice.cli.workflow_command_selection import (
    acquire_workflow_selection,
    evaluate_workflow_selection,
    train_workflow_selection,
    tune_workflow_selection,
)


def test_cli_values_build_acquire_workflow_selection() -> None:
    selection = acquire_workflow_selection(
        surface="surface",
        chain="chain",
        problem="problem",
        features="features",
        provider="provider",
        storage_root=Path("/tmp/spice"),
        dry_run=True,
    )

    assert selection.surface == "surface"
    assert selection.chain == "chain"
    assert selection.problem == "problem"
    assert selection.features == "features"
    assert selection.provider == "provider"
    assert selection.storage_root == Path("/tmp/spice")
    assert selection.dry_run is True


def test_cli_values_build_train_workflow_selection() -> None:
    selection = train_workflow_selection(
        surface="surface",
        chain="chain",
        problem="problem",
        features="features",
        objective="objective",
        evaluation="evaluation",
        model="model",
        tuning_space="space",
        training="training",
        split="split",
        tuning="tuning",
        study="study",
        variant="baseline",
        dataset_id="cor_test",
        study_id="std_test",
    )

    assert selection.surface == "surface"
    assert selection.objective == "objective"
    assert selection.dataset_id == "cor_test"
    assert selection.study_id == "std_test"
    assert selection.variant == "baseline"


def test_cli_values_build_tune_workflow_selection() -> None:
    selection = tune_workflow_selection(
        surface="surface",
        chain="chain",
        problem="problem",
        features="features",
        objective="objective",
        evaluation="evaluation",
        model="model",
        tuning_space="space",
        training="training",
        split="split",
        tuning="tuning",
        study="study",
        dataset_id="cor_test",
        trial_count=3,
    )

    assert selection.surface == "surface"
    assert selection.dataset_id == "cor_test"
    assert selection.trial_count == 3


def test_cli_values_build_evaluate_workflow_selection() -> None:
    selection = evaluate_workflow_selection(
        artifact_id="art_test",
        dataset_id="cor_test",
        evaluation="poisson_replay",
        delay_seconds=12,
        batch_size=64,
    )

    assert selection.artifact_id == "art_test"
    assert selection.dataset_id == "cor_test"
    assert selection.evaluation == "poisson_replay"
    assert selection.delay_seconds == 12
    assert selection.batch_size == 64
