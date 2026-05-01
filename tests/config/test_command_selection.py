from __future__ import annotations

from pathlib import Path

from spice.config.command_selection import (
    build_acquire_command_selection,
    build_evaluate_command_selection,
    build_train_command_selection,
    build_tune_command_selection,
)


def test_acquire_command_selection_keeps_operator_fields(tmp_path: Path) -> None:
    selection = build_acquire_command_selection(
        surface="current_row_fee_dynamics",
        chain="avalanche",
        problem=None,
        features="core_fee_dynamics",
        provider="publicnode",
        storage_root=tmp_path / "outputs",
        dry_run=True,
    )

    assert selection.surface == "current_row_fee_dynamics"
    assert selection.chain == "avalanche"
    assert selection.features == "core_fee_dynamics"
    assert selection.provider == "publicnode"
    assert selection.storage_root == tmp_path / "outputs"
    assert selection.dry_run is True


def test_train_command_selection_filters_sparse_cli_values() -> None:
    selection = build_train_command_selection(
        surface="current_row_fee_dynamics",
        chain=None,
        problem=None,
        features=None,
        objective=None,
        evaluation=None,
        model=None,
        tuning_space=None,
        training=None,
        split=None,
        tuning=None,
        study="experiment",
        dataset_id="cor_123",
        study_id=None,
        variant="baseline",
    )

    assert selection.surface == "current_row_fee_dynamics"
    assert selection.study == "experiment"
    assert selection.dataset_id == "cor_123"
    assert selection.study_id is None
    assert selection.variant == "baseline"


def test_tune_command_selection_keeps_trial_count() -> None:
    selection = build_tune_command_selection(
        surface="current_row_fee_dynamics",
        chain=None,
        problem=None,
        features=None,
        objective=None,
        evaluation=None,
        model=None,
        tuning_space=None,
        training=None,
        split=None,
        tuning=None,
        study=None,
        dataset_id="cor_123",
        trial_count=2,
    )

    assert selection.surface == "current_row_fee_dynamics"
    assert selection.dataset_id == "cor_123"
    assert selection.trial_count == 2


def test_evaluate_command_selection_is_id_shaped() -> None:
    selection = build_evaluate_command_selection(
        artifact_id="art_123",
        dataset_id="cor_123",
        evaluation="poisson_replay_2h",
        delay_seconds=12,
        batch_size=64,
    )

    assert selection.artifact_id == "art_123"
    assert selection.dataset_id == "cor_123"
    assert selection.evaluation == "poisson_replay_2h"
    assert selection.delay_seconds == 12
    assert selection.batch_size == 64
    assert selection.storage_root is None
