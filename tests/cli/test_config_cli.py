from __future__ import annotations

import stat
from typing import cast

import pytest
import yaml
from typer.testing import CliRunner

from spice.cli import app
from spice.config import (
    EvaluateConfig,
    TrainConfig,
    WorkflowSelections,
    WorkflowTask,
    resolve_workflow_config,
)
from spice.core.errors import ConfigResolutionError

runner = CliRunner()


def test_config_list_and_show_commands(isolate_conf_root) -> None:
    isolate_conf_root()

    list_result = runner.invoke(app, ["config", "list", "dataset"])
    assert list_result.exit_code == 0, list_result.stdout
    assert "icdcs_2026" in list_result.stdout.splitlines()

    show_result = runner.invoke(app, ["config", "show", "dataset", "icdcs_2026"])
    assert show_result.exit_code == 0, show_result.stdout
    assert yaml.safe_load(show_result.stdout) == {
        "name": "icdcs_2026",
        "evaluation_date": "2025-11-09",
    }


def test_config_edit_seeds_missing_file_and_uses_editor(
    tmp_path, isolate_conf_root, monkeypatch
) -> None:
    conf_root = isolate_conf_root()
    log_path = tmp_path / "editor.log"
    editor_path = tmp_path / "fake-editor"
    editor_path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                f'echo "$1" > "{log_path}"',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    editor_path.chmod(editor_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("EDITOR", str(editor_path))
    monkeypatch.delenv("VISUAL", raising=False)

    result = runner.invoke(app, ["config", "edit", "problem", "phase2_problem"])

    assert result.exit_code == 0, result.stdout
    created_path = conf_root / "problem" / "phase2_problem.yaml"
    assert created_path.exists()
    assert log_path.read_text(encoding="utf-8").strip() == str(created_path)
    assert yaml.safe_load(created_path.read_text(encoding="utf-8"))["id"] == "phase2_problem"


def test_removed_group_is_gone_and_legacy_task_key_is_rejected(tmp_path, isolate_conf_root) -> None:
    conf_root = isolate_conf_root()

    list_result = runner.invoke(app, ["config", "list", "training"])
    assert list_result.exit_code != 0

    legacy_preset = conf_root / "preset" / "legacy.yaml"
    legacy_preset.write_text(
        yaml.safe_dump({"task": {"id": "legacy_problem"}}, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ConfigResolutionError, match="Extra inputs are not permitted"):
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowSelections(
                preset="legacy",
                storage_root=tmp_path / "outputs",
            ),
        )


def test_evaluate_loader_uses_delay_seconds_and_named_override(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        EvaluateConfig,
        load_workflow_config(
            WorkflowTask.EVALUATE,
            workspace=tmp_path,
            preset="icdcs_2026",
            override={
                "problem": {
                    "id": "test_problem",
                    "lookback_seconds": 120,
                    "sample_count": 24,
                    "max_delay_seconds": 24,
                    "compiler": {"id": "timestamp_native"},
                },
                "feature_set": "time_native_baseline",
                "delay_seconds": 12,
                "evaluation": {
                    "evaluator": {
                        "id": "poisson_replay",
                        "window_seconds": 600,
                        "arrival_rate_per_second": 0.02,
                        "repetitions": 3,
                        "seed": 2026,
                    }
                },
            },
            chain="ethereum",
            model="lstm",
        ),
    )

    assert config.problem.id == "test_problem"
    assert config.problem.max_delay_seconds == 24
    assert config.delay_seconds == 12
    assert config.feature_set.id == "time_native_baseline"
    assert config.evaluation.evaluator.id == "poisson_replay"


def test_train_loader_resolves_production_preset(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            preset="icdcs_2026",
            override={
                "dataset": {"evaluation_date": "2025-11-09"},
                "training": {
                    "device": "cpu",
                    "batch_size": 8,
                    "max_epochs": 1,
                    "log_every_n_steps": 1,
                    "precision": "fp32",
                    "compile": "off",
                    "early_stopping": {"patience": 1, "min_delta": 0.0},
                },
                "split": {
                    "train_fraction": 0.8,
                    "validation_fraction": 0.1,
                },
            },
        ),
    )

    assert config.problem.id == "icdcs_2026"
    assert config.problem.compiler.id == "estimated_block"
    assert config.dataset_builder.id == "standard_temporal"
    assert config.feature_set.id == "icdcs_2026"
    assert config.prediction.id == "icdcs_2026"
    assert config.model.id == "lstm"
    assert config.model.hidden_size == 256
