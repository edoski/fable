from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from spice.cli import app
from spice.config import (
    load_acquire_config,
    load_simulate_config,
    load_train_config,
    load_tune_config,
)

runner = CliRunner()


def _stderr_text(result) -> str:
    return result.stdout + result.stderr


def test_config_list_and_show_commands(tmp_path, isolate_conf_root) -> None:
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


def test_old_task_group_and_runtime_key_are_rejected(tmp_path, isolate_conf_root) -> None:
    isolate_conf_root()

    list_result = runner.invoke(app, ["config", "list", "task"])
    assert list_result.exit_code != 0

    config_path = tmp_path / "legacy_train.yaml"
    config_path.write_text(
        yaml.safe_dump({"task": {"id": "legacy_problem"}}, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown top-level config fields: task"):
        load_train_config(
            preset="icdcs_2026",
            config_path=config_path,
            storage_root=tmp_path / "outputs",
        )


def test_config_create_update_and_unset_commands(tmp_path, isolate_conf_root) -> None:
    conf_root = isolate_conf_root()

    create_chain = runner.invoke(
        app,
        [
            "config",
            "create",
            "chain",
            "phase2_chain",
            "--set",
            "runtime.chain_id=7070",
            "--set",
            "runtime.uses_poa_extra_data=false",
        ],
    )
    assert create_chain.exit_code == 0, create_chain.stdout
    assert yaml.safe_load((conf_root / "chain" / "phase2_chain.yaml").read_text()) == {
        "name": "phase2_chain",
        "runtime": {
            "chain_id": 7070,
            "uses_poa_extra_data": False,
        },
    }

    create_preset = runner.invoke(
        app,
        [
            "config",
            "create",
            "preset",
            "phase2_preset",
            "--set",
            "dataset=icdcs_2026",
            "--set",
            "problem=icdcs_2026",
            "--set",
            "execution=icdcs_2026",
            "--set",
            "chain=ethereum",
            "--set",
            "provider=publicnode",
            "--set",
            "feature_set=icdcs_2026",
            "--set",
            "model=lstm",
            "--set",
            "prediction=candidate_slate_current",
            "--set",
            "acquisition=default",
            "--set",
            "training=icdcs_2026",
            "--set",
            "split=default",
            "--set",
            "simulation=icdcs_2026",
            "--set",
            "tuning=default",
            "--set",
            "study.name=default",
            "--set",
            "storage.root=outputs_phase2",
        ],
    )
    assert create_preset.exit_code == 0, create_preset.stdout

    update_preset = runner.invoke(
        app,
        [
            "config",
            "update",
            "preset",
            "phase2_preset",
            "--unset",
            "storage",
        ],
    )
    assert update_preset.exit_code == 0, update_preset.stdout
    assert yaml.safe_load((conf_root / "preset" / "phase2_preset.yaml").read_text()) == {
        "dataset": "icdcs_2026",
        "problem": "icdcs_2026",
        "execution": "icdcs_2026",
        "chain": "ethereum",
        "provider": "publicnode",
        "model": "lstm",
        "prediction": "candidate_slate_current",
        "feature_set": "icdcs_2026",
        "acquisition": "default",
        "training": "icdcs_2026",
        "split": "default",
        "simulation": "icdcs_2026",
        "tuning": "default",
        "study": {"name": "default"},
    }

    missing_unset = runner.invoke(
        app,
        [
            "config",
            "update",
            "preset",
            "phase2_preset",
            "--unset",
            "storage",
        ],
    )
    assert missing_unset.exit_code == 1, missing_unset.stdout
    assert "Missing unset path: storage" in _stderr_text(missing_unset)


def test_config_create_validates_cross_references(tmp_path, isolate_conf_root) -> None:
    isolate_conf_root()

    bad_provider = runner.invoke(
        app,
        [
            "config",
            "create",
            "provider",
            "broken_provider",
            "--set",
            "rpc.timeout_seconds=30.0",
            "--set",
            "rpc.retry_count=5",
            "--set",
            "rpc.backoff_factor=0.125",
            "--set",
            "chains.ghost.endpoint.url=https://ghost.example.test",
        ],
    )
    assert bad_provider.exit_code == 1, bad_provider.stdout
    assert "declares unknown chains: ghost" in _stderr_text(bad_provider)

    bad_preset = runner.invoke(
        app,
        [
            "config",
            "create",
            "preset",
            "broken_preset",
            "--set",
            "dataset=missing_dataset",
        ],
    )
    assert bad_preset.exit_code == 1, bad_preset.stdout
    assert (
        "preset.dataset references unknown dataset spec: missing_dataset"
        in _stderr_text(bad_preset)
    )


def test_config_delete_blocks_on_dependents_and_force_bypasses(
    tmp_path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()

    create_chain = runner.invoke(
        app,
        [
            "config",
            "create",
            "chain",
            "phase2_chain",
            "--set",
            "runtime.chain_id=9090",
            "--set",
            "runtime.uses_poa_extra_data=false",
        ],
    )
    assert create_chain.exit_code == 0, create_chain.stdout

    create_provider = runner.invoke(
        app,
        [
            "config",
            "create",
            "provider",
            "phase2_provider",
            "--set",
            "rpc.timeout_seconds=30.0",
            "--set",
            "rpc.retry_count=5",
            "--set",
            "rpc.backoff_factor=0.125",
            "--set",
            "chains.phase2_chain.endpoint.url=https://phase2.example.test",
        ],
    )
    assert create_provider.exit_code == 0, create_provider.stdout

    blocked_delete = runner.invoke(
        app,
        ["config", "delete", "chain", "phase2_chain"],
    )
    assert blocked_delete.exit_code == 1, blocked_delete.stdout
    assert "Cannot delete chain spec: phase2_chain" in _stderr_text(blocked_delete)
    assert "provider: phase2_provider" in _stderr_text(blocked_delete)

    forced_delete = runner.invoke(
        app,
        ["config", "delete", "chain", "phase2_chain", "--force"],
    )
    assert forced_delete.exit_code == 0, forced_delete.stdout
    assert not (conf_root / "chain" / "phase2_chain.yaml").exists()

    preset_block = runner.invoke(
        app,
        ["config", "delete", "dataset", "icdcs_2026"],
    )
    assert preset_block.exit_code == 1, preset_block.stdout
    assert "preset: icdcs_2026" in _stderr_text(preset_block)


def test_config_cli_created_specs_resolve_in_runtime_commands(
    tmp_path,
    isolate_conf_root,
    monkeypatch,
) -> None:
    isolate_conf_root()

    feature_outputs = yaml.safe_dump(
        yaml.safe_load(
            runner.invoke(app, ["config", "show", "feature-set", "icdcs_2026"]).stdout
        )["outputs"],
        default_flow_style=True,
        sort_keys=False,
    ).strip()

    commands = [
        [
            "config",
            "create",
            "dataset",
            "phase2_dataset",
            "--set",
            "evaluation_date=2025-11-09",
        ],
        [
            "config",
            "create",
            "problem",
            "phase2_task",
            "--set",
            "lookback_seconds=120",
            "--set",
            "sample_count=24",
            "--set",
            "max_supported_delay_seconds=36",
            "--set",
            "compiler.id=timestamp_native",
        ],
        [
            "config",
            "create",
            "execution",
            "phase2_execution",
            "--set",
            "requested_delay_seconds=36",
        ],
        [
            "config",
            "create",
            "feature-set",
            "phase2_feature_set",
            "--set",
            "family.id=block_native",
            "--set",
            f"outputs={feature_outputs}",
        ],
        [
            "config",
            "create",
            "preset",
            "phase2_preset",
            "--set",
            "dataset=phase2_dataset",
            "--set",
            "problem=phase2_task",
            "--set",
            "execution=phase2_execution",
            "--set",
            "chain=ethereum",
            "--set",
            "provider=publicnode",
            "--set",
            "feature_set=phase2_feature_set",
            "--set",
            "model=lstm",
            "--set",
            "prediction=candidate_slate_current",
            "--set",
            "acquisition=default",
            "--set",
            "training=icdcs_2026",
            "--set",
            "split=default",
            "--set",
            "simulation=icdcs_2026",
            "--set",
            "tuning=default",
            "--set",
            "study.name=default",
            "--set",
            "artifact.variant=baseline",
        ],
    ]
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.stdout

    captured: dict[str, object] = {}

    def _capture_acquire(**kwargs) -> None:
        captured["acquire"] = load_acquire_config(
            preset=kwargs["preset"],
            config_path=kwargs["config"],
            dataset=kwargs["dataset"],
            problem=kwargs["problem"],
            chain=kwargs["chain"],
            provider=kwargs["provider"],
            feature_set=kwargs["feature_set"],
            acquisition=kwargs["acquisition_profile"],
            storage_root=kwargs["storage_root"],
            dry_run=kwargs["dry_run"],
        )

    def _capture_train(**kwargs) -> None:
        captured["train"] = load_train_config(
            preset=kwargs["preset"],
            config_path=kwargs["config"],
            dataset=kwargs["dataset"],
            problem=kwargs["problem"],
            chain=kwargs["chain"],
            model=kwargs["model"],
            feature_set=kwargs["feature_set"],
            prediction=kwargs["prediction"],
            training=kwargs["training_profile"],
            split=kwargs["split"],
            storage_root=kwargs["storage_root"],
            variant=kwargs["variant"],
            study=kwargs["study"],
        )

    def _capture_tune(**kwargs) -> None:
        captured["tune"] = load_tune_config(
            preset=kwargs["preset"],
            config_path=kwargs["config"],
            dataset=kwargs["dataset"],
            problem=kwargs["problem"],
            chain=kwargs["chain"],
            model=kwargs["model"],
            feature_set=kwargs["feature_set"],
            prediction=kwargs["prediction"],
            training=kwargs["training_profile"],
            split=kwargs["split"],
            tuning=kwargs["tuning_profile"],
            tuning_space=kwargs["tuning_space"],
            storage_root=kwargs["storage_root"],
            study=kwargs["study"],
            trial_count=kwargs["trial_count"],
        )

    def _capture_simulate(**kwargs) -> None:
        captured["simulate"] = load_simulate_config(
            preset=kwargs["preset"],
            config_path=kwargs["config"],
            dataset=kwargs["dataset"],
            problem=kwargs["problem"],
            chain=kwargs["chain"],
            model=kwargs["model"],
            feature_set=kwargs["feature_set"],
            prediction=kwargs["prediction"],
            training=kwargs["training_profile"],
            simulation=kwargs["simulation_profile"],
            execution=kwargs["execution"],
            storage_root=kwargs["storage_root"],
            variant=kwargs["variant"],
            study=kwargs["study"],
        )

    monkeypatch.setattr("spice.cli.commands.workflows._run_acquire", _capture_acquire)
    monkeypatch.setattr("spice.cli.commands.workflows._run_train", _capture_train)
    monkeypatch.setattr("spice.cli.commands.workflows._run_tune", _capture_tune)
    monkeypatch.setattr("spice.cli.commands.workflows._run_simulate", _capture_simulate)

    assert runner.invoke(app, ["acquire", "--preset", "phase2_preset"]).exit_code == 0
    assert runner.invoke(app, ["train", "--preset", "phase2_preset"]).exit_code == 0
    assert runner.invoke(app, ["tune", "--preset", "phase2_preset"]).exit_code == 0
    assert runner.invoke(app, ["simulate", "--preset", "phase2_preset"]).exit_code == 0

    assert captured["acquire"].dataset.name == "phase2_dataset"
    assert captured["acquire"].problem.id == "phase2_task"
    assert captured["train"].feature_set.id == "phase2_feature_set"
    assert captured["train"].model.id == "lstm"
    assert captured["tune"].study.name == "default"
    assert captured["simulate"].execution.id == "phase2_execution"
