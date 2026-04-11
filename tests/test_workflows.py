from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from spice.core.console import NullReporter, create_reporter
from spice.workflows.acquire import run as run_acquire
from spice.workflows.simulate import run as run_simulate
from spice.workflows.train import run as run_train
from spice.workflows.tune import run as run_tune
from tests.support import (
    base_overrides,
    compose_experiment,
    make_evaluation_rows,
    make_history_rows,
    write_dataset_dir,
)


def _plain_reporter(stream: StringIO):
    return create_reporter(Console(file=stream, force_terminal=False, width=120))


def _rich_reporter(stream: StringIO):
    return create_reporter(Console(file=stream, force_terminal=True, width=120))


def test_train_and_simulate_workflows_write_reports(tmp_path) -> None:
    train_config = compose_experiment("train", overrides=base_overrides(tmp_path))
    simulate_config = compose_experiment("simulate", overrides=base_overrides(tmp_path))

    history_dir = Path(train_config.paths.history_dir)
    evaluation_dir = Path(simulate_config.paths.evaluation_dir)
    write_dataset_dir(history_dir, make_history_rows())
    write_dataset_dir(evaluation_dir, make_evaluation_rows())

    run_train(train_config, reporter=NullReporter())
    run_simulate(simulate_config, reporter=NullReporter())

    train_report = (
        tmp_path
        / "artifacts"
        / "models"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "lstm"
        / "36s"
        / "train_report.json"
    )
    simulation_report = (
        tmp_path
        / "artifacts"
        / "models"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "lstm"
        / "36s"
        / "simulation_report.json"
    )
    assert train_report.is_file()
    assert simulation_report.is_file()


def test_train_workflow_plain_output_filters_lightning_noise(tmp_path) -> None:
    config = compose_experiment("train", overrides=base_overrides(tmp_path))
    history_dir = Path(config.paths.history_dir)
    write_dataset_dir(history_dir, make_history_rows())
    stream = StringIO()
    reporter = _plain_reporter(stream)

    run_train(config, reporter=reporter)
    reporter.close()

    output = stream.getvalue()
    assert "load history dataset started" in output
    assert "prepare training dataset finished" in output
    assert "training started" in output
    assert "training finished: best_epoch=" in output
    assert "evaluate model finished" in output
    assert "GPU available" not in output
    assert "TPU available" not in output
    assert "litlogger" not in output.lower()
    assert "LeafSpec" not in output
    assert "train_dataloader" not in output
    assert "val_dataloader" not in output


def test_train_workflow_interactive_output_uses_shared_console(tmp_path) -> None:
    config = compose_experiment("train", overrides=base_overrides(tmp_path))
    history_dir = Path(config.paths.history_dir)
    write_dataset_dir(history_dir, make_history_rows())
    stream = StringIO()
    reporter = _rich_reporter(stream)

    run_train(config, reporter=reporter)
    reporter.close()

    output = stream.getvalue()
    assert "load history dataset" in output
    assert "training started" in output
    assert "best_epoch=" in output
    assert "GPU available" not in output
    assert "litlogger" not in output.lower()


def test_train_workflow_creates_local_mlflow_run(tmp_path) -> None:
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path) + ["tracking.enabled=true"],
    )
    history_dir = Path(config.paths.history_dir)
    write_dataset_dir(history_dir, make_history_rows())

    run_train(config, reporter=NullReporter())

    mlruns_dir = tmp_path / "artifacts" / "mlruns"
    assert mlruns_dir.is_dir()
    assert (mlruns_dir / "mlflow.db").is_file()
    assert (mlruns_dir / "artifacts").is_dir()


def test_train_applies_best_tuning_params_and_cleans_stale_outputs(tmp_path) -> None:
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path) + ["tuning.apply_best_params=true"],
    )
    history_dir = Path(config.paths.history_dir)
    write_dataset_dir(history_dir, make_history_rows())

    artifact_dir = (
        tmp_path
        / "artifacts"
        / "models"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "lstm"
        / "36s"
    )
    best_params_path = artifact_dir / "tuning" / "best_params.json"
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.write_text(
        json.dumps(
            {
                "kind": "tuning_best_params",
                "params": {
                    "model.hidden_size": 64,
                    "training.learning_rate": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )
    stale_checkpoint = artifact_dir / "checkpoints" / "stale.ckpt"
    stale_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stale_checkpoint.write_text("stale", encoding="utf-8")
    stale_simulation = artifact_dir / "simulation_report.json"
    stale_simulation.write_text("stale", encoding="utf-8")

    run_train(config, reporter=NullReporter())

    artifact_payload = json.loads((artifact_dir / "artifact.json").read_text(encoding="utf-8"))
    assert artifact_payload["model"]["hidden_size"] == 64
    assert not stale_checkpoint.exists()
    assert not stale_simulation.exists()


def test_tune_workflow_writes_optuna_summary(tmp_path) -> None:
    config = compose_experiment("tune", overrides=base_overrides(tmp_path))
    config.tuning.trial_count = 2
    config.tuning.enable_pruning = False
    config.training.max_epochs = 1
    config.tracking.enabled = False
    config.tuning.search_space = {
        "training.learning_rate": [1e-4, 3e-4],
        "model.hidden_size": [64, 128],
    }

    history_dir = Path(config.paths.history_dir)
    write_dataset_dir(history_dir, make_history_rows())
    stale_trial = (
        tmp_path
        / "artifacts"
        / "models"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "lstm"
        / "36s"
        / "tuning"
        / "trials"
        / "trial-999"
        / "stale.txt"
    )
    stale_trial.parent.mkdir(parents=True, exist_ok=True)
    stale_trial.write_text("stale", encoding="utf-8")

    run_tune(config)

    tuning_root = (
        tmp_path
        / "artifacts"
        / "models"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "lstm"
        / "36s"
        / "tuning"
    )
    study_path = tuning_root / "study.json"
    trials_path = tuning_root / "trials.json"
    best_params_path = tuning_root / "best_params.json"
    assert study_path.is_file()
    assert trials_path.is_file()
    assert best_params_path.is_file()
    assert not stale_trial.exists()
    assert (tuning_root / "trials" / "trial-000" / "train_report.json").is_file()

    study_payload = json.loads(study_path.read_text(encoding="utf-8"))
    trials_payload = json.loads(trials_path.read_text(encoding="utf-8"))
    best_params_payload = json.loads(best_params_path.read_text(encoding="utf-8"))
    assert study_payload["kind"] == "tuning_study"
    assert study_payload["dataset_id"] == "icdcs_2025_11_09"
    assert study_payload["trial_counts"]["total"] == 2
    assert len(trials_payload) == 2
    assert best_params_payload["kind"] == "tuning_best_params"
    assert best_params_payload["params"]


def test_tune_workflow_plain_output_bridges_optuna_logs(tmp_path) -> None:
    config = compose_experiment("tune", overrides=base_overrides(tmp_path))
    config.tuning.trial_count = 2
    config.tuning.enable_pruning = False
    config.training.max_epochs = 1
    config.tracking.enabled = False
    config.tuning.search_space = {
        "training.learning_rate": [1e-4, 3e-4],
        "model.hidden_size": [64, 128],
    }
    history_dir = Path(config.paths.history_dir)
    write_dataset_dir(history_dir, make_history_rows())
    stream = StringIO()
    reporter = _plain_reporter(stream)

    run_tune(config, reporter=reporter)
    reporter.close()

    output = stream.getvalue()
    assert "tune study started" in output
    assert "A new study created" in output
    assert "Trial 0 finished" in output
    assert "Trial 1 finished" in output
    assert "trial 1/2 started" not in output
    assert "tune study finished: best_value=" in output


def test_acquire_dry_run_plain_output_is_human_summary(tmp_path) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["acquisition.dry_run=true"],
    )
    config.tracking.enabled = False
    stream = StringIO()
    reporter = _plain_reporter(stream)

    run_acquire(config, reporter=reporter)
    reporter.close()

    output = stream.getvalue()
    assert "acquire dry run" in output
    assert "dataset: icdcs_2025_11_09" in output
    assert "history window:" in output
    assert "evaluation expected:" in output
    assert '{"dataset_id"' not in output


def test_simulate_workflow_plain_output_is_sparse(tmp_path) -> None:
    train_config = compose_experiment("train", overrides=base_overrides(tmp_path))
    simulate_config = compose_experiment("simulate", overrides=base_overrides(tmp_path))
    history_dir = Path(train_config.paths.history_dir)
    evaluation_dir = Path(simulate_config.paths.evaluation_dir)
    write_dataset_dir(history_dir, make_history_rows())
    write_dataset_dir(evaluation_dir, make_evaluation_rows())
    run_train(train_config, reporter=NullReporter())
    stream = StringIO()
    reporter = _plain_reporter(stream)

    run_simulate(simulate_config, reporter=reporter)
    reporter.close()

    output = stream.getvalue()
    predict_lines = [line for line in output.splitlines() if line.startswith("predict offsets:")]
    assert "simulate repetitions finished: total_events=" in output
    assert predict_lines
    assert len(predict_lines) < 15
