from __future__ import annotations

import json

from spice.core.console import NullReporter
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


class RecordingReporter(NullReporter):
    def __init__(self) -> None:
        self.started: list[str] = []
        self.updated: list[tuple[int, int | None, int | None, str | None]] = []
        self.finished: list[tuple[int, str | None]] = []
        self.logged: list[tuple[str, str]] = []
        self._next_task_id = 1

    def log(self, message: str, *, level: str = "info") -> None:
        self.logged.append((level, message))

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        self.started.append(name)
        return task_id

    def update_task(
        self,
        task_id: int,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
    ) -> None:
        self.updated.append((task_id, completed, advance, message))

    def finish_task(self, task_id: int, *, message: str | None = None) -> None:
        self.finished.append((task_id, message))


def test_train_and_simulate_workflows_write_reports(tmp_path) -> None:
    train_config = compose_experiment("train", overrides=base_overrides(tmp_path))
    simulate_config = compose_experiment("simulate", overrides=base_overrides(tmp_path))

    history_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "history"
    )
    evaluation_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "evaluation"
    )
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


def test_train_workflow_reports_standardized_progress(tmp_path) -> None:
    config = compose_experiment("train", overrides=base_overrides(tmp_path))
    history_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "history"
    )
    write_dataset_dir(history_dir, make_history_rows())
    reporter = RecordingReporter()

    run_train(config, reporter=reporter)

    assert "load history dataset" in reporter.started
    assert "prepare training dataset" in reporter.started
    assert "train epochs" in reporter.started
    assert "evaluate model" in reporter.started
    assert "write training artifact" in reporter.started
    assert any(message and "loss=" in message for _, _, _, message in reporter.updated)


def test_train_workflow_creates_local_mlflow_run(tmp_path) -> None:
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path) + ["tracking.enabled=true"],
    )
    history_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "history"
    )
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
    history_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "history"
    )
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

    history_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "history"
    )
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


def test_tune_workflow_reports_study_progress(tmp_path) -> None:
    config = compose_experiment("tune", overrides=base_overrides(tmp_path))
    config.tuning.trial_count = 2
    config.tuning.enable_pruning = False
    config.training.max_epochs = 1
    config.tracking.enabled = False
    config.tuning.search_space = {
        "training.learning_rate": [1e-4, 3e-4],
        "model.hidden_size": [64, 128],
    }
    history_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "history"
    )
    write_dataset_dir(history_dir, make_history_rows())
    reporter = RecordingReporter()

    run_tune(config, reporter=reporter)

    assert "tune study" in reporter.started
    assert "write tuning summary" in reporter.started
    assert any(message and "complete" in message for _, _, _, message in reporter.updated)
    assert any("trial 1/2 started" in message for _, message in reporter.logged)


def test_simulate_workflow_reports_standardized_progress(tmp_path) -> None:
    train_config = compose_experiment("train", overrides=base_overrides(tmp_path))
    simulate_config = compose_experiment("simulate", overrides=base_overrides(tmp_path))
    history_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "history"
    )
    evaluation_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "evaluation"
    )
    write_dataset_dir(history_dir, make_history_rows())
    write_dataset_dir(evaluation_dir, make_evaluation_rows())
    run_train(train_config, reporter=NullReporter())
    reporter = RecordingReporter()

    run_simulate(simulate_config, reporter=reporter)

    assert "load inference inputs" in reporter.started
    assert "prepare inference dataset" in reporter.started
    assert "predict offsets" in reporter.started
    assert "simulate repetitions" in reporter.started
    assert "write simulation report" in reporter.started
