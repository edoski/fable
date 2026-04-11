from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from spice.acquisition.rpc import BlockPullPlan, BlockRange, TimestampRange
from spice.acquisition.windowing import required_history_block_count
from spice.core.console import NullReporter, create_reporter
from spice.workflows._tuning import TuningBestParamsReport, TuningStudyReport, TuningTrialRecord
from spice.workflows.acquire import run as run_acquire
from spice.workflows.simulate import run as run_simulate
from spice.workflows.train import run as run_train
from spice.workflows.tune import run as run_tune
from tests.support import (
    base_overrides,
    compose_experiment,
    make_block_rows,
    make_evaluation_rows,
    make_history_rows,
    write_dataset_dir,
)


def _reporter(stream: StringIO, *, interactive: bool):
    return create_reporter(Console(file=stream, force_terminal=interactive, width=120))


def _artifact_dir(config) -> Path:
    return config.paths.artifact_root


def _train_report_path(config) -> Path:
    return config.paths.train_report_path


def _simulation_report_path(config) -> Path:
    return config.paths.simulation_report_path


def _tuning_root(config) -> Path:
    return config.paths.tuning_root


def _seed_train_history(config) -> Path:
    history_dir = config.paths.history_dir
    write_dataset_dir(history_dir, make_history_rows())
    return history_dir


def _seed_simulation_inputs(train_config, simulate_config) -> tuple[Path, Path]:
    history_dir = _seed_train_history(train_config)
    evaluation_dir = simulate_config.paths.evaluation_dir
    write_dataset_dir(evaluation_dir, make_evaluation_rows())
    return history_dir, evaluation_dir


@pytest.mark.parametrize("interactive", [False, True], ids=["plain", "rich"])
def test_train_workflow_reporter_output_filters_native_noise(
    tmp_path,
    interactive: bool,
) -> None:
    config = compose_experiment("train", overrides=base_overrides(tmp_path))
    _seed_train_history(config)
    stream = StringIO()
    reporter = _reporter(stream, interactive=interactive)

    run_train(config, reporter=reporter)
    reporter.close()

    output = stream.getvalue()
    assert "load history dataset" in output
    assert "prepare training dataset" in output
    assert "training started" in output
    assert "training finished: best_epoch=" in output
    assert "evaluate model finished" in output
    assert "GPU available" not in output
    assert "TPU available" not in output
    assert "litlogger" not in output.lower()
    assert "LeafSpec" not in output
    assert "train_dataloader" not in output
    assert "val_dataloader" not in output


def test_train_applies_best_tuning_params_cleans_stale_outputs_and_tracks(tmp_path) -> None:
    config = compose_experiment("train", overrides=base_overrides(tmp_path))
    config.tuning.apply_best_params = True
    config.tracking.enabled = True
    _seed_train_history(config)

    artifact_dir = _artifact_dir(config)
    train_report_path = _train_report_path(config)
    mlruns_dir = config.paths.mlruns_dir
    best_params_path = config.paths.tuning_best_params_path
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.write_text(
        json.dumps(
            {
                "kind": "tuning_best_params",
                "study_name": config.tuning.study_name,
                "chain": config.chain.name.value,
                "dataset_id": config.dataset.id,
                "family": config.model.family.value,
                "max_delay_seconds": config.dataset.temporal.max_delay_seconds,
                "lookback_seconds": config.dataset.temporal.lookback_seconds,
                "anchor_count": config.dataset.sampling.anchor_count,
                "objective_metric": config.tuning.objective_metric.value,
                "direction": config.tuning.direction.value,
                "trial": {
                    "number": 0,
                    "value": 1.234,
                },
                "params": {
                    "model": {
                        "hidden_size": 64,
                    },
                    "training": {
                        "learning_rate": 0.001,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    stale_checkpoint = artifact_dir / "checkpoints" / "stale.ckpt"
    stale_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stale_checkpoint.write_text("stale", encoding="utf-8")
    stale_simulation = _simulation_report_path(config)
    stale_simulation.write_text("stale", encoding="utf-8")

    run_train(config, reporter=NullReporter())

    artifact_payload = json.loads((artifact_dir / "artifact.json").read_text(encoding="utf-8"))
    assert artifact_payload["model"]["hidden_size"] == 64
    assert (artifact_dir / "artifact.json").is_file()
    assert (artifact_dir / "model.pt").is_file()
    assert train_report_path.is_file()
    assert not stale_checkpoint.exists()
    assert not stale_simulation.exists()
    assert mlruns_dir.is_dir()
    assert (mlruns_dir / "mlflow.db").is_file()
    assert (mlruns_dir / "artifacts").is_dir()


def test_tune_workflow_writes_optuna_summary(tmp_path) -> None:
    config = compose_experiment("tune", overrides=base_overrides(tmp_path))
    config.tuning.trial_count = 2
    config.tuning.enable_pruning = False
    config.training.max_epochs = 1
    config.tracking.enabled = False
    config.tuning.search_space = {
        "training": {
            "learning_rate": [1e-4, 3e-4],
        },
        "model": {
            "hidden_size": [64, 128],
        },
    }

    _seed_train_history(config)
    tuning_root = _tuning_root(config)
    stale_trial = tuning_root / "trials" / "trial-999" / "stale.txt"
    stale_trial.parent.mkdir(parents=True, exist_ok=True)
    stale_trial.write_text("stale", encoding="utf-8")

    run_tune(config)

    study_path = tuning_root / "study.json"
    trials_path = tuning_root / "trials.json"
    best_params_path = tuning_root / "best_params.json"
    assert study_path.is_file()
    assert trials_path.is_file()
    assert best_params_path.is_file()
    assert not stale_trial.exists()
    assert (tuning_root / "trials" / "trial-000" / "train_report.json").is_file()

    study_report = TuningStudyReport.model_validate_json(study_path.read_text(encoding="utf-8"))
    trial_records = [
        TuningTrialRecord.model_validate(payload)
        for payload in json.loads(trials_path.read_text(encoding="utf-8"))
    ]
    best_params_report = TuningBestParamsReport.model_validate_json(
        best_params_path.read_text(encoding="utf-8")
    )
    assert study_report.kind == "tuning_study"
    assert study_report.dataset_id == "icdcs_2025_11_09"
    assert study_report.trial_counts.total == 2
    assert len(trial_records) == 2
    assert trial_records[0].params.model is not None
    assert best_params_report.kind == "tuning_best_params"
    assert (
        best_params_report.params.model is not None
        or best_params_report.params.training is not None
    )


def test_acquire_success_output_is_small_summary(tmp_path, monkeypatch) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "dataset.temporal.lookback_seconds=24",
            "dataset.temporal.max_delay_seconds=12",
            "dataset.sampling.anchor_count=4",
        ],
    )
    required_history_blocks = required_history_block_count(config)
    block_time_seconds = int(config.chain.block_time_seconds)
    expected_history_start = (
        config.dataset.window.start_timestamp - required_history_blocks * block_time_seconds
    )

    class FakeSummaryBlockClient:
        def __init__(self, provider, chain) -> None:
            del provider
            self.chain = chain

        async def close(self) -> None:
            return None

        async def plan_history_window(
            self,
            *,
            end_timestamp: int,
            required_history_blocks: int,
            chunk_size: int,
        ) -> BlockPullPlan:
            del chunk_size
            return BlockPullPlan(
                window=TimestampRange(start=expected_history_start, end=end_timestamp),
                block_range=BlockRange(start=100, end=100 + required_history_blocks),
                expected_rows=required_history_blocks,
                expected_files=1,
            )

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            del chunk_size
            return BlockPullPlan(
                window=window,
                block_range=BlockRange(start=10_001, end=10_033),
                expected_rows=32,
                expected_files=1,
            )

        async def pull_block_range(
            self,
            output_dir: Path,
            *,
            plan: BlockPullPlan,
            chunk_size: int,
            rpc_controller,
            reporter,
        ) -> BlockPullPlan:
            del chunk_size, rpc_controller, reporter
            rows = make_block_rows(
                plan.expected_rows,
                start_block=plan.block_range.start,
                start_timestamp=plan.window.start,
                block_time_seconds=block_time_seconds,
                include_gas_limit=True,
            )
            write_dataset_dir(output_dir, rows)
            return plan

    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", FakeSummaryBlockClient)

    stream = StringIO()
    reporter = _reporter(stream, interactive=True)

    run_acquire(config, reporter=reporter)
    reporter.close()

    output = stream.getvalue()
    assert "acquisition summary" in output
    assert "icdcs_2025_11_09" in output
    assert "Ethereum" in output
    assert f"{required_history_blocks:,} blocks in 1 file" in output
    assert "32 blocks in 1 file" in output
    assert "metadata" not in output
    assert "required history" not in output
    assert "validation" not in output
    assert "rpc" not in output
    assert "write dataset metadata finished" not in output
    assert "validate dataset history finished" not in output
    assert "validate dataset evaluation finished" not in output


def test_simulate_workflow_plain_output_is_sparse(tmp_path) -> None:
    train_config = compose_experiment("train", overrides=base_overrides(tmp_path))
    simulate_config = compose_experiment("simulate", overrides=base_overrides(tmp_path))
    _seed_simulation_inputs(train_config, simulate_config)
    run_train(train_config, reporter=NullReporter())
    stream = StringIO()
    reporter = _reporter(stream, interactive=False)

    run_simulate(simulate_config, reporter=reporter)
    reporter.close()

    output = stream.getvalue()
    predict_lines = [line for line in output.splitlines() if line.startswith("predict offsets:")]
    assert _simulation_report_path(simulate_config).is_file()
    assert "simulate repetitions finished: total_events=" in output
    assert predict_lines
    assert len(predict_lines) < 15
