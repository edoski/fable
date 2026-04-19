from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
from typer.testing import CliRunner

from spice.cli import app
from spice.config import (
    EvaluateConfig,
    TrainConfig,
    TuneConfig,
    WorkflowTask,
)
from spice.core.reporting import NullReporter
from spice.features import compile_feature_contract
from spice.storage.catalog import CatalogStudyRecord
from spice.storage.layout import resolve_workflow_paths
from spice.temporal.contracts import compile_problem_contract
from spice.workflows.evaluate import run as run_evaluate
from spice.workflows.train import run as run_train
from spice.workflows.tune import run as run_tune

runner = CliRunner()


def _load_test_train_config(
    tmp_path: Path,
    load_workflow_config,
    *,
    override: dict[str, object] | None = None,
) -> TrainConfig:
    return cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            preset="icdcs_2026",
            override=override,
        ),
    )


def _load_test_tune_config(
    tmp_path: Path,
    load_workflow_config,
    *,
    override: dict[str, object] | None = None,
) -> TuneConfig:
    return cast(
        TuneConfig,
        load_workflow_config(
            WorkflowTask.TUNE,
            workspace=tmp_path,
            preset="icdcs_2026",
            override=override,
        ),
    )


def _load_test_evaluate_config(
    tmp_path: Path,
    load_workflow_config,
    *,
    override: dict[str, object] | None = None,
) -> EvaluateConfig:
    return cast(
        EvaluateConfig,
        load_workflow_config(
            WorkflowTask.EVALUATE,
            workspace=tmp_path,
            preset="icdcs_2026",
            override=override,
        ),
    )


def _seed_dataset(path: Path, rows: list[dict[str, int]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    parquet_path = path / "blocks.parquet"
    pl.DataFrame(rows).write_parquet(parquet_path)
    return parquet_path


def _seed_history_dataset(config) -> Path:
    paths = resolve_workflow_paths(config)
    feature_contract = compile_feature_contract(feature_set=config.feature_set)
    contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
        chain_runtime=config.chain.runtime,
    )
    block_interval_seconds = 12
    row_count = max(
        128,
        ((contract.required_history_seconds + contract.max_delay_seconds + 12) // 12)
        + contract.warmup_rows
        + contract.sample_count
        + 16,
    )
    rows = [
        {
            "block_number": index,
            "timestamp": 1_000 + index * block_interval_seconds,
            "base_fee_per_gas": 1_000_000_000,
            "gas_used": 18_000_000,
            "gas_limit": 30_000_000,
            "chain_id": config.chain.runtime.chain_id,
        }
        for index in range(1, row_count + 1)
    ]
    return _seed_dataset(paths.history_dir, rows)


def _seed_evaluation_dataset(config) -> Path:
    paths = resolve_workflow_paths(config)
    rows = [
        {
            "block_number": index,
            "timestamp": config.evaluation_window_start_timestamp + (index - 10_001) * 12,
            "base_fee_per_gas": 1_000_000_000,
            "gas_used": 18_000_000,
            "gas_limit": 30_000_000,
            "chain_id": config.chain.runtime.chain_id,
        }
        for index in range(10_001, 10_065)
    ]
    return _seed_dataset(paths.evaluation_dir, rows)


def test_show_command_smoke(tmp_path, load_workflow_config, model_workflow_override) -> None:
    override = model_workflow_override()
    train_config = _load_test_train_config(tmp_path, load_workflow_config, override=override)
    evaluate_config = _load_test_evaluate_config(tmp_path, load_workflow_config, override=override)
    _seed_history_dataset(train_config)
    _seed_evaluation_dataset(evaluate_config)
    run_train(train_config, reporter=NullReporter())
    run_evaluate(evaluate_config, reporter=NullReporter())

    result = runner.invoke(
        app,
        [
            "show",
            "artifact",
            "--chain",
            train_config.chain.name,
            "--dataset",
            train_config.dataset.name,
            "--feature-set",
            train_config.feature_set.id,
            "--model",
            train_config.model.id,
            "--problem",
            train_config.problem.id,
            "--variant",
            train_config.artifact.variant.value,
            "--storage-root",
            str(tmp_path / "outputs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "artifact summary" in result.stdout
    assert train_config.model.id in result.stdout
    assert "evaluation" in result.stdout


def test_show_study_config_detail_smoke(
    tmp_path,
    load_workflow_config,
    model_workflow_override,
    tune_override,
) -> None:
    config = _load_test_tune_config(
        tmp_path,
        load_workflow_config,
        override=model_workflow_override() | tune_override(),
    )
    _seed_history_dataset(config)
    run_tune(config, reporter=NullReporter())

    result = runner.invoke(
        app,
        [
            "show",
            "study",
            "--chain",
            config.chain.name,
            "--dataset",
            config.dataset.name,
            "--feature-set",
            config.feature_set.id,
            "--model",
            config.model.id,
            "--problem",
            config.problem.id,
            "--study",
            config.study.name,
            "--storage-root",
            str(tmp_path / "outputs"),
            "--detail",
            "config",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "study summary" in result.stdout
    assert "tuning space" in result.stdout
    assert "learning rate" in result.stdout


def test_show_detail_rejects_invalid_value_at_parse_time() -> None:
    result = runner.invoke(app, ["show", "artifact", "--detail", "bogus"])

    assert result.exit_code != 0
    assert "Invalid value for '--detail'" in result.output


def test_show_missing_match_renders_compact_operator_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "spice.cli.commands.storage.list_dataset_records",
        lambda *_args, **_kwargs: [],
    )

    result = runner.invoke(app, ["show", "dataset", "--dataset", "missing"])

    assert result.exit_code == 1
    assert "No dataset matches found" in result.output
    assert "Traceback" not in result.output


def test_show_detail_guides_narrowing_flags(monkeypatch, tmp_path: Path) -> None:
    records = [
        CatalogStudyRecord(
            study_id="study_a",
            study_name="baseline",
            dataset_id="dataset",
            dataset_name="icdcs_2026",
            chain_name="ethereum",
            feature_set_id="icdcs_2026",
            prediction_id="candidate_offset_selection",
            model_id="lstm",
            problem_id="icdcs_2026",
            root_path=tmp_path / "study-a",
            state_db_path=tmp_path / "study-a.sqlite",
        ),
        CatalogStudyRecord(
            study_id="study_b",
            study_name="ablation",
            dataset_id="dataset",
            dataset_name="icdcs_2026",
            chain_name="ethereum",
            feature_set_id="icdcs_2026",
            prediction_id="candidate_offset_selection",
            model_id="lstm",
            problem_id="icdcs_2026",
            root_path=tmp_path / "study-b",
            state_db_path=tmp_path / "study-b.sqlite",
        ),
    ]
    monkeypatch.setattr(
        "spice.cli.commands.storage.list_study_records",
        lambda *_args, **_kwargs: records,
    )

    result = runner.invoke(
        app,
        [
            "show",
            "study",
            "--chain",
            "ethereum",
            "--dataset",
            "icdcs_2026",
            "--feature-set",
            "icdcs_2026",
            "--model",
            "lstm",
            "--problem",
            "icdcs_2026",
            "--detail",
            "trials",
        ],
    )

    assert result.exit_code == 1
    assert "--study" in result.output


def test_delete_artifact_command_smoke(
    tmp_path,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = _load_test_train_config(
        tmp_path,
        load_workflow_config,
        override=model_workflow_override(),
    )
    paths = resolve_workflow_paths(config)
    _seed_history_dataset(config)
    run_train(config, reporter=NullReporter())

    result = runner.invoke(
        app,
        [
            "delete",
            "artifact",
            "--chain",
            config.chain.name,
            "--dataset",
            config.dataset.name,
            "--feature-set",
            config.feature_set.id,
            "--model",
            config.model.id,
            "--problem",
            config.problem.id,
            "--variant",
            config.artifact.variant.value,
            "--storage-root",
            str(tmp_path / "outputs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert not paths.artifact_root.exists()
