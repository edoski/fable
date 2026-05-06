from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spice.benchmarks.planning import plan_benchmark
from spice.config import EvaluateConfig, TrainConfig, TuneConfig, WorkflowTask
from spice.core.errors import ConfigResolutionError
from spice.storage.catalog.records import CatalogStudyRecord
from spice.storage.catalog.index import upsert_catalog_record
from spice.storage.engine import state_db_path
from spice.storage.root_identity import produced_artifact_id, produced_study_id

ETH_DATASET_ID = "cor_9a73b1e88edb488afb1e"


def _materialize(isolate_conf_root, payload: dict[str, object]):
    conf_root = isolate_conf_root()
    benchmark_path = conf_root / "benchmark" / "materialization_case.yaml"
    benchmark_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return plan_benchmark("materialization_case")


def test_materialization_injects_study_id_for_tuned_train_dependency(
    isolate_conf_root,
) -> None:
    entries = _materialize(
        isolate_conf_root,
        {
            "cases": [
                {
                    "id": "case",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "dataset_id": ETH_DATASET_ID,
                        "study": "case_study",
                    },
                    "steps": [
                        {"id": "tune", "workflow": "tune"},
                        {
                            "id": "train",
                            "workflow": "train",
                            "after": ["tune"],
                            "set": {"variant": "tuned"},
                        },
                    ],
                }
            ]
        },
    )

    tune = next(entry for entry in entries if entry.workflow is WorkflowTask.TUNE)
    train = next(entry for entry in entries if entry.workflow is WorkflowTask.TRAIN)

    assert isinstance(tune.config, TuneConfig)
    assert isinstance(train.config, TrainConfig)
    assert train.config.study_id == produced_study_id(tune.config)
    assert train.config.dataset_id is None
    assert train.root_ledger.root_id(role="consumed", root_kind="study") == (train.config.study_id)
    assert tune.root_ledger.produced_study_id() == produced_study_id(tune.config)
    assert train.selection.study == "case_study"


def test_materialization_rejects_tuned_train_without_study_source(
    isolate_conf_root,
) -> None:
    with pytest.raises(ConfigResolutionError, match="tune dependency or explicit study_id"):
        _materialize(
            isolate_conf_root,
            {
                "cases": [
                    {
                        "id": "case",
                        "base": {
                            "surface": "current_row_fee_dynamics",
                            "dataset_id": ETH_DATASET_ID,
                        },
                        "steps": [
                            {
                                "id": "train",
                                "workflow": "train",
                                "set": {"variant": "tuned"},
                            },
                        ],
                    }
                ]
            },
        )


def test_materialization_injects_artifact_and_dataset_for_artifact_from(
    isolate_conf_root,
) -> None:
    entries = _materialize(
        isolate_conf_root,
        {
            "cases": [
                {
                    "id": "case",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "dataset_id": ETH_DATASET_ID,
                    },
                    "steps": [
                        {
                            "id": "train",
                            "workflow": "train",
                            "set": {"variant": "baseline"},
                        },
                        {
                            "id": "evaluate",
                            "workflow": "evaluate",
                            "artifact_from": "train",
                            "set": {"evaluation": "poisson_replay_2h"},
                        },
                    ],
                }
            ]
        },
    )

    train = next(entry for entry in entries if entry.workflow is WorkflowTask.TRAIN)
    evaluate = next(entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE)

    assert isinstance(train.config, TrainConfig)
    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.dependencies.artifact_from_run_id == train.run_id
    assert train.root_ledger.produced_artifact_id() == evaluate.config.artifact_id
    assert evaluate.config.dataset_id == ETH_DATASET_ID
    assert evaluate.config.artifact_id == produced_artifact_id(
        train.config,
        dataset_id=ETH_DATASET_ID,
    )
    assert evaluate.root_ledger.consumed_dataset_id() == ETH_DATASET_ID
    assert evaluate.root_ledger.consumed_artifact_id() == evaluate.config.artifact_id
    assert evaluate.root_ledger.artifact_source_dataset_id() == ETH_DATASET_ID


def test_materialization_preserves_explicit_evaluate_dataset_id_with_artifact_from(
    isolate_conf_root,
) -> None:
    evaluate_dataset_id = "cor_cross_corpus_same_chain"

    entries = _materialize(
        isolate_conf_root,
        {
            "cases": [
                {
                    "id": "case",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "dataset_id": ETH_DATASET_ID,
                    },
                    "steps": [
                        {
                            "id": "train",
                            "workflow": "train",
                            "set": {"variant": "baseline"},
                        },
                        {
                            "id": "evaluate",
                            "workflow": "evaluate",
                            "artifact_from": "train",
                            "set": {
                                "dataset_id": evaluate_dataset_id,
                                "evaluation": "poisson_replay_2h",
                            },
                        },
                    ],
                }
            ]
        },
    )

    train = next(entry for entry in entries if entry.workflow is WorkflowTask.TRAIN)
    evaluate = next(entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE)

    assert isinstance(train.config, TrainConfig)
    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.config.dataset_id == evaluate_dataset_id
    assert evaluate.config.artifact_id == produced_artifact_id(
        train.config,
        dataset_id=ETH_DATASET_ID,
    )
    assert evaluate.root_ledger.consumed_dataset_id() == evaluate_dataset_id
    assert evaluate.root_ledger.consumed_artifact_id() == evaluate.config.artifact_id
    assert evaluate.root_ledger.artifact_source_dataset_id() == ETH_DATASET_ID


def test_materialization_uses_catalog_dataset_for_explicit_tuned_study(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    storage_root = tmp_path / "outputs"
    study_root = storage_root / "studies" / "ethereum" / "std_existing"
    upsert_catalog_record(
        storage_root,
        CatalogStudyRecord(
            study_id="std_existing",
            study_name="existing",
            dataset_id=ETH_DATASET_ID,
            dataset_name="icdcs_2026",
            chain_name="ethereum",
            features_id="core_fee_dynamics",
            prediction_id="icdcs_2026",
            model_id="lstm",
            problem_id="current_row_nominal",
            root_path=study_root,
            state_db_path=state_db_path(study_root),
        ),
    )

    entries = _materialize(
        isolate_conf_root,
        {
            "cases": [
                {
                    "id": "case",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "storage_root": str(storage_root),
                    },
                    "steps": [
                        {
                            "id": "train",
                            "workflow": "train",
                            "set": {
                                "variant": "tuned",
                                "study_id": "std_existing",
                            },
                        },
                        {
                            "id": "evaluate",
                            "workflow": "evaluate",
                            "artifact_from": "train",
                            "set": {"evaluation": "poisson_replay_2h"},
                        },
                    ],
                }
            ]
        },
    )

    train = next(entry for entry in entries if entry.workflow is WorkflowTask.TRAIN)
    evaluate = next(entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE)

    assert isinstance(train.config, TrainConfig)
    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.config.dataset_id == ETH_DATASET_ID
    assert evaluate.config.artifact_id == produced_artifact_id(
        train.config,
        dataset_id=ETH_DATASET_ID,
    )


def test_materialization_preserves_selection_ledger_context(
    isolate_conf_root,
) -> None:
    entries = _materialize(
        isolate_conf_root,
        {
            "cases": [
                {
                    "id": "case",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "dataset_id": ETH_DATASET_ID,
                        "objective": "validation_total_loss",
                    },
                    "steps": [
                        {
                            "id": "train",
                            "workflow": "train",
                            "set": {"variant": "baseline"},
                        },
                        {
                            "id": "evaluate",
                            "workflow": "evaluate",
                            "artifact_from": "train",
                            "set": {"evaluation": "poisson_replay_2h"},
                        },
                    ],
                }
            ]
        },
    )

    evaluate = next(entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE)

    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.selection.surface == "current_row_fee_dynamics"
    assert evaluate.selection.objective == "validation_total_loss"
    assert evaluate.root_ledger.consumed_dataset_id() == ETH_DATASET_ID
    assert evaluate.root_ledger.consumed_artifact_id() == evaluate.config.artifact_id
