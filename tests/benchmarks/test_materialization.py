from __future__ import annotations

from pathlib import Path

from spice.benchmarks.materialization import materialize_benchmark_plan
from spice.benchmarks.planning import plan_benchmark_workflow_selections
from spice.benchmarks.schema import BenchmarkSpec
from spice.config import EvaluateConfig, TrainConfig, WorkflowTask
from spice.storage.catalog.store import upsert_study_record
from spice.storage.engine import state_db_path
from spice.storage.layout import catalog_db_path
from spice.storage.workflow_roots import produced_artifact_id, produced_study_id

ETH_DATASET_ID = "cor_9a73b1e88edb488afb1e"


def _materialize(payload: dict[str, object]):
    spec = BenchmarkSpec.model_validate(payload)
    return materialize_benchmark_plan(plan_benchmark_workflow_selections(spec))


def test_materialization_injects_study_id_for_tuned_train_dependency(
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    entries = _materialize(
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
        }
    )

    tune = next(entry for entry in entries if entry.workflow is WorkflowTask.TUNE)
    train = next(entry for entry in entries if entry.workflow is WorkflowTask.TRAIN)

    assert isinstance(train.config, TrainConfig)
    assert train.config.study_id == produced_study_id(tune.config)
    assert train.config.dataset_id is None
    assert train.selection["study_id"] == train.config.study_id


def test_materialization_injects_artifact_and_dataset_for_artifact_from(
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    entries = _materialize(
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
        }
    )

    train = next(entry for entry in entries if entry.workflow is WorkflowTask.TRAIN)
    evaluate = next(entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE)

    assert isinstance(train.config, TrainConfig)
    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.artifact_from == train.run_id
    assert evaluate.config.dataset_id == ETH_DATASET_ID
    assert evaluate.config.artifact_id == produced_artifact_id(
        train.config,
        dataset_id=ETH_DATASET_ID,
    )
    assert evaluate.selection["dataset_id"] == ETH_DATASET_ID
    assert evaluate.selection["artifact_id"] == evaluate.config.artifact_id


def test_materialization_preserves_explicit_evaluate_dataset_id_with_artifact_from(
    isolate_conf_root,
) -> None:
    isolate_conf_root()
    evaluate_dataset_id = "cor_cross_corpus_same_chain"

    entries = _materialize(
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
        }
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
    assert evaluate.selection["dataset_id"] == evaluate_dataset_id
    assert evaluate.selection["artifact_id"] == evaluate.config.artifact_id


def test_materialization_uses_catalog_dataset_for_explicit_tuned_study(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()
    storage_root = tmp_path / "outputs"
    study_root = storage_root / "studies" / "ethereum" / "std_existing"
    upsert_study_record(
        catalog_db_path(storage_root),
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
    )

    entries = _materialize(
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
        }
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
    isolate_conf_root()

    entries = _materialize(
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
        }
    )

    evaluate = next(entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE)

    assert evaluate.selection["surface"] == "current_row_fee_dynamics"
    assert evaluate.selection["objective"] == "validation_total_loss"
    assert evaluate.selection["dataset_id"] == ETH_DATASET_ID
    assert evaluate.selection["artifact_id"] == evaluate.config.artifact_id
