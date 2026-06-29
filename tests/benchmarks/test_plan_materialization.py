from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spice.benchmarks.plan_materialization import materialize_benchmark_plan
from spice.config import EvaluateConfig, TrainConfig, TuneConfig, WorkflowTask
from spice.core.errors import ConfigResolutionError
from spice.storage.catalog.index import upsert_catalog_record
from spice.storage.workflow_root_materialization import produced_artifact_id, produced_study_id
from tests.catalog_helpers import artifact_record, study_record

ETH_DATASET_ID = "cor_9a73b1e88edb488afb1e"
EVALUATION_WINDOW = {"start": "2026-02-03T14:00:00Z", "duration_seconds": 7200}


def _with_evaluation_windows(payload: object) -> object:
    if isinstance(payload, dict):
        if payload.get("workflow") == "evaluate":
            step_set = payload.setdefault("set", {})
            if isinstance(step_set, dict):
                step_set.setdefault("evaluation_window", EVALUATION_WINDOW)
        for value in payload.values():
            _with_evaluation_windows(value)
    elif isinstance(payload, list):
        for value in payload:
            _with_evaluation_windows(value)
    return payload


def _materialize(isolate_conf_root, payload: dict[str, object]):
    conf_root = isolate_conf_root()
    _with_evaluation_windows(payload)
    benchmark_path = conf_root / "benchmark" / "materialization_case.yaml"
    benchmark_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return materialize_benchmark_plan("materialization_case")


def test_plan_materialization_derives_study_id_for_tuned_train_dependency(
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
                        "corpus_id": ETH_DATASET_ID,
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
    assert train.config.corpus_id is None
    assert train.root_facts.consumed_study_id == train.config.study_id
    consumed_study = next(
        entry
        for entry in train.root_ledger.entries
        if entry.role == "consumed" and entry.root_kind == "study"
    )
    assert consumed_study.corpus_id == ETH_DATASET_ID
    assert tune.root_facts.produced_study_id == produced_study_id(tune.config)
    assert train.selection.study == "case_study"


def test_plan_materialization_rejects_tuned_train_without_study_source(
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
                            "corpus_id": ETH_DATASET_ID,
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


def test_plan_materialization_rejects_ambiguous_tuned_train_dependency(
    isolate_conf_root,
) -> None:
    with pytest.raises(ConfigResolutionError, match="multiple tune dependencies"):
        _materialize(
            isolate_conf_root,
            {
                "cases": [
                    {
                        "id": "case",
                        "base": {
                            "surface": "current_row_fee_dynamics",
                            "corpus_id": ETH_DATASET_ID,
                        },
                        "steps": [
                            {
                                "id": "tune_a",
                                "workflow": "tune",
                                "set": {"study": "study_a"},
                            },
                            {
                                "id": "tune_b",
                                "workflow": "tune",
                                "set": {"study": "study_b"},
                            },
                            {
                                "id": "train",
                                "workflow": "train",
                                "after": ["tune_a", "tune_b"],
                                "set": {"variant": "tuned"},
                            },
                        ],
                    }
                ]
            },
        )


def test_plan_materialization_derives_artifact_and_dataset_for_artifact_from(
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
                        "corpus_id": ETH_DATASET_ID,
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
                            "set": {"evaluator": "poisson_replay"},
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
    assert train.root_facts.produced_artifact_id == evaluate.config.artifact_id
    assert evaluate.config.corpus_id == ETH_DATASET_ID
    assert evaluate.config.artifact_id == produced_artifact_id(
        train.config,
        corpus_id=ETH_DATASET_ID,
    )
    assert evaluate.root_facts.consumed_corpus_id == ETH_DATASET_ID
    assert evaluate.root_facts.consumed_artifact_id == evaluate.config.artifact_id
    assert evaluate.root_facts.artifact_source_corpus_id == ETH_DATASET_ID


def test_plan_materialization_preserves_explicit_evaluate_corpus_id_with_artifact_from(
    isolate_conf_root,
) -> None:
    evaluate_corpus_id = "cor_cross_corpus_same_chain"

    entries = _materialize(
        isolate_conf_root,
        {
            "cases": [
                {
                    "id": "case",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "corpus_id": ETH_DATASET_ID,
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
                                "corpus_id": evaluate_corpus_id,
                                "evaluator": "poisson_replay",
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
    assert evaluate.config.corpus_id == evaluate_corpus_id
    assert evaluate.config.artifact_id == produced_artifact_id(
        train.config,
        corpus_id=ETH_DATASET_ID,
    )
    consumed_artifact = next(
        entry
        for entry in evaluate.root_ledger.entries
        if entry.role == "consumed" and entry.root_kind == "artifact"
    )
    assert consumed_artifact.corpus_id == ETH_DATASET_ID
    assert evaluate.root_facts.consumed_corpus_id == evaluate_corpus_id
    assert evaluate.root_facts.consumed_artifact_id == evaluate.config.artifact_id
    assert evaluate.root_facts.artifact_source_corpus_id == ETH_DATASET_ID


def test_plan_materialization_uses_catalog_dataset_for_explicit_tuned_study(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    storage_root = tmp_path / "outputs"
    study_root = storage_root / "studies" / "ethereum" / "std_existing"
    upsert_catalog_record(
        storage_root,
        study_record(
            study_root,
            study_id="std_existing",
            study_name="existing",
            corpus_id=ETH_DATASET_ID,
            corpus_name="icdcs_2026",
            chain_name="ethereum",
            features_id="core_fee_dynamics",
            prediction_id="icdcs_2026",
            model_id="lstm",
            problem_id="current_row_nominal",
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
                            "set": {"evaluator": "poisson_replay"},
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
    consumed_study = next(
        entry
        for entry in train.root_ledger.entries
        if entry.role == "consumed" and entry.root_kind == "study"
    )
    assert consumed_study.corpus_id == ETH_DATASET_ID
    assert evaluate.config.corpus_id == ETH_DATASET_ID
    assert evaluate.config.artifact_id == produced_artifact_id(
        train.config,
        corpus_id=ETH_DATASET_ID,
    )


def test_plan_materialization_records_explicit_artifact_source_corpus_from_catalog(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    storage_root = tmp_path / "outputs"
    artifact_root = storage_root / "artifacts" / "ethereum" / "art_existing"
    upsert_catalog_record(
        storage_root,
        artifact_record(
            artifact_root,
            artifact_id="art_existing",
            corpus_id=ETH_DATASET_ID,
            corpus_name="icdcs_2026",
            chain_name="ethereum",
            features_id="core_fee_dynamics",
            prediction_id="icdcs_2026",
            model_id="lstm",
            problem_id="current_row_nominal",
            variant="baseline",
            study_id=None,
            study_name=None,
        ),
    )
    evaluate_corpus_id = "cor_cross_corpus_same_chain"

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
                            "id": "evaluate",
                            "workflow": "evaluate",
                            "set": {
                                "corpus_id": evaluate_corpus_id,
                                "artifact_id": "art_existing",
                                "evaluator": "poisson_replay",
                            },
                        },
                    ],
                }
            ]
        },
    )

    evaluate = next(entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE)
    consumed_artifact = next(
        entry
        for entry in evaluate.root_ledger.entries
        if entry.role == "consumed" and entry.root_kind == "artifact"
    )

    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.root_facts.consumed_corpus_id == evaluate_corpus_id
    assert evaluate.root_facts.consumed_artifact_corpus_id == ETH_DATASET_ID
    assert consumed_artifact.corpus_id == ETH_DATASET_ID


def test_plan_materialization_preserves_selection_ledger_context(
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
                        "corpus_id": ETH_DATASET_ID,
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
                            "set": {"evaluator": "poisson_replay"},
                        },
                    ],
                }
            ]
        },
    )

    evaluate = next(entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE)

    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.selection.surface == "current_row_fee_dynamics"
    assert evaluate.root_facts.consumed_corpus_id == ETH_DATASET_ID
    assert evaluate.root_facts.consumed_artifact_id == evaluate.config.artifact_id


def test_evaluations_suite_fans_out_evaluate_steps_and_sets_train_cutoff(
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    (conf_root / "evaluations" / "suite.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "suite",
                "items": [
                    {
                        "id": "early",
                        "start": "2026-01-20T00:00:00Z",
                        "duration_seconds": 3600,
                    },
                    {
                        "id": "late",
                        "start": "2026-02-03T14:00:00Z",
                        "duration_seconds": 7200,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    benchmark_path = conf_root / "benchmark" / "suite_case.yaml"
    benchmark_path.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "id": "case",
                        "base": {
                            "surface": "current_row_fee_dynamics",
                            "corpus_id": ETH_DATASET_ID,
                            "evaluations": "suite",
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
                                "set": {"evaluator": "poisson_replay"},
                            },
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    entries = materialize_benchmark_plan("suite_case")
    train = next(entry for entry in entries if entry.workflow is WorkflowTask.TRAIN)
    evaluations = [entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE]

    assert isinstance(train.config, TrainConfig)
    assert train.config.training_cutoff_timestamp == 1768867200
    assert [entry.dimension_labels["evaluations"] for entry in evaluations] == [
        "early",
        "late",
    ]
    assert [entry.config.evaluation_window.start_timestamp for entry in evaluations] == [
        1768867200,
        1770127200,
    ]


def test_block_evaluations_suite_fans_out_without_tags(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    storage_root = tmp_path / "outputs"
    artifact_root = storage_root / "artifacts" / "ethereum" / "art_test"
    upsert_catalog_record(
        storage_root,
        artifact_record(
            artifact_root,
            artifact_id="art_test",
            corpus_id=ETH_DATASET_ID,
            corpus_name="icdcs_2026",
            chain_name="ethereum",
            features_id="core_fee_dynamics",
            prediction_id="icdcs_2026",
            model_id="lstm",
            problem_id="current_row_nominal",
            variant="baseline",
        ),
    )
    (conf_root / "evaluations" / "block_suite.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "block_suite",
                "items": [
                    {
                        "id": "block_a",
                        "start_block": 12_000,
                        "block_count": 1_200,
                    },
                    {
                        "id": "block_b",
                        "start_block": 13_200,
                        "block_count": 1_200,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    benchmark_path = conf_root / "benchmark" / "block_suite_case.yaml"
    benchmark_path.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "id": "case",
                        "base": {
                            "artifact_id": "art_test",
                            "corpus_id": ETH_DATASET_ID,
                            "evaluations": "block_suite",
                            "evaluator": "block_poisson_replay",
                            "storage_root": str(storage_root),
                        },
                        "steps": [
                            {
                                "id": "evaluate",
                                "workflow": "evaluate",
                            },
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    entries = materialize_benchmark_plan("block_suite_case")

    assert [entry.dimension_labels["evaluations"] for entry in entries] == [
        "block_a",
        "block_b",
    ]
    assert [entry.config.evaluation_window.start_block for entry in entries] == [
        12_000,
        13_200,
    ]
    assert [entry.config.evaluation_window.block_count for entry in entries] == [
        1_200,
        1_200,
    ]


def test_problem_grid_keeps_selection_problem_id_with_inline_workflow_problem(
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
                        "corpus_id": ETH_DATASET_ID,
                    },
                    "dimensions": {
                        "problems": [
                            {
                                "grid": {
                                    "base": "current_row_nominal",
                                    "fields": {
                                        "lookback_seconds": [600],
                                    },
                                }
                            }
                        ]
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
                            "set": {"evaluator": "poisson_replay"},
                        },
                    ],
                }
            ]
        },
    )
    problem_id = "current_row_nominal__lookback_seconds-600"
    train = next(entry for entry in entries if entry.workflow is WorkflowTask.TRAIN)
    evaluate = next(entry for entry in entries if entry.workflow is WorkflowTask.EVALUATE)

    assert isinstance(train.config, TrainConfig)
    assert isinstance(evaluate.config, EvaluateConfig)
    assert train.config.problem.id == problem_id
    assert train.config.problem.lookback_seconds == 600
    assert train.selection.problem == problem_id
    assert evaluate.selection.problem == problem_id
