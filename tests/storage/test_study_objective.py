from __future__ import annotations

from typing import cast

from spice.config import TuneConfig, WorkflowTask
from spice.corpus.metadata import (
    ChainMetadata,
    DatasetCoverageMetadata,
    DatasetIdentity,
    DatasetManifest,
    DatasetRequestMetadata,
    DatasetValidationMetadata,
    DatasetWindowMetadata,
)
from spice.storage.root_producer_handles import produced_study_id
from spice.storage.study_optuna import open_tuning_study
from tests.root_handle_helpers import corpus_handle, study_handle

TEST_DATASET_ID = "cor_9a73b1e88edb488afb1e"


def _corpus_manifest(config: TuneConfig) -> DatasetManifest:
    window = DatasetWindowMetadata(start_timestamp=1, end_timestamp=2)
    return DatasetManifest(
        dataset=DatasetIdentity(id=TEST_DATASET_ID, name=config.dataset.name),
        chain=ChainMetadata(name=config.chain.name, runtime=config.chain.runtime),
        request=DatasetRequestMetadata(history=window, evaluation=window),
        coverage=DatasetCoverageMetadata(history=window, evaluation=window),
        validation=DatasetValidationMetadata(history=None, evaluation=None),
    )


def test_tuning_objective_controls_study_direction(
    tmp_path,
    load_workflow_config,
    model_workflow_override,
    tune_override,
) -> None:
    override = model_workflow_override() | tune_override()
    override["tuning"] = {
        "trial_count": 2,
        "timeout_seconds": None,
        "sampler_seed": 2026,
        "enable_pruning": False,
    }
    override["objective"] = {
        "id": "validation",
        "metric_id": "offset_accuracy",
        "direction": "maximize",
    }
    config = cast(
        TuneConfig,
        load_workflow_config(
            WorkflowTask.TUNE,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override=override,
        ),
    )

    corpus = corpus_handle(
        config.storage.root,
        chain_name=config.chain.name,
        dataset_id=TEST_DATASET_ID,
        dataset_name=config.dataset.name,
    )
    study = study_handle(
        config.storage.root,
        corpus=corpus,
        study_id=produced_study_id(config),
        study_name=config.study.name,
    )
    access = open_tuning_study(
        study,
        config=config,
        corpus=corpus,
        corpus_manifest=_corpus_manifest(config),
    )

    assert access.study.direction.name == "MAXIMIZE"
    assert access.manifest.objective.metric_id == "offset_accuracy"
