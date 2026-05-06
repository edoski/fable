from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from optuna.trial import TrialState

from spice.config import TuneConfig, WorkflowTask
from spice.corpus.metadata import (
    BlockRangeMetadata,
    ChainMetadata,
    CompactValidationReport,
    CorpusAcquisitionSourceRequirements,
    CorpusSplitManifest,
    CorpusSplitManifests,
    DatasetIdentity,
    DatasetManifest,
    SplitCoverageMetadata,
    SplitMaterializationMetadata,
    SplitRequestMetadata,
    TimestampRangeMetadata,
)
from spice.modeling import tuning_execution
from spice.modeling.tuning_execution import (
    OpenTuningExecution,
    TuningExecutionCallbacks,
    open_tuning_execution,
    run_tuning_execution,
)
from spice.storage.workflow_root_materialization import produced_study_id
from tests.root_handle_helpers import corpus_handle, study_handle, tune_roots

TEST_DATASET_ID = "cor_9a73b1e88edb488afb1e"


def _corpus_manifest(config: TuneConfig) -> DatasetManifest:
    split = CorpusSplitManifest(
        kind="history",
        request=SplitRequestMetadata(
            start_timestamp=1,
            end_timestamp=2,
            start_block=1,
            end_block=2,
        ),
        coverage=SplitCoverageMetadata(
            first_timestamp=1,
            last_timestamp=2,
            first_block=1,
            last_block=1,
            rows=1,
        ),
        validation=CompactValidationReport(
            status="clean",
            rows=1,
            block_range=BlockRangeMetadata(first=1, last=1),
            timestamp_range=TimestampRangeMetadata(first=1, last=2),
        ),
        materialization=SplitMaterializationMetadata(outcome="reused", file_count=1),
    )
    return DatasetManifest(
        dataset=DatasetIdentity(id=TEST_DATASET_ID, name=config.dataset.name),
        chain=ChainMetadata(name=config.chain.name, runtime=config.chain.runtime),
        splits=CorpusSplitManifests(history=split, evaluation=split),
        source_requirements=CorpusAcquisitionSourceRequirements(
            required_columns=frozenset(
                {"block_number", "timestamp", "chain_id", "base_fee_per_gas"}
            ),
            optional_enrichments=frozenset(),
            temporal_unit="block",
            ordering_key="block_number",
            partition_key="chain_id",
        ),
    )


def _load_tune_config(
    tmp_path,
    load_workflow_config,
    model_workflow_override,
    tune_override,
) -> TuneConfig:
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
    return cast(
        TuneConfig,
        load_workflow_config(
            WorkflowTask.TUNE,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override=override,
        ),
    )


def test_tuning_execution_controls_study_direction(
    tmp_path,
    load_workflow_config,
    model_workflow_override,
    tune_override,
) -> None:
    config = _load_tune_config(
        tmp_path,
        load_workflow_config,
        model_workflow_override,
        tune_override,
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
    roots = tune_roots(config.storage.root, corpus=corpus, study=study)

    opened = open_tuning_execution(
        config,
        roots=roots,
        corpus_manifest=_corpus_manifest(config),
    )

    assert opened.study.direction.name == "MAXIMIZE"
    assert opened.manifest.objective.metric_id == "offset_accuracy"


def test_tuning_execution_reports_resume_trials_and_best_improvements(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
    tune_override,
) -> None:
    config = _load_tune_config(
        tmp_path,
        load_workflow_config,
        model_workflow_override,
        tune_override,
    )
    corpus = corpus_handle(
        tmp_path / "outputs",
        chain_name=config.chain.name,
        dataset_id=TEST_DATASET_ID,
        dataset_name=config.dataset.name,
    )
    roots = tune_roots(
        tmp_path / "outputs",
        corpus=corpus,
        study=study_handle(
            tmp_path / "outputs",
            corpus=corpus,
            study_id="std_test",
            study_name=config.study.name,
        ),
    )

    class FakeStudy:
        def __init__(self) -> None:
            self.trials: list[Any] = [
                SimpleNamespace(
                    number=0,
                    state=TrialState.COMPLETE,
                    value=0.2,
                    user_attrs={"spice_best_epoch": 2},
                )
            ]
            self.best_trial = self.trials[0]

        def optimize(self, objective, *, n_trials: int, timeout, callbacks) -> None:
            del objective, timeout
            for index in range(n_trials):
                trial = SimpleNamespace(
                    number=index + 1,
                    state=TrialState.COMPLETE,
                    value=0.3 + index,
                    user_attrs={"spice_best_epoch": index + 3},
                )
                self.trials.append(trial)
                self.best_trial = trial
                for callback in callbacks:
                    callback(self, trial)

    monkeypatch.setattr(tuning_execution, "build_study_summary", lambda *_args: object())
    opened = OpenTuningExecution(
        manifest=cast(Any, object()),
        study=cast(Any, FakeStudy()),
        existing_trial_count=1,
        target_trial_count=3,
        remaining_trial_count=2,
    )
    events: list[tuple[str, object]] = []

    summary = run_tuning_execution(
        opened,
        config=config,
        roots=roots,
        corpus_manifest=_corpus_manifest(config),
        callbacks=TuningExecutionCallbacks(
            on_resume=lambda existing, target: events.append(("resume", (existing, target))),
            on_study_start=lambda remaining: events.append(("start", remaining)),
            on_trial_complete=lambda progress: events.append(("trial", progress)),
            on_best_improved=lambda progress: events.append(("best", progress)),
        ),
    )

    assert summary is not None
    assert events[0] == ("resume", (1, 3))
    assert events[1] == ("start", 2)
    assert [event[0] for event in events[2:]] == ["trial", "best", "trial", "best"]
