from __future__ import annotations

from dataclasses import replace

import pytest

from spice.core.errors import StateLayoutError
from spice.evaluation import EvaluationRun, PoissonReplayEvaluatorConfig, coerce_evaluator_config
from spice.metrics import MetricDescriptor, MetricSet, WindowMetricSummary
from spice.modeling.results import (
    EvaluationConfigSnapshot,
    EvaluationExecutionProvenance,
    EvaluationRuntimeSummary,
    SplitSizes,
    TrainingEpochRecord,
    TrainingRuntimeSummary,
)
from spice.storage.artifact import (
    _evaluation_storage_id,
    list_evaluation_runs,
    list_evaluation_summaries,
    list_training_epochs,
    load_artifact_manifest,
    load_evaluation_summary,
    load_training_summary,
    record_evaluation_state,
    upsert_evaluation_state,
    write_artifact_manifest,
    write_training_state,
)
from spice.storage.artifact_codecs import (
    ARTIFACT_MANIFEST_CODEC,
    EVALUATION_RUN_CODEC,
    EVALUATION_SUMMARY_CODEC,
    TRAINING_SUMMARY_CODEC,
)
from spice.storage.engine import DATASET_ROOT_KIND, ensure_state_db
from spice.storage.schema import DATASET_TABLES
from tests.artifact_helpers import manifest as _manifest


def test_training_artifact_summary_round_trip(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    manifest = _manifest()
    summary = TrainingRuntimeSummary(
        n_rows_available=128,
        n_rows_used=96,
        split_sizes=SplitSizes(train_samples=16, validation_samples=4, test_samples=4),
        best_epoch=2,
        best_objective_metric_id="total_loss",
        best_objective_value=0.25,
        best_validation_metrics=MetricSet(values={"total_loss": 0.25}),
        test_metrics=MetricSet(values={"total_loss": 0.3}),
    )
    epoch_rows = [
        TrainingEpochRecord(
            epoch=1,
            train_metrics=MetricSet(values={"total_loss": 0.4}),
            validation_metrics=MetricSet(values={"total_loss": 0.35}),
            objective_metrics=MetricSet(values={"total_loss": 0.35}),
        ),
        TrainingEpochRecord(
            epoch=2,
            train_metrics=MetricSet(values={"total_loss": 0.3}),
            validation_metrics=MetricSet(values={"total_loss": 0.25}),
            objective_metrics=MetricSet(values={"total_loss": 0.25}),
        ),
    ]

    write_artifact_manifest(db_path, manifest=manifest)
    write_training_state(
        db_path,
        summary=summary,
        epoch_rows=epoch_rows,
    )

    loaded_manifest = load_artifact_manifest(db_path)
    loaded_summary = load_training_summary(db_path)
    loaded_epochs = list_training_epochs(db_path)

    assert loaded_manifest == manifest
    assert loaded_summary is not None
    assert loaded_summary.manifest == manifest
    assert loaded_summary.runtime == summary
    assert loaded_epochs == epoch_rows


def test_artifact_manifest_codec_round_trips() -> None:
    manifest = _manifest()

    assert ARTIFACT_MANIFEST_CODEC.decode(ARTIFACT_MANIFEST_CODEC.encode(manifest)) == manifest


def test_training_summary_codec_round_trips() -> None:
    summary = TrainingRuntimeSummary(
        n_rows_available=128,
        n_rows_used=96,
        split_sizes=SplitSizes(train_samples=16, validation_samples=4, test_samples=4),
        best_epoch=2,
        best_objective_metric_id="total_loss",
        best_objective_value=0.25,
        best_validation_metrics=MetricSet(values={"total_loss": 0.25}),
        test_metrics=MetricSet(values={"total_loss": 0.3}),
    )

    assert TRAINING_SUMMARY_CODEC.decode(TRAINING_SUMMARY_CODEC.encode(summary)) == summary


def test_evaluation_run_codec_round_trips() -> None:
    run = EvaluationRun(
        n_events=2,
        metrics={"profit_over_baseline": 0.2},
        metadata={"mode": "poisson_replay"},
    )

    assert EVALUATION_RUN_CODEC.decode(EVALUATION_RUN_CODEC.encode(run)) == run


def test_evaluation_summary_codec_round_trips() -> None:
    summary = _evaluation_summary(0.2)

    assert EVALUATION_SUMMARY_CODEC.decode(EVALUATION_SUMMARY_CODEC.encode(summary)) == summary


def test_evaluation_run_codec_rejects_bool_metadata() -> None:
    with pytest.raises(TypeError, match="metadata"):
        EVALUATION_RUN_CODEC.encode(
            EvaluationRun(
                n_events=1,
                metrics={"profit_over_baseline": 0.2},
                metadata={"mode": True},
            )
        )


def _evaluation_summary(value: float, *, seed: int = 2026) -> EvaluationRuntimeSummary:
    evaluation_config = coerce_evaluator_config(
        {
            "id": "poisson_replay",
            "window_seconds": 7200,
            "repetitions": 50,
            "arrival_rate_per_second": 0.05,
            "seed": seed,
        }
    )
    return EvaluationRuntimeSummary(
        delay_seconds=24,
        evaluator_id="poisson_replay",
        evaluation_config=EvaluationConfigSnapshot.from_config(evaluation_config),
        metric_descriptors=(
            MetricDescriptor(
                id="profit_over_baseline",
                label="profit over baseline",
                role="primary",
            ),
        ),
        n_history_rows=128,
        n_evaluation_rows=64,
        sample_count=24,
        metrics=MetricSet(values={"profit_over_baseline": value}),
        window_metrics={"profit_over_baseline": WindowMetricSummary(mean=value, std=0.01)},
        total_events=2,
        runs=[
            EvaluationRun(
                n_events=2,
                metrics={"profit_over_baseline": value},
                metadata={"mode": "poisson_replay"},
            ),
        ],
    )


def test_remote_evaluation_provenance_creates_distinct_summaries(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    write_artifact_manifest(db_path, manifest=_manifest())
    first = _evaluation_summary(0.1)
    second = _evaluation_summary(0.1)

    first_id, _ = upsert_evaluation_state(
        db_path,
        summary=replace(
            first,
            execution_provenance=EvaluationExecutionProvenance(
                execution_ref="slurm:1",
                job_id="1",
                log_path="/logs/spice-evaluate-1.out",
                workflow_task="evaluate",
                target="disi_l40",
            ),
        ),
    )
    second_id, _ = upsert_evaluation_state(
        db_path,
        summary=replace(
            second,
            execution_provenance=EvaluationExecutionProvenance(
                execution_ref="slurm:2",
                job_id="2",
                log_path="/logs/spice-evaluate-2.out",
                workflow_task="evaluate",
                target="disi_l40",
            ),
        ),
    )

    summaries = list_evaluation_summaries(db_path)

    assert first_id != second_id
    assert {
        summary.runtime.execution_provenance.execution_ref
        for summary in summaries
        if summary.runtime.execution_provenance is not None
    } == {
        "slurm:1",
        "slurm:2",
    }


def test_evaluation_storage_identity_uses_execution_ref_not_log_metadata() -> None:
    summary = _evaluation_summary(0.1)
    first = replace(
        summary,
        execution_provenance=EvaluationExecutionProvenance(
            execution_ref="slurm:1",
            job_id="1",
            log_path="/logs/one.out",
            workflow_task="evaluate",
            target="disi_l40",
        ),
    )
    same_execution = replace(
        summary,
        execution_provenance=EvaluationExecutionProvenance(
            execution_ref="slurm:1",
            job_id="999",
            log_path="/logs/other.out",
            workflow_task="evaluate",
            target="other",
        ),
    )
    other_execution = replace(
        summary,
        execution_provenance=EvaluationExecutionProvenance(
            execution_ref="slurm:2",
            job_id="2",
            log_path="/logs/two.out",
            workflow_task="evaluate",
            target="disi_l40",
        ),
    )

    assert _evaluation_storage_id(first) == _evaluation_storage_id(same_execution)
    assert _evaluation_storage_id(first) != _evaluation_storage_id(other_execution)


def test_evaluation_artifact_summaries_round_trip_and_coexist(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    manifest = _manifest()
    base_summary = _evaluation_summary(0.2, seed=2026)
    replay_summary = _evaluation_summary(0.1, seed=2027)

    write_artifact_manifest(db_path, manifest=manifest)
    evaluation_id, recorded_at = upsert_evaluation_state(
        db_path,
        summary=base_summary,
    )

    loaded_summary = load_evaluation_summary(db_path)
    loaded_runs = list_evaluation_runs(db_path)

    assert loaded_summary is not None
    assert loaded_summary.evaluation_storage_id == evaluation_id
    assert loaded_summary.recorded_at == recorded_at
    assert loaded_summary.manifest == manifest
    assert loaded_summary.runtime == base_summary
    assert loaded_runs == base_summary.runs

    replay_evaluation_id, _ = upsert_evaluation_state(
        db_path,
        summary=replay_summary,
    )

    summaries = list_evaluation_summaries(db_path)

    with pytest.raises(StateLayoutError, match="Multiple evaluation summaries stored"):
        load_evaluation_summary(db_path)
    with pytest.raises(StateLayoutError, match="Multiple evaluation summaries stored"):
        list_evaluation_runs(db_path)
    assert {summary.evaluation_storage_id for summary in summaries} == {
        evaluation_id,
        replay_evaluation_id,
    }
    assert load_evaluation_summary(db_path, evaluation_storage_id=evaluation_id) is not None
    assert load_evaluation_summary(db_path, evaluation_storage_id=replay_evaluation_id) is not None
    assert list_evaluation_runs(db_path, evaluation_storage_id=evaluation_id) == base_summary.runs
    assert (
        list_evaluation_runs(
            db_path,
            evaluation_storage_id=replay_evaluation_id,
        )
        == replay_summary.runs
    )


def test_record_evaluation_state_returns_loaded_summary(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    manifest = _manifest()
    summary = _evaluation_summary(0.2)

    write_artifact_manifest(db_path, manifest=manifest)

    loaded = record_evaluation_state(db_path, summary=summary)

    assert loaded.evaluation_storage_id == _evaluation_storage_id(summary)
    assert loaded.recorded_at > 0
    assert loaded.manifest == manifest
    assert loaded.runtime == summary


def test_evaluation_config_snapshot_freezes_storage_identity() -> None:
    config = coerce_evaluator_config(
        {
            "id": "poisson_replay",
            "window_seconds": 7200,
            "repetitions": 50,
            "arrival_rate_per_second": 0.05,
            "seed": 2026,
        }
    )
    summary = replace(
        _evaluation_summary(0.2),
        evaluation_config=EvaluationConfigSnapshot.from_config(config),
    )
    evaluation_id = _evaluation_storage_id(summary)

    assert isinstance(config, PoissonReplayEvaluatorConfig)
    config.seed = 9999

    assert summary.evaluation_config.payload()["seed"] == 2026
    assert _evaluation_storage_id(summary) == evaluation_id
    assert summary.evaluation_config == EvaluationConfigSnapshot.from_payload(
        {
            "id": "poisson_replay",
            "window_seconds": 7200,
            "repetitions": 50,
            "arrival_rate_per_second": 0.05,
            "seed": 2026,
        }
    )


def test_evaluation_config_snapshot_rejects_non_canonical_payload() -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        EvaluationConfigSnapshot.from_payload(
            {
                "id": "poisson_replay",
                "window_seconds": "7200",
                "repetitions": 50,
                "arrival_rate_per_second": 0.05,
                "seed": 2026,
            }
        )


def test_evaluation_run_listing_rejects_non_artifact_root_kind(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    ensure_state_db(db_path, root_kind=DATASET_ROOT_KIND, tables=DATASET_TABLES)

    with pytest.raises(StateLayoutError, match="root kind mismatch"):
        list_evaluation_runs(db_path)
