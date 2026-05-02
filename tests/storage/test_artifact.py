from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from spice.config import (
    ArtifactVariant,
    ChainSpec,
    PredictionConfig,
    SplitConfig,
    StudyConfig,
    TrainingConfig,
    coerce_features_config,
    coerce_problem_spec,
)
from spice.core.errors import StateLayoutError
from spice.evaluation import EvaluationRun, PoissonReplayEvaluatorConfig, coerce_evaluator_config
from spice.features import compile_feature_contract
from spice.metrics import MetricDescriptor, MetricSet, WindowMetricSummary
from spice.modeling.dataset_builders import (
    coerce_dataset_builder_config,
    fixed_sequence_temporal_runtime_metadata,
)
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.representations import sequence_input_contract
from spice.modeling.results import (
    EvaluationConfigSnapshot,
    EvaluationExecutionProvenance,
    EvaluationRuntimeSummary,
    SplitSizes,
    TrainingArtifactManifest,
    TrainingEpochRecord,
    TrainingRuntimeSummary,
)
from spice.objectives import coerce_objective_config
from spice.prediction import (
    compile_prediction_contract,
)
from spice.semantics import (
    ArtifactSemantics,
    DatasetBuilderSemantics,
    InputNormalizationSemantics,
    ObjectiveSemantics,
)
from spice.storage.artifact import (
    _evaluation_storage_id,
    list_evaluation_runs,
    list_evaluation_summaries,
    list_training_epochs,
    load_artifact_manifest,
    load_evaluation_summary,
    load_training_summary,
    upsert_evaluation_state,
    write_artifact_manifest,
    write_training_state,
)
from spice.storage.artifact_codecs import (
    evaluation_run_from_payload,
    training_summary_from_payload,
)
from spice.storage.engine import DATASET_ROOT_KIND, ensure_state_db
from spice.storage.schema import DATASET_TABLES
from spice.storage.semantics_codecs import (
    artifact_semantics_from_payload,
    artifact_semantics_payload,
)
from spice.temporal.compilers.observed_time_window import ObservedTimeWindowRuntimeMetadata
from spice.temporal.contracts import compile_problem_contract
from spice.temporal.scaling import ScalerStats


def _prediction_config():
    return PredictionConfig.model_validate(
        {
            "id": "icdcs_2026",
            "family_id": "min_block_fee_multitask",
        }
    )


def _prediction_contract():
    prediction = _prediction_config()
    return compile_prediction_contract(
        prediction_id=prediction.id,
        family_id=prediction.family_id,
    )


def _features_config():
    return coerce_features_config(
        {
            "id": "core_fee_dynamics",
            "outputs": [
                "log_base_fee_per_gas",
                "log_prev_gas_used",
            ],
        }
    )


def _dataset_builder_config():
    return coerce_dataset_builder_config({"id": "fixed_sequence_temporal"})


def _objective_config():
    return coerce_objective_config(
        {
            "id": "validation",
            "metric_id": "total_loss",
            "direction": "minimize",
        }
    )


def _problem_config():
    return coerce_problem_spec(
        {
            "id": "test_problem",
            "lookback_seconds": 120,
            "sample_count": 24,
            "max_delay_seconds": 36,
            "compiler": {
                "id": "observed_time_window",
                "slot_spacing": {"id": "nominal"},
            },
            "execution_policy": {"id": "strict_deadline_miss"},
        }
    )


def _model_config():
    return LstmModelConfig(
        input_projection_dim=8,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
        head_hidden_dim=8,
    )


def _split_config():
    return SplitConfig(train_fraction=0.8, validation_fraction=0.1)


def _training_config():
    return TrainingConfig.model_validate(
        {
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "batch_size": 8,
            "max_epochs": 2,
            "early_stopping": {"patience": 1, "min_delta": 0.0},
            "gradient_clip_norm": 1.0,
            "seed": 2026,
            "deterministic": True,
            "log_every_n_steps": 1,
            "input_normalization": {"id": "window_weighted_standard"},
        }
    )


def _manifest(
    *,
    prediction_factory=_prediction_config,
    prediction_contract_factory=_prediction_contract,
) -> TrainingArtifactManifest:
    prediction = prediction_factory()
    prediction_contract = prediction_contract_factory()
    features = _features_config()
    feature_contract = compile_feature_contract(features=features)
    problem = _problem_config()
    problem_contract = compile_problem_contract(
        problem=problem,
        feature_contract=feature_contract,
        chain_runtime=ChainSpec.model_validate(
            {
                "name": "ethereum",
                "runtime": {
                    "chain_id": 1,
                    "uses_poa_extra_data": False,
                    "nominal_block_time_seconds": 12.0,
                },
            }
        ).runtime,
    )
    model = _model_config()
    representation_contract = sequence_input_contract()
    return TrainingArtifactManifest(
        artifact_id="artifact-1",
        dataset_builder=_dataset_builder_config(),
        prediction=prediction,
        objective=_objective_config(),
        chain_name="ethereum",
        dataset_id="current_row_fee_dynamics",
        dataset_name="current_row_fee_dynamics",
        problem=problem,
        variant=ArtifactVariant.BASELINE,
        study=StudyConfig(name="default"),
        study_id=None,
        features=features,
        model=model,
        split=_split_config(),
        training=_training_config(),
        scaler=ScalerStats(means=[0.0, 1.0], scales=[1.0, 1.0]),
        builder_runtime_metadata=fixed_sequence_temporal_runtime_metadata(
            compiler_id=problem_contract.compiler_id,
            compiler_runtime_metadata=ObservedTimeWindowRuntimeMetadata(
                slot_spacing_id="nominal",
                slot_spacing_seconds=12.0,
                capability_action_count=4,
            ),
            sequence_length=16,
            median_dt_seconds=12.0,
            min_sequence_length=8,
            max_sequence_length=64,
        ),
        semantics=ArtifactSemantics(
            problem=problem_contract.semantics,
            execution_policy=problem_contract.execution_policy.semantics,
            objective=ObjectiveSemantics(
                objective_id="validation",
                metric_id="total_loss",
                direction="minimize",
                benchmark_id=None,
            ),
            feature=feature_contract.semantics,
            prediction=prediction_contract.semantics,
            input_normalization=InputNormalizationSemantics(
                input_normalization_id="window_weighted_standard"
            ),
            representation=representation_contract.semantics,
            dataset_builder=DatasetBuilderSemantics(dataset_builder_id="fixed_sequence_temporal"),
            max_candidate_slots=2,
        ),
    )


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


def test_training_summary_payload_rejects_loose_scalar_types() -> None:
    with pytest.raises(StateLayoutError, match="training summary"):
        training_summary_from_payload(
            {
                "n_rows_available": "128",
                "n_rows_used": True,
                "train_samples": 10,
                "validation_samples": 5,
                "test_samples": 5,
                "best_epoch": 1,
                "best_objective_metric_id": "total_loss",
                "best_objective_value": "0.25",
                "best_validation_metrics": {"total_loss": "0.25"},
                "test_metrics": {"total_loss": 0.3},
            }
        )


def test_evaluation_run_payload_rejects_loose_scalar_types() -> None:
    with pytest.raises(StateLayoutError, match="evaluation run"):
        evaluation_run_from_payload(
            {
                "n_events": True,
                "metrics": {"profit_over_baseline": "0.2"},
                "metadata": {"mode": True},
            }
        )


def _evaluation_summary(value: float, *, seed: int = 2026) -> EvaluationRuntimeSummary:
    evaluation_config = coerce_evaluator_config(
        {
            "id": "poisson_replay_2h",
            "window_seconds": 7200,
            "repetitions": 50,
            "arrival_rate_per_second": 0.05,
            "seed": seed,
        }
    )
    return EvaluationRuntimeSummary(
        delay_seconds=24,
        evaluation_id="poisson_replay_2h",
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
        window_metrics={
            "profit_over_baseline": WindowMetricSummary(mean=value, std=0.01)
        },
        total_events=2,
        runs=[
            EvaluationRun(
                n_events=2,
                metrics={"profit_over_baseline": value},
                metadata={"mode": "poisson_replay_2h"},
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
    assert loaded_summary.evaluation_id == evaluation_id
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
    assert {summary.evaluation_id for summary in summaries} == {
        evaluation_id,
        replay_evaluation_id,
    }
    assert load_evaluation_summary(db_path, evaluation_id=evaluation_id) is not None
    assert load_evaluation_summary(db_path, evaluation_id=replay_evaluation_id) is not None
    assert list_evaluation_runs(db_path, evaluation_id=evaluation_id) == base_summary.runs
    assert list_evaluation_runs(db_path, evaluation_id=replay_evaluation_id) == replay_summary.runs


def test_evaluation_config_snapshot_freezes_storage_identity() -> None:
    config = coerce_evaluator_config(
        {
            "id": "poisson_replay_2h",
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
            "id": "poisson_replay_2h",
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
                "id": "poisson_replay_2h",
                "window_seconds": "7200",
                "repetitions": 50,
                "arrival_rate_per_second": 0.05,
                "seed": 2026,
            }
        )


def test_artifact_semantics_payload_rejects_loose_scalar_types() -> None:
    payload = artifact_semantics_payload(_manifest().semantics)
    problem_payload = cast(dict[str, object], payload["problem"])
    problem_payload["sample_count"] = "24"

    with pytest.raises(StateLayoutError, match="semantics payload"):
        artifact_semantics_from_payload(payload)


def test_evaluation_run_listing_rejects_non_artifact_root_kind(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    ensure_state_db(db_path, root_kind=DATASET_ROOT_KIND, tables=DATASET_TABLES)

    with pytest.raises(StateLayoutError, match="root kind mismatch"):
        list_evaluation_runs(db_path)
