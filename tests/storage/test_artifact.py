from __future__ import annotations

import pytest

from spice.config import (
    ArtifactVariant,
    PredictionConfig,
    SplitConfig,
    StudyConfig,
    TrainingConfig,
    coerce_dataset_builder_config,
    coerce_feature_set_config,
    coerce_problem_spec,
)
from spice.core.errors import StateLayoutError
from spice.evaluation import EvaluationRun
from spice.features import compile_feature_contract
from spice.modeling.dataset_builders import standard_temporal_runtime_metadata
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.representations import sequence_input_contract
from spice.modeling.results import (
    EvaluationRuntimeSummary,
    SplitSizes,
    TrainingArtifactManifest,
    TrainingEpochRecord,
    TrainingRuntimeSummary,
)
from spice.objectives import coerce_objective_config
from spice.prediction import (
    MetricDescriptor,
    MetricSet,
    compile_prediction_contract,
)
from spice.semantics import (
    ArtifactSemantics,
    DatasetBuilderSemantics,
    InputNormalizationSemantics,
    ObjectiveSemantics,
)
from spice.storage.artifact import (
    list_evaluation_runs,
    list_evaluation_summaries,
    list_training_epochs,
    load_artifact_manifest,
    load_evaluation_summary,
    load_training_summary,
    write_artifact_manifest,
    write_evaluation_state,
    write_training_state,
)
from spice.storage.engine import RootKind
from spice.temporal.compilers.estimated_block import EstimatedBlockRuntimeMetadata
from spice.temporal.contracts import compile_problem_contract
from spice.temporal.scaling import ScalerStats


def _prediction_config():
    return PredictionConfig.model_validate(
        {
            "id": "candidate_offset_selection",
            "family_id": "candidate_offset_selection",
        }
    )


def _prediction_contract():
    prediction = _prediction_config()
    return compile_prediction_contract(
        prediction_id=prediction.id,
        family_id=prediction.family_id,
    )


def _feature_set_config():
    return coerce_feature_set_config(
        {
            "id": "timestamp_features_baseline",
            "family": {"id": "timestamp_features"},
            "outputs": [
                "seconds_since_previous_block",
                "elapsed_seconds",
            ],
        }
    )


def _dataset_builder_config():
    return coerce_dataset_builder_config({"id": "standard_temporal"})


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
            "compiler": {"id": "estimated_block"},
            "realization_policy": {"id": "strict_deadline_miss"},
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
    feature_set = _feature_set_config()
    feature_contract = compile_feature_contract(feature_set=feature_set)
    problem = _problem_config()
    problem_contract = compile_problem_contract(
        problem=problem,
        feature_contract=feature_contract,
    )
    model = _model_config()
    representation_contract = sequence_input_contract()
    return TrainingArtifactManifest(
        artifact_id="artifact-1",
        dataset_builder=_dataset_builder_config(),
        prediction=prediction,
        objective=_objective_config(),
        chain_name="ethereum",
        dataset_id="same_block_closed",
        dataset_name="same_block_closed",
        problem=problem,
        variant=ArtifactVariant.BASELINE,
        study=StudyConfig(name="default"),
        study_id=None,
        feature_set=feature_set,
        model=model,
        split=_split_config(),
        training=_training_config(),
        scaler=ScalerStats(means=[0.0, 1.0], scales=[1.0, 1.0]),
        builder_runtime_metadata=standard_temporal_runtime_metadata(
            compiler_id=problem_contract.compiler_id,
            compiler_runtime_metadata=EstimatedBlockRuntimeMetadata(
                calibrated_interval_seconds=12.0,
                lookback_interval_seconds=12.0,
                candidate_interval_seconds=12.0,
                lookback_steps=10,
                capability_candidate_count=4,
            ),
        ),
        semantics=ArtifactSemantics(
            problem=problem_contract.semantics,
            realization_policy=problem_contract.realization_policy.semantics,
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
            dataset_builder=DatasetBuilderSemantics(dataset_builder_id="standard_temporal"),
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

    write_artifact_manifest(db_path, manifest=manifest, root_kind=RootKind.ARTIFACT)
    write_training_state(
        db_path,
        root_kind=RootKind.ARTIFACT,
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


def _evaluation_summary(evaluation_id: str, value: float) -> EvaluationRuntimeSummary:
    return EvaluationRuntimeSummary(
        delay_seconds=24,
        evaluation_id=evaluation_id,
        evaluation_config={
            "id": evaluation_id,
            "sampler": "fullset",
            "aggregation": {"id": "total_ratio"},
        },
        metric_descriptors=(
            MetricDescriptor(
                id="profit_over_baseline",
                label="profit over baseline",
                role="score",
            ),
        ),
        n_history_rows=128,
        n_evaluation_rows=64,
        sample_count=24,
        metrics=MetricSet(values={"profit_over_baseline": value}),
        window_metrics={},
        total_events=2,
        runs=[
            EvaluationRun(
                n_events=2,
                metrics={"profit_over_baseline": value},
                metadata={"mode": evaluation_id},
            ),
        ],
    )


def test_evaluation_artifact_summaries_round_trip_and_coexist(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    manifest = _manifest()
    base_summary = _evaluation_summary("fullset", 0.2)
    replay_summary = _evaluation_summary("poisson_replay_2h_mean", 0.1)

    write_artifact_manifest(db_path, manifest=manifest, root_kind=RootKind.ARTIFACT)
    evaluation_id, recorded_at = write_evaluation_state(
        db_path,
        root_kind=RootKind.ARTIFACT,
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

    replay_evaluation_id, _ = write_evaluation_state(
        db_path,
        root_kind=RootKind.ARTIFACT,
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
