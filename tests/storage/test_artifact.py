from __future__ import annotations

import pytest

from spice.config import (
    ArtifactVariant,
    StudyConfig,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from spice.features import FeaturePrerequisites, compile_feature_contract
from spice.modeling.artifacts import validate_artifact_semantics
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.results import (
    ArtifactChainMetadata,
    SimulationSummaryRecord,
    SplitSizes,
    TrainingArtifactManifest,
    TrainingEpochRecord,
    TrainingSummary,
)
from spice.modeling.summary import simulation_summary_sections
from spice.prediction import (
    MetricSet,
    PredictionSimulationRun,
    WindowMetricSummary,
    compile_prediction_contract,
)
from spice.storage.artifact import (
    list_simulation_runs,
    list_training_epochs,
    load_artifact_manifest,
    load_simulation_summary,
    load_training_summary,
    write_artifact_manifest,
    write_simulation_state,
    write_training_state,
)
from spice.storage.engine import RootKind
from spice.temporal.scaling import ScalerStats


def _prediction_config():
    return coerce_prediction_config(
        {
            "id": "candidate_offset_selection",
            "family": {
                "id": "candidate_offset_selection",
            },
        }
    )


def _paper_prediction_config():
    return coerce_prediction_config(
        {
            "id": "icdcs_2026_paper",
            "family": {
                "id": "min_block_fee_multitask",
                "classification_loss_weight": 1.0,
                "regression_loss_weight": 1.0,
                "class_weighting": "inverse_frequency",
            },
        }
    )


def _prediction_contract():
    prediction = _prediction_config()
    return compile_prediction_contract(
        prediction_id=prediction.id,
        family_config=prediction.family,
    )


def _paper_prediction_contract():
    prediction = _paper_prediction_config()
    return compile_prediction_contract(
        prediction_id=prediction.id,
        family_config=prediction.family,
    )


def _feature_set_config():
    return coerce_feature_set_config(
        {
            "id": "icdcs_2026_time_native",
            "family": {
                "id": "time_native",
            },
            "outputs": [
                "seconds_since_previous_block",
                "elapsed_seconds",
            ],
        }
    )


def test_training_artifact_summary_round_trip(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    prediction = _prediction_config()
    contract = _prediction_contract()
    feature_set = _feature_set_config()
    manifest = TrainingArtifactManifest(
        artifact_id="artifact-1",
        prediction=prediction,
        training_metric_descriptors=list(contract.training_metric_descriptors),
        chain=ArtifactChainMetadata(name="ethereum"),
        dataset_id="icdcs_2026",
        dataset_name="icdcs_2026",
        problem=coerce_problem_spec(
            {
                "id": "test_problem",
                "lookback_seconds": 120,
                "sample_count": 24,
                "max_supported_delay_seconds": 36,
                "compiler": {"id": "estimated_block"},
            }
        ),
        variant=ArtifactVariant.BASELINE,
        study=StudyConfig(name="default"),
        study_id=None,
        feature_set=feature_set,
        feature_prerequisites=FeaturePrerequisites(history_seconds=120, warmup_rows=0),
        max_candidate_slots=2,
        feature_graph_fingerprint="fingerprint-1",
        model=LstmModelConfig(
            input_projection_dim=8,
            hidden_size=16,
            num_layers=2,
            dropout=0.1,
            head_hidden_dim=8,
        ),
        scaler=ScalerStats(means=[0.0, 1.0], scales=[1.0, 1.0]),
        compiler_runtime_metadata={
            "effective_block_interval_seconds": 12.0,
            "lookback_steps": 10,
            "capability_candidate_count": 4,
        },
    )
    summary = TrainingSummary(
        artifact_id=manifest.artifact_id,
        prediction_id=manifest.prediction_id,
        prediction_family_id=manifest.prediction_family_id,
        training_metric_descriptors=list(manifest.training_metric_descriptors),
        chain=manifest.chain.name,
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        variant=manifest.variant,
        study=manifest.study,
        study_id=manifest.study_id,
        model_id=manifest.model.id,
        problem_id=manifest.problem_id,
        max_supported_delay_seconds=manifest.max_supported_delay_seconds,
        lookback_seconds=manifest.lookback_seconds,
        feature_family_id=manifest.feature_family_id,
        feature_prerequisites=manifest.feature_prerequisites,
        sample_count=manifest.sample_count,
        max_candidate_slots=manifest.max_candidate_slots,
        n_rows_available=128,
        n_rows_used=96,
        split_sizes=SplitSizes(train_samples=16, validation_samples=4, test_samples=4),
        best_epoch=2,
        resolved_device="cpu",
        resolved_precision="32-true",
        compiled=False,
        representation_id="sequence_inputs",
        storage_mode_id="materialized_dense",
        batch_planner_id="signature_bucketed",
        best_validation_metrics=MetricSet(
            values={
                "total_loss": 0.25,
                "exact_optimum_hit_rate": 0.5,
                "cost_over_optimum": 0.1,
                "profit_over_baseline": 0.2,
            }
        ),
        test_metrics=MetricSet(
            values={
                "total_loss": 0.3,
                "exact_optimum_hit_rate": 0.4,
                "cost_over_optimum": 0.15,
                "profit_over_baseline": 0.15,
            }
        ),
    )
    epoch_rows = [
        TrainingEpochRecord(
            epoch=1,
            train_metrics=MetricSet(
                values={
                    "total_loss": 0.4,
                    "exact_optimum_hit_rate": 0.3,
                    "cost_over_optimum": 0.2,
                    "profit_over_baseline": 0.1,
                }
            ),
            validation_metrics=MetricSet(
                values={
                    "total_loss": 0.35,
                    "exact_optimum_hit_rate": 0.4,
                    "cost_over_optimum": 0.15,
                    "profit_over_baseline": 0.12,
                }
            ),
        ),
        TrainingEpochRecord(
            epoch=2,
            train_metrics=MetricSet(
                values={
                    "total_loss": 0.3,
                    "exact_optimum_hit_rate": 0.35,
                    "cost_over_optimum": 0.17,
                    "profit_over_baseline": 0.14,
                }
            ),
            validation_metrics=MetricSet(
                values={
                    "total_loss": 0.25,
                    "exact_optimum_hit_rate": 0.5,
                    "cost_over_optimum": 0.1,
                    "profit_over_baseline": 0.2,
                }
            ),
        ),
    ]

    write_artifact_manifest(db_path, manifest=manifest, root_kind=RootKind.ARTIFACT)
    write_training_state(
        db_path, root_kind=RootKind.ARTIFACT, summary=summary, epoch_rows=epoch_rows
    )

    loaded_manifest = load_artifact_manifest(db_path)
    loaded_summary = load_training_summary(db_path)
    loaded_epochs = list_training_epochs(db_path)

    assert loaded_manifest == manifest
    assert loaded_manifest.feature_set == feature_set
    assert loaded_summary == summary
    assert loaded_epochs == epoch_rows


def test_artifact_validation_uses_full_feature_set_and_catches_feature_drift() -> None:
    feature_set = _feature_set_config()
    feature_contract = compile_feature_contract(feature_set=feature_set)
    prediction = _prediction_config()
    model = LstmModelConfig(
        input_projection_dim=8,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
        head_hidden_dim=8,
    )
    manifest = TrainingArtifactManifest(
        artifact_id="artifact-1",
        prediction=prediction,
        training_metric_descriptors=list(_prediction_contract().training_metric_descriptors),
        chain=ArtifactChainMetadata(name="ethereum"),
        dataset_id="icdcs_2026",
        dataset_name="icdcs_2026",
        problem=coerce_problem_spec(
            {
                "id": "test_problem",
                "lookback_seconds": 120,
                "sample_count": 24,
                "max_supported_delay_seconds": 36,
                "compiler": {"id": "estimated_block"},
            }
        ),
        variant=ArtifactVariant.BASELINE,
        study=StudyConfig(name="default"),
        study_id=None,
        feature_set=feature_set,
        feature_prerequisites=feature_contract.feature_prerequisites,
        max_candidate_slots=2,
        feature_graph_fingerprint=feature_contract.feature_graph_fingerprint,
        model=model,
        scaler=ScalerStats(means=[0.0, 1.0], scales=[1.0, 1.0]),
        compiler_runtime_metadata={
            "effective_block_interval_seconds": 12.0,
            "lookback_steps": 10,
            "capability_candidate_count": 4,
        },
    )

    with pytest.raises(
        ValueError,
        match="Configured feature_set does not match the trained artifact semantics",
    ):
        validate_artifact_semantics(
            manifest,
            problem=manifest.problem,
            feature_set=coerce_feature_set_config(
                {
                    "id": "icdcs_2026_time_native",
                    "family": {"id": "time_native"},
                    "outputs": ["elapsed_seconds"],
                }
            ),
            prediction=manifest.prediction,
            model=manifest.model,
        )

    drifted_manifest = TrainingArtifactManifest(
        artifact_id=manifest.artifact_id,
        prediction=manifest.prediction,
        training_metric_descriptors=list(manifest.training_metric_descriptors),
        chain=manifest.chain,
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        problem=manifest.problem,
        variant=manifest.variant,
        study=manifest.study,
        study_id=manifest.study_id,
        feature_set=manifest.feature_set,
        feature_prerequisites=manifest.feature_prerequisites,
        max_candidate_slots=manifest.max_candidate_slots,
        feature_graph_fingerprint="stale-fingerprint",
        model=manifest.model,
        scaler=manifest.scaler,
        compiler_runtime_metadata=dict(manifest.compiler_runtime_metadata),
    )

    with pytest.raises(
        ValueError,
        match="Current feature graph does not match the trained artifact manifest",
    ):
        validate_artifact_semantics(
            drifted_manifest,
            problem=manifest.problem,
            feature_set=feature_set,
            prediction=manifest.prediction,
            model=manifest.model,
        )


def test_simulation_artifact_summary_round_trip(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    contract = _prediction_contract()
    summary = SimulationSummaryRecord(
        artifact_id="artifact-1",
        prediction_id="candidate_offset_selection",
        prediction_family_id="candidate_offset_selection",
        simulation_metric_descriptors=list(contract.simulation_metric_descriptors),
        chain="ethereum",
        dataset_id="icdcs_2026",
        dataset_name="icdcs_2026",
        variant=ArtifactVariant.BASELINE,
        study=StudyConfig(name="default"),
        study_id=None,
        model_id="lstm",
        problem_id="test_problem",
        max_supported_delay_seconds=36,
        requested_delay_seconds=36,
        lookback_seconds=120,
        feature_family_id="time_native",
        feature_prerequisites=FeaturePrerequisites(history_seconds=120, warmup_rows=0),
        simulation_window_seconds=600,
        arrival_rate_per_second=0.02,
        repetitions=3,
        n_history_rows=128,
        n_evaluation_rows=64,
        sample_count=24,
        max_candidate_slots=2,
        metrics=MetricSet(
            values={
                "profit_over_baseline": 0.2,
                "cost_over_optimum": 0.15,
                "baseline_cost_over_optimum": 0.25,
                "realized_fee_sum": 9.0,
                "baseline_fee_sum": 10.0,
                "optimum_fee_sum": 8.0,
            }
        ),
        window_metrics={
            "profit_over_baseline": WindowMetricSummary(mean=0.2, std=0.01),
            "cost_over_optimum": WindowMetricSummary(mean=0.15, std=0.02),
            "baseline_cost_over_optimum": WindowMetricSummary(mean=0.25, std=0.03),
        },
        total_events=6,
        runs=[
            PredictionSimulationRun(
                window_start_timestamp=1_000.0,
                window_end_timestamp=1_600.0,
                n_arrivals=3,
                n_events=2,
                metrics={
                    "profit_over_baseline": 0.2,
                    "cost_over_optimum": 0.1,
                    "baseline_cost_over_optimum": 0.15,
                    "realized_fee_sum": 4.0,
                    "baseline_fee_sum": 5.0,
                    "optimum_fee_sum": 3.5,
                },
            ),
            PredictionSimulationRun(
                window_start_timestamp=2_000.0,
                window_end_timestamp=2_600.0,
                n_arrivals=4,
                n_events=4,
                metrics={
                    "profit_over_baseline": 0.18,
                    "cost_over_optimum": 0.2,
                    "baseline_cost_over_optimum": 0.28,
                    "realized_fee_sum": 5.0,
                    "baseline_fee_sum": 5.0,
                    "optimum_fee_sum": 4.5,
                },
            ),
        ],
    )

    write_simulation_state(db_path, root_kind=RootKind.ARTIFACT, summary=summary)

    loaded_summary = load_simulation_summary(db_path)
    loaded_runs = list_simulation_runs(db_path)

    assert loaded_summary == summary
    assert loaded_runs == summary.runs


@pytest.mark.parametrize(
    "contract_factory",
    [_prediction_contract, _paper_prediction_contract],
)
def test_simulation_summary_sections_use_simulation_descriptors_and_window_metrics(
    contract_factory,
) -> None:
    contract = contract_factory()
    summary = SimulationSummaryRecord(
        artifact_id="artifact-1",
        prediction_id=contract.prediction_id,
        prediction_family_id=contract.prediction_family_id,
        simulation_metric_descriptors=list(contract.simulation_metric_descriptors),
        chain="ethereum",
        dataset_id="icdcs_2026",
        dataset_name="icdcs_2026",
        variant=ArtifactVariant.BASELINE,
        study=None,
        study_id=None,
        model_id="lstm",
        problem_id="test_problem",
        max_supported_delay_seconds=36,
        requested_delay_seconds=24,
        lookback_seconds=120,
        feature_family_id="block_native",
        feature_prerequisites=FeaturePrerequisites(history_seconds=120, warmup_rows=200),
        simulation_window_seconds=600,
        arrival_rate_per_second=0.02,
        repetitions=3,
        n_history_rows=128,
        n_evaluation_rows=64,
        sample_count=24,
        max_candidate_slots=3,
        metrics=MetricSet(
            values={
                "profit_over_baseline": 0.2,
                "cost_over_optimum": 0.15,
                "baseline_cost_over_optimum": 0.25,
                "realized_fee_sum": 9.0,
                "baseline_fee_sum": 10.0,
                "optimum_fee_sum": 8.0,
            }
        ),
        window_metrics={
            "profit_over_baseline": WindowMetricSummary(mean=0.2, std=0.01),
            "cost_over_optimum": WindowMetricSummary(mean=0.15, std=0.02),
        },
        total_events=6,
        runs=[],
    )

    sections = dict(simulation_summary_sections(summary))

    assert "results" in sections
    assert ("profit over baseline", "0.2000") in sections["results"]
    assert ("cost over optimum", "0.1500") in sections["results"]
    assert "window metrics" in sections
    assert ("profit over baseline", "mean=0.2000 std=0.0100") in sections["window metrics"]
