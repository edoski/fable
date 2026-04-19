from __future__ import annotations

import pytest

from spice.config import (
    ArtifactVariant,
    StudyConfig,
    coerce_dataset_builder_config,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from spice.core.errors import SpiceOperatorError
from spice.evaluation import EvaluationRun
from spice.features import compile_feature_contract
from spice.modeling.artifacts import validate_artifact_semantics
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.families.registry import resolve_model_representation_id
from spice.modeling.representations import compile_representation_contract
from spice.modeling.results import (
    EvaluationRuntimeSummary,
    LoadedEvaluationSummary,
    SplitSizes,
    TrainingArtifactManifest,
    TrainingEpochRecord,
    TrainingRuntimeSummary,
)
from spice.modeling.summary import evaluation_summary_sections
from spice.prediction import (
    MetricDescriptor,
    MetricSet,
    WindowMetricSummary,
    compile_prediction_contract,
)
from spice.semantics import (
    ArtifactSemantics,
    DatasetBuilderSemantics,
    InputNormalizationSemantics,
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
from spice.temporal.contracts import EstimatedBlockRuntimeMetadata, compile_problem_contract
from spice.temporal.scaling import ScalerStats


def _prediction_config():
    return coerce_prediction_config(
        {
            "id": "candidate_offset_selection",
            "family": {"id": "candidate_offset_selection"},
        }
    )


def _paper_prediction_config():
    return coerce_prediction_config(
        {
            "id": "icdcs_2026",
            "family": {
                "id": "min_block_fee_multitask",
                "classification_loss_weight": 1.0,
                "regression_loss_weight": 0.5,
                "class_weighting": "inverse_frequency",
                "fee_target_normalization": "zscore_train_split",
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
            "id": "time_native_baseline",
            "family": {"id": "time_native"},
            "outputs": [
                "seconds_since_previous_block",
                "elapsed_seconds",
            ],
        }
    )


def _dataset_builder_config():
    return coerce_dataset_builder_config({"id": "standard_temporal"})


def _problem_config():
    return coerce_problem_spec(
        {
            "id": "test_problem",
            "lookback_seconds": 120,
            "sample_count": 24,
            "max_delay_seconds": 36,
            "compiler": {"id": "estimated_block"},
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
    representation_contract = compile_representation_contract(
        resolve_model_representation_id(model)
    )
    return TrainingArtifactManifest(
        artifact_id="artifact-1",
        dataset_builder=_dataset_builder_config(),
        prediction=prediction,
        chain_name="ethereum",
        dataset_id="icdcs_2026",
        dataset_name="icdcs_2026",
        problem=problem,
        variant=ArtifactVariant.BASELINE,
        study=StudyConfig(name="default"),
        study_id=None,
        feature_set=feature_set,
        model=model,
        scaler=ScalerStats(means=[0.0, 1.0], scales=[1.0, 1.0]),
        builder_runtime_metadata=EstimatedBlockRuntimeMetadata(
            calibrated_interval_seconds=12.0,
            lookback_interval_seconds=12.0,
            candidate_interval_seconds=12.0,
            lookback_steps=10,
            capability_candidate_count=4,
        ),
        semantics=ArtifactSemantics(
            problem=problem_contract.semantics,
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
        resolved_device="cpu",
        resolved_precision="32-true",
        compiled=False,
        loader_strategy_id="host_dataloader",
        input_storage_mode_id="materialized_host",
        target_storage_mode_id="materialized_host",
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
            train_metrics=MetricSet(values={"total_loss": 0.4}),
            validation_metrics=MetricSet(values={"total_loss": 0.35}),
        ),
        TrainingEpochRecord(
            epoch=2,
            train_metrics=MetricSet(values={"total_loss": 0.3}),
            validation_metrics=MetricSet(values={"total_loss": 0.25}),
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


def test_artifact_validation_catches_feature_drift() -> None:
    manifest = _manifest()
    feature_set = manifest.feature_set

    with pytest.raises(
        SpiceOperatorError,
        match="Configured feature_set does not match the trained artifact semantics",
    ):
        validate_artifact_semantics(
            manifest,
            problem=manifest.problem,
            dataset_builder=manifest.dataset_builder,
            feature_set=coerce_feature_set_config(
                {
                    "id": "time_native_baseline",
                    "family": {"id": "time_native"},
                    "outputs": ["elapsed_seconds"],
                }
            ),
            prediction=manifest.prediction,
            model=manifest.model,
        )

    drifted_manifest = TrainingArtifactManifest(
        artifact_id=manifest.artifact_id,
        dataset_builder=manifest.dataset_builder,
        prediction=manifest.prediction,
        chain_name=manifest.chain_name,
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        problem=manifest.problem,
        variant=manifest.variant,
        study=manifest.study,
        study_id=manifest.study_id,
        feature_set=manifest.feature_set,
        model=manifest.model,
        scaler=manifest.scaler,
        builder_runtime_metadata=manifest.builder_runtime_metadata,
        semantics=ArtifactSemantics(
            problem=manifest.semantics.problem,
            feature=manifest.semantics.feature.__class__(
                feature_set_id=manifest.semantics.feature.feature_set_id,
                feature_family_id=manifest.semantics.feature.feature_family_id,
                feature_names=manifest.semantics.feature.feature_names,
                feature_graph_fingerprint="stale-fingerprint",
                feature_prerequisites=manifest.semantics.feature.feature_prerequisites,
            ),
            prediction=manifest.semantics.prediction,
            input_normalization=manifest.semantics.input_normalization,
            representation=manifest.semantics.representation,
            dataset_builder=manifest.semantics.dataset_builder,
            max_candidate_slots=manifest.semantics.max_candidate_slots,
        ),
    )

    with pytest.raises(
        SpiceOperatorError,
        match="Current feature graph does not match the trained artifact manifest",
    ):
        validate_artifact_semantics(
            drifted_manifest,
            problem=manifest.problem,
            dataset_builder=manifest.dataset_builder,
            feature_set=feature_set,
            prediction=manifest.prediction,
            model=manifest.model,
        )


def test_evaluation_artifact_summary_round_trip(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    manifest = _manifest()
    summary = EvaluationRuntimeSummary(
        delay_seconds=24,
        evaluator_id="poisson_replay",
        evaluator_config={
            "id": "poisson_replay",
            "window_seconds": 600,
            "arrival_rate_per_second": 0.02,
            "repetitions": 3,
            "seed": 2026,
        },
        metric_descriptors=(
            MetricDescriptor(
                id="profit_over_baseline",
                label="profit over baseline",
                role="score",
            ),
            MetricDescriptor(
                id="cost_over_optimum",
                label="cost over optimum",
                role="score",
            ),
            MetricDescriptor(
                id="baseline_cost_over_optimum",
                label="baseline cost over optimum",
                role="diagnostic",
            ),
            MetricDescriptor(
                id="realized_fee_sum",
                label="realized fee sum",
                role="diagnostic",
            ),
            MetricDescriptor(
                id="baseline_fee_sum",
                label="baseline fee sum",
                role="diagnostic",
            ),
            MetricDescriptor(
                id="optimum_fee_sum",
                label="optimum fee sum",
                role="diagnostic",
            ),
        ),
        n_history_rows=128,
        n_evaluation_rows=64,
        sample_count=24,
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
        runs=[
            EvaluationRun(
                n_events=2,
                metrics={"profit_over_baseline": 0.2},
                metadata={
                    "window_start_timestamp": 1_000.0,
                    "window_end_timestamp": 1_600.0,
                    "n_arrivals": 3,
                },
            ),
            EvaluationRun(
                n_events=4,
                metrics={"profit_over_baseline": 0.18},
                metadata={
                    "window_start_timestamp": 2_000.0,
                    "window_end_timestamp": 2_600.0,
                    "n_arrivals": 4,
                },
            ),
        ],
    )

    write_artifact_manifest(db_path, manifest=manifest, root_kind=RootKind.ARTIFACT)
    evaluation_id, recorded_at = write_evaluation_state(
        db_path,
        root_kind=RootKind.ARTIFACT,
        summary=summary,
    )

    loaded_summary = load_evaluation_summary(db_path)
    loaded_runs = list_evaluation_runs(db_path)
    listed_summaries = list_evaluation_summaries(db_path)

    assert loaded_summary is not None
    assert loaded_summary.evaluation_id == evaluation_id
    assert loaded_summary.recorded_at == recorded_at
    assert loaded_summary.manifest == manifest
    assert loaded_summary.runtime == summary
    assert loaded_runs == summary.runs
    assert [item.evaluation_id for item in listed_summaries] == [evaluation_id]


def test_multiple_evaluation_summaries_can_coexist_per_artifact(tmp_path) -> None:
    db_path = tmp_path / ".spice" / "state.sqlite"
    manifest = _manifest()
    base_summary = EvaluationRuntimeSummary(
        delay_seconds=24,
        evaluator_id="paper_fullset",
        evaluator_config={"id": "paper_fullset"},
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
        metrics=MetricSet(values={"profit_over_baseline": 0.2}),
        window_metrics={},
        total_events=6,
        runs=[
            EvaluationRun(
                n_events=2,
                metrics={"profit_over_baseline": 0.2},
                metadata={"mode": "fullset"},
            )
        ],
    )
    replay_summary = EvaluationRuntimeSummary(
        delay_seconds=24,
        evaluator_id="poisson_replay",
        evaluator_config={
            "id": "poisson_replay",
            "window_seconds": 600,
            "arrival_rate_per_second": 0.02,
            "repetitions": 3,
            "seed": 2026,
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
        metrics=MetricSet(values={"profit_over_baseline": 0.1}),
        window_metrics={},
        total_events=4,
        runs=[
            EvaluationRun(
                n_events=4,
                metrics={"profit_over_baseline": 0.1},
                metadata={"window_start_timestamp": 1_000.0},
            )
        ],
    )

    write_artifact_manifest(db_path, manifest=manifest, root_kind=RootKind.ARTIFACT)
    paper_evaluation_id, _ = write_evaluation_state(
        db_path,
        root_kind=RootKind.ARTIFACT,
        summary=base_summary,
    )
    replay_evaluation_id, _ = write_evaluation_state(
        db_path,
        root_kind=RootKind.ARTIFACT,
        summary=replay_summary,
    )

    summaries = list_evaluation_summaries(db_path)

    assert {summary.evaluation_id for summary in summaries} == {
        paper_evaluation_id,
        replay_evaluation_id,
    }
    assert load_evaluation_summary(db_path, evaluation_id=paper_evaluation_id) is not None
    assert load_evaluation_summary(db_path, evaluation_id=replay_evaluation_id) is not None
    assert list_evaluation_runs(db_path, evaluation_id=paper_evaluation_id) == base_summary.runs
    assert list_evaluation_runs(db_path, evaluation_id=replay_evaluation_id) == replay_summary.runs


@pytest.mark.parametrize(
    ("prediction_factory", "contract_factory"),
    [
        (_prediction_config, _prediction_contract),
        (_paper_prediction_config, _paper_prediction_contract),
    ],
)
def test_evaluation_summary_sections_use_runtime_descriptors_and_window_metrics(
    prediction_factory,
    contract_factory,
) -> None:
    summary = LoadedEvaluationSummary(
        evaluation_id="poisson_replay-24s-test",
        recorded_at=1_234,
        manifest=_manifest(
            prediction_factory=prediction_factory,
            prediction_contract_factory=contract_factory,
        ),
        runtime=EvaluationRuntimeSummary(
            delay_seconds=24,
            evaluator_id="poisson_replay",
            evaluator_config={
                "id": "poisson_replay",
                "window_seconds": 600,
                "arrival_rate_per_second": 0.02,
                "repetitions": 3,
                "seed": 2026,
            },
            metric_descriptors=(
                MetricDescriptor(
                    id="profit_over_baseline",
                    label="profit over baseline",
                    role="score",
                ),
                MetricDescriptor(
                    id="cost_over_optimum",
                    label="cost over optimum",
                    role="score",
                ),
                MetricDescriptor(
                    id="baseline_cost_over_optimum",
                    label="baseline cost over optimum",
                    role="diagnostic",
                ),
                MetricDescriptor(
                    id="realized_fee_sum",
                    label="realized fee sum",
                    role="diagnostic",
                ),
                MetricDescriptor(
                    id="baseline_fee_sum",
                    label="baseline fee sum",
                    role="diagnostic",
                ),
                MetricDescriptor(
                    id="optimum_fee_sum",
                    label="optimum fee sum",
                    role="diagnostic",
                ),
            ),
            n_history_rows=128,
            n_evaluation_rows=64,
            sample_count=24,
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
        ),
    )

    sections = dict(evaluation_summary_sections(summary))

    assert "results" in sections
    assert ("profit over baseline", "0.2000") in sections["results"]
    assert ("cost over optimum", "0.1500") in sections["results"]
    assert "window metrics" in sections
    assert ("profit over baseline", "mean=0.2000 std=0.0100") in sections["window metrics"]
