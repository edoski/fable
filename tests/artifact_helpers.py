from __future__ import annotations

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
from spice.features import compile_feature_contract
from spice.modeling.dataset_builders import (
    coerce_dataset_builder_config,
    fixed_sequence_temporal_runtime_metadata,
)
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.representations import sequence_input_contract
from spice.modeling.results import TrainingArtifactManifest
from spice.objectives import coerce_objective_config
from spice.prediction import compile_prediction_contract
from spice.semantics import (
    ArtifactSemantics,
    DatasetBuilderSemantics,
    InputNormalizationSemantics,
    ObjectiveSemantics,
)
from spice.temporal import TemporalCapability
from spice.temporal.compilers.observed_time_window import ObservedTimeWindowRuntimeMetadata
from spice.temporal.contracts import compile_problem_contract
from spice.temporal.input_normalization import ScalerStats


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


def manifest(
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
    temporal_capability = TemporalCapability(
        compiler_id=problem_contract.compiler_id,
        max_delay_seconds=36,
        action_width=4,
        compiler_runtime_metadata=ObservedTimeWindowRuntimeMetadata(
            slot_spacing_id="nominal",
            slot_spacing_seconds=12.0,
        ),
    )
    return TrainingArtifactManifest(
        artifact_id="artifact-1",
        dataset_builder=_dataset_builder_config(),
        prediction=prediction,
        objective=_objective_config(),
        evaluation=None,
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
            sequence_length=16,
            median_dt_seconds=12.0,
            min_sequence_length=8,
            max_sequence_length=64,
        ),
        temporal_capability=temporal_capability,
        semantics=ArtifactSemantics(
            problem=problem_contract.semantics,
            execution_policy=problem_contract.execution_policy.semantics,
            objective=ObjectiveSemantics(
                objective_id="validation",
                metric_id="total_loss",
                direction="minimize",
                evaluator_id=None,
            ),
            feature=feature_contract.semantics,
            prediction=prediction_contract.semantics,
            input_normalization=InputNormalizationSemantics(
                input_normalization_id="window_weighted_standard"
            ),
            representation=representation_contract.semantics,
            dataset_builder=DatasetBuilderSemantics(dataset_builder_id="fixed_sequence_temporal"),
            temporal_capability=temporal_capability.semantics,
        ),
    )
