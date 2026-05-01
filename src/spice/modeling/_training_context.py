"""Shared compiled training contracts for train/tune provenance."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import ChainRuntimeSpec, TrainConfig, TuneConfig
from ..features import CompiledFeatureContract, compile_feature_contract
from ..prediction import CompiledPredictionContract, compile_prediction_contract
from ..temporal.contracts import CompiledProblemContract, compile_problem_contract
from ..temporal.input_normalization import (
    CompiledInputNormalizationContract,
    compile_input_normalization_contract,
)
from .dataset_builders import (
    CompiledDatasetBuilderContract,
    compile_dataset_builder_contract,
)
from .objective_runtime import CompiledObjectiveRuntime, compile_objective_runtime
from .representations import (
    CompiledRepresentationContract,
    sequence_input_contract,
)


@dataclass(frozen=True, slots=True)
class CompiledTrainingContext:
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract
    prediction_contract: CompiledPredictionContract
    objective_runtime: CompiledObjectiveRuntime
    dataset_builder_contract: CompiledDatasetBuilderContract
    input_normalization_contract: CompiledInputNormalizationContract
    representation_contract: CompiledRepresentationContract


def compile_training_context(
    config: TrainConfig | TuneConfig,
    *,
    chain_runtime: ChainRuntimeSpec | None = None,
) -> CompiledTrainingContext:
    feature_contract = compile_feature_contract(features=config.features)
    prediction_contract = compile_prediction_contract(
        prediction_id=config.prediction.id,
        family_id=config.prediction.family_id,
    )
    objective_runtime = compile_objective_runtime(
        config.objective,
        evaluation=config.evaluation,
        prediction_metric_descriptors=prediction_contract.training_metric_descriptors,
    )
    return CompiledTrainingContext(
        feature_contract=feature_contract,
        problem_contract=compile_problem_contract(
            problem=config.problem,
            feature_contract=feature_contract,
            chain_runtime=chain_runtime or config.chain.runtime,
        ),
        prediction_contract=prediction_contract,
        objective_runtime=objective_runtime,
        dataset_builder_contract=compile_dataset_builder_contract(config.dataset_builder),
        input_normalization_contract=compile_input_normalization_contract(
            config.training.input_normalization
        ),
        representation_contract=sequence_input_contract(),
    )
