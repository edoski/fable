"""Training and inference dataset preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config.models import (
    ArtifactVariant,
    ChainRuntimeSpec,
    ChainSpec,
    FeaturesConfig,
    PredictionConfig,
    ProblemSpec,
    SplitConfig,
    StudyConfig,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
)
from ..corpus.io import load_block_frame
from ..corpus.metadata import DatasetManifest
from ..evaluation import CompiledEvaluatorContract, EvaluatorConfig, compile_evaluator_contract
from ..features import CompiledFeatureContract, compile_feature_contract
from ..objectives import ObjectiveConfig
from ..prediction import CompiledPredictionContract, compile_prediction_contract
from ..temporal.contracts import CompiledProblemContract, compile_problem_contract
from ..temporal.input_normalization import (
    CompiledInputNormalizationContract,
    compile_input_normalization_contract,
)
from .dataset_builders import (
    CompiledDatasetBuilderContract,
    DatasetBuilderConfig,
    PreparedTrainingDataset,
    TrainingDatasetPreparationContext,
    TrainingDatasetPreparationFacts,
    compile_dataset_builder_contract,
)
from .families.base import ModelConfig
from .families.registry import build_model
from .objective_runtime import CompiledObjectiveRuntime, compile_objective_runtime
from .representations import CompiledRepresentationContract, compile_representation_contract
from .training_run import TrainingRunResult
from .training_runner import (
    EarlyStopCallback,
    EpochEndCallback,
    TrainingCallbacks,
    TrainingFitSpec,
    run_training_fit,
)

if TYPE_CHECKING:
    from ..storage.workflow_roots import ArtifactRootHandle, CorpusRootHandle, StudyRootHandle


@dataclass(slots=True)
class TrainingSpec:
    chain: ChainSpec
    dataset_id: str
    dataset_name: str
    artifact_id: str
    problem: ProblemSpec
    dataset_builder: DatasetBuilderConfig
    dataset_builder_contract: CompiledDatasetBuilderContract
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract
    features: FeaturesConfig
    prediction: PredictionConfig
    objective: ObjectiveConfig
    evaluation: EvaluatorConfig | None
    prediction_contract: CompiledPredictionContract
    objective_runtime: CompiledObjectiveRuntime
    input_normalization_contract: CompiledInputNormalizationContract
    representation_contract: CompiledRepresentationContract
    model: ModelConfig
    split: SplitConfig
    training: TrainingConfig
    variant: ArtifactVariant = ArtifactVariant.BASELINE
    study: StudyConfig | None = None
    study_id: str | None = None


@dataclass(frozen=True, slots=True)
class CompiledTrainingContext:
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract
    prediction_contract: CompiledPredictionContract
    objective_runtime: CompiledObjectiveRuntime
    evaluator_contract: CompiledEvaluatorContract | None
    dataset_builder_contract: CompiledDatasetBuilderContract
    input_normalization_contract: CompiledInputNormalizationContract
    representation_contract: CompiledRepresentationContract


def build_artifact_training_spec(
    config: TrainConfig,
    *,
    corpus: CorpusRootHandle,
    artifact: ArtifactRootHandle,
    corpus_manifest: DatasetManifest | None = None,
) -> TrainingSpec:
    return _build_training_spec(
        config,
        corpus=corpus,
        artifact_id=artifact.artifact_id,
        variant=artifact.variant,
        study=config.study if artifact.variant is ArtifactVariant.TUNED else None,
        study_id=artifact.study_id,
        corpus_manifest=corpus_manifest,
    )


def build_trial_training_spec(
    config: TuneConfig,
    *,
    corpus: CorpusRootHandle,
    study: StudyRootHandle,
    corpus_manifest: DatasetManifest | None = None,
) -> TrainingSpec:
    return _build_training_spec(
        config,
        corpus=corpus,
        artifact_id=study.study_id,
        variant=ArtifactVariant.TUNED,
        study=config.study,
        study_id=study.study_id,
        corpus_manifest=corpus_manifest,
    )


def _build_training_spec(
    config: TrainConfig | TuneConfig,
    *,
    corpus: CorpusRootHandle,
    artifact_id: str,
    variant: ArtifactVariant,
    study: StudyConfig | None,
    study_id: str | None,
    corpus_manifest: DatasetManifest | None,
) -> TrainingSpec:
    chain = (
        ChainSpec(name=corpus_manifest.chain.name, runtime=corpus_manifest.chain.runtime)
        if corpus_manifest is not None
        else config.chain
    )
    context = compile_training_context(
        config,
        chain_runtime=None if corpus_manifest is None else corpus_manifest.chain.runtime,
    )
    return TrainingSpec(
        chain=chain,
        dataset_id=corpus.dataset_id,
        dataset_name=(
            corpus.dataset_name if corpus_manifest is None else corpus_manifest.dataset.name
        ),
        artifact_id=artifact_id,
        problem=config.problem,
        dataset_builder=config.dataset_builder,
        dataset_builder_contract=context.dataset_builder_contract,
        feature_contract=context.feature_contract,
        problem_contract=context.problem_contract,
        features=config.features,
        prediction=config.prediction,
        objective=config.objective,
        evaluation=config.evaluation,
        prediction_contract=context.prediction_contract,
        objective_runtime=context.objective_runtime,
        input_normalization_contract=context.input_normalization_contract,
        representation_contract=context.representation_contract,
        model=config.model,
        variant=variant,
        study=study if variant is ArtifactVariant.TUNED else None,
        study_id=study_id if variant is ArtifactVariant.TUNED else None,
        split=config.split,
        training=config.training,
    )


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
    evaluator_contract = (
        None if config.evaluation is None else compile_evaluator_contract(config.evaluation)
    )
    objective_runtime = compile_objective_runtime(
        config.objective,
        evaluator_contract=evaluator_contract,
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
        evaluator_contract=evaluator_contract,
        dataset_builder_contract=compile_dataset_builder_contract(config.dataset_builder),
        input_normalization_contract=compile_input_normalization_contract(
            config.training.input_normalization
        ),
        representation_contract=compile_representation_contract(),
    )


def run_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
    on_prepare_complete: Callable[[PreparedTrainingDataset], None] | None = None,
    on_fit_start: Callable[[], None] | None = None,
    on_epoch_end: EpochEndCallback | None = None,
    on_early_stop: EarlyStopCallback | None = None,
) -> TrainingRunResult:
    blocks = load_block_frame(history_block_path)
    prepared = spec.dataset_builder_contract.prepare_training_dataset(
        blocks,
        facts=TrainingDatasetPreparationFacts(split=spec.split),
        context=TrainingDatasetPreparationContext(
            feature_contract=spec.feature_contract,
            problem_contract=spec.problem_contract,
            input_normalization_contract=spec.input_normalization_contract,
        ),
    )
    if on_prepare_complete is not None:
        on_prepare_complete(prepared)
    model = build_model(
        prepared.n_features,
        spec.prediction_contract.build_output_spec(prepared.temporal_capability.action_width),
        spec.model,
    )
    if on_fit_start is not None:
        on_fit_start()
    training_result = run_training_fit(
        TrainingFitSpec(
            model=model,
            prediction_contract=spec.prediction_contract,
            objective_runtime=spec.objective_runtime,
            representation_contract=spec.representation_contract,
            prepared=prepared,
            training_config=spec.training,
        ),
        callbacks=TrainingCallbacks(
            on_epoch_end=on_epoch_end,
            on_early_stop=on_early_stop,
        ),
    )
    return TrainingRunResult(
        model=model,
        prepared=prepared,
        training_result=training_result,
    )
