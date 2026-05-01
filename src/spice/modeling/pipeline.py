"""Training and inference dataset preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config.models import (
    ArtifactVariant,
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
from ..features import CompiledFeatureContract
from ..objectives import ObjectiveConfig
from ..prediction import CompiledPredictionContract
from ..storage.workflow_roots import ArtifactRootHandle, CorpusRootHandle, StudyRootHandle
from ..temporal.contracts import CompiledProblemContract
from ..temporal.input_normalization import CompiledInputNormalizationContract
from ._training_context import compile_training_context
from .dataset_builders import (
    CompiledDatasetBuilderContract,
    DatasetBuilderConfig,
    PreparedTrainingDataset,
    TrainingDatasetPreparationSpec,
)
from .families.base import ModelConfig
from .families.registry import build_model
from .objective_runtime import CompiledObjectiveRuntime
from .representations import CompiledRepresentationContract
from .training_run import TrainingRunResult
from .training_runner import (
    EarlyStopCallback,
    EpochEndCallback,
    TrainingCallbacks,
    TrainingFitSpec,
    run_training_fit,
)


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


def _training_preparation_spec(spec: TrainingSpec) -> TrainingDatasetPreparationSpec:
    return TrainingDatasetPreparationSpec(
        dataset_builder=spec.dataset_builder,
        feature_contract=spec.feature_contract,
        problem_contract=spec.problem_contract,
        sample_count=spec.problem.sample_count,
        lookback_seconds=spec.problem.lookback_seconds,
        split=spec.split,
        input_normalization_contract=spec.input_normalization_contract,
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
        spec=_training_preparation_spec(spec),
    )
    if on_prepare_complete is not None:
        on_prepare_complete(prepared)
    model = build_model(
        prepared.n_features,
        spec.prediction_contract.build_output_spec(prepared.max_candidate_slots),
        spec.model,
    )
    if on_fit_start is not None:
        on_fit_start()
    training_result = run_training_fit(
        TrainingFitSpec(
            model=model,
            model_config=spec.model,
            prediction_contract=spec.prediction_contract,
            objective_runtime=spec.objective_runtime,
            execution_policy=prepared.execution_policy,
            representation_contract=spec.representation_contract,
            store=prepared.store,
            train_sample_indices=prepared.split_indices.train,
            validation_sample_indices=prepared.split_indices.validation,
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
        prediction_training_state=training_result.prediction_training_state,
    )
