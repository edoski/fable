"""Training and inference dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..config import (
    ArtifactVariant,
    ChainSpec,
    DatasetBuilderConfig,
    FeatureSetConfig,
    ModelConfig,
    PredictionConfig,
    ProblemSpec,
    SplitConfig,
    StudyConfig,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
)
from ..core.reporting import NullReporter, Reporter
from ..corpus.io import load_block_frame
from ..features import (
    CompiledFeatureContract,
    compile_feature_contract,
)
from ..modeling.dataset_builders import (
    CompiledDatasetBuilderContract,
    compile_dataset_builder_contract,
)
from ..prediction import CompiledPredictionContract, compile_prediction_contract
from ..semantics import FeatureSemantics
from ..storage.layout import resolve_workflow_paths
from ..temporal.contracts import (
    CompiledProblemContract,
    ProblemRuntimeMetadata,
    compile_problem_contract,
)
from ..temporal.input_normalization import (
    CompiledInputNormalizationContract,
    compile_input_normalization_contract,
)
from ..temporal.problem_store import (
    CompiledProblemStore,
    DatasetSplitIndices,
    IntVector,
)
from ..temporal.scaling import ScalerStats
from ._runtime import CompiledRepresentationContract
from .families.registry import build_model, resolve_model_representation_id
from .models import TemporalModel
from .representations import compile_representation_contract
from .training import TrainingResult, train_model


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
    contract: CompiledProblemContract
    feature_set: FeatureSetConfig
    prediction: PredictionConfig
    prediction_contract: CompiledPredictionContract
    input_normalization_contract: CompiledInputNormalizationContract
    representation_contract: CompiledRepresentationContract
    model: ModelConfig
    split: SplitConfig
    training: TrainingConfig
    variant: ArtifactVariant = ArtifactVariant.BASELINE
    study: StudyConfig | None = None
    study_id: str | None = None


@dataclass(slots=True)
class InferencePreparationSpec:
    feature_contract: CompiledFeatureContract
    contract: CompiledProblemContract
    delay_seconds: int
    builder_runtime_metadata: ProblemRuntimeMetadata
    scaler: ScalerStats
    max_candidate_slots: int
    window_start_timestamp: int
    window_end_timestamp: int


def build_training_spec(config: TrainConfig | TuneConfig) -> TrainingSpec:
    paths = resolve_workflow_paths(config)
    variant = ArtifactVariant.TUNED if isinstance(config, TuneConfig) else config.artifact.variant
    feature_contract = compile_feature_contract(feature_set=config.feature_set)
    contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
        chain_runtime=config.chain.runtime,
    )
    prediction_contract = compile_prediction_contract(
        prediction_id=config.prediction.id,
        family_config=config.prediction.family,
    )
    dataset_builder_contract = compile_dataset_builder_contract(config.dataset_builder)
    input_normalization_contract = compile_input_normalization_contract(
        config.training.input_normalization
    )
    if config.workflow.value not in prediction_contract.supported_workflows:
        raise ValueError(
            f"prediction family {prediction_contract.prediction_family_id} does not support "
            f"{config.workflow.value}"
        )
    return TrainingSpec(
        chain=config.chain,
        dataset_id=paths.corpus_id,
        dataset_name=config.dataset.name,
        artifact_id=(
            paths.artifact_id
            if paths.artifact_id is not None
            else paths.study_id or "trial"
        ),
        problem=config.problem,
        dataset_builder=config.dataset_builder,
        dataset_builder_contract=dataset_builder_contract,
        feature_contract=feature_contract,
        contract=contract,
        feature_set=config.feature_set,
        prediction=config.prediction,
        prediction_contract=prediction_contract,
        input_normalization_contract=input_normalization_contract,
        representation_contract=compile_representation_contract(
            resolve_model_representation_id(config.model)
        ),
        model=config.model,
        variant=variant,
        study=config.study if variant is ArtifactVariant.TUNED else None,
        study_id=paths.study_id if variant is ArtifactVariant.TUNED else None,
        split=config.split,
        training=config.training,
    )


@dataclass(slots=True)
class PreparedTrainingDataset:
    n_rows_available: int
    n_rows_used: int
    sample_count: int
    feature: FeatureSemantics
    store: CompiledProblemStore
    split_indices: DatasetSplitIndices
    scaler: ScalerStats
    builder_runtime_metadata: ProblemRuntimeMetadata

    @property
    def n_features(self) -> int:
        return self.store.n_features

    @property
    def max_candidate_slots(self) -> int:
        return self.store.max_candidate_slots


@dataclass(slots=True)
class PreparedInferenceDataset:
    n_history_rows: int
    n_evaluation_rows: int
    sample_count: int
    feature: FeatureSemantics
    store: CompiledProblemStore
    sample_indices: IntVector

    @property
    def n_features(self) -> int:
        return self.store.n_features


@dataclass(slots=True)
class TrainingRunResult:
    model: TemporalModel
    prepared: PreparedTrainingDataset
    training_result: TrainingResult
    prediction_training_state: object | None


@dataclass(slots=True)
class TrainingStageReporters:
    load: Reporter
    prepare: Reporter
    build: Reporter
    fit: Reporter
    evaluate: Reporter

    @classmethod
    def shared(cls, reporter: Reporter) -> TrainingStageReporters:
        return cls(
            load=reporter,
            prepare=reporter,
            build=reporter,
            fit=reporter,
            evaluate=reporter,
        )


def selected_row_span(store: CompiledProblemStore, sample_indices: IntVector) -> tuple[int, int]:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    first_sample = int(sample_indices[0])
    last_sample = int(sample_indices[-1])
    start = int(store.context_start_rows[first_sample])
    end = int(store.candidate_end_rows[last_sample])
    return start, end


def prepare_training_dataset(
    blocks: pl.DataFrame,
    *,
    spec: TrainingSpec,
) -> PreparedTrainingDataset:
    return spec.dataset_builder_contract.prepare_training_dataset(blocks, spec=spec)


def prepare_inference_dataset(
    history_blocks: pl.DataFrame,
    evaluation_blocks: pl.DataFrame,
    *,
    dataset_builder_contract: CompiledDatasetBuilderContract,
    feature_contract: CompiledFeatureContract,
    contract: CompiledProblemContract,
    delay_seconds: int,
    builder_runtime_metadata: ProblemRuntimeMetadata,
    scaler: ScalerStats,
    max_candidate_slots: int,
    window_start_timestamp: int,
    window_end_timestamp: int,
) -> PreparedInferenceDataset:
    return dataset_builder_contract.prepare_inference_dataset(
        history_blocks,
        evaluation_blocks,
        spec=InferencePreparationSpec(
            feature_contract=feature_contract,
            contract=contract,
            delay_seconds=delay_seconds,
            builder_runtime_metadata=builder_runtime_metadata,
            scaler=scaler,
            max_candidate_slots=max_candidate_slots,
            window_start_timestamp=window_start_timestamp,
            window_end_timestamp=window_end_timestamp,
        ),
    )


def run_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
    artifact_dir: Path,
    stage_reporters: TrainingStageReporters | None = None,
    reporter: Reporter | None = None,
) -> TrainingRunResult:
    reporter = reporter or NullReporter()
    active_reporters = stage_reporters or TrainingStageReporters.shared(reporter)
    load_task = active_reporters.load.start_task("load history dataset")
    blocks = load_block_frame(history_block_path)
    active_reporters.load.finish_task(load_task, message=str(history_block_path))
    prepare_task = active_reporters.prepare.start_task("prepare training dataset")
    prepared = prepare_training_dataset(blocks, spec=spec)
    active_reporters.prepare.finish_task(
        prepare_task,
        message=f"rows={prepared.n_rows_used} samples={prepared.sample_count}",
    )
    build_task = active_reporters.build.start_task("build model")
    model = build_model(
        prepared.n_features,
        spec.prediction_contract.build_output_spec(prepared.max_candidate_slots),
        spec.model,
    )
    active_reporters.build.finish_task(build_task, message=spec.model.id)
    training_result = train_model(
        model,
        model_config=spec.model,
        prediction_contract=spec.prediction_contract,
        representation_contract=spec.representation_contract,
        store=prepared.store,
        train_sample_indices=prepared.split_indices.train,
        validation_sample_indices=prepared.split_indices.validation,
        training_config=spec.training,
        artifact_dir=artifact_dir,
        reporter=active_reporters.fit,
    )
    return TrainingRunResult(
        model=model,
        prepared=prepared,
        training_result=training_result,
        prediction_training_state=training_result.prediction_training_state,
    )
