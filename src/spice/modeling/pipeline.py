"""Training and inference dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl

from ..config import (
    ArtifactVariant,
    ChainSpec,
    FeatureSetConfig,
    ModelConfig,
    SplitConfig,
    StudyConfig,
    TrainingConfig,
)
from ..core.console import NullReporter, Reporter
from ..data.datasets import (
    DatasetGeometry,
    DatasetSplitIndices,
    IntVector,
    TemporalDatasetStore,
    build_temporal_store,
    chronological_split_indices,
    filter_sample_indices_by_timestamp_window,
    history_context_slice,
    trim_history_for_sample_count,
)
from ..data.io import load_block_frame
from ..data.normalization import ScalerStats, fit_standard_scaler, transform_feature_matrix
from ..features import FeatureSelection, build_feature_table, feature_warmup_blocks
from ..planning.geometry import derive_dataset_geometry, minimum_history_context_blocks
from .evaluation import EpochMetrics
from .models import TemporalModel
from .registry import build_model
from .torch_datasets import build_class_weights
from .training import TrainingResult, evaluate_model, train_model


@dataclass(slots=True)
class TrainingSpec:
    chain: ChainSpec
    dataset_id: str
    feature_set: FeatureSetConfig
    model: ModelConfig
    max_delay_seconds: int
    lookback_seconds: int
    history_context_blocks: int
    sample_count: int
    split: SplitConfig
    training: TrainingConfig
    variant: ArtifactVariant = ArtifactVariant.BASELINE
    study: StudyConfig | None = None


@dataclass(slots=True)
class PreparedTrainingDataset:
    n_blocks_available: int
    n_blocks_used: int
    sample_count: int
    feature_set_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    store: TemporalDatasetStore
    split_indices: DatasetSplitIndices
    scaler: ScalerStats
    geometry: DatasetGeometry

    @property
    def n_features(self) -> int:
        return self.store.n_features

    @property
    def action_count(self) -> int:
        return self.store.action_count


@dataclass(slots=True)
class PreparedInferenceDataset:
    n_history_context_blocks: int
    n_evaluation_blocks: int
    sample_count: int
    feature_set_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    store: TemporalDatasetStore
    sample_indices: IntVector
    geometry: DatasetGeometry

    @property
    def n_features(self) -> int:
        return self.store.n_features


@dataclass(slots=True)
class TrainingRunResult:
    model: TemporalModel
    prepared: PreparedTrainingDataset
    training_result: TrainingResult
    test_metrics: EpochMetrics


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


def _slice_frame(frame: pl.DataFrame, selection: slice) -> pl.DataFrame:
    start = 0 if selection.start is None else selection.start
    stop = frame.height if selection.stop is None else selection.stop
    return frame.slice(start, stop - start)


def prepare_training_dataset(
    blocks: pl.DataFrame,
    *,
    spec: TrainingSpec,
) -> PreparedTrainingDataset:
    selection = FeatureSelection(
        feature_set_id=spec.feature_set.id,
        feature_names=tuple(spec.feature_set.outputs),
    )
    minimum_context = minimum_history_context_blocks(
        lookback_seconds=spec.lookback_seconds,
        block_time_seconds=spec.chain.runtime.block_time_seconds,
        feature_warmup_blocks=feature_warmup_blocks(selection.feature_names),
    )
    if minimum_context > spec.history_context_blocks:
        raise ValueError(
            "Configured dataset.history_context_blocks is too small for the selected feature set: "
            f"need at least {minimum_context}, got {spec.history_context_blocks}"
        )
    geometry = derive_dataset_geometry(
        lookback_seconds=spec.lookback_seconds,
        max_delay_seconds=spec.max_delay_seconds,
        block_time_seconds=spec.chain.runtime.block_time_seconds,
        history_context_blocks=minimum_context,
    )
    trimmed_blocks = _slice_frame(
        blocks.sort("block_number"),
        trim_history_for_sample_count(
            blocks.height,
            sample_count=spec.sample_count,
            geometry=geometry,
        ),
    )
    feature_table = build_feature_table(trimmed_blocks, selection=selection)
    store = build_temporal_store(
        feature_table,
        lookback_steps=geometry.lookback_steps,
        action_count=geometry.action_count,
    )
    if store.n_samples != spec.sample_count:
        raise RuntimeError(
            "Training dataset preparation produced an unexpected number of samples; "
            f"expected {spec.sample_count}, got {store.n_samples}"
        )
    split_indices = chronological_split_indices(store.n_samples, spec.split)
    scaler = fit_standard_scaler(
        store.feature_matrix,
        sample_row_indices=store.sample_row_indices,
        sample_indices=split_indices.train,
        lookback_steps=geometry.lookback_steps,
    )
    scaled_store = replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )
    return PreparedTrainingDataset(
        n_blocks_available=blocks.height,
        n_blocks_used=trimmed_blocks.height,
        sample_count=store.n_samples,
        feature_set_id=feature_table.feature_set_id,
        feature_names=feature_table.feature_names,
        feature_graph_fingerprint=feature_table.feature_graph_fingerprint,
        store=scaled_store,
        split_indices=split_indices,
        scaler=scaler,
        geometry=geometry,
    )


def prepare_inference_dataset(
    history_blocks: pl.DataFrame,
    evaluation_blocks: pl.DataFrame,
    *,
    selection: FeatureSelection,
    geometry: DatasetGeometry,
    scaler: ScalerStats,
    window_start_timestamp: int,
    window_end_timestamp: int,
) -> PreparedInferenceDataset:
    context_blocks = _slice_frame(
        history_blocks.sort("block_number"),
        history_context_slice(history_blocks.height, geometry=geometry),
    )
    combined_blocks = pl.concat([context_blocks, evaluation_blocks.sort("block_number")])
    feature_table = build_feature_table(combined_blocks, selection=selection)
    store = build_temporal_store(
        feature_table,
        lookback_steps=geometry.lookback_steps,
        action_count=geometry.action_count,
    )
    sample_indices = filter_sample_indices_by_timestamp_window(
        store,
        start_timestamp=window_start_timestamp,
        end_timestamp=window_end_timestamp,
    )
    if sample_indices.size == 0:
        raise ValueError("Evaluation dataset produced no valid inference examples")

    scaled_store = replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )
    return PreparedInferenceDataset(
        n_history_context_blocks=context_blocks.height,
        n_evaluation_blocks=evaluation_blocks.height,
        sample_count=int(sample_indices.shape[0]),
        feature_set_id=feature_table.feature_set_id,
        feature_names=feature_table.feature_names,
        feature_graph_fingerprint=feature_table.feature_graph_fingerprint,
        store=scaled_store,
        sample_indices=sample_indices,
        geometry=geometry,
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
        message=f"blocks={prepared.n_blocks_used} samples={prepared.sample_count}",
    )
    build_task = active_reporters.build.start_task("build model")
    model = build_model(prepared.n_features, prepared.action_count, spec.model)
    active_reporters.build.finish_task(build_task, message=spec.model.id)
    training_result = train_model(
        model,
        model_config=spec.model,
        store=prepared.store,
        train_sample_indices=prepared.split_indices.train,
        validation_sample_indices=prepared.split_indices.validation,
        lookback_steps=prepared.geometry.lookback_steps,
        training_config=spec.training,
        artifact_dir=artifact_dir,
        reporter=active_reporters.fit,
    )
    class_weights = build_class_weights(
        prepared.store.class_labels,
        prepared.split_indices.train,
        prepared.action_count,
    )
    test_metrics = evaluate_model(
        model,
        store=prepared.store,
        sample_indices=prepared.split_indices.test,
        lookback_steps=prepared.geometry.lookback_steps,
        training_config=spec.training,
        class_weights=class_weights,
        reporter=active_reporters.evaluate,
    )
    return TrainingRunResult(
        model=model,
        prepared=prepared,
        training_result=training_result,
        test_metrics=test_metrics,
    )
