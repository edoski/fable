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
    TaskSpec,
    TrainingConfig,
)
from ..core.console import NullReporter, Reporter
from ..data.datasets import (
    DatasetSplitIndices,
    IntVector,
    TemporalDatasetStore,
    build_temporal_store,
    chronological_split_indices,
    filter_sample_indices_by_timestamp_window,
    tail_sample_indices,
)
from ..data.io import load_block_frame
from ..data.normalization import ScalerStats, fit_standard_scaler, transform_feature_matrix
from ..features import FeatureSelection, build_feature_table, make_feature_selection
from ..planning.contracts import ResolvedTaskContract
from ..planning.geometry import DelayWindow
from .models import TemporalModel
from .registry import build_model
from .training import TrainingResult, train_model


@dataclass(slots=True)
class TrainingSpec:
    chain: ChainSpec
    dataset_id: str
    dataset_name: str
    artifact_id: str
    task: TaskSpec
    contract: ResolvedTaskContract
    feature_set: FeatureSetConfig
    model: ModelConfig
    split: SplitConfig
    training: TrainingConfig
    variant: ArtifactVariant = ArtifactVariant.BASELINE
    study: StudyConfig | None = None
    study_id: str | None = None


@dataclass(slots=True)
class PreparedTrainingDataset:
    n_rows_available: int
    n_rows_used: int
    sample_count: int
    feature_set_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    store: TemporalDatasetStore
    split_indices: DatasetSplitIndices
    scaler: ScalerStats

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
    feature_set_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    store: TemporalDatasetStore
    sample_indices: IntVector

    @property
    def n_features(self) -> int:
        return self.store.n_features


@dataclass(slots=True)
class TrainingRunResult:
    model: TemporalModel
    prepared: PreparedTrainingDataset
    training_result: TrainingResult


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


def _selected_row_span(store: TemporalDatasetStore, sample_indices: IntVector) -> tuple[int, int]:
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
    selection = make_feature_selection(
        feature_set_id=spec.feature_set.id,
        feature_names=tuple(spec.feature_set.outputs),
    )
    sorted_blocks = blocks.sort("block_number")
    if sorted_blocks.height == 0:
        raise ValueError("Training dataset is empty")
    feature_table = build_feature_table(sorted_blocks, selection=selection)
    if feature_table.feature_history_seconds != spec.contract.feature_history_seconds:
        raise ValueError(
            "Resolved feature history does not match the current feature graph: "
            f"expected {spec.contract.feature_history_seconds}, "
            f"got {feature_table.feature_history_seconds}"
        )
    capability_window = spec.contract.capability_window
    store = build_temporal_store(
        feature_table,
        window=capability_window,
    )
    selected_sample_indices = tail_sample_indices(store, sample_count=spec.task.sample_count)
    split_positions = chronological_split_indices(spec.task.sample_count, spec.split)
    split_indices = DatasetSplitIndices(
        train=selected_sample_indices[split_positions.train],
        validation=selected_sample_indices[split_positions.validation],
        test=selected_sample_indices[split_positions.test],
    )
    scaler = fit_standard_scaler(
        store.feature_matrix,
        context_start_rows=store.context_start_rows,
        anchor_rows=store.anchor_rows,
        sample_indices=split_indices.train,
    )
    scaled_store = replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )
    used_start, used_end = _selected_row_span(store, selected_sample_indices)
    return PreparedTrainingDataset(
        n_rows_available=sorted_blocks.height,
        n_rows_used=used_end - used_start,
        sample_count=spec.task.sample_count,
        feature_set_id=feature_table.feature_set_id,
        feature_names=feature_table.feature_names,
        feature_graph_fingerprint=feature_table.feature_graph_fingerprint,
        store=scaled_store,
        split_indices=split_indices,
        scaler=scaler,
    )


def prepare_inference_dataset(
    history_blocks: pl.DataFrame,
    evaluation_blocks: pl.DataFrame,
    *,
    selection: FeatureSelection,
    window: DelayWindow,
    scaler: ScalerStats,
    max_candidate_slots: int,
    window_start_timestamp: int,
    window_end_timestamp: int,
) -> PreparedInferenceDataset:
    sorted_history_blocks = history_blocks.sort("block_number")
    if sorted_history_blocks.height == 0:
        raise ValueError("History dataset is empty")
    combined_blocks = pl.concat([sorted_history_blocks, evaluation_blocks.sort("block_number")])
    feature_table = build_feature_table(combined_blocks, selection=selection)
    store = build_temporal_store(
        feature_table,
        window=window,
        max_candidate_slots=max_candidate_slots,
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
        n_history_rows=history_blocks.height,
        n_evaluation_rows=evaluation_blocks.height,
        sample_count=int(sample_indices.shape[0]),
        feature_set_id=feature_table.feature_set_id,
        feature_names=feature_table.feature_names,
        feature_graph_fingerprint=feature_table.feature_graph_fingerprint,
        store=scaled_store,
        sample_indices=sample_indices,
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
    model = build_model(prepared.n_features, prepared.max_candidate_slots, spec.model)
    active_reporters.build.finish_task(build_task, message=spec.model.id)
    training_result = train_model(
        model,
        model_config=spec.model,
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
    )
