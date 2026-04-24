"""Current production dataset-preparation path."""

from __future__ import annotations

from dataclasses import replace

import polars as pl

from ...core.errors import ConfigResolutionError
from ...temporal.problem_store import (
    DatasetSplitIndices,
    chronological_split_indices,
    filter_sample_indices_by_timestamp_window,
    tail_sample_indices,
)
from ...temporal.scaling import transform_feature_matrix
from ..pipeline import (
    InferencePreparationSpec,
    PreparedInferenceDataset,
    PreparedTrainingDataset,
    TrainingSpec,
    selected_row_span,
)
from .base import (
    CompiledDatasetBuilderContract,
    StandardTemporalDatasetBuilderConfig,
    compiler_runtime_metadata_from_builder_payload,
    standard_temporal_runtime_metadata,
)


def prepare_training_dataset(blocks: pl.DataFrame, spec: TrainingSpec) -> PreparedTrainingDataset:
    sorted_blocks = blocks.sort("block_number")
    if sorted_blocks.height == 0:
        raise ValueError("Training dataset is empty")
    feature_table = spec.feature_contract.build_table(sorted_blocks)
    if feature_table.feature_prerequisites != spec.contract.feature_prerequisites:
        raise ValueError(
            "Resolved feature prerequisites do not match the current feature graph: "
            f"expected {spec.contract.feature_prerequisites.model_dump(mode='json')}, "
            f"got {feature_table.feature_prerequisites.model_dump(mode='json')}"
        )
    store, compiler_runtime_metadata = spec.contract.build_capability_store(feature_table)
    selected_sample_indices = tail_sample_indices(store, sample_count=spec.problem.sample_count)
    split_positions = chronological_split_indices(int(selected_sample_indices.shape[0]), spec.split)
    split_indices = DatasetSplitIndices(
        train=selected_sample_indices[split_positions.train],
        validation=selected_sample_indices[split_positions.validation],
        test=selected_sample_indices[split_positions.test],
    )
    scaler = spec.input_normalization_contract.fit_scaler(
        store.feature_matrix,
        context_start_rows=store.context_start_rows,
        anchor_rows=store.anchor_rows,
        sample_indices=split_indices.train,
    )
    scaled_store = replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )
    used_start, used_end = selected_row_span(store, selected_sample_indices)
    return PreparedTrainingDataset(
        n_rows_available=sorted_blocks.height,
        n_rows_used=used_end - used_start,
        sample_count=int(selected_sample_indices.shape[0]),
        feature=spec.feature_contract.semantics,
        realization_policy=spec.contract.realization_policy,
        store=scaled_store,
        split_indices=split_indices,
        scaler=scaler,
        builder_runtime_metadata=standard_temporal_runtime_metadata(
            compiler_id=spec.contract.compiler_id,
            compiler_runtime_metadata=compiler_runtime_metadata
        ),
    )


def prepare_inference_dataset(
    history_blocks: pl.DataFrame,
    evaluation_blocks: pl.DataFrame,
    spec: InferencePreparationSpec,
) -> PreparedInferenceDataset:
    sorted_history_blocks = history_blocks.sort("block_number")
    if sorted_history_blocks.height == 0:
        raise ValueError("History dataset is empty")
    combined_blocks = pl.concat([sorted_history_blocks, evaluation_blocks.sort("block_number")])
    feature_table = spec.feature_contract.build_table(combined_blocks)
    store = spec.contract.build_delay_store(
        feature_table,
        spec.delay_seconds,
        compiler_runtime_metadata=compiler_runtime_metadata_from_builder_payload(
            spec.builder_runtime_metadata,
            compiler_id=spec.contract.compiler_id,
        ),
        max_candidate_slots=spec.max_candidate_slots,
    )
    sample_indices = filter_sample_indices_by_timestamp_window(
        store,
        start_timestamp=spec.window_start_timestamp,
        end_timestamp=spec.window_end_timestamp,
    )
    if sample_indices.size == 0:
        raise ValueError("Evaluation dataset produced no valid inference examples")
    scaled_store = replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, spec.scaler),
    )
    return PreparedInferenceDataset(
        n_history_rows=history_blocks.height,
        n_evaluation_rows=evaluation_blocks.height,
        sample_count=int(sample_indices.shape[0]),
        feature=spec.feature_contract.semantics,
        realization_policy=spec.contract.realization_policy,
        store=scaled_store,
        sample_indices=sample_indices,
    )


def compile_dataset_builder(
    config: StandardTemporalDatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    if config.id != "standard_temporal":
        raise ConfigResolutionError("dataset_builder.id must be standard_temporal")
    return CompiledDatasetBuilderContract(
        dataset_builder_id="standard_temporal",
        prepare_training_fn=prepare_training_dataset,
        prepare_inference_fn=prepare_inference_dataset,
    )
