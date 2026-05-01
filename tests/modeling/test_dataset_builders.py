from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl

from spice.config import TrainConfig, WorkflowTask
from spice.modeling.dataset_builders import (
    ArtifactInferencePreparationSpec,
    FixedSequenceTemporalBuilderRuntimeMetadata,
    TrainingDatasetPreparationSpec,
)
from spice.modeling.pipeline import build_artifact_training_spec
from tests.dataset_helpers import (
    make_block_rows,
    required_dataset_blocks,
    synthetic_block_interval_seconds,
)
from tests.root_handle_helpers import artifact_handle, corpus_handle


def _make_history_rows_with_margin(config: TrainConfig, *, extra_rows: int = 32):
    block_interval_seconds = synthetic_block_interval_seconds(config.chain.name)
    count = required_dataset_blocks(config) + extra_rows
    return make_block_rows(
        count,
        start_block=1,
        start_timestamp=config.evaluation_window_start_timestamp - count * block_interval_seconds,
        chain_id=config.chain.runtime.chain_id,
        block_interval_seconds=block_interval_seconds,
    )


def _make_rows_with_tail_cadence_shift(
    count: int,
    *,
    fast_start_offset: int,
    chain_id: int,
) -> list[dict[str, int]]:
    rows = make_block_rows(
        count,
        start_block=1,
        start_timestamp=0,
        chain_id=chain_id,
        block_interval_seconds=12,
    )
    timestamp = 0
    for offset, row in enumerate(rows):
        row["timestamp"] = timestamp
        timestamp += 1 if offset >= fast_start_offset else 12
    return rows


def _spec(config: TrainConfig):
    assert config.dataset_id is not None
    corpus = corpus_handle(
        config.storage.root,
        chain_name=config.chain.name,
        dataset_id=config.dataset_id,
        dataset_name=config.dataset.name,
    )
    return build_artifact_training_spec(
        config,
        corpus=corpus,
        artifact=artifact_handle(config.storage.root, corpus=corpus),
    )


def _training_prep_spec(spec) -> TrainingDatasetPreparationSpec:
    return TrainingDatasetPreparationSpec(
        dataset_builder=spec.dataset_builder,
        feature_contract=spec.feature_contract,
        problem_contract=spec.problem_contract,
        sample_count=spec.problem.sample_count,
        lookback_seconds=spec.problem.lookback_seconds,
        split=spec.split,
        input_normalization_contract=spec.input_normalization_contract,
    )


def _prepare_training_dataset(rows, *, spec):
    return spec.dataset_builder_contract.prepare_training_dataset(
        pl.DataFrame(rows),
        spec=_training_prep_spec(spec),
    )


def test_fixed_context_dataset_builder_prepares_seq_len_without_builder_owned_class_state(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override={
                "problem": {
                    "id": "test_fixed_context_problem",
                    "lookback_seconds": 600,
                    "sample_count": 4096,
                    "max_delay_seconds": 36,
                    "compiler": {
                        "id": "observed_time_window",
                        "slot_spacing": {"id": "nominal"},
                    },
                    "execution_policy": {
                        "id": "strict_deadline_miss",
                    },
                }
            },
        ),
    )
    spec = _spec(config)
    prepared = _prepare_training_dataset(_make_history_rows_with_margin(config), spec=spec)

    assert config.dataset_builder.id == "fixed_sequence_temporal"
    assert isinstance(
        prepared.builder_runtime_metadata,
        FixedSequenceTemporalBuilderRuntimeMetadata,
    )
    assert prepared.builder_runtime_metadata.sequence_length >= 64
    assert prepared.builder_runtime_metadata.median_dt_seconds > 0.0
    assert prepared.max_candidate_slots == prepared.store.max_candidate_slots
    assert prepared.sample_count == config.problem.sample_count
    assert prepared.store.n_samples > prepared.sample_count
    assert prepared.split_indices.train.size > 0
    assert prepared.split_indices.validation.size > 0
    assert prepared.split_indices.test.size > 0


def test_fixed_context_dataset_builder_keeps_candidate_window_arrays_aligned_after_seq_trim(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override={
                "problem": {
                    "id": "test_fixed_context_problem",
                    "lookback_seconds": 600,
                    "sample_count": 4096,
                    "max_delay_seconds": 36,
                    "compiler": {
                        "id": "observed_time_window",
                        "slot_spacing": {"id": "nominal"},
                    },
                    "execution_policy": {
                        "id": "strict_deadline_miss",
                    },
                }
            },
        ),
    )

    spec = _spec(config)
    prepared = _prepare_training_dataset(_make_history_rows_with_margin(config), spec=spec)

    assert prepared.store.anchor_rows.shape == prepared.store.candidate_start_rows.shape
    assert prepared.store.anchor_rows.shape == prepared.store.candidate_end_rows.shape
    np.testing.assert_array_equal(prepared.store.anchor_rows, prepared.store.candidate_start_rows)
    np.testing.assert_array_less(
        prepared.store.candidate_start_rows,
        prepared.store.candidate_end_rows,
    )


def test_fixed_sequence_length_calibration_uses_train_sample_rows_only(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override={
                "dataset_builder": {
                    "id": "fixed_sequence_temporal",
                    "min_sequence_length": 1,
                    "max_sequence_length": 200,
                },
                "features": {
                    "id": "core_fee_dynamics",
                    "outputs": ["log_base_fee_per_gas"],
                },
                "problem": {
                    "id": "test_fixed_context_problem",
                    "lookback_seconds": 120,
                    "sample_count": 120,
                    "max_delay_seconds": 600,
                    "compiler": {
                        "id": "observed_time_window",
                        "slot_spacing": {"id": "nominal"},
                    },
                    "execution_policy": {
                        "id": "strict_deadline_miss",
                    },
                },
            },
        ),
    )
    rows = _make_rows_with_tail_cadence_shift(
        420,
        fast_start_offset=320,
        chain_id=config.chain.runtime.chain_id,
    )

    spec = _spec(config)
    prepared = _prepare_training_dataset(rows, spec=spec)

    assert prepared.builder_runtime_metadata.median_dt_seconds == 12.0
    assert prepared.builder_runtime_metadata.sequence_length == 10


def test_fixed_sequence_inference_prep_reconstructs_artifact_runtime_state(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override={
                "dataset_builder": {
                    "id": "fixed_sequence_temporal",
                    "min_sequence_length": 4,
                    "max_sequence_length": 64,
                },
                "features": {
                    "id": "core_fee_dynamics",
                    "outputs": ["log_base_fee_per_gas"],
                },
                "problem": {
                    "id": "test_fixed_context_problem",
                    "lookback_seconds": 120,
                    "sample_count": 120,
                    "max_delay_seconds": 36,
                    "compiler": {
                        "id": "observed_time_window",
                        "slot_spacing": {"id": "nominal"},
                    },
                    "execution_policy": {
                        "id": "strict_deadline_miss",
                    },
                },
            },
        ),
    )
    spec = _spec(config)
    history_rows = make_block_rows(
        240,
        start_block=1,
        start_timestamp=0,
        chain_id=config.chain.runtime.chain_id,
        block_interval_seconds=12,
    )
    trained = _prepare_training_dataset(history_rows, spec=spec)
    evaluation_rows = make_block_rows(
        40,
        start_block=241,
        start_timestamp=240 * 12,
        chain_id=config.chain.runtime.chain_id,
        block_interval_seconds=12,
    )
    evaluation_start = evaluation_rows[0]["timestamp"]
    evaluation_end = evaluation_rows[-1]["timestamp"]

    prepared = spec.dataset_builder_contract.prepare_inference_dataset(
        pl.DataFrame(history_rows),
        pl.DataFrame(evaluation_rows),
        spec=ArtifactInferencePreparationSpec(
            feature_contract=spec.feature_contract,
            problem_contract=spec.problem_contract,
            delay_seconds=12,
            builder_runtime_metadata=trained.builder_runtime_metadata,
            scaler=trained.scaler,
            max_candidate_slots=trained.max_candidate_slots,
            evaluation_start_timestamp=evaluation_start,
            evaluation_end_timestamp=evaluation_end,
        ),
    )

    sample_timestamps = prepared.store.sample_timestamps(prepared.sample_indices)
    assert prepared.n_history_rows == len(history_rows)
    assert prepared.n_evaluation_rows == len(evaluation_rows)
    assert prepared.sample_count > 0
    assert prepared.store.max_candidate_slots == trained.max_candidate_slots
    assert int(sample_timestamps.min()) >= evaluation_start
    assert int(sample_timestamps.max()) <= evaluation_end
