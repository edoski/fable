from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl

from spice.config import TrainConfig, WorkflowTask
from spice.modeling.dataset_builders import FixedContextTemporalBuilderRuntimeMetadata
from spice.modeling.pipeline import build_training_spec, prepare_training_dataset
from spice.storage.layout import resolve_workflow_paths
from tests.dataset_helpers import make_history_rows


def test_fixed_context_dataset_builder_prepares_seq_len_without_builder_owned_class_state(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            surface="same_block_closed",
            override={
                "problem": {
                    "id": "test_fixed_context_problem",
                    "lookback_seconds": 600,
                    "sample_count": 4096,
                    "max_delay_seconds": 36,
                    "compiler": {
                        "id": "estimated_block",
                        "lookback_interval_source": "nominal_chain_runtime",
                        "candidate_interval_source": "nominal_chain_runtime",
                        "calibrated_interval_statistic": "mean",
                    },
                    "realization_policy": {
                        "id": "strict_deadline_miss",
                    },
                }
            },
        ),
    )
    prepared = prepare_training_dataset(
        pl.DataFrame(make_history_rows(config)),
        spec=build_training_spec(config, paths=resolve_workflow_paths(config)),
    )

    assert config.dataset_builder.id == "fixed_context_temporal"
    assert isinstance(prepared.builder_runtime_metadata, FixedContextTemporalBuilderRuntimeMetadata)
    assert prepared.builder_runtime_metadata.sequence_length >= 64
    assert prepared.builder_runtime_metadata.median_dt_seconds > 0.0
    assert prepared.max_candidate_slots == prepared.store.max_candidate_slots
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
            surface="same_block_closed",
            override={
                "problem": {
                    "id": "test_fixed_context_problem",
                    "lookback_seconds": 600,
                    "sample_count": 4096,
                    "max_delay_seconds": 36,
                    "compiler": {
                        "id": "estimated_block",
                        "lookback_interval_source": "nominal_chain_runtime",
                        "candidate_interval_source": "nominal_chain_runtime",
                        "calibrated_interval_statistic": "mean",
                    },
                    "realization_policy": {
                        "id": "strict_deadline_miss",
                    },
                }
            },
        ),
    )

    prepared = prepare_training_dataset(
        pl.DataFrame(make_history_rows(config)),
        spec=build_training_spec(config, paths=resolve_workflow_paths(config)),
    )

    assert prepared.store.anchor_rows.shape == prepared.store.candidate_start_rows.shape
    assert prepared.store.anchor_rows.shape == prepared.store.candidate_end_rows.shape
    np.testing.assert_array_equal(prepared.store.anchor_rows, prepared.store.candidate_start_rows)
    np.testing.assert_array_less(
        prepared.store.candidate_start_rows,
        prepared.store.candidate_end_rows,
    )
