from __future__ import annotations

from typing import cast

import polars as pl

from spice.config import TrainConfig, WorkflowTask
from spice.modeling.pipeline import build_training_spec, prepare_training_dataset
from tests.dataset_helpers import make_history_rows


def test_professor_dataset_builder_prepares_seq_len_without_builder_owned_class_state(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            preset="icdcs_2026_professor",
            override={
                "problem": {
                    "id": "test_professor_problem",
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
        spec=build_training_spec(config),
    )

    assert config.dataset_builder.id == "professor_temporal"
    assert int(prepared.builder_runtime_metadata["seq_len"]) >= 64
    assert float(prepared.builder_runtime_metadata["median_dt_seconds"]) > 0.0
    assert prepared.max_candidate_slots == prepared.store.max_candidate_slots
    assert prepared.split_indices.train.size > 0
    assert prepared.split_indices.validation.size > 0
    assert prepared.split_indices.test.size > 0
