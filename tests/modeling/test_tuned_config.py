from __future__ import annotations

import pytest
from optuna.trial import FixedTrial

from spice.config import (
    TrainWorkflowRequest,
    TunedParameterSet,
    WorkflowTask,
    coerce_problem_spec,
    resolve_workflow_config,
)
from spice.modeling.families.transformer import (
    TransformerModelConfig,
    TransformerTunedModelParams,
)
from spice.modeling.tuned_config import coerce_tuning_space_config, sample_tuned_parameters
from spice.modeling.tuning import apply_tuned_parameters


def _transformer_model() -> TransformerModelConfig:
    return TransformerModelConfig(
        d_model=256,
        nhead=4,
        transformer_layers=4,
        feedforward_dim=512,
        head_hidden_dim=256,
        dropout=0.2,
    )


def _problem():
    return coerce_problem_spec(
        {
            "id": "test_problem",
            "lookback_seconds": 900,
            "sample_count": 1024,
            "max_delay_seconds": 36,
            "compiler": {
                "id": "timestamp_future_window",
                "action_interval_estimator": {"id": "nominal"},
            },
            "realization_policy": {"id": "strict_deadline_miss"},
        }
    )


def test_sample_transformer_large_capacity_params_derives_feedforward_dim() -> None:
    tuning_space = coerce_tuning_space_config(
        {
            "training": {
                "learning_rate": [0.0001],
                "weight_decay": [0.001],
                "batch_size": [128],
            },
            "model": {
                "id": "transformer",
                "d_model": [512],
                "nhead": [8],
                "transformer_layers": [6],
                "feedforward_multiplier": [4],
                "head_hidden_dim": [1024],
                "dropout": [0.1],
            },
        },
        model_config=_transformer_model(),
        problem_config=_problem(),
    )
    assert tuning_space is not None

    params = sample_tuned_parameters(
        FixedTrial(
            {
                "training.learning_rate": 0.0001,
                "training.weight_decay": 0.001,
                "training.batch_size": 128,
                "model.d_model": 512,
                "model.nhead": 8,
                "model.transformer_layers": 6,
                "model.feedforward_multiplier": 4,
                "model.head_hidden_dim": 1024,
                "model.dropout": 0.1,
            }
        ),
        tuning_space=tuning_space,
    )

    assert params.training is not None
    assert params.training.batch_size == 128
    assert isinstance(params.model, TransformerTunedModelParams)
    assert params.model.d_model == 512
    assert params.model.nhead == 8
    assert params.model.transformer_layers == 6
    assert params.model.feedforward_dim == 2048
    assert params.model.head_hidden_dim == 1024


def test_apply_transformer_tuned_params_updates_model_config(tmp_path) -> None:
    config = resolve_workflow_config(
        WorkflowTask.TRAIN,
        TrainWorkflowRequest(
            surface="same_block_closed",
            model="transformer",
            variant="baseline",
            storage_root=tmp_path / "outputs",
        ),
    )

    tuned = apply_tuned_parameters(
        config,
        TunedParameterSet(
            model=TransformerTunedModelParams(
                d_model=512,
                nhead=8,
                transformer_layers=6,
                feedforward_dim=2048,
                head_hidden_dim=1024,
            )
        ),
    )

    assert isinstance(tuned.model, TransformerModelConfig)
    assert tuned.model.d_model == 512
    assert tuned.model.nhead == 8
    assert tuned.model.transformer_layers == 6
    assert tuned.model.feedforward_dim == 2048
    assert tuned.model.head_hidden_dim == 1024


def test_transformer_tuning_space_rejects_incompatible_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="d_model values must be divisible"):
        coerce_tuning_space_config(
            {
                "model": {
                    "id": "transformer",
                    "d_model": [384],
                    "nhead": [7],
                },
            },
            model_config=_transformer_model(),
            problem_config=_problem(),
        )
