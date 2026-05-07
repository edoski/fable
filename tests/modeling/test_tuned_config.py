from __future__ import annotations

from types import SimpleNamespace

import pytest
from optuna.trial import FixedTrial

from spice.config import (
    TrainWorkflowSelection,
    TunedParameterSet,
    coerce_problem_spec,
    resolve_workflow_config,
)
from spice.config.models import TunedTrainingParams, TuningProblemSearchSpace, TuningSpaceConfig
from spice.core.errors import ConfigResolutionError
from spice.modeling.families.lstm import (
    LstmModelConfig,
    LstmTunedModelParams,
)
from spice.modeling.families.registry import apply_model_tuned_parameters
from spice.modeling.families.transformer import (
    TransformerModelConfig,
    TransformerTunedModelParams,
    TransformerTuningSpaceModelConfig,
)
from spice.modeling.families.transformer_lstm import (
    TransformerLstmModelConfig,
    TransformerLstmTunedModelParams,
)
from spice.modeling.tuned_config import (
    coerce_tuned_parameter_set,
    coerce_tuning_space_config,
    sample_tuned_parameters,
)
from spice.modeling.tuning import apply_study_best_params
from tests.root_handle_helpers import corpus_handle, study_handle


def _transformer_model() -> TransformerModelConfig:
    return TransformerModelConfig(
        d_model=256,
        nhead=4,
        transformer_layers=4,
        feedforward_dim=512,
        head_hidden_dim=256,
        dropout=0.2,
    )


def _lstm_model() -> LstmModelConfig:
    return LstmModelConfig(
        input_projection_dim=128,
        hidden_size=256,
        num_layers=2,
        head_hidden_dim=128,
        dropout=0.2,
    )


def test_tuning_coercers_preserve_typed_identity() -> None:
    tuning_space = coerce_tuning_space_config(
        {"model": {"id": "lstm", "hidden_size": [64]}},
        model_config=_lstm_model(),
        problem_config=_problem(),
    )
    tuned_parameters = coerce_tuned_parameter_set(
        {"model": {"id": "lstm", "hidden_size": 64}},
        model_id="lstm",
    )

    assert (
        coerce_tuning_space_config(
            tuning_space,
            model_config=_lstm_model(),
            problem_config=_problem(),
        )
        == tuning_space
    )
    assert coerce_tuned_parameter_set(tuned_parameters, model_id="lstm") is tuned_parameters


def _transformer_lstm_model() -> TransformerLstmModelConfig:
    return TransformerLstmModelConfig(
        hidden_size=256,
        num_layers=2,
        d_model=256,
        nhead=4,
        transformer_layers=4,
        feedforward_dim=512,
        head_hidden_dim=256,
        dropout=0.2,
    )


def _model_config(model_id: str):
    if model_id == "lstm":
        return _lstm_model()
    if model_id == "transformer":
        return _transformer_model()
    if model_id == "transformer_lstm":
        return _transformer_lstm_model()
    raise AssertionError(f"unknown model id {model_id}")


def _problem():
    return coerce_problem_spec(
        {
            "id": "test_problem",
            "lookback_seconds": 900,
            "sample_count": 1024,
            "max_delay_seconds": 36,
            "compiler": {
                "id": "observed_time_window",
                "slot_spacing": {"id": "nominal"},
            },
            "execution_policy": {"id": "strict_deadline_miss"},
        }
    )


@pytest.mark.parametrize(
    ("model_id", "model_space", "trial_values", "params_type", "expected"),
    [
        (
            "lstm",
            {
                "input_projection_dim": [64],
                "hidden_size": [128],
                "num_layers": [3],
                "head_hidden_dim": [256],
                "dropout": [0.1],
            },
            {
                "model.input_projection_dim": 64,
                "model.hidden_size": 128,
                "model.num_layers": 3,
                "model.head_hidden_dim": 256,
                "model.dropout": 0.1,
            },
            LstmTunedModelParams,
            {
                "input_projection_dim": 64,
                "hidden_size": 128,
                "num_layers": 3,
                "head_hidden_dim": 256,
                "dropout": 0.1,
            },
        ),
        (
            "transformer",
            {
                "d_model": [512],
                "nhead": [8],
                "transformer_layers": [6],
                "feedforward_multiplier": [4],
                "head_hidden_dim": [1024],
                "dropout": [0.1],
            },
            {
                "model.d_model": 512,
                "model.nhead": 8,
                "model.transformer_layers": 6,
                "model.feedforward_multiplier": 4,
                "model.head_hidden_dim": 1024,
                "model.dropout": 0.1,
            },
            TransformerTunedModelParams,
            {
                "d_model": 512,
                "nhead": 8,
                "transformer_layers": 6,
                "feedforward_dim": 2048,
                "head_hidden_dim": 1024,
                "dropout": 0.1,
            },
        ),
        (
            "transformer_lstm",
            {
                "hidden_size": [384],
                "num_layers": [3],
                "d_model": [512],
                "nhead": [8],
                "transformer_layers": [6],
                "feedforward_multiplier": [4],
                "head_hidden_dim": [1024],
                "dropout": [0.1],
            },
            {
                "model.hidden_size": 384,
                "model.num_layers": 3,
                "model.d_model": 512,
                "model.nhead": 8,
                "model.transformer_layers": 6,
                "model.feedforward_multiplier": 4,
                "model.head_hidden_dim": 1024,
                "model.dropout": 0.1,
            },
            TransformerLstmTunedModelParams,
            {
                "hidden_size": 384,
                "num_layers": 3,
                "d_model": 512,
                "nhead": 8,
                "transformer_layers": 6,
                "feedforward_dim": 2048,
                "head_hidden_dim": 1024,
                "dropout": 0.1,
            },
        ),
    ],
)
def test_sample_model_tuned_params_uses_registered_family_adapter(
    model_id: str,
    model_space: dict[str, object],
    trial_values: dict[str, object],
    params_type: type[object],
    expected: dict[str, object],
) -> None:
    tuning_space = coerce_tuning_space_config(
        {
            "model": {"id": model_id, **model_space},
        },
        model_config=_model_config(model_id),
        problem_config=_problem(),
    )
    assert tuning_space is not None

    params = sample_tuned_parameters(
        FixedTrial(trial_values),
        tuning_space=tuning_space,
    )

    assert isinstance(params.model, params_type)
    assert params.model.model_dump(exclude={"id"}, exclude_none=True) == expected


def test_sample_tuned_parameters_samples_shared_training_and_problem_params() -> None:
    tuning_space = coerce_tuning_space_config(
        {
            "training": {
                "learning_rate": [0.0001],
                "weight_decay": [0.001],
                "batch_size": [128],
            },
            "problem": {"lookback_seconds": [1800]},
            "model": {"id": "lstm"},
        },
        model_config=_lstm_model(),
        problem_config=_problem(),
    )
    assert tuning_space is not None

    params = sample_tuned_parameters(
        FixedTrial(
            {
                "training.learning_rate": 0.0001,
                "training.weight_decay": 0.001,
                "training.batch_size": 128,
                "problem.lookback_seconds": 1800,
            }
        ),
        tuning_space=tuning_space,
    )

    assert params.training is not None
    assert params.training.learning_rate == 0.0001
    assert params.training.weight_decay == 0.001
    assert params.training.batch_size == 128
    assert params.problem is not None
    assert params.problem.lookback_seconds == 1800
    assert params.model is None


@pytest.mark.parametrize(
    ("base_config", "tuned_params", "expected"),
    [
        (
            _lstm_model(),
            LstmTunedModelParams(hidden_size=512, dropout=0.15),
            {"hidden_size": 512, "num_layers": 2, "dropout": 0.15},
        ),
        (
            _transformer_model(),
            TransformerTunedModelParams(d_model=512, nhead=8, feedforward_dim=2048),
            {"d_model": 512, "nhead": 8, "feedforward_dim": 2048, "transformer_layers": 4},
        ),
        (
            _transformer_lstm_model(),
            TransformerLstmTunedModelParams(hidden_size=384, d_model=512, nhead=8),
            {"hidden_size": 384, "d_model": 512, "nhead": 8, "num_layers": 2},
        ),
    ],
)
def test_apply_tuned_parameters_uses_registered_family_adapter(
    base_config,
    tuned_params,
    expected: dict[str, object],
) -> None:
    tuned = apply_model_tuned_parameters(base_config, tuned_params)

    for field, value in expected.items():
        assert getattr(tuned, field) == value


def test_apply_tuned_parameters_revalidates_model_config() -> None:
    with pytest.raises(ValueError, match="d_model must be divisible"):
        apply_model_tuned_parameters(
            _transformer_model(),
            TransformerTunedModelParams(d_model=384, nhead=7),
        )


def test_apply_study_best_params_uses_manifest_study_name(tmp_path, monkeypatch) -> None:
    config = resolve_workflow_config(
        TrainWorkflowSelection(
            surface="current_row_fee_dynamics",
            study_id="std_manifest",
            variant="tuned",
            storage_root=tmp_path / "outputs",
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "spice.modeling.tuning.load_study_manifest",
        lambda _path: SimpleNamespace(study_name="manifest_study"),
    )

    def fake_validate(config, **_kwargs):
        captured["validated_study"] = config.study.name

    monkeypatch.setattr(
        "spice.modeling.tuning.validate_tuned_artifact_definition",
        fake_validate,
    )

    def fake_load_best_params(_path, *, study_name: str):
        captured["loaded_study"] = study_name
        return TunedParameterSet(training=TunedTrainingParams(learning_rate=0.0002))

    monkeypatch.setattr("spice.modeling.tuning.load_best_params", fake_load_best_params)

    corpus = corpus_handle(
        tmp_path / "outputs",
        chain_name=config.chain.name,
        dataset_id="cor_9a73b1e88edb488afb1e",
        dataset_name=config.dataset.name,
    )
    study = study_handle(
        tmp_path / "outputs",
        corpus=corpus,
        study_id="std_manifest",
        study_name=config.study.name,
    )

    applied = apply_study_best_params(
        config,
        study=study,
        corpus=corpus,
    )

    assert captured == {
        "validated_study": "manifest_study",
        "loaded_study": "manifest_study",
    }
    assert applied.config.study.name == "manifest_study"


def test_apply_study_best_params_rejects_mismatched_manifest_before_loading_params(
    tmp_path,
    monkeypatch,
) -> None:
    config = resolve_workflow_config(
        TrainWorkflowSelection(
            surface="current_row_fee_dynamics",
            study_id="std_manifest",
            variant="tuned",
            storage_root=tmp_path / "outputs",
        ),
    )
    corpus = corpus_handle(
        tmp_path / "outputs",
        chain_name=config.chain.name,
        dataset_id="cor_9a73b1e88edb488afb1e",
        dataset_name=config.dataset.name,
    )
    study = study_handle(
        tmp_path / "outputs",
        corpus=corpus,
        study_id="std_manifest",
        study_name=config.study.name,
    )
    monkeypatch.setattr(
        "spice.modeling.tuning.load_study_manifest",
        lambda _path: SimpleNamespace(study_name="manifest_study"),
    )
    monkeypatch.setattr(
        "spice.modeling.tuning.validate_tuned_artifact_definition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ConfigResolutionError("definition mismatch")
        ),
    )

    def fail_load_best_params(*_args, **_kwargs):
        raise AssertionError("load_best_params should not run after manifest mismatch")

    monkeypatch.setattr("spice.modeling.tuning.load_best_params", fail_load_best_params)

    with pytest.raises(ConfigResolutionError, match="definition mismatch"):
        apply_study_best_params(config, study=study, corpus=corpus)


@pytest.mark.parametrize("model_id", ["transformer", "transformer_lstm"])
@pytest.mark.parametrize(
    ("model_space", "message"),
    [
        ({"d_model": [383], "nhead": [1]}, "d_model values must be even"),
        ({"d_model": [384], "nhead": [7]}, "d_model values must be divisible"),
        (
            {"feedforward_multiplier": [4]},
            "feedforward_multiplier requires d_model",
        ),
    ],
)
def test_attention_family_tuning_space_validates_shared_attention_constraints(
    model_id: str,
    model_space: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        coerce_tuning_space_config(
            {"model": {"id": model_id, **model_space}},
            model_config=_model_config(model_id),
            problem_config=_problem(),
        )


def test_tuning_space_rejects_model_id_mismatch() -> None:
    with pytest.raises(ConfigResolutionError, match="tuning_space.model.id must match model.id"):
        coerce_tuning_space_config(
            {"model": {"id": "transformer", "d_model": [256]}},
            model_config=_lstm_model(),
            problem_config=_problem(),
        )


def test_typed_tuning_space_rejects_model_id_mismatch() -> None:
    with pytest.raises(ConfigResolutionError, match="tuning_space.model.id must match model.id"):
        coerce_tuning_space_config(
            TuningSpaceConfig(
                model=TransformerTuningSpaceModelConfig(d_model=[256], nhead=[4])
            ),
            model_config=_lstm_model(),
            problem_config=_problem(),
        )


def test_tuning_space_requires_problem_context_for_typed_problem_group() -> None:
    with pytest.raises(ConfigResolutionError, match="problem_config is required"):
        coerce_tuning_space_config(
            {
                "model": {"id": "transformer"},
                "problem": TuningProblemSearchSpace(lookback_seconds=[900]),
            },
            model_config=_transformer_model(),
            problem_config=None,
        )


def test_typed_tuning_space_requires_problem_context_for_typed_problem_group() -> None:
    with pytest.raises(ConfigResolutionError, match="problem_config is required"):
        coerce_tuning_space_config(
            TuningSpaceConfig(
                model=TransformerTuningSpaceModelConfig(),
                problem=TuningProblemSearchSpace(lookback_seconds=[900]),
            ),
            model_config=_transformer_model(),
            problem_config=None,
        )
