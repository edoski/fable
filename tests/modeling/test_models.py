from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import polars as pl
import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from spice.addresses import artifact_checkpoint_path
from spice.config import (
    AdamWMethod,
    BaselineSource,
    CorpusDefinition,
    CorpusRequest,
    ExperimentSemantics,
    FitMethod,
    LossDefinition,
    LstmCapacity,
    LstmDefinition,
    LstmMethod,
    LstmMethodSpace,
    OriginWindow,
    SelectedStudySource,
    StudyDefinition,
    TrainingDefinition,
    TrainRequest,
    TransformerDefinition,
    TransformerLstmDefinition,
    TuneRequest,
)
from spice.corpus.contract import Corpus, FinalizedAnchor
from spice.min_block_fee import (
    ClassificationLossState,
    MinBlockFeeOutput,
    TargetState,
    min_block_fee_loss,
)
from spice.modeling import (
    ArtifactAssociation,
    FitDeployment,
    load_artifact,
    train,
)
from spice.modeling import artifacts as modeling_artifacts
from spice.temporal.features import FeatureState
from spice.temporal.history import prepare_fit_history

ARTIFACT_ID = UUID("10000000-0000-4000-8000-000000000001")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")
_BASE_FEES = np.array(
    [11, 12, 10, 4, 9, 4, 8, 3, 5, 6, 10, 6, 2, 2],
    dtype=np.int64,
)


def _experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=OriginWindow(
            role="training",
            first_parent_block=12,
            last_parent_block=15,
        ),
        validation_window=OriginWindow(
            role="validation",
            first_parent_block=20,
            last_parent_block=21,
        ),
        context_blocks=3,
        horizon_blocks=2,
        ordered_features=("log_base_fee_per_gas", "gas_utilization"),
        loss=LossDefinition(
            classification_algorithm="cross_entropy",
            classification_weighting="unweighted",
            regression_algorithm="smooth_l1",
            regression_threshold=0.6,
            classification_scale=1.2,
            regression_scale=0.7,
        ),
    )


def _request() -> TrainRequest:
    definition = TrainingDefinition(
        experiment=_experiment(),
        model=LstmDefinition(
            family="lstm",
            hidden=5,
            layers=1,
            head_hidden=3,
            dropout=0.1,
        ),
        optimizer=AdamWMethod(learning_rate=0.002, weight_decay=0.003),
        training_batch=3,
        fit=FitMethod(
            accumulation=2,
            gradient_clip_norm=0.4,
            scheduler="none",
            seed=19,
            max_epochs=4,
            validate_every_completed_epoch=1,
            patience=1,
            min_delta=0.02,
            improvement="strict_lower",
            restore="earliest_best",
        ),
    )
    return TrainRequest(
        workflow="train",
        artifact_id=ARTIFACT_ID,
        source=BaselineSource(
            kind="baseline",
            corpus_id=CORPUS_ID,
            training_definition=definition,
        ),
    )


def _corpus() -> Corpus:
    blocks = np.arange(10, 24, dtype=np.int64)
    return Corpus(
        request=_corpus_request(),
        finalized_anchor=FinalizedAnchor(block_number=23, block_hash="a" * 64),
        blocks=pl.DataFrame(
            {
                "block_number": blocks,
                "timestamp": blocks * 11,
                "chain_id": np.ones(blocks.size, dtype=np.int64),
                "base_fee_per_gas": _BASE_FEES,
                "gas_used": 30 + np.arange(blocks.size, dtype=np.int64),
                "gas_limit": np.full(blocks.size, 100, dtype=np.int64),
                "tx_count": 4 + np.arange(blocks.size, dtype=np.int64),
            },
            schema={
                "block_number": pl.Int64,
                "timestamp": pl.Int64,
                "chain_id": pl.Int64,
                "base_fee_per_gas": pl.Int64,
                "gas_used": pl.Int64,
                "gas_limit": pl.Int64,
                "tx_count": pl.Int64,
            },
        ),
    )


def _corpus_request() -> CorpusRequest:
    return CorpusRequest(
        corpus_id=CORPUS_ID,
        definition=CorpusDefinition(chain_id=1, first_block=10, last_block=23),
    )


def _deployment() -> FitDeployment:
    return FitDeployment(
        deterministic=True,
        benchmark=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False,
        float32_matmul_precision="highest",
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
    )


def _candidate_request(method: LstmMethod) -> TuneRequest:
    return TuneRequest(
        workflow="tune",
        study_id=UUID("40000000-0000-4000-8000-000000000001"),
        corpus_id=CORPUS_ID,
        study_definition=StudyDefinition(
            experiment=method_experiment(),
            method_space=LstmMethodSpace(family="lstm", methods=(method,)),
        ),
    )


def method_experiment() -> ExperimentSemantics:
    experiment = _experiment()
    return ExperimentSemantics(
        training_window=experiment.training_window,
        validation_window=experiment.validation_window,
        context_blocks=experiment.context_blocks,
        horizon_blocks=experiment.horizon_blocks,
        ordered_features=experiment.ordered_features,
        loss=LossDefinition(
            classification_algorithm="cross_entropy",
            classification_weighting="unweighted",
            regression_algorithm="smooth_l1",
            regression_threshold=0.6,
            classification_scale=0.0,
            regression_scale=0.0,
        ),
    )


def _definition(
    model: LstmDefinition | TransformerDefinition | TransformerLstmDefinition,
) -> TrainingDefinition:
    return TrainingDefinition(
        experiment=_experiment(),
        model=model,
        optimizer=AdamWMethod(learning_rate=0.002, weight_decay=0.003),
        training_batch=3,
        fit=FitMethod(
            accumulation=1,
            gradient_clip_norm=0.8,
            scheduler="none",
            seed=29,
            max_epochs=1,
            validate_every_completed_epoch=1,
            patience=0,
            min_delta=0.0,
            improvement="strict_lower",
            restore="earliest_best",
        ),
    )


def _train_request(
    artifact_id: UUID,
    model: LstmDefinition | TransformerDefinition | TransformerLstmDefinition,
) -> TrainRequest:
    return TrainRequest(
        workflow="train",
        artifact_id=artifact_id,
        source=BaselineSource(
            kind="baseline",
            corpus_id=CORPUS_ID,
            training_definition=_definition(model),
        ),
    )


def test_artifact_association_is_strict_json_safe_and_request_owned() -> None:
    association = ArtifactAssociation(
        request=_request(),
        feature_state=FeatureState(
            means=(1.0, 2.0),
            standard_deviations=(0.5, 0.25),
        ),
        target_state=TargetState(mean=3.0, standard_deviation=0.75),
    )

    raw = association.model_dump(mode="json", exclude_none=True)

    assert set(raw) == {"request", "feature_state", "target_state"}
    assert raw["request"]["artifact_id"] == str(ARTIFACT_ID)
    assert ArtifactAssociation.model_validate_json(
        json.dumps(raw, allow_nan=False),
        strict=True,
    ) == association


def test_selected_association_uses_exact_method_and_result_index() -> None:
    method = LstmMethod(
        family="lstm",
        capacity=LstmCapacity(hidden=5, layers=1, head_hidden=3),
        dropout=0.15,
        optimizer=AdamWMethod(learning_rate=0.003, weight_decay=0.004),
        training_batch=5,
        fit=FitMethod(
            accumulation=3,
            gradient_clip_norm=0.7,
            scheduler="none",
            seed=43,
            max_epochs=7,
            validate_every_completed_epoch=2,
            patience=2,
            min_delta=0.03,
            improvement="strict_lower",
            restore="earliest_best",
        ),
    )
    request = TrainRequest(
        workflow="train",
        artifact_id=ARTIFACT_ID,
        source=SelectedStudySource(
            kind="selected_study",
            corpus_id=CORPUS_ID,
            study_id=UUID("50000000-0000-4000-8000-000000000001"),
            study_result_index=2,
            experiment=_experiment(),
        ),
    )

    association = ArtifactAssociation(
        request=request,
        feature_state=FeatureState(
            means=(1.0, 2.0),
            standard_deviations=(0.5, 0.25),
        ),
        target_state=TargetState(mean=3.0, standard_deviation=0.75),
        study_result_index=2,
        method=method,
    )

    assert association.method == method
    assert association.study_result_index == 2


def test_artifact_association_rejects_only_owned_mismatches() -> None:
    feature_state = FeatureState(
        means=(1.0, 2.0),
        standard_deviations=(0.5, 0.25),
    )
    target_state = TargetState(mean=3.0, standard_deviation=0.75)
    method = LstmMethod(
        family="lstm",
        capacity=LstmCapacity(hidden=5, layers=1, head_hidden=3),
        dropout=0.1,
        optimizer=AdamWMethod(learning_rate=0.002, weight_decay=0.003),
        training_batch=3,
        fit=FitMethod(
            accumulation=2,
            gradient_clip_norm=0.4,
            scheduler="none",
            seed=19,
            max_epochs=4,
            validate_every_completed_epoch=1,
            patience=1,
            min_delta=0.02,
            improvement="strict_lower",
            restore="earliest_best",
        ),
    )

    with pytest.raises(ValidationError, match="baseline artifacts"):
        ArtifactAssociation(
            request=_request(),
            feature_state=feature_state,
            target_state=target_state,
            study_result_index=0,
            method=method,
        )
    with pytest.raises(ValidationError, match="feature state width"):
        ArtifactAssociation(
            request=_request(),
            feature_state=FeatureState(
                means=(1.0,),
                standard_deviations=(0.5,),
            ),
            target_state=target_state,
        )

    experiment = _experiment()
    corrected_experiment = ExperimentSemantics(
        training_window=experiment.training_window,
        validation_window=experiment.validation_window,
        context_blocks=experiment.context_blocks,
        horizon_blocks=experiment.horizon_blocks,
        ordered_features=experiment.ordered_features,
        loss=LossDefinition(
            classification_algorithm="cross_entropy",
            classification_weighting="corrected_inverse_frequency",
            regression_algorithm="smooth_l1",
            regression_threshold=0.6,
            classification_scale=1.2,
            regression_scale=0.7,
        ),
    )
    corrected_request = TrainRequest(
        workflow="train",
        artifact_id=ARTIFACT_ID,
        source=BaselineSource(
            kind="baseline",
            corpus_id=CORPUS_ID,
            training_definition=TrainingDefinition(
                experiment=corrected_experiment,
                model=LstmDefinition(
                    family="lstm",
                    hidden=5,
                    layers=1,
                    head_hidden=3,
                    dropout=0.1,
                ),
                optimizer=method.optimizer,
                training_batch=method.training_batch,
                fit=method.fit,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="support width"):
        ArtifactAssociation(
            request=corrected_request,
            feature_state=feature_state,
            target_state=target_state,
            classification_state=ClassificationLossState(class_support=(2,)),
        )


def test_transformer_encoder_layers_have_independent_matrix_initialization() -> None:
    torch.manual_seed(71)
    encoder = modeling_artifacts._encoder(
        width=4,
        heads=2,
        feedforward=7,
        layers=2,
        dropout=0.1,
    )
    matrices = [
        [parameter for parameter in layer.parameters() if parameter.ndim > 1]
        for layer in encoder.layers
    ]

    assert matrices[0]
    assert all(
        not torch.equal(first, second)
        for first, second in zip(matrices[0], matrices[1], strict=True)
    )


@pytest.mark.parametrize(
    "model",
    [
        LstmDefinition(
            family="lstm",
            hidden=5,
            layers=1,
            head_hidden=3,
            dropout=0.1,
        ),
        TransformerDefinition(
            family="transformer",
            model_width=4,
            attention_heads=2,
            transformer_layers=1,
            feedforward_width=7,
            head_hidden=3,
            dropout=0.1,
        ),
        TransformerLstmDefinition(
            family="transformer_lstm",
            model_width=4,
            attention_heads=2,
            transformer_layers=1,
            feedforward_width=7,
            lstm_hidden=5,
            lstm_layers=1,
            head_hidden=3,
            dropout=0.1,
        ),
    ],
)
def test_all_three_models_train_load_and_apply_direct_loss(
    model: LstmDefinition | TransformerDefinition | TransformerLstmDefinition,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_fit_history(_corpus(), _experiment())
    artifact_id = UUID(
        {
            "lstm": "30000000-0000-4000-8000-000000000001",
            "transformer": "30000000-0000-4000-8000-000000000002",
            "transformer_lstm": "30000000-0000-4000-8000-000000000003",
        }[model.family]
    )
    request = _train_request(artifact_id, model)
    real_trainer: Any = modeling_artifacts.pl.Trainer

    def cpu_trainer(**kwargs: Any) -> Any:
        kwargs["accelerator"] = "cpu"
        return real_trainer(**kwargs)

    monkeypatch.setattr(modeling_artifacts.pl, "Trainer", cpu_trainer)

    train(request, prepared, tmp_path, _deployment())
    association, loaded_model = load_artifact(tmp_path, artifact_id)

    assert association.request == request
    assert isinstance(loaded_model, nn.Module)
    batches = list(DataLoader(prepared.training, batch_size=3, shuffle=False))
    assert [batch["inputs"].shape[0] for batch in batches] == [3, 1]
    for batch in batches:
        output = loaded_model(batch["inputs"])
        assert isinstance(output, MinBlockFeeOutput)
        assert output.action_logits.shape == (batch["inputs"].shape[0], 2)
        assert output.minimum_fee_z.shape == (batch["inputs"].shape[0],)
        assert torch.isfinite(output.action_logits).all()
        assert torch.isfinite(output.minimum_fee_z).all()
        loss = min_block_fee_loss(
            output,
            label=batch["label"],
            target=batch["target"],
            loss_definition=_experiment().loss,
            classification_state=None,
        )
        assert torch.isfinite(loss.mean_total)

    with pytest.raises((TypeError, ValueError), match="float32"):
        loaded_model(batches[0]["inputs"].to(torch.float64))
    with pytest.raises(ValueError, match="trailing shape"):
        loaded_model(batches[0]["inputs"][:, :-1])

    if model.family == "lstm":
        mismatched_id = UUID("30000000-0000-4000-8000-000000000009")
        artifact_checkpoint_path(tmp_path, artifact_id).rename(
            artifact_checkpoint_path(tmp_path, mismatched_id)
        )
        with pytest.raises(ValueError, match="embedded artifact ID"):
            load_artifact(tmp_path, mismatched_id)


def test_request_and_deployment_drive_native_lightning_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = LstmMethod(
        family="lstm",
        capacity=LstmCapacity(hidden=5, layers=1, head_hidden=3),
        dropout=0.0,
        optimizer=AdamWMethod(learning_rate=0.004, weight_decay=0.002),
        training_batch=4,
        fit=FitMethod(
            accumulation=1,
            gradient_clip_norm=0.0,
            scheduler="none",
            seed=37,
            max_epochs=4,
            validate_every_completed_epoch=1,
            patience=1,
            min_delta=0.0,
            improvement="strict_lower",
            restore="earliest_best",
        ),
    )
    request = _candidate_request(method)
    prepared = prepare_fit_history(_corpus(), request.study_definition.experiment)
    real_trainer: Any = modeling_artifacts.pl.Trainer
    trainer_kwargs: list[dict[str, object]] = []
    fit_kwargs: list[dict[str, object]] = []
    train_orders: list[list[int]] = []
    metric_dtypes: list[torch.dtype] = []
    clip_maxima: list[float] = []
    real_clip: Any = torch.nn.utils.clip_grad_norm_

    def clip_spy(parameters: Any, max_norm: float, **kwargs: Any) -> Any:
        clip_maxima.append(float(max_norm))
        return real_clip(parameters, max_norm, **kwargs)

    class TrainerSpy:
        def __init__(self, **kwargs: Any) -> None:
            trainer_kwargs.append(dict(kwargs))
            assert kwargs["accelerator"] == "gpu"
            assert kwargs["devices"] == 1
            assert kwargs["precision"] == "32-true"
            kwargs["accelerator"] = "cpu"
            self._trainer = real_trainer(**kwargs)

        def fit(self, module: Any, **kwargs: Any) -> None:
            loader = kwargs["train_dataloaders"]
            assert isinstance(loader, DataLoader)
            assert loader.generator is not None
            state = loader.generator.get_state()
            train_orders.append(list(loader.sampler))
            loader.generator.set_state(state)
            validation = kwargs["val_dataloaders"]
            assert isinstance(validation, DataLoader)
            assert isinstance(loader.sampler, RandomSampler)
            assert isinstance(validation.sampler, SequentialSampler)
            fit_kwargs.append(dict(kwargs))
            on_validation_epoch_end = module.on_validation_epoch_end

            def validation_end_spy() -> None:
                metric_dtypes.append(module.validation_total.dtype)
                on_validation_epoch_end()

            module.on_validation_epoch_end = validation_end_spy
            self._trainer.fit(module, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self._trainer, name)

    monkeypatch.setattr(modeling_artifacts.pl, "Trainer", TrainerSpy)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_spy)
    scratch = tmp_path / "candidate"

    first = modeling_artifacts._run_candidate(
        request, method, prepared, scratch, _deployment()
    )
    second = modeling_artifacts._run_candidate(
        request, method, prepared, scratch, _deployment()
    )

    assert first.method == method
    assert first.objective == 0.0
    assert first.selected_epoch == 1
    assert first.completed_epochs == 2
    assert second.method == method
    assert second.objective == first.objective
    assert second.selected_epoch == first.selected_epoch
    assert first.completed_epochs < second.completed_epochs <= method.fit.max_epochs
    assert train_orders[0] == train_orders[1]
    assert set(train_orders[0]) == set(range(len(prepared.training)))
    assert "ckpt_path" not in fit_kwargs[0]
    assert fit_kwargs[1]["ckpt_path"] == scratch / "last.ckpt"
    assert fit_kwargs[1]["weights_only"] is True
    assert trainer_kwargs[0]["max_epochs"] == method.fit.max_epochs
    assert trainer_kwargs[0]["accumulate_grad_batches"] == method.fit.accumulation
    assert trainer_kwargs[0]["gradient_clip_val"] == method.fit.gradient_clip_norm
    assert trainer_kwargs[0]["check_val_every_n_epoch"] == (
        method.fit.validate_every_completed_epoch
    )
    assert trainer_kwargs[0]["deterministic"] is True
    assert metric_dtypes and all(dtype == torch.float64 for dtype in metric_dtypes)
    assert clip_maxima and all(math.isinf(maximum) for maximum in clip_maxima)
