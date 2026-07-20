from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Self
from uuid import UUID

import numpy as np
import polars as pl
import pytest
import torch
from pydantic import ValidationError
from torch import nn

from fable.addresses import (
    corpus_blocks_path,
    corpus_directory,
    corpus_json_path,
    evaluation_directory,
    evaluation_json_path,
    evaluation_observations_path,
)
from fable.config import (
    AdamWMethod,
    BaselineSource,
    CorpusDefinition,
    CorpusRequest,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LossDefinition,
    LstmCapacity,
    LstmDefinition,
    LstmMethod,
    OriginWindow,
    SelectedStudySource,
    TrainingDefinition,
    TrainRequest,
)
from fable.evaluation import EvaluationDeployment, evaluate
from fable.min_block_fee import (
    ClassificationLossState,
    MinBlockFeeOutput,
    TargetState,
)
from fable.modeling import ArtifactAssociation
from fable.temporal.features import FeatureState

evaluation_module = importlib.import_module("fable.evaluation.evaluate")

_Weighting = Literal["unweighted", "corrected_inverse_frequency"]

_CORPUS_ID = UUID("10000000-0000-4000-8000-000000000001")
_OTHER_CORPUS_ID = UUID("10000000-0000-4000-8000-000000000002")
_ARTIFACT_ID = UUID("20000000-0000-4000-8000-000000000001")
_EVALUATION_ID = UUID("30000000-0000-4000-8000-000000000001")
_STUDY_ID = UUID("40000000-0000-4000-8000-000000000001")

_BASE_FEES = np.array(
    [50, 40, 30, 20, 10, 15, 25, 35, 45, 90, 80, 60, 70, 50, 40, 55, 30, 30, 65, 45, 75],
    dtype=np.int64,
)
_TIMESTAMPS = np.array(
    [
        100,
        112,
        124,
        136,
        148,
        160,
        172,
        184,
        196,
        208,
        220,
        232,
        232,
        256,
        268,
        280,
        292,
        292,
        316,
        328,
        340,
    ],
    dtype=np.int64,
)
_LOGITS = torch.tensor(
    [
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0],
        [0.0, 2.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 1.0, 2.0],
    ],
    dtype=torch.float32,
)
_PREDICTED_Z = torch.tensor([0.5, -0.5, 1.0, 0.0, 2.0], dtype=torch.float32)
_OBSERVATION_SCHEMA = pl.Schema(
    {
        "origin_block": pl.Int64,
        "origin_timestamp": pl.Int64,
        "selected_action_k": pl.Int64,
        "earliest_hindsight_action_k": pl.Int64,
        "classification_loss_contribution": pl.Float64,
        "predicted_hindsight_minimum_base_fee_z": pl.Float32,
        "previous_closed_parent_base_fee_per_gas": pl.Int64,
        "closed_parent_base_fee_per_gas": pl.Int64,
        "immediate_k0_base_fee_per_gas": pl.Int64,
        "selected_target_base_fee_per_gas": pl.Int64,
        "hindsight_minimum_base_fee_per_gas": pl.Int64,
        "selected_action_wait_seconds": pl.Int64,
        "full_horizon_elapsed_seconds": pl.Int64,
    }
)


def _loss(weighting: _Weighting) -> LossDefinition:
    return LossDefinition(
        classification_algorithm="cross_entropy",
        classification_weighting=weighting,
        regression_algorithm="smooth_l1",
        regression_threshold=1.0,
        classification_scale=2.0,
        regression_scale=0.5,
    )


def _experiment(weighting: _Weighting = "unweighted") -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=OriginWindow(
            role="training",
            first_parent_block=12,
            last_parent_block=15,
        ),
        validation_window=OriginWindow(
            role="validation",
            first_parent_block=20,
            last_parent_block=24,
        ),
        context_blocks=3,
        horizon_blocks=3,
        ordered_features=("log_base_fee_per_gas",),
        loss=_loss(weighting),
    )


def _method() -> LstmMethod:
    return LstmMethod(
        family="lstm",
        capacity=LstmCapacity(hidden=4, layers=1, head_hidden=3),
        dropout=0.0,
        optimizer=AdamWMethod(learning_rate=0.01, weight_decay=0.0),
        training_batch=7,
        fit=FitMethod(
            accumulation=1,
            gradient_clip_norm=1.0,
            scheduler="none",
            seed=17,
            max_epochs=2,
            validate_every_completed_epoch=1,
            patience=1,
            min_delta=0.0,
            improvement="strict_lower",
            restore="earliest_best",
        ),
    )


def _association(
    source_kind: str,
    weighting: _Weighting,
    *,
    experiment: ExperimentSemantics | None = None,
) -> ArtifactAssociation:
    experiment = experiment or _experiment(weighting)
    classification_state = (
        None if weighting == "unweighted" else ClassificationLossState(class_support=(5, 10, 20))
    )
    if source_kind == "baseline":
        source = BaselineSource(
            kind="baseline",
            corpus_id=_CORPUS_ID,
            training_definition=TrainingDefinition(
                experiment=experiment,
                model=LstmDefinition(
                    family="lstm",
                    hidden=4,
                    layers=1,
                    head_hidden=3,
                    dropout=0.0,
                ),
                optimizer=_method().optimizer,
                training_batch=7,
                fit=_method().fit,
            ),
        )
        return ArtifactAssociation(
            request=TrainRequest(
                workflow="train",
                artifact_id=_ARTIFACT_ID,
                source=source,
            ),
            feature_state=FeatureState(means=(0.0,), standard_deviations=(1.0,)),
            target_state=TargetState(mean=0.0, standard_deviation=1.0),
            classification_state=classification_state,
        )

    method = _method()
    return ArtifactAssociation(
        request=TrainRequest(
            workflow="train",
            artifact_id=_ARTIFACT_ID,
            source=SelectedStudySource(
                kind="selected_study",
                corpus_id=_CORPUS_ID,
                study_id=_STUDY_ID,
                study_result_index=2,
                experiment=experiment,
            ),
        ),
        feature_state=FeatureState(means=(0.0,), standard_deviations=(1.0,)),
        target_state=TargetState(mean=0.0, standard_deviation=1.0),
        classification_state=classification_state,
        study_result_index=2,
        method=method,
    )


def _write_corpus(storage_root: Path, corpus_id: UUID) -> None:
    request = CorpusRequest(
        corpus_id=corpus_id,
        definition=CorpusDefinition(chain_id=9, first_block=10, last_block=30),
    )
    corpus_directory(storage_root, corpus_id).mkdir(parents=True)
    corpus_json_path(storage_root, corpus_id).write_text(
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "finalized_anchor": {
                    "block_number": 30,
                    "block_hash": "a" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    blocks = np.arange(10, 31, dtype=np.int64)
    pl.DataFrame(
        {
            "block_number": blocks,
            "timestamp": _TIMESTAMPS,
            "chain_id": np.full(blocks.size, 9, dtype=np.int64),
            "base_fee_per_gas": _BASE_FEES,
            "gas_used": np.arange(30, 51, dtype=np.int64),
            "gas_limit": np.full(blocks.size, 100, dtype=np.int64),
            "tx_count": np.arange(5, 26, dtype=np.int64),
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
    ).write_parquet(corpus_blocks_path(storage_root, corpus_id))


def _request(
    *,
    corpus_id: UUID = _CORPUS_ID,
    window: OriginWindow | None = None,
) -> EvaluateRequest:
    return EvaluateRequest(
        workflow="evaluate",
        evaluation_id=_EVALUATION_ID,
        artifact_id=_ARTIFACT_ID,
        corpus_id=corpus_id,
        window=window or _experiment().validation_window,
    )


def _deployment() -> EvaluationDeployment:
    return EvaluationDeployment(
        batch_size=3,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False,
        deterministic=False,
        benchmark=True,
        float32_matmul_precision="high",
        cuda_matmul_allow_tf32=True,
        cudnn_allow_tf32=True,
    )


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval()
        self.cursor = 0
        self.batch_sizes: list[int] = []
        self.transfers = 0

    def to(self, *args: Any, **kwargs: Any) -> Self:
        assert torch.device(args[0]) == torch.device("cpu")
        assert torch.get_float32_matmul_precision() == "high"
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
        self.transfers += 1
        return self

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        assert torch.is_inference_mode_enabled()
        assert not self.training
        size = inputs.shape[0]
        start = self.cursor
        self.cursor += size
        self.batch_sizes.append(size)
        return MinBlockFeeOutput(
            action_logits=_LOGITS[start : start + size],
            minimum_fee_z=_PREDICTED_Z[start : start + size],
        )


def _classification_contributions(weighting: _Weighting) -> list[float]:
    labels = [2, 2, 1, 2, 1]
    weights = {0: 7 / 3, 1: 7 / 6, 2: 7 / 12}
    return [
        (math.log(sum(math.exp(float(value)) for value in logits)) - float(logits[label]))
        * (1.0 if weighting == "unweighted" else weights[label])
        * 2.0
        for logits, label in zip(_LOGITS.tolist(), labels, strict=True)
    ]


@pytest.mark.parametrize(
    ("source_kind", "weighting"),
    [("baseline", "corrected_inverse_frequency"), ("selected", "unweighted")],
)
def test_evaluate_publishes_exact_observations_through_one_full_and_tail_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    weighting: _Weighting,
) -> None:
    _write_corpus(tmp_path, _CORPUS_ID)
    association = _association(source_kind, weighting)
    model = _Model()
    monkeypatch.setattr(
        evaluation_module,
        "load_artifact",
        lambda storage_root, artifact_id: (association, model),
    )
    monkeypatch.setattr(evaluation_module, "_DEVICE", torch.device("cpu"))
    request = _request()

    evaluate(request, tmp_path, _deployment())

    assert model.transfers == 1
    assert model.batch_sizes == [3, 2]
    assert evaluation_json_path(tmp_path, _EVALUATION_ID).read_text() == request.model_dump_json()

    observations = pl.read_parquet(evaluation_observations_path(tmp_path, _EVALUATION_ID))
    assert observations.schema == _OBSERVATION_SCHEMA
    assert observations.null_count().row(0) == (0,) * len(_OBSERVATION_SCHEMA)
    assert observations["origin_block"].to_list() == [20, 21, 22, 23, 24]
    assert observations["origin_timestamp"].to_list() == [220, 232, 232, 256, 268]
    assert observations["selected_action_k"].to_list() == [0, 2, 1, 0, 2]
    assert observations["earliest_hindsight_action_k"].to_list() == [2, 2, 1, 2, 1]
    assert observations["predicted_hindsight_minimum_base_fee_z"].to_list() == pytest.approx(
        [0.5, -0.5, 1.0, 0.0, 2.0]
    )
    assert observations["classification_loss_contribution"].to_list() == pytest.approx(
        _classification_contributions(weighting)
    )
    assert observations.select(
        "previous_closed_parent_base_fee_per_gas",
        "closed_parent_base_fee_per_gas",
        "immediate_k0_base_fee_per_gas",
        "selected_target_base_fee_per_gas",
        "hindsight_minimum_base_fee_per_gas",
        "selected_action_wait_seconds",
        "full_horizon_elapsed_seconds",
    ).rows() == [
        (90, 80, 60, 60, 50, 0, 36),
        (80, 60, 70, 40, 40, 24, 36),
        (60, 70, 50, 40, 40, 24, 48),
        (70, 50, 40, 40, 30, 0, 36),
        (50, 40, 55, 30, 30, 24, 24),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"batch_size": 0},
        {"float32_matmul_precision": "medium"},
    ],
)
def test_evaluation_deployment_rejects_invalid_batch_or_matmul_policy(
    payload: dict[str, object],
) -> None:
    valid = _deployment().model_dump()
    valid.update(payload)

    with pytest.raises(ValidationError):
        EvaluationDeployment.model_validate(valid)


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("source_corpus", ValueError, "artifact source Corpus"),
        ("previous_parent", ValueError, "previous closed parent"),
        ("occupied_canonical", FileExistsError, "evaluations"),
    ],
)
def test_evaluate_rejects_owned_association_and_publication_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    error: type[Exception],
    match: str,
) -> None:
    corpus_id = _OTHER_CORPUS_ID if case == "source_corpus" else _CORPUS_ID
    _write_corpus(tmp_path, corpus_id)
    if case == "previous_parent":
        experiment = ExperimentSemantics(
            training_window=OriginWindow(
                role="training",
                first_parent_block=0,
                last_parent_block=1,
            ),
            validation_window=OriginWindow(
                role="validation",
                first_parent_block=10,
                last_parent_block=10,
            ),
            context_blocks=1,
            horizon_blocks=3,
            ordered_features=("log_base_fee_per_gas",),
            loss=_loss("unweighted"),
        )
        association = _association(
            "baseline",
            "unweighted",
            experiment=experiment,
        )
    else:
        experiment = _experiment()
        association = _association("baseline", "unweighted")
    model = _Model()
    monkeypatch.setattr(
        evaluation_module,
        "load_artifact",
        lambda storage_root, artifact_id: (association, model),
    )
    monkeypatch.setattr(evaluation_module, "_DEVICE", torch.device("cpu"))
    if case == "previous_parent":
        window = experiment.validation_window
    else:
        window = _experiment().validation_window
    request = _request(corpus_id=corpus_id, window=window)
    if case == "occupied_canonical":
        evaluation_directory(tmp_path, _EVALUATION_ID).mkdir(parents=True)

    with pytest.raises(error, match=match):
        evaluate(request, tmp_path, _deployment())

    if case == "occupied_canonical":
        scratch = tmp_path / "evaluations" / f".{_EVALUATION_ID}"
        assert sorted(path.name for path in scratch.iterdir()) == [
            "evaluation.json",
            "observations.parquet",
        ]
