from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self
from uuid import UUID

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

import fable.evaluation as evaluation_module
from fable.addresses import (
    corpus_blocks_path,
    corpus_directory,
    corpus_json_path,
    evaluation_directory,
    evaluation_json_path,
    evaluation_observations_path,
)
from fable.config import (
    BlockWindow,
    CorpusDefinition,
    CorpusRequest,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    SelectedStudySource,
    TrainRequest,
)
from fable.min_block_fee import MinBlockFeeOutput, TargetState
from fable.modeling import ArtifactAssociation
from fable.temporal import FeatureState

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
    [[2.0, 2.0, 0.0], [0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [2.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
    dtype=torch.float32,
)
_PREDICTED_Z = torch.tensor([0.1, -0.5, 1.0, 0.0, 2.0], dtype=torch.float32)
_OBSERVATION_SCHEMA = pl.Schema(
    {
        "origin_block": pl.Int64,
        "predicted_action_k": pl.Int64,
        "predicted_minimum_log_base_fee": pl.Float64,
        "minimum_action_k": pl.Int64,
        "immediate_base_fee_per_gas": pl.Int64,
        "immediate_effective_priority_fee_per_gas_p50": pl.Int64,
        "selected_base_fee_per_gas": pl.Int64,
        "selected_effective_priority_fee_per_gas_p50": pl.Int64,
        "deadline_base_fee_per_gas": pl.Int64,
        "deadline_effective_priority_fee_per_gas_p50": pl.Int64,
        "minimum_base_fee_per_gas": pl.Int64,
    }
)


@pytest.fixture(autouse=True)
def _use_single_process_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation_module._runtime, "NUM_WORKERS", 0)


def _experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=BlockWindow(first_parent_block=10, last_parent_block=11),
        validation_window=BlockWindow(first_parent_block=15, last_parent_block=16),
        context_blocks=3,
        horizon_blocks=3,
        ordered_features=("log_base_fee_per_gas",),
    )


def _method() -> Method:
    return Method(
        model=LstmDefinition(family="lstm", hidden=4, layers=1, head_hidden=3, dropout=0.0),
        fit=FitMethod(
            learning_rate=0.01,
            weight_decay=0.0,
            accumulation=1,
            gradient_clip_norm=1.0,
            seed=17,
            max_epochs=2,
            validate_every_completed_epoch=1,
            patience=1,
            min_delta=0.0,
        ),
    )


def _association(experiment: ExperimentSemantics | None = None) -> ArtifactAssociation:
    experiment = experiment or _experiment()
    method = _method()
    return ArtifactAssociation(
        request=TrainRequest(
            workflow="train",
            artifact_id=_ARTIFACT_ID,
            source=SelectedStudySource(
                corpus_id=_CORPUS_ID,
                study_id=_STUDY_ID,
                study_result_index=2,
                experiment=experiment,
            ),
        ),
        feature_state=FeatureState(means=(0.0,), standard_deviations=(1.0,)),
        target_state=TargetState(mean=10.0, standard_deviation=0.25),
        method=method,
    )


def _write_corpus(storage_root: Path, corpus_id: UUID) -> None:
    request = CorpusRequest(
        corpus_id=corpus_id, definition=CorpusDefinition(chain_id=9, first_block=10, last_block=30)
    )
    corpus_directory(storage_root, corpus_id).mkdir(parents=True)
    corpus_json_path(storage_root, corpus_id).write_text(
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "finalized_anchor": {"block_number": 30, "block_hash": "a" * 64},
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
            "effective_priority_fee_per_gas_p50": np.arange(blocks.size, dtype=np.int64),
            "effective_priority_fee_per_gas_p90": 2 * np.arange(blocks.size, dtype=np.int64),
        }
    ).write_parquet(corpus_blocks_path(storage_root, corpus_id))


def _request(
    *, corpus_id: UUID = _CORPUS_ID, testing_window: BlockWindow | None = None
) -> EvaluateRequest:
    return EvaluateRequest(
        workflow="evaluate",
        evaluation_id=_EVALUATION_ID,
        artifact_id=_ARTIFACT_ID,
        corpus_id=corpus_id,
        testing_window=testing_window or BlockWindow(first_parent_block=20, last_parent_block=24),
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
        assert torch.backends.cudnn.allow_tf32
        self.transfers += 1
        return self

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled(inputs.device.type)
        assert not self.training
        size = inputs.shape[0]
        start = self.cursor
        self.cursor += size
        self.batch_sizes.append(size)
        return MinBlockFeeOutput(
            action_logits=_LOGITS[start : start + size],
            minimum_fee_z=_PREDICTED_Z[start : start + size],
        )


def test_evaluate_publishes_exact_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_corpus(tmp_path, _CORPUS_ID)
    association = _association()
    model = _Model()
    monkeypatch.setattr(
        evaluation_module, "load_artifact", lambda storage_root, artifact_id: (association, model)
    )
    monkeypatch.setattr(evaluation_module, "_DEVICE", torch.device("cpu"))
    request = _request()

    evaluation_module.evaluate(request, tmp_path)

    assert model.transfers == 1
    assert model.batch_sizes == [5]
    assert evaluation_json_path(tmp_path, _EVALUATION_ID).read_text() == request.model_dump_json()

    observations = pl.read_parquet(evaluation_observations_path(tmp_path, _EVALUATION_ID))
    assert observations.schema == _OBSERVATION_SCHEMA
    assert observations.rows() == [
        (20, 0, 10.025000000372529, 2, 60, 11, 60, 11, 50, 13, 50),
        (21, 2, 9.875, 2, 70, 12, 40, 14, 40, 14, 40),
        (22, 1, 10.25, 1, 50, 13, 40, 14, 55, 15, 40),
        (23, 0, 10.0, 2, 40, 14, 40, 14, 30, 16, 30),
        (24, 2, 10.5, 1, 55, 15, 30, 17, 30, 17, 30),
    ]


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("source_corpus", ValueError, "artifact source Corpus"),
        ("occupied_canonical", FileExistsError, "evaluations"),
    ],
)
def test_evaluate_rejects_owned_association_and_publication_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, error: type[Exception], match: str
) -> None:
    corpus_id = _OTHER_CORPUS_ID if case == "source_corpus" else _CORPUS_ID
    _write_corpus(tmp_path, corpus_id)
    association = _association()
    model = _Model()
    monkeypatch.setattr(
        evaluation_module, "load_artifact", lambda storage_root, artifact_id: (association, model)
    )
    monkeypatch.setattr(evaluation_module, "_DEVICE", torch.device("cpu"))
    request = _request(corpus_id=corpus_id)
    if case == "occupied_canonical":
        evaluation_directory(tmp_path, _EVALUATION_ID).mkdir(parents=True)

    with pytest.raises(error, match=match):
        evaluation_module.evaluate(request, tmp_path)

    if case == "occupied_canonical":
        scratch = tmp_path / "evaluations" / f".{_EVALUATION_ID}"
        assert sorted(path.name for path in scratch.iterdir()) == [
            "evaluation.json",
            "observations.parquet",
        ]
