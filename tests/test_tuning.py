from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest
from pytest import MonkeyPatch

from fable import tuning
from fable.config import (
    BlockWindow,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    TuneRequest,
)
from fable.study import RetainedResult

STUDY_ID = UUID("10000000-0000-4000-8000-000000000001")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")

FIT = FitMethod(
    learning_rate=3e-4,
    weight_decay=1e-4,
    accumulation=1,
    gradient_clip_norm=0.75,
    seed=17,
    max_epochs=12,
    validate_every_completed_epoch=1,
    patience=4,
    min_delta=0.01,
)
METHOD = Method(
    model=LstmDefinition(
        family="lstm",
        hidden=16,
        layers=1,
        head_hidden=8,
        dropout=0.2,
    ),
    fit=FIT,
)
OTHER_METHOD = METHOD.model_copy(
    update={"fit": FIT.model_copy(update={"seed": 18})},
)
EXPERIMENT = ExperimentSemantics(
    training_window=BlockWindow(
        first_parent_block=10,
        last_parent_block=20,
    ),
    validation_window=BlockWindow(
        first_parent_block=30,
        last_parent_block=35,
    ),
    context_blocks=3,
    horizon_blocks=2,
    ordered_features=("log_base_fee_per_gas",),
)
REQUEST = TuneRequest(
    workflow="tune",
    study_id=STUDY_ID,
    corpus_id=CORPUS_ID,
    experiment=EXPERIMENT,
    methods=(METHOD, OTHER_METHOD),
)


def test_run_candidate_publishes_indexed_result_and_removes_scratch(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    result = RetainedResult(
        objective=0.4,
        selected_epoch=3,
        completed_epochs=8,
    )
    study_scratch = tmp_path / "studies" / f".{STUDY_ID}"
    candidate_scratch = study_scratch / "candidate-1"

    def run_fit(*_: object) -> RetainedResult:
        candidate_scratch.mkdir(parents=True)
        return result

    monkeypatch.setattr(tuning, "fit_candidate", run_fit)

    tuning.run_candidate(tmp_path, REQUEST, 1)

    retained = json.loads((study_scratch / "result-1.json").read_bytes())
    assert retained == {
        "request": REQUEST.model_dump(mode="json"),
        "method_index": 1,
        "result": result.model_dump(mode="json"),
    }
    assert not (study_scratch / "candidate-1").exists()


def test_run_candidate_preserves_last_checkpoint_after_retention_failure(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    scratch = tmp_path / "studies" / f".{STUDY_ID}" / "candidate-0"
    result = RetainedResult(
        objective=0.4,
        selected_epoch=3,
        completed_epochs=8,
    )

    def run_fit(*_: object) -> RetainedResult:
        scratch.mkdir(parents=True)
        (scratch / "last.ckpt").write_bytes(b"checkpoint")
        return result

    def retain_result(*_: object) -> None:
        raise RuntimeError("retention failed")

    monkeypatch.setattr(tuning, "fit_candidate", run_fit)
    monkeypatch.setattr(tuning, "retain_result", retain_result)

    with pytest.raises(RuntimeError, match="retention failed"):
        tuning.run_candidate(tmp_path, REQUEST, 0)

    assert (scratch / "last.ckpt").read_bytes() == b"checkpoint"
