from __future__ import annotations

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
from fable.study import RetainedResult, Study

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
    methods=(METHOD,),
)
MULTI_METHOD_REQUEST = REQUEST.model_copy(
    update={"methods": (METHOD, OTHER_METHOD)},
)
RESULT = RetainedResult(
    method=METHOD,
    objective=0.4,
    selected_epoch=3,
    completed_epochs=8,
)


def test_run_candidate_publishes_result_and_removes_candidate_scratch(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    corpus = object()
    prepared = object()

    def load_corpus(storage_root: Path, corpus_id: UUID) -> object:
        assert (storage_root, corpus_id) == (tmp_path, CORPUS_ID)
        return corpus

    def prepare_fit_history(
        loaded_corpus: object,
        experiment: ExperimentSemantics,
    ) -> object:
        assert (loaded_corpus, experiment) == (corpus, EXPERIMENT)
        return prepared

    def run_fit(
        request: TuneRequest,
        method: Method,
        preparation: object,
        scratch: Path,
    ) -> RetainedResult:
        assert scratch.is_dir()
        assert (request, method, preparation) == (
            REQUEST,
            METHOD,
            prepared,
        )
        return RESULT

    monkeypatch.setattr(tuning, "load_corpus", load_corpus)
    monkeypatch.setattr(tuning, "prepare_fit_history", prepare_fit_history)
    monkeypatch.setattr(tuning, "_run_candidate", run_fit)

    tuning.run_candidate(tmp_path, REQUEST, METHOD)

    scratch = tmp_path / "studies" / f".{STUDY_ID}" / "candidate-0"
    result_path = scratch.parent / "result-0.json"
    assert Study.model_validate_json(result_path.read_bytes(), strict=True) == Study(
        request=REQUEST,
        trials=(RESULT,),
    )
    assert not scratch.exists()


@pytest.mark.parametrize(
    ("method", "method_index"),
    [(METHOD, 0), (OTHER_METHOD, 1)],
)
def test_run_candidate_uses_method_index_scratch(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    method: Method,
    method_index: int,
) -> None:
    def run_fit(
        request: TuneRequest,
        active_method: Method,
        preparation: object,
        scratch: Path,
    ) -> RetainedResult:
        expected = tmp_path / "studies" / f".{STUDY_ID}" / f"candidate-{method_index}"
        assert scratch == expected
        return RESULT.model_copy(update={"method": active_method})

    monkeypatch.setattr(tuning, "load_corpus", lambda *_: object())
    monkeypatch.setattr(tuning, "prepare_fit_history", lambda *_: object())
    monkeypatch.setattr(tuning, "_run_candidate", run_fit)
    monkeypatch.setattr(tuning, "retain_result", lambda *_: None)

    tuning.run_candidate(tmp_path, MULTI_METHOD_REQUEST, method)


@pytest.mark.parametrize("failure", ["fit", "retention"])
def test_run_candidate_preserves_scratch_after_failure(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    failure: str,
) -> None:
    scratch = tmp_path / "studies" / f".{STUDY_ID}" / "candidate-0"

    def run_fit(*_: object) -> RetainedResult:
        (scratch / "last.ckpt").write_bytes(b"checkpoint")
        if failure == "fit":
            raise RuntimeError("fit failed")
        return RESULT

    def retain_result(*_: object) -> None:
        if failure == "retention":
            raise RuntimeError("retention failed")

    monkeypatch.setattr(tuning, "load_corpus", lambda *_: object())
    monkeypatch.setattr(tuning, "prepare_fit_history", lambda *_: object())
    monkeypatch.setattr(tuning, "_run_candidate", run_fit)
    monkeypatch.setattr(tuning, "retain_result", retain_result)

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        tuning.run_candidate(tmp_path, REQUEST, METHOD)

    assert (scratch / "last.ckpt").read_bytes() == b"checkpoint"
