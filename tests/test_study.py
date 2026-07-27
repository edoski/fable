from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from fable.addresses import study_json_path
from fable.config import (
    BlockWindow,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    SelectedStudySource,
    TuneRequest,
)
from fable.study import (
    RetainedResult,
    Study,
    load_selected_method,
    load_study,
    publish_study,
    retain_result,
)

STUDY_ID = UUID("10000000-0000-4000-8000-000000000001")
OTHER_STUDY_ID = UUID("10000000-0000-4000-8000-000000000002")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")
OTHER_CORPUS_ID = UUID("20000000-0000-4000-8000-000000000002")

FIT = FitMethod(
    learning_rate=3e-4,
    weight_decay=1e-4,
    accumulation=3,
    gradient_clip_norm=0.75,
    seed=17,
    max_epochs=12,
    validate_every_completed_epoch=2,
    patience=4,
    min_delta=0.01,
)

LSTM_METHOD = Method(
    model=LstmDefinition(
        family="lstm",
        hidden=256,
        layers=1,
        head_hidden=128,
        dropout=0.2,
    ),
    fit=FIT,
)
OTHER_LSTM_METHOD = LSTM_METHOD.model_copy(
    update={"fit": FIT.model_copy(update={"learning_rate": 1e-4})},
)
RESULT = RetainedResult(
    method=LSTM_METHOD,
    objective=0.5,
    selected_epoch=2,
    completed_epochs=5,
)


def _experiment(*, shift: int = 0) -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=BlockWindow(
            first_parent_block=100 + shift,
            last_parent_block=199 + shift,
        ),
        validation_window=BlockWindow(
            first_parent_block=220 + shift,
            last_parent_block=249 + shift,
        ),
        context_blocks=200,
        horizon_blocks=5,
        ordered_features=("base_fee", "gas_used"),
    )


def _request(
    methods: tuple[Method, ...] = (LSTM_METHOD,),
    *,
    corpus_id: UUID = CORPUS_ID,
) -> TuneRequest:
    return TuneRequest(
        workflow="tune",
        study_id=STUDY_ID,
        corpus_id=corpus_id,
        experiment=_experiment(),
        methods=methods,
    )


def test_retain_publish_and_load_selected_method_in_request_order(
    tmp_path: Path,
) -> None:
    request = _request((LSTM_METHOD, OTHER_LSTM_METHOD))
    first = RetainedResult(
        method=LSTM_METHOD,
        objective=-0.4,
        selected_epoch=3,
        completed_epochs=8,
    )
    second = RetainedResult(
        method=OTHER_LSTM_METHOD,
        objective=-0.3,
        selected_epoch=4,
        completed_epochs=9,
    )

    retain_result(tmp_path, request, 1, second)
    retain_result(tmp_path, request, 0, first)

    publish_study(tmp_path, STUDY_ID)
    source = SelectedStudySource(
        kind="selected_study",
        corpus_id=CORPUS_ID,
        study_id=STUDY_ID,
        study_result_index=1,
        experiment=_experiment(shift=1_000),
    )

    selected = load_selected_method(tmp_path, source)
    canonical_path = study_json_path(tmp_path, STUDY_ID)
    canonical = load_study(tmp_path, STUDY_ID)

    assert canonical == Study(request=request, trials=(first, second))
    assert selected == OTHER_LSTM_METHOD
    assert not (tmp_path / "studies" / f".{STUDY_ID}").exists()
    assert not canonical_path.with_name(f".{canonical_path.name}").exists()


@pytest.mark.parametrize(
    ("selected_epoch", "completed_epochs", "message"),
    [
        (3, 2, "selected_epoch must not exceed completed_epochs"),
        (1, 13, "completed_epochs must not exceed method.fit.max_epochs"),
    ],
)
def test_retained_result_rejects_invalid_epoch_bounds(
    selected_epoch: int,
    completed_epochs: int,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        RetainedResult(
            method=LSTM_METHOD,
            objective=0.5,
            selected_epoch=selected_epoch,
            completed_epochs=completed_epochs,
        )


def test_study_rejects_method_outside_request() -> None:
    outside = LSTM_METHOD.model_copy(
        update={"model": LSTM_METHOD.model.model_copy(update={"dropout": 0.3})}
    )

    with pytest.raises(ValidationError, match="Method is outside the TuneRequest"):
        Study(
            request=_request(),
            trials=(
                RetainedResult(
                    method=outside,
                    objective=0.5,
                    selected_epoch=1,
                    completed_epochs=1,
                ),
            ),
        )


def test_publish_study_rejects_missing_result(tmp_path: Path) -> None:
    request = _request((LSTM_METHOD, OTHER_LSTM_METHOD))
    retain_result(tmp_path, request, 0, RESULT)

    with pytest.raises(ValueError, match="result files do not match TuneRequest methods"):
        publish_study(tmp_path, STUDY_ID)

    assert not study_json_path(tmp_path, STUDY_ID).exists()


def test_publish_study_rejects_mismatched_result_request(tmp_path: Path) -> None:
    request = _request((LSTM_METHOD, OTHER_LSTM_METHOD))
    conflicting = _request(
        (LSTM_METHOD, OTHER_LSTM_METHOD),
        corpus_id=OTHER_CORPUS_ID,
    )
    second = RetainedResult(
        method=OTHER_LSTM_METHOD,
        objective=0.4,
        selected_epoch=3,
        completed_epochs=8,
    )
    retain_result(tmp_path, request, 0, RESULT)
    retain_result(tmp_path, conflicting, 1, second)

    with pytest.raises(ValueError, match="result requests must be identical"):
        publish_study(tmp_path, STUDY_ID)

    assert not study_json_path(tmp_path, STUDY_ID).exists()


def test_publish_study_rejects_result_for_wrong_method_index(tmp_path: Path) -> None:
    request = _request((LSTM_METHOD, OTHER_LSTM_METHOD))
    retain_result(tmp_path, request, 0, RESULT)
    scratch = tmp_path / "studies" / f".{STUDY_ID}"
    (scratch / "result-1.json").write_text(
        Study(request=request, trials=(RESULT,)).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="result method does not match request index"):
        publish_study(tmp_path, STUDY_ID)

    assert not study_json_path(tmp_path, STUDY_ID).exists()


def test_publish_study_rejects_result_with_multiple_trials(tmp_path: Path) -> None:
    request = _request()
    scratch = tmp_path / "studies" / f".{STUDY_ID}"
    scratch.mkdir(parents=True)
    (scratch / "result-0.json").write_text(
        Study(request=request, trials=(RESULT, RESULT)).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one trial"):
        publish_study(tmp_path, STUDY_ID)

    assert not study_json_path(tmp_path, STUDY_ID).exists()


def test_load_selected_method_rejects_corpus_mismatch(tmp_path: Path) -> None:
    canonical = study_json_path(tmp_path, STUDY_ID)
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        Study(request=_request(), trials=(RESULT,)).model_dump_json(),
        encoding="utf-8",
    )
    source = SelectedStudySource(
        kind="selected_study",
        corpus_id=OTHER_CORPUS_ID,
        study_id=STUDY_ID,
        study_result_index=0,
        experiment=_experiment(),
    )

    with pytest.raises(ValueError, match="Corpus ID does not match"):
        load_selected_method(tmp_path, source)


def test_load_study_rejects_non_strict_json(tmp_path: Path) -> None:
    canonical = study_json_path(tmp_path, STUDY_ID)
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        Study(request=_request(), trials=(RESULT,))
        .model_dump_json()
        .replace('"objective":0.5', '"objective":"0.5"'),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_study(tmp_path, STUDY_ID)


def test_load_study_rejects_embedded_id_mismatch(tmp_path: Path) -> None:
    canonical = study_json_path(tmp_path, STUDY_ID)
    canonical.parent.mkdir(parents=True)
    request = _request().model_copy(update={"study_id": OTHER_STUDY_ID})
    canonical.write_text(
        Study(request=request, trials=(RESULT,)).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Study ID does not match requested Study ID"):
        load_study(tmp_path, STUDY_ID)


def test_publish_study_preserves_canonical_created_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retain_result(tmp_path, _request(), 0, RESULT)
    canonical = study_json_path(tmp_path, STUDY_ID)
    real_link = os.link

    def create_collision(source: Path, target: Path) -> None:
        canonical.write_text("occupied", encoding="utf-8")
        real_link(source, target)

    monkeypatch.setattr(os, "link", create_collision)

    with pytest.raises(FileExistsError):
        publish_study(tmp_path, STUDY_ID)

    assert canonical.read_text(encoding="utf-8") == "occupied"
    assert not (tmp_path / "studies" / f".{STUDY_ID}").exists()


def test_published_study_survives_hidden_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retain_result(tmp_path, _request(), 0, RESULT)
    canonical = study_json_path(tmp_path, STUDY_ID)
    hidden = canonical.with_name(f".{canonical.name}")
    cleanup_attempted = False
    real_unlink = Path.unlink

    def fail_hidden_cleanup(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        nonlocal cleanup_attempted
        if path == hidden:
            cleanup_attempted = True
            raise OSError("cleanup failed")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_hidden_cleanup)

    publish_study(tmp_path, STUDY_ID)

    assert cleanup_attempted
    assert canonical.is_file()
    assert hidden.is_file()
