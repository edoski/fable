from __future__ import annotations

from pathlib import Path
from uuid import UUID

import polars as pl
import pytest
import torch
from pydantic import ValidationError

from kairos.addresses import (
    study_json_path,
    study_trial_checkpoint_path,
    study_trial_observations_path,
)
from kairos.config import (
    BlockWindow,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    SelectedStudySource,
    TrainingDefinition,
    TuneRequest,
)
from kairos.observations import OBSERVATION_SCHEMA
from kairos.study import (
    RetainedResult,
    Study,
    load_selected_method,
    load_study,
    publish_study,
    reduce_study,
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
    model=LstmDefinition(family="lstm", hidden=256, layers=1, head_hidden=128, dropout=0.2), fit=FIT
)
OTHER_LSTM_METHOD = LSTM_METHOD.model_copy(
    update={"fit": FIT.model_copy(update={"learning_rate": 1e-4})}
)
RESULT = RetainedResult(objective=0.5, selected_epoch=2, completed_epochs=5)


def _observations(objective: float) -> pl.DataFrame:
    minimum = 10
    selected = minimum + round(minimum * objective)
    return pl.DataFrame(
        [
            {
                "origin_block": 220,
                "predicted_action_k": 1,
                "predicted_minimum_log_base_fee": 2.0,
                "minimum_action_k": 0,
                "immediate_base_fee_per_gas": 20,
                "immediate_effective_priority_fee_per_gas_p50": 2,
                "selected_base_fee_per_gas": selected,
                "selected_effective_priority_fee_per_gas_p50": 1,
                "deadline_base_fee_per_gas": 15,
                "deadline_effective_priority_fee_per_gas_p50": 1,
                "minimum_base_fee_per_gas": minimum,
            }
        ],
        schema=OBSERVATION_SCHEMA,
    )


def _retain(
    storage_root: Path, request: TuneRequest, method_index: int, result: RetainedResult
) -> None:
    checkpoint = storage_root / f"checkpoint-{method_index}.ckpt"
    definition = TrainingDefinition(
        experiment=request.experiment, method=request.method_at(method_index)
    )
    torch.save(
        {"hyper_parameters": {"association": definition.model_dump(mode="json")}}, checkpoint
    )
    retain_result(
        storage_root, request, method_index, result, checkpoint, _observations(result.objective)
    )


def _write_canonical_study(storage_root: Path, study: Study) -> None:
    path = study_json_path(storage_root, study.request.study_id)
    path.parent.mkdir(parents=True)
    path.write_text(study.model_dump_json(), encoding="utf-8")
    for index, result in enumerate(study.trials):
        checkpoint = study_trial_checkpoint_path(storage_root, study.request.study_id, index)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        _observations(result.objective).write_parquet(
            study_trial_observations_path(storage_root, study.request.study_id, index)
        )


def _experiment(*, shift: int = 0) -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=BlockWindow(first_parent_block=100 + shift, last_parent_block=199 + shift),
        validation_window=BlockWindow(
            first_parent_block=220 + shift, last_parent_block=249 + shift
        ),
        context_blocks=200,
        horizon_blocks=5,
        ordered_features=("log_base_fee_per_gas", "gas_utilization"),
    )


def _request(
    methods: tuple[Method, ...] = (LSTM_METHOD,), *, corpus_id: UUID = CORPUS_ID
) -> TuneRequest:
    return TuneRequest(
        workflow="tune",
        study_id=STUDY_ID,
        corpus_id=corpus_id,
        experiment=_experiment(),
        methods=methods,
    )


def test_retain_publish_and_load_selected_method_in_request_order(tmp_path: Path) -> None:
    request = _request((LSTM_METHOD, OTHER_LSTM_METHOD))
    first = RetainedResult(objective=0.4, selected_epoch=3, completed_epochs=8)
    second = RetainedResult(objective=0.3, selected_epoch=4, completed_epochs=9)

    _retain(tmp_path, request, 1, second)
    _retain(tmp_path, request, 0, first)

    publish_study(tmp_path, STUDY_ID)
    source = SelectedStudySource(
        corpus_id=CORPUS_ID,
        study_id=STUDY_ID,
        study_result_index=1,
        experiment=_experiment(shift=1_000),
    )

    selected = load_selected_method(tmp_path, source)
    canonical = load_study(tmp_path, STUDY_ID)

    assert canonical == Study(request=request, trials=(first, second))
    assert selected == OTHER_LSTM_METHOD
    reduced = reduce_study(tmp_path, STUDY_ID)
    assert reduced["method_index"].to_list() == [0, 1]
    assert reduced["base_fee_optimality_gap"].to_list() == [0.4, 0.3]
    canonical_directory = study_json_path(tmp_path, STUDY_ID).parent
    assert {path.name for path in canonical_directory.iterdir()} == {"study.json", "trials"}
    assert {path.name for path in (canonical_directory / "trials" / "0").iterdir()} == {
        "selected.ckpt",
        "validation.parquet",
    }
    assert not (tmp_path / "studies" / f".{STUDY_ID}").exists()


def test_retained_result_rejects_invalid_epoch_bounds() -> None:
    with pytest.raises(ValidationError, match="selected_epoch must not exceed completed_epochs"):
        RetainedResult(objective=0.5, selected_epoch=3, completed_epochs=2)


@pytest.mark.parametrize(
    ("trials", "message"),
    [
        ((RESULT, RESULT), "trials must align with request methods"),
        (
            (RetainedResult(objective=0.5, selected_epoch=1, completed_epochs=13),),
            "completed_epochs must not exceed method.fit.max_epochs",
        ),
    ],
)
def test_study_rejects_trials_outside_request_contract(
    trials: tuple[RetainedResult, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Study(request=_request(), trials=trials)


def test_study_selects_the_earliest_minimum() -> None:
    request = _request((LSTM_METHOD, OTHER_LSTM_METHOD))
    first = RESULT.model_copy(update={"objective": 0.4})
    second = RESULT.model_copy(update={"objective": 0.4})
    study = Study(request=request, trials=(first, second))

    assert study.best_result() == (0, first)


def test_publish_study_rejects_missing_result(tmp_path: Path) -> None:
    request = _request((LSTM_METHOD, OTHER_LSTM_METHOD))
    _retain(tmp_path, request, 0, RESULT)

    with pytest.raises(ValueError, match="retained trials do not match TuneRequest methods"):
        publish_study(tmp_path, STUDY_ID)

    assert not study_json_path(tmp_path, STUDY_ID).exists()


def test_publish_study_rejects_mismatched_result_request(tmp_path: Path) -> None:
    request = _request((LSTM_METHOD, OTHER_LSTM_METHOD))
    conflicting = _request((LSTM_METHOD, OTHER_LSTM_METHOD), corpus_id=OTHER_CORPUS_ID)
    second = RetainedResult(objective=0.4, selected_epoch=3, completed_epochs=8)
    _retain(tmp_path, request, 0, RESULT)
    _retain(tmp_path, conflicting, 1, second)

    with pytest.raises(ValueError, match="result requests must be identical"):
        publish_study(tmp_path, STUDY_ID)

    assert not study_json_path(tmp_path, STUDY_ID).exists()


def test_load_selected_method_rejects_corpus_mismatch(tmp_path: Path) -> None:
    _write_canonical_study(tmp_path, Study(request=_request(), trials=(RESULT,)))
    source = SelectedStudySource(
        corpus_id=OTHER_CORPUS_ID, study_id=STUDY_ID, study_result_index=0, experiment=_experiment()
    )

    with pytest.raises(ValueError, match="Corpus ID does not match"):
        load_selected_method(tmp_path, source)


def test_load_study_rejects_non_strict_json(tmp_path: Path) -> None:
    study = Study(request=_request(), trials=(RESULT,))
    _write_canonical_study(tmp_path, study)
    canonical = study_json_path(tmp_path, STUDY_ID)
    canonical.write_text(
        study.model_dump_json().replace('"objective":0.5', '"objective":"0.5"'), encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_study(tmp_path, STUDY_ID)


def test_load_study_rejects_embedded_id_mismatch(tmp_path: Path) -> None:
    request = _request().model_copy(update={"study_id": OTHER_STUDY_ID})
    study = Study(request=request, trials=(RESULT,))
    canonical = study_json_path(tmp_path, STUDY_ID)
    canonical.parent.mkdir(parents=True)
    canonical.write_text(study.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="Study ID does not match requested Study ID"):
        load_study(tmp_path, STUDY_ID)


def test_publish_study_preserves_canonical_created_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _retain(tmp_path, _request(), 0, RESULT)
    canonical = study_json_path(tmp_path, STUDY_ID).parent
    real_rename = Path.rename

    def create_collision(source: Path, target: Path) -> Path:
        if target == canonical:
            canonical.mkdir()
            (canonical / "occupied").write_text("occupied", encoding="utf-8")
        return real_rename(source, target)

    monkeypatch.setattr(Path, "rename", create_collision)

    with pytest.raises(FileExistsError):
        publish_study(tmp_path, STUDY_ID)

    assert (canonical / "occupied").read_text(encoding="utf-8") == "occupied"
    assert (tmp_path / "studies" / f".{STUDY_ID}").is_dir()
