from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from spice.addresses import study_json_path
from spice.config import (
    AdamWMethod,
    ExperimentSemantics,
    FitMethod,
    LossDefinition,
    LstmCapacity,
    LstmDefinition,
    LstmMethod,
    LstmMethodSpace,
    Method,
    MethodSpace,
    OriginWindow,
    SelectedStudySource,
    StudyDefinition,
    TrainingDefinition,
    TransformerCapacity,
    TransformerDefinition,
    TransformerLstmCapacity,
    TransformerLstmDefinition,
    TransformerLstmMethod,
    TransformerLstmMethodSpace,
    TransformerMethod,
    TransformerMethodSpace,
    TuneRequest,
)
from spice.study import (
    RetainedResult,
    Study,
    apply_method,
    materialize_selected_training,
    publish_study,
    retain_result,
    training_definition_from_method,
)

STUDY_ID = UUID("10000000-0000-4000-8000-000000000001")
OTHER_STUDY_ID = UUID("10000000-0000-4000-8000-000000000002")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")
OTHER_CORPUS_ID = UUID("20000000-0000-4000-8000-000000000002")

FIT = FitMethod(
    accumulation=3,
    gradient_clip_norm=0.75,
    scheduler="none",
    seed=17,
    max_epochs=12,
    validate_every_completed_epoch=2,
    patience=4,
    min_delta=0.01,
    improvement="strict_lower",
    restore="earliest_best",
)

LSTM_CAPACITY = LstmCapacity(hidden=256, layers=1, head_hidden=128)
TRANSFORMER_CAPACITY = TransformerCapacity(
    model_width=192,
    attention_heads=4,
    transformer_layers=3,
    feedforward_width=384,
    head_hidden=192,
)
TRANSFORMER_LSTM_CAPACITY = TransformerLstmCapacity(
    model_width=192,
    attention_heads=4,
    transformer_layers=3,
    feedforward_width=384,
    lstm_hidden=192,
    lstm_layers=1,
    head_hidden=192,
)

LSTM_METHOD = LstmMethod(
    family="lstm",
    capacity=LSTM_CAPACITY,
    dropout=0.2,
    optimizer=AdamWMethod(learning_rate=3e-4, weight_decay=1e-4),
    training_batch=48,
    fit=FIT,
)
TRANSFORMER_METHOD = TransformerMethod(
    family="transformer",
    capacity=TRANSFORMER_CAPACITY,
    dropout=0.2,
    optimizer=AdamWMethod(learning_rate=1e-4, weight_decay=1e-4),
    training_batch=48,
    fit=FIT,
)
TRANSFORMER_LSTM_METHOD = TransformerLstmMethod(
    family="transformer_lstm",
    capacity=TRANSFORMER_LSTM_CAPACITY,
    dropout=0.2,
    optimizer=AdamWMethod(learning_rate=1e-4, weight_decay=1e-4),
    training_batch=48,
    fit=FIT,
)

LSTM_SPACE = LstmMethodSpace(
    family="lstm",
    methods=(LSTM_METHOD,),
)
TRANSFORMER_SPACE = TransformerMethodSpace(
    family="transformer",
    methods=(TRANSFORMER_METHOD,),
)
TRANSFORMER_LSTM_SPACE = TransformerLstmMethodSpace(
    family="transformer_lstm",
    methods=(TRANSFORMER_LSTM_METHOD,),
)


def _experiment(*, shift: int = 0) -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=OriginWindow(
            role="training",
            first_parent_block=100 + shift,
            last_parent_block=199 + shift,
        ),
        validation_window=OriginWindow(
            role="validation",
            first_parent_block=220 + shift,
            last_parent_block=249 + shift,
        ),
        context_blocks=200,
        horizon_blocks=5,
        ordered_features=("base_fee", "gas_used"),
        loss=LossDefinition(
            classification_algorithm="cross_entropy",
            classification_weighting="unweighted",
            regression_algorithm="smooth_l1",
            regression_threshold=0.75,
            classification_scale=1.25,
            regression_scale=0.5,
        ),
    )


def _request(
    space: MethodSpace = LSTM_SPACE,
    *,
    study_id: UUID = STUDY_ID,
    corpus_id: UUID = CORPUS_ID,
) -> TuneRequest:
    return TuneRequest(
        workflow="tune",
        study_id=study_id,
        corpus_id=corpus_id,
        study_definition=StudyDefinition(experiment=_experiment(), method_space=space),
    )


@pytest.mark.parametrize(
    ("space", "method", "model"),
    [
        (
            LSTM_SPACE,
            LSTM_METHOD,
            LstmDefinition(
                family="lstm",
                hidden=256,
                layers=1,
                head_hidden=128,
                dropout=0.2,
            ),
        ),
        (
            TRANSFORMER_SPACE,
            TRANSFORMER_METHOD,
            TransformerDefinition(
                family="transformer",
                model_width=192,
                attention_heads=4,
                transformer_layers=3,
                feedforward_width=384,
                head_hidden=192,
                dropout=0.2,
            ),
        ),
        (
            TRANSFORMER_LSTM_SPACE,
            TRANSFORMER_LSTM_METHOD,
            TransformerLstmDefinition(
                family="transformer_lstm",
                model_width=192,
                attention_heads=4,
                transformer_layers=3,
                feedforward_width=384,
                lstm_hidden=192,
                lstm_layers=1,
                head_hidden=192,
                dropout=0.2,
            ),
        ),
    ],
)
def test_composes_all_three_method_families(
    space: MethodSpace,
    method: Method,
    model: LstmDefinition | TransformerDefinition | TransformerLstmDefinition,
) -> None:
    request = _request(space)

    pure = training_definition_from_method(request.study_definition.experiment, method)

    assert pure == TrainingDefinition(
        experiment=request.study_definition.experiment,
        model=model,
        optimizer=method.optimizer,
        training_batch=method.training_batch,
        fit=method.fit,
    )
    assert apply_method(request, method) == pure


def test_retain_publish_and_materialize_selected_training(tmp_path: Path) -> None:
    request = _request()
    first = RetainedResult(
        method=LSTM_METHOD,
        objective=-0.4,
        selected_epoch=3,
        completed_epochs=8,
    )
    duplicate_at_cap = RetainedResult(
        method=LSTM_METHOD,
        objective=-0.4,
        selected_epoch=LSTM_METHOD.fit.max_epochs,
        completed_epochs=LSTM_METHOD.fit.max_epochs,
    )

    retain_result(tmp_path, request, first)
    retain_result(tmp_path, request, duplicate_at_cap)
    progress = tmp_path / "studies" / f".{STUDY_ID}" / "progress.json"
    assert Study.model_validate_json(progress.read_bytes(), strict=True) == Study(
        request=request,
        trials=(first, duplicate_at_cap),
    )

    publish_study(tmp_path, STUDY_ID)
    source = SelectedStudySource(
        kind="selected_study",
        corpus_id=CORPUS_ID,
        study_id=STUDY_ID,
        study_result_index=1,
        experiment=_experiment(shift=1_000),
    )

    selected = materialize_selected_training(tmp_path, source)

    assert not progress.exists()
    assert selected.study_result_index == 1
    assert selected.method == LSTM_METHOD
    assert selected.training_definition == training_definition_from_method(
        source.experiment,
        LSTM_METHOD,
    )


@pytest.mark.parametrize(
    "case",
    [
        "nonfinite_objective",
        "selected_after_completed",
        "completed_above_max",
        "method_outside_space",
        "conflicting_append",
        "study_mismatch",
        "corpus_mismatch",
        "result_index_out_of_range",
        "canonical_exists",
    ],
)
def test_rejects_invalid_study_operations(case: str, tmp_path: Path) -> None:
    request = _request()
    result = RetainedResult(
        method=LSTM_METHOD,
        objective=0.5,
        selected_epoch=2,
        completed_epochs=5,
    )

    outside = LstmMethod(
        family="lstm",
        capacity=LSTM_CAPACITY,
        dropout=0.3,
        optimizer=LSTM_METHOD.optimizer,
        training_batch=48,
        fit=FIT,
    )
    source: SelectedStudySource | None = None
    if case == "conflicting_append":
        retain_result(tmp_path, request, result)
    elif case == "study_mismatch":
        canonical = study_json_path(tmp_path, OTHER_STUDY_ID)
        canonical.parent.mkdir(parents=True)
        canonical.write_text(Study(request=request, trials=(result,)).model_dump_json())
        source = SelectedStudySource(
            kind="selected_study",
            corpus_id=CORPUS_ID,
            study_id=OTHER_STUDY_ID,
            study_result_index=0,
            experiment=_experiment(),
        )
    elif case == "corpus_mismatch":
        canonical = study_json_path(tmp_path, STUDY_ID)
        canonical.parent.mkdir(parents=True)
        canonical.write_text(Study(request=request, trials=(result,)).model_dump_json())
        source = SelectedStudySource(
            kind="selected_study",
            corpus_id=OTHER_CORPUS_ID,
            study_id=STUDY_ID,
            study_result_index=0,
            experiment=_experiment(),
        )
    elif case == "result_index_out_of_range":
        canonical = study_json_path(tmp_path, STUDY_ID)
        canonical.parent.mkdir(parents=True)
        canonical.write_text(Study(request=request, trials=(result,)).model_dump_json())
        source = SelectedStudySource(
            kind="selected_study",
            corpus_id=CORPUS_ID,
            study_id=STUDY_ID,
            study_result_index=1,
            experiment=_experiment(),
        )
    elif case == "canonical_exists":
        retain_result(tmp_path, request, result)
        canonical = study_json_path(tmp_path, STUDY_ID)
        canonical.write_text("occupied")

    error = (
        FileExistsError
        if case == "canonical_exists"
        else IndexError
        if case == "result_index_out_of_range"
        else ValueError
    )
    with pytest.raises(error):
        if case == "nonfinite_objective":
            RetainedResult(
                method=LSTM_METHOD,
                objective=float("nan"),
                selected_epoch=1,
                completed_epochs=1,
            )
        elif case == "selected_after_completed":
            RetainedResult(
                method=LSTM_METHOD,
                objective=0.5,
                selected_epoch=3,
                completed_epochs=2,
            )
        elif case == "completed_above_max":
            RetainedResult(
                method=LSTM_METHOD,
                objective=0.5,
                selected_epoch=1,
                completed_epochs=13,
            )
        elif case == "method_outside_space":
            Study(
                request=request,
                trials=(
                    RetainedResult(
                        method=outside,
                        objective=0.5,
                        selected_epoch=1,
                        completed_epochs=1,
                    ),
                ),
            )
        elif case == "conflicting_append":
            retain_result(tmp_path, _request(corpus_id=OTHER_CORPUS_ID), result)
        elif case in {"study_mismatch", "corpus_mismatch", "result_index_out_of_range"}:
            assert source is not None
            materialize_selected_training(tmp_path, source)
        else:
            publish_study(tmp_path, STUDY_ID)
