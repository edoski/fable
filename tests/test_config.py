from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from fable.config import (
    WORKFLOW_REQUEST_ADAPTER,
    BlockWindow,
    CorpusDefinition,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    SelectedStudySource,
    TrainRequest,
    TransformerDefinition,
    TuneRequest,
)

STUDY_ID = UUID("00000000-0000-4000-8000-000000000002")
CORPUS_ID = UUID("00000000-0000-4000-8000-000000000001")


def _window(first: int = 210, last: int = 249) -> BlockWindow:
    return BlockWindow(
        first_parent_block=first,
        last_parent_block=last,
    )


def _experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=_window(100, 199),
        validation_window=_window(),
        context_blocks=20,
        horizon_blocks=10,
        ordered_features=("log_base_fee_per_gas", "gas_utilization"),
    )


def _method() -> Method:
    return Method(
        model=LstmDefinition(
            family="lstm",
            hidden=32,
            layers=1,
            head_hidden=16,
            dropout=0.2,
        ),
        fit=FitMethod(
            learning_rate=0.001,
            weight_decay=0.0,
            accumulation=2,
            gradient_clip_norm=0.75,
            seed=17,
            max_epochs=9,
            validate_every_completed_epoch=2,
            patience=3,
            min_delta=0.01,
        ),
    )


def _invalid_cases() -> tuple[tuple[type[object], dict[str, object], str], ...]:
    experiment = _experiment()
    method = _method()
    return (
        (
            CorpusDefinition,
            {"chain_id": 1, "first_block": 2, "last_block": 1},
            "last_block must not precede first_block",
        ),
        (
            TransformerDefinition,
            {
                "family": "transformer",
                "model_width": 31,
                "attention_heads": 1,
                "transformer_layers": 1,
                "feedforward_width": 32,
                "head_hidden": 8,
                "dropout": 0.2,
            },
            "model_width must be even",
        ),
        (
            TransformerDefinition,
            {
                "family": "transformer",
                "model_width": 30,
                "attention_heads": 4,
                "transformer_layers": 1,
                "feedforward_width": 32,
                "head_hidden": 8,
                "dropout": 0.2,
            },
            "model_width must be divisible by attention_heads",
        ),
        (
            TuneRequest,
            {
                "workflow": "tune",
                "study_id": STUDY_ID,
                "corpus_id": CORPUS_ID,
                "experiment": experiment,
                "methods": (method, method),
            },
            "methods must not contain duplicates",
        ),
        (
            TuneRequest,
            {
                "workflow": "tune",
                "study_id": STUDY_ID,
                "corpus_id": CORPUS_ID,
                "experiment": experiment,
                "methods": (
                    method,
                    Method(
                        model=TransformerDefinition(
                            family="transformer",
                            model_width=32,
                            attention_heads=4,
                            transformer_layers=1,
                            feedforward_width=64,
                            head_hidden=8,
                            dropout=0.2,
                        ),
                        fit=method.fit,
                    ),
                ),
            },
            "methods must use one model family",
        ),
        (
            ExperimentSemantics,
            {
                **experiment.model_dump(),
                "validation_window": BlockWindow(
                    first_parent_block=209,
                    last_parent_block=249,
                ),
            },
            "validation_window must follow complete training outcomes",
        ),
        (
            ExperimentSemantics,
            {
                **experiment.model_dump(),
                "ordered_features": ("unsupported",),
            },
            "Input should be",
        ),
    )


@pytest.mark.parametrize(("value_type", "payload", "message"), _invalid_cases())
def test_domain_contract_rejects_invalid_values(
    value_type: type[object],
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        value_type(**payload)


def test_workflow_request_rejects_tune_json() -> None:
    with pytest.raises(ValidationError, match="tune"):
        WORKFLOW_REQUEST_ADAPTER.validate_json('{"workflow":"tune"}')


def test_request_defaults_mint_and_persist_destination_identity() -> None:
    experiment = _experiment()
    train = TrainRequest(
        source=SelectedStudySource(
            corpus_id=CORPUS_ID,
            study_id=STUDY_ID,
            study_result_index=0,
            experiment=experiment,
        )
    )
    tune = TuneRequest(
        corpus_id=CORPUS_ID,
        experiment=experiment,
        methods=(_method(),),
    )
    evaluate = EvaluateRequest(
        artifact_id=train.artifact_id,
        corpus_id=CORPUS_ID,
        testing_window=_window(),
    )

    for request, destination_field in (
        (train, "artifact_id"),
        (tune, "study_id"),
        (evaluate, "evaluation_id"),
    ):
        serialized = request.model_dump_json()
        payload = json.loads(serialized)

        assert payload["workflow"] == request.workflow
        assert payload[destination_field] == str(getattr(request, destination_field))
        assert getattr(request, destination_field).version == 4
        assert type(request).model_validate_json(serialized, strict=True) == request
