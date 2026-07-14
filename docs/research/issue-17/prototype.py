"""Interactive shell for the disposable Issue 17 logic prototype."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any
from uuid import UUID

from prototype_logic import (
    FIXED_FIT,
    LSTM_METHOD_SPACE,
    METHOD_ADAPTER,
    TRANSFORMER_LSTM_METHOD_SPACE,
    TRANSFORMER_METHOD_SPACE,
    AdamWMethod,
    ExperimentSemantics,
    LstmCapacity,
    LstmMethod,
    Method,
    MethodNotApprovedError,
    MethodSpace,
    OriginWindow,
    StudyDefinition,
    TransformerLstmMethod,
    TransformerMethod,
    TuneRequest,
    apply_method,
    construct_fit_module,
)
from pydantic import ValidationError


def synthetic_experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=OriginWindow(
            role="training",
            first_parent_block=1_000,
            last_parent_block=1_999,
        ),
        validation_window=OriginWindow(
            role="validation",
            first_parent_block=2_200,
            last_parent_block=2_499,
        ),
        context_blocks=200,
        horizon_blocks=5,
        ordered_features=(
            "log_base_fee_per_gas",
            "gas_utilization",
            "log_exact_forming_base_fee_per_gas",
        ),
        classification_loss="unweighted",
    )


def tune_request(method_space: MethodSpace, *, study_id: str) -> TuneRequest:
    return TuneRequest(
        workflow="tune",
        study_id=UUID(study_id),
        corpus_id=UUID("00000000-0000-4000-8000-000000000001"),
        study_definition=StudyDefinition(
            experiment=synthetic_experiment(),
            method_space=method_space,
        ),
    )


def approved_methods() -> tuple[tuple[TuneRequest, Method], ...]:
    return (
        (
            tune_request(
                LSTM_METHOD_SPACE,
                study_id="00000000-0000-4000-8000-000000000101",
            ),
            LstmMethod(
                family="lstm",
                capacity=LSTM_METHOD_SPACE.capacities[1],
                dropout=0.2,
                optimizer=AdamWMethod(
                    learning_rate=3e-4,
                    weight_decay=1e-4,
                ),
                training_batch=64,
                fit=FIXED_FIT,
            ),
        ),
        (
            tune_request(
                TRANSFORMER_METHOD_SPACE,
                study_id="00000000-0000-4000-8000-000000000102",
            ),
            TransformerMethod(
                family="transformer",
                capacity=TRANSFORMER_METHOD_SPACE.capacities[1],
                dropout=0.2,
                optimizer=AdamWMethod(
                    learning_rate=3e-4,
                    weight_decay=1e-4,
                ),
                training_batch=64,
                fit=FIXED_FIT,
            ),
        ),
        (
            tune_request(
                TRANSFORMER_LSTM_METHOD_SPACE,
                study_id="00000000-0000-4000-8000-000000000103",
            ),
            TransformerLstmMethod(
                family="transformer_lstm",
                capacity=TRANSFORMER_LSTM_METHOD_SPACE.capacities[1],
                dropout=0.2,
                optimizer=AdamWMethod(
                    learning_rate=3e-4,
                    weight_decay=1e-4,
                ),
                training_batch=64,
                fit=FIXED_FIT,
            ),
        ),
    )


def _validation_codes(error: ValidationError) -> list[str]:
    return sorted({str(item["type"]) for item in error.errors()})


def _schema_rejection(payload: dict[str, Any]) -> list[str]:
    try:
        METHOD_ADAPTER.validate_python(payload)
    except ValidationError as error:
        return _validation_codes(error)
    raise AssertionError("invalid Method payload unexpectedly passed")


def run_observations() -> dict[str, object]:
    constructions: list[dict[str, object]] = []
    for request, method in approved_methods():
        training = apply_method(request, method)
        constructions.append(asdict(construct_fit_module(training)))

    lstm_request, lstm_method = approved_methods()[0]
    transformer_request, transformer_method = approved_methods()[1]

    unknown_payload = lstm_method.model_dump(mode="python")
    unknown_payload["model.hidden"] = 512
    unknown_rejection = _schema_rejection(unknown_payload)

    partial_payload = lstm_method.model_dump(mode="python")
    del partial_payload["fit"]
    partial_rejection = _schema_rejection(partial_payload)

    invalid_cross_field = transformer_method.model_dump(mode="python")
    invalid_cross_field["capacity"]["model_width"] = 250
    cross_field_rejection = _schema_rejection(invalid_cross_field)

    try:
        apply_method(lstm_request, transformer_method)
    except MethodNotApprovedError as error:
        family_mismatch = str(error)
    else:
        raise AssertionError("family mismatch unexpectedly passed")

    out_of_space = LstmMethod(
        family="lstm",
        capacity=LstmCapacity(projection=512, hidden=512, layers=2, head_hidden=256),
        dropout=0.2,
        optimizer=lstm_method.optimizer,
        training_batch=64,
        fit=FIXED_FIT,
    )
    try:
        apply_method(lstm_request, out_of_space)
    except MethodNotApprovedError as error:
        capacity_rejection = str(error)
    else:
        raise AssertionError("out-of-space capacity unexpectedly passed")

    first = apply_method(lstm_request, lstm_method)
    second = apply_method(lstm_request, lstm_method)
    assert first == second
    assert first.experiment == lstm_request.study_definition.experiment

    batch_32_method = LstmMethod(
        family="lstm",
        capacity=LSTM_METHOD_SPACE.capacities[1],
        dropout=lstm_method.dropout,
        optimizer=lstm_method.optimizer,
        training_batch=32,
        fit=FIXED_FIT,
    )
    batch_32_definition = apply_method(
        lstm_request,
        batch_32_method,
    )
    assert batch_32_definition.training_batch == 32

    wider_request_payload = lstm_request.model_dump(mode="python")
    wider_experiment = wider_request_payload["study_definition"]["experiment"]
    wider_experiment["horizon_blocks"] = 10
    wider_experiment["ordered_features"] = (
        *lstm_request.study_definition.experiment.ordered_features,
        "hour_sin",
    )
    wider_request = TuneRequest.model_validate(wider_request_payload)
    derived = asdict(construct_fit_module(apply_method(wider_request, lstm_method)))

    assert transformer_request.study_definition.method_space == TRANSFORMER_METHOD_SPACE
    assert [item["family"] for item in constructions] == [
        "lstm",
        "transformer",
        "transformer_lstm",
    ]
    assert all(item["input_features"] == 3 for item in constructions)
    assert all(item["action_count"] == 5 for item in constructions)
    assert derived["input_features"] == 4
    assert derived["action_count"] == 10

    return {
        "approved_families": [item["family"] for item in constructions],
        "complete_method_application": "pass",
        "constructor_match": "pass",
        "construction_dimensions": [
            {
                "family": item["family"],
                "input_features": item["input_features"],
                "context_blocks": item["context_blocks"],
                "action_count": item["action_count"],
            }
            for item in constructions
        ],
        "derived_dimension_change": {
            "input_features": derived["input_features"],
            "context_blocks": derived["context_blocks"],
            "action_count": derived["action_count"],
        },
        "unknown_parameter": unknown_rejection,
        "partial_method": partial_rejection,
        "cross_field_constraint": cross_field_rejection,
        "family_mismatch": family_mismatch,
        "capacity_outside_space": capacity_rejection,
        "pure_repeat": first == second,
        "input_unchanged": first.experiment == lstm_request.study_definition.experiment,
        "training_batch_axis": batch_32_definition.training_batch,
        "method_space_owner": "TuneRequest.study_definition",
        "free_method_space_arguments": 0,
        "feedforward_multiplier_fields": 0,
        "parameter_path_fields": 0,
        "checks": "pass",
    }


def _render(section: str) -> None:
    observations = run_observations()
    os.system("clear")
    print("\033[1mIssue 17 — typed Method application + direct construction\033[0m")
    print("\033[2mSynthetic values only; disposable planning evidence.\033[0m\n")
    if section == "all":
        print(json.dumps(observations, indent=2))
    else:
        print(json.dumps({section: observations[section]}, indent=2))
    print(
        "\n\033[1m[a]\033[0m all  \033[1m[c]\033[0m constructors  "
        "\033[1m[d]\033[0m derived dimensions  \033[1m[r]\033[0m rejections  "
        "\033[1m[p]\033[0m purity  \033[1m[q]\033[0m quit"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run bounded synthetic probes")
    args = parser.parse_args()
    if args.all:
        print(json.dumps(run_observations(), indent=2))
        return

    sections = {
        "a": "all",
        "c": "construction_dimensions",
        "d": "derived_dimension_change",
        "r": "unknown_parameter",
        "p": "pure_repeat",
    }
    selected = "all"
    while True:
        _render(selected)
        key = input("\n> ").strip().lower()
        if key == "q":
            return
        selected = sections.get(key, selected)


if __name__ == "__main__":
    main()
