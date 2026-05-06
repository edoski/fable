from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from spice.config import typed_groups as typed
from spice.config.groups import (
    ensure_named_group_file,
    load_named_group_payload,
)


def test_typed_group_loaders_return_owner_concrete_types() -> None:
    problem = typed.load(typed.PROBLEM, "current_row_nominal")
    model = typed.load(typed.MODEL, "lstm")
    builder = typed.load(typed.DATASET_BUILDER, "fixed_sequence_temporal")
    evaluator = typed.load(typed.EVALUATION, "poisson_replay")
    training = typed.load(typed.TRAINING, "default")

    assert type(problem.compiler).__name__ == "ObservedTimeWindowCompilerConfig"
    assert type(problem.compiler.slot_spacing).__name__ == (
        "ObservedTimeWindowNominalSlotSpacingConfig"
    )
    assert type(problem.execution_policy).__name__ == "StrictDeadlineMissConfig"
    assert type(model).__name__ == "LstmModelConfig"
    assert type(builder).__name__ == "FixedSequenceTemporalDatasetBuilderConfig"
    assert type(evaluator).__name__ == "PoissonReplayEvaluatorConfig"
    assert type(training.input_normalization).__name__ == "RowStandardConfig"


@pytest.mark.parametrize(
    ("loader", "name", "identity"),
    [
        (lambda: typed.load(typed.DATASET, "icdcs_2026"), "icdcs_2026", "name"),
        (lambda: typed.load(typed.CHAIN, "avalanche"), "avalanche", "name"),
        (lambda: typed.load(typed.FEATURES, "core_fee_dynamics"), "core_fee_dynamics", "id"),
        (lambda: typed.load(typed.PROVIDER, "publicnode"), "publicnode", "name"),
        (lambda: typed.load(typed.PREDICTION, "icdcs_2026"), "icdcs_2026", "id"),
        (lambda: typed.load(typed.EXECUTION, "disi_l40"), "disi_l40", "id"),
        (lambda: typed.load(typed.SURFACE, "current_row_fee_dynamics"), "ethereum", "chain"),
    ],
)
def test_context_free_typed_loader_covers_named_group_shapes(
    loader: Callable[[], Any],
    name: str,
    identity: str,
) -> None:
    assert getattr(loader(), identity) == name


def test_typed_loader_does_not_use_raw_canonical_payload_loader(monkeypatch) -> None:
    def fail_raw_loader(*_args, **_kwargs):
        raise AssertionError("typed loading must not use raw canonical payload loading")

    monkeypatch.setattr("spice.config.groups.load_named_group_payload", fail_raw_loader)

    problem = typed.load(typed.PROBLEM, "current_row_nominal")

    assert problem.id == "current_row_nominal"


def test_raw_payload_loader_returns_canonical_dicts() -> None:
    dataset = load_named_group_payload("icdcs_2026", "dataset")
    model = load_named_group_payload("lstm", "model")

    assert type(dataset) is dict
    assert dataset == {
        "name": "icdcs_2026",
        "evaluation_date": "2025-11-09",
    }
    assert type(model) is dict
    assert model["id"] == "lstm"


def test_seeded_problem_template_loads_raw_and_typed(isolate_conf_root) -> None:
    isolate_conf_root()

    ensure_named_group_file("problem", "seeded_problem")

    raw = load_named_group_payload("seeded_problem", "problem")
    problem = typed.load(typed.PROBLEM, "seeded_problem")

    assert raw["id"] == "seeded_problem"
    assert problem.id == "seeded_problem"
    assert type(problem.compiler).__name__ == "ObservedTimeWindowCompilerConfig"
