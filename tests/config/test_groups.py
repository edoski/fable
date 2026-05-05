from __future__ import annotations

from spice.config.groups import (
    ensure_named_group_file,
    load_named_group_payload,
)
from spice.config.typed_registry import (
    load_chain_spec,
    load_dataset_builder_config,
    load_dataset_spec,
    load_evaluator_config,
    load_execution_spec,
    load_features_config,
    load_model_config,
    load_objective_config,
    load_prediction_config,
    load_problem_spec,
    load_provider_spec,
    load_split_config,
    load_surface_frame,
    load_training_config,
    load_tuning_config,
)


def test_typed_group_loaders_return_owner_concrete_types() -> None:
    problem = load_problem_spec("current_row_nominal")
    model = load_model_config("lstm")
    builder = load_dataset_builder_config("fixed_sequence_temporal")
    evaluator = load_evaluator_config("poisson_replay_2h")
    training = load_training_config("default")

    assert type(problem.compiler).__name__ == "ObservedTimeWindowCompilerConfig"
    assert type(problem.compiler.slot_spacing).__name__ == (
        "ObservedTimeWindowNominalSlotSpacingConfig"
    )
    assert type(problem.execution_policy).__name__ == "StrictDeadlineMissConfig"
    assert type(model).__name__ == "LstmModelConfig"
    assert type(builder).__name__ == "FixedSequenceTemporalDatasetBuilderConfig"
    assert type(evaluator).__name__ == "PoissonReplayEvaluatorConfig"
    assert type(training.input_normalization).__name__ == "RowStandardConfig"


def test_context_free_typed_group_loaders_cover_resolution_groups() -> None:
    assert load_dataset_spec("icdcs_2026").name == "icdcs_2026"
    assert load_chain_spec("avalanche").name == "avalanche"
    assert load_features_config("core_fee_dynamics").id == "core_fee_dynamics"
    assert load_provider_spec("publicnode").name == "publicnode"
    assert load_objective_config("validation_total_loss").id == "validation"
    assert load_prediction_config("icdcs_2026").id == "icdcs_2026"
    assert load_split_config("default").train_fraction == 0.8
    assert load_tuning_config("default").trial_count == 20
    assert load_execution_spec("disi_l40").id == "disi_l40"
    assert load_surface_frame("current_row_fee_dynamics").chain == "ethereum"


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
    typed = load_problem_spec("seeded_problem")

    assert raw["id"] == "seeded_problem"
    assert typed.id == "seeded_problem"
    assert type(typed.compiler).__name__ == "ObservedTimeWindowCompilerConfig"
