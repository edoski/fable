from __future__ import annotations

from collections.abc import Callable

import pytest

from spice.config.groups import load_named_group_payload
from spice.config.models import (
    TrainingConfig,
    TuningSpaceConfig,
    coerce_features_config,
    coerce_problem_spec,
)
from spice.core.errors import ConfigResolutionError
from spice.evaluation import EvaluatorConfig, coerce_evaluator_config
from spice.modeling.dataset_builders import (
    coerce_builder_runtime_metadata,
    coerce_dataset_builder_config,
)
from spice.modeling.families.base import ModelConfig, ModelTuningSpaceConfig
from spice.modeling.families.registry import coerce_model_config
from spice.modeling.tuned_config import (
    coerce_tuned_parameter_set,
    coerce_tuning_space_config,
)
from spice.objectives import coerce_objective_config
from spice.temporal.compilers import ProblemCompilerConfig, coerce_problem_compiler_config
from spice.temporal.execution_policy import (
    ExecutionPolicyConfig,
    coerce_execution_policy_config,
)
from spice.temporal.execution_policy.strict_deadline_miss import StrictDeadlineMissConfig
from spice.temporal.input_normalization import (
    InputNormalizationConfig,
    coerce_input_normalization_config,
)
from spice.temporal.input_normalization.row_standard import RowStandardConfig


@pytest.mark.parametrize(
    ("coerce", "message"),
    [
        (coerce_problem_spec, "problem must be a mapping or ProblemSpec"),
        (coerce_features_config, "features must be a mapping or FeaturesConfig"),
        (coerce_evaluator_config, "evaluation must be a mapping or EvaluatorConfig"),
        (coerce_objective_config, "objective must be a mapping or ObjectiveConfig"),
        (coerce_model_config, "model must be a mapping or ModelConfig"),
        (
            coerce_dataset_builder_config,
            "dataset_builder must be a mapping or DatasetBuilderConfig",
        ),
        (
            coerce_problem_compiler_config,
            "problem.compiler must be a mapping or ProblemCompilerConfig",
        ),
        (
            coerce_execution_policy_config,
            "problem.execution_policy must be a mapping or ExecutionPolicyConfig",
        ),
        (
            coerce_input_normalization_config,
            "training.input_normalization must be a mapping or InputNormalizationConfig",
        ),
        (
            lambda payload: coerce_tuning_space_config(
                payload,
                model_config=coerce_model_config(load_named_group_payload("lstm", "model")),
                problem_config=coerce_problem_spec(
                    load_named_group_payload("current_row_nominal", "problem")
                ),
            ),
            "tuning_space must be a mapping or TuningSpaceConfig",
        ),
        (
            coerce_tuned_parameter_set,
            "tuned parameters must be a mapping or TunedParameterSet",
        ),
        (
            lambda payload: coerce_builder_runtime_metadata(
                "fixed_sequence_temporal",
                payload,
            ),
            "builder runtime metadata must be a mapping or BuilderRuntimeMetadata",
        ),
        (
            lambda payload: coerce_problem_compiler_config(
                {"id": "observed_time_window", "slot_spacing": payload}
            ),
            "observed_time_window.slot_spacing must be a mapping "
            "or ObservedTimeWindowSlotSpacingConfig",
        ),
    ],
)
def test_owner_coercers_reject_non_mapping_payloads(
    coerce: Callable[[object], object],
    message: str,
) -> None:
    with pytest.raises(ConfigResolutionError, match=message):
        coerce([])


def test_owner_coercers_preserve_typed_config_identity() -> None:
    problem = coerce_problem_spec(load_named_group_payload("current_row_nominal", "problem"))
    features = coerce_features_config(load_named_group_payload("core_fee_dynamics", "features"))
    evaluator = coerce_evaluator_config(load_named_group_payload("poisson_replay_2h", "evaluation"))
    objective = coerce_objective_config(
        load_named_group_payload("validation_total_loss", "objective")
    )
    model = coerce_model_config(load_named_group_payload("lstm", "model"))
    builder = coerce_dataset_builder_config(
        load_named_group_payload("fixed_sequence_temporal", "dataset_builder")
    )
    training = TrainingConfig.model_validate(load_named_group_payload("default", "training"))
    tuning_space = coerce_tuning_space_config(
        load_named_group_payload("lstm_default", "tuning_space"),
        model_config=model,
        problem_config=problem,
    )
    tuned_parameters = coerce_tuned_parameter_set(
        {"model": {"id": "lstm", "hidden_size": 64}},
        model_id="lstm",
    )
    metadata = coerce_builder_runtime_metadata(
        "fixed_sequence_temporal",
        {
            "sequence_length": 64,
            "median_dt_seconds": 1.0,
            "min_sequence_length": 64,
            "max_sequence_length": 4096,
        },
    )

    assert coerce_problem_spec(problem) is problem
    assert coerce_features_config(features) is features
    assert coerce_evaluator_config(evaluator) is evaluator
    assert coerce_objective_config(objective) is objective
    assert coerce_model_config(model) is model
    assert coerce_dataset_builder_config(builder) is builder
    assert coerce_problem_compiler_config(problem.compiler) is problem.compiler
    assert coerce_execution_policy_config(problem.execution_policy) is problem.execution_policy
    assert coerce_input_normalization_config(
        training.input_normalization
    ) is training.input_normalization
    assert coerce_tuning_space_config(
        tuning_space,
        model_config=model,
        problem_config=problem,
    ) == tuning_space
    assert (
        coerce_tuned_parameter_set(tuned_parameters, model_id="lstm")
        is tuned_parameters
    )
    assert (
        coerce_builder_runtime_metadata("fixed_sequence_temporal", metadata)
        is metadata
    )


def test_owner_coercers_redispatch_abstract_selector_configs() -> None:
    execution_policy = ExecutionPolicyConfig(id="strict_deadline_miss")
    input_normalization = InputNormalizationConfig(id="row_standard")
    model = coerce_model_config(load_named_group_payload("lstm", "model"))
    problem = coerce_problem_spec(load_named_group_payload("current_row_nominal", "problem"))
    tuning_space = TuningSpaceConfig(model=ModelTuningSpaceConfig[str](id="lstm"))

    coerced_policy = coerce_execution_policy_config(execution_policy)
    coerced_normalization = coerce_input_normalization_config(input_normalization)
    coerced_tuning_space = coerce_tuning_space_config(
        tuning_space,
        model_config=model,
        problem_config=problem,
    )

    assert coerced_policy is not execution_policy
    assert isinstance(coerced_policy, StrictDeadlineMissConfig)
    assert coerced_normalization is not input_normalization
    assert isinstance(coerced_normalization, RowStandardConfig)
    assert coerced_tuning_space is not tuning_space
    assert type(coerced_tuning_space.model).__name__ == "LstmTuningSpaceModelConfig"


@pytest.mark.parametrize(
    ("coerce", "payload", "message"),
    [
        (
            coerce_evaluator_config,
            EvaluatorConfig(id="poisson_replay_2h"),
            "Field required",
        ),
        (
            coerce_model_config,
            ModelConfig[str](id="lstm"),
            "Field required",
        ),
        (
            coerce_problem_compiler_config,
            ProblemCompilerConfig(id="observed_time_window"),
            "slot_spacing",
        ),
    ],
)
def test_incomplete_abstract_selector_configs_fail_at_owner_boundary(
    coerce: Callable[[object], object],
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ConfigResolutionError, match=message):
        coerce(payload)
