"""Workflow selection resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar, overload

from pydantic import ValidationError

from ..core.config_model import ConfigModel
from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig
from ..modeling.families.base import ModelConfig
from ..modeling.tuned_config import coerce_tuning_space_config
from ..objectives import ObjectiveConfig
from .groups import load_named_group_payload
from .models import (
    AcquireConfig,
    ArtifactConfig,
    ChainSpec,
    DatasetBuilderConfig,
    DatasetSpec,
    EvaluateConfig,
    FeaturesConfig,
    PredictionConfig,
    ProblemSpec,
    ResolvedRpcEndpointConfig,
    SplitConfig,
    StorageSpec,
    StudyConfig,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
    TuningConfig,
    TuningSpaceConfig,
    WorkflowTask,
)
from .selection_application import (
    AppliedAcquireSurfaceSelection,
    AppliedModelSurfaceSelection,
    AppliedSurfaceSelection,
    AppliedTrainSurfaceSelection,
    AppliedTuneSurfaceSelection,
    apply_surface_selection,
)
from .selections import (
    AcquireWorkflowSelection,
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
    WorkflowSelection,
    workflow_selection_from_values,
    workflow_selection_type,
)
from .typed_registry import (
    load_chain_spec,
    load_dataset_builder_config,
    load_dataset_spec,
    load_evaluator_config,
    load_features_config,
    load_model_config,
    load_objective_config,
    load_prediction_config,
    load_problem_spec,
    load_provider_spec,
    load_split_config,
    load_training_config,
    load_tuning_config,
)

_TUNING_SPACE_GROUP = "tuning_space"
ConfigModelT = TypeVar("ConfigModelT", bound=ConfigModel)

WorkflowConfig = AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig


@dataclass(frozen=True, slots=True)
class ModelWorkflowBase:
    dataset: DatasetSpec
    chain: ChainSpec
    storage: StorageSpec
    problem: ProblemSpec
    model: ModelConfig[str]
    dataset_builder: DatasetBuilderConfig
    features: FeaturesConfig
    prediction: PredictionConfig
    objective: ObjectiveConfig
    evaluation: EvaluatorConfig | None
    study: StudyConfig
    artifact: ArtifactConfig


@dataclass(frozen=True, slots=True)
class ModelWorkflowSpine:
    training: TrainingConfig
    split: SplitConfig
    tuning: TuningConfig | None
    tuning_space: TuningSpaceConfig | None


@overload
def resolve_workflow_command_config(
    workflow_kind: Literal[WorkflowTask.ACQUIRE],
    values: Mapping[str, object | None],
) -> AcquireConfig: ...


@overload
def resolve_workflow_command_config(
    workflow_kind: Literal[WorkflowTask.TRAIN],
    values: Mapping[str, object | None],
) -> TrainConfig: ...


@overload
def resolve_workflow_command_config(
    workflow_kind: Literal[WorkflowTask.TUNE],
    values: Mapping[str, object | None],
) -> TuneConfig: ...


@overload
def resolve_workflow_command_config(
    workflow_kind: Literal[WorkflowTask.EVALUATE],
    values: Mapping[str, object | None],
) -> EvaluateConfig: ...


@overload
def resolve_workflow_command_config(
    workflow_kind: WorkflowTask,
    values: Mapping[str, object | None],
) -> WorkflowConfig: ...


def resolve_workflow_command_config(
    workflow_kind: WorkflowTask,
    values: Mapping[str, object | None],
) -> WorkflowConfig:
    """Resolve sparse command values into one validated workflow config."""

    return resolve_workflow_config(
        workflow_kind,
        workflow_selection_from_values(workflow_kind, values),
    )


def load_named_tuning_space(
    name: str,
    *,
    model_config: ModelConfig[str],
    problem_config: ProblemSpec,
) -> TuningSpaceConfig:
    tuning_space = coerce_tuning_space_config(
        load_named_group_payload(name, _TUNING_SPACE_GROUP),
        model_config=model_config,
        problem_config=problem_config,
    )
    if tuning_space is None:
        raise ConfigResolutionError(f"tuning space {name} resolved to None")
    return tuning_space


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.ACQUIRE],
    selection: AcquireWorkflowSelection,
) -> AcquireConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.TRAIN],
    selection: TrainWorkflowSelection,
) -> TrainConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.TUNE],
    selection: TuneWorkflowSelection,
) -> TuneConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.EVALUATE],
    selection: EvaluateWorkflowSelection,
) -> EvaluateConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: WorkflowTask,
    selection: WorkflowSelection,
) -> WorkflowConfig: ...


def resolve_workflow_config(
    workflow_kind: WorkflowTask,
    selection: WorkflowSelection,
) -> WorkflowConfig:
    """Resolve one workflow selection into one validated workflow config."""

    try:
        _validate_selection_kind(workflow_kind, selection)
        if workflow_kind is WorkflowTask.EVALUATE:
            if not isinstance(selection, EvaluateWorkflowSelection):
                raise ConfigResolutionError("evaluate requires EvaluateWorkflowSelection")
            return _resolve_evaluate_config(selection)
        if isinstance(selection, EvaluateWorkflowSelection):
            raise ConfigResolutionError("evaluate selection cannot resolve surface workflows")
        applied = apply_surface_selection(selection)
        return _resolve_applied_surface_selection(workflow_kind, applied)
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc


def _resolve_applied_surface_selection(
    workflow: WorkflowTask,
    applied: AppliedSurfaceSelection,
) -> WorkflowConfig:
    if workflow is WorkflowTask.ACQUIRE:
        if not isinstance(applied, AppliedAcquireSurfaceSelection):
            raise ConfigResolutionError("acquire requires AppliedAcquireSurfaceSelection")
        return _resolve_acquire_config(applied)
    if workflow is WorkflowTask.TRAIN:
        if not isinstance(applied, AppliedTrainSurfaceSelection):
            raise ConfigResolutionError("train requires AppliedTrainSurfaceSelection")
        return _resolve_train_config(applied)
    if workflow is WorkflowTask.TUNE:
        if not isinstance(applied, AppliedTuneSurfaceSelection):
            raise ConfigResolutionError("tune requires AppliedTuneSurfaceSelection")
        return _resolve_tune_config(applied)
    raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}")


def _validate_selection_kind(workflow: WorkflowTask, selection: WorkflowSelection) -> None:
    expected = workflow_selection_type(workflow)
    if not isinstance(selection, expected):
        raise ConfigResolutionError(
            f"{workflow.value} requires {expected.__name__}, got {type(selection).__name__}"
        )


def _resolve_problem(name: str | ProblemSpec) -> ProblemSpec:
    if isinstance(name, ProblemSpec):
        return name
    return load_problem_spec(name)


def _resolve_evaluation(name: str | None) -> EvaluatorConfig | None:
    if name is None:
        return None
    return load_evaluator_config(name)


def _resolve_objective(name: str, *, evaluation_name: str | None) -> ObjectiveConfig:
    objective = load_objective_config(name)
    expected_evaluation = _benchmark_evaluation_name(objective)
    if (
        evaluation_name is not None
        and expected_evaluation is not None
        and expected_evaluation != evaluation_name
    ):
        raise ConfigResolutionError(
            f"objective {name} requires evaluation {expected_evaluation}, "
            f"got {evaluation_name}"
        )
    return objective


def _benchmark_evaluation_name(config: ObjectiveConfig) -> str | None:
    if config.id != "evaluation":
        return None
    if config.benchmark_id is None:
        raise ConfigResolutionError("evaluation objective.benchmark_id must be a named benchmark")
    return config.benchmark_id


def _resolve_storage(storage: StorageSpec | None) -> StorageSpec:
    return storage or StorageSpec()


def _resolve_study(study: StudyConfig | None) -> StudyConfig:
    return study or StudyConfig()


def _resolve_artifact(artifact: ArtifactConfig | None) -> ArtifactConfig:
    return artifact or ArtifactConfig()


def _resolve_model_workflow_base(
    applied: AppliedModelSurfaceSelection,
    *,
    validate_objective_benchmark: bool,
) -> ModelWorkflowBase:
    dataset = load_dataset_spec(applied.dataset)
    chain = load_chain_spec(applied.chain)
    storage = _resolve_storage(applied.storage)
    problem = _resolve_problem(applied.problem)
    model = load_model_config(_required(applied.model, "model"))
    dataset_builder = load_dataset_builder_config(applied.dataset_builder)
    features = load_features_config(_required(applied.features, "features"))
    prediction = load_prediction_config(applied.prediction)
    objective = _resolve_objective(
        _required(applied.objective, "objective"),
        evaluation_name=applied.evaluation if validate_objective_benchmark else None,
    )
    evaluation = _resolve_evaluation(applied.evaluation)
    study = _resolve_study(applied.study)
    artifact = _resolve_artifact(applied.artifact)
    return ModelWorkflowBase(
        dataset=dataset,
        chain=chain,
        storage=storage,
        problem=problem,
        model=model,
        dataset_builder=dataset_builder,
        features=features,
        prediction=prediction,
        objective=objective,
        evaluation=evaluation,
        study=study,
        artifact=artifact,
    )


def _resolve_model_workflow_spine(
    applied: AppliedModelSurfaceSelection,
    *,
    model: ModelConfig[str],
    problem: ProblemSpec,
    artifact: ArtifactConfig,
    require_tuning: bool,
    allow_tuned_variant: bool,
) -> ModelWorkflowSpine:
    training = load_training_config(applied.training)
    split = load_split_config(applied.split)
    if require_tuning or (allow_tuned_variant and artifact.variant.value == "tuned"):
        tuning = load_tuning_config(applied.tuning)
        tuning_space = load_named_tuning_space(
            _required(applied.tuning_space, "tuning.space"),
            model_config=model,
            problem_config=problem,
        )
        return ModelWorkflowSpine(
            training=training,
            split=split,
            tuning=tuning,
            tuning_space=tuning_space,
        )
    return ModelWorkflowSpine(
        training=training,
        split=split,
        tuning=None,
        tuning_space=None,
    )


def _resolve_acquire_config(applied: AppliedAcquireSurfaceSelection) -> AcquireConfig:
    dataset = load_dataset_spec(applied.dataset)
    chain = load_chain_spec(applied.chain)
    storage = _resolve_storage(applied.storage)
    problem = _resolve_problem(applied.problem)
    provider = load_provider_spec(applied.provider)
    endpoint = provider.endpoint_config_for(chain.name)
    rpc_endpoint = ResolvedRpcEndpointConfig(
        provider_name=provider.name,
        url=endpoint.url,
        reference=endpoint.reference or endpoint.url,
        timeout_seconds=provider.transport.timeout_seconds,
        retry_count=provider.transport.retry_count,
        backoff_factor=provider.transport.backoff_factor,
    )
    features = load_features_config(_required(applied.features, "features"))
    if provider.acquisition is None:
        raise ConfigResolutionError(
            f"provider {provider.name} must define acquisition settings"
        )
    acquisition = provider.acquisition
    if applied.dry_run is not None:
        acquisition = _validated_config_update(acquisition, dry_run=applied.dry_run)
    return AcquireConfig(
        chain=chain,
        dataset=dataset,
        storage=storage,
        problem=problem,
        features=features,
        rpc_endpoint=rpc_endpoint,
        acquisition=acquisition,
    )


def _resolve_train_config(applied: AppliedTrainSurfaceSelection) -> TrainConfig:
    base = _resolve_model_workflow_base(applied, validate_objective_benchmark=True)
    spine = _resolve_model_workflow_spine(
        applied,
        model=base.model,
        problem=base.problem,
        artifact=base.artifact,
        require_tuning=False,
        allow_tuned_variant=True,
    )
    return TrainConfig(
        chain=base.chain,
        dataset=base.dataset,
        storage=base.storage,
        dataset_id=applied.dataset_id,
        study_id=applied.study_id,
        problem=base.problem,
        model=base.model,
        dataset_builder=base.dataset_builder,
        features=base.features,
        prediction=base.prediction,
        objective=base.objective,
        evaluation=base.evaluation,
        study=base.study,
        artifact=base.artifact,
        training=spine.training,
        split=spine.split,
        tuning=spine.tuning,
        tuning_space=spine.tuning_space,
    )


def _resolve_tune_config(applied: AppliedTuneSurfaceSelection) -> TuneConfig:
    base = _resolve_model_workflow_base(applied, validate_objective_benchmark=True)
    spine = _resolve_model_workflow_spine(
        applied,
        model=base.model,
        problem=base.problem,
        artifact=base.artifact,
        require_tuning=True,
        allow_tuned_variant=False,
    )
    assert spine.tuning is not None
    assert spine.tuning_space is not None
    tuning = spine.tuning
    if applied.trial_count is not None:
        tuning = _validated_config_update(tuning, trial_count=applied.trial_count)
    return TuneConfig(
        chain=base.chain,
        dataset=base.dataset,
        storage=base.storage,
        dataset_id=_required(applied.dataset_id, "dataset_id"),
        problem=base.problem,
        model=base.model,
        dataset_builder=base.dataset_builder,
        features=base.features,
        prediction=base.prediction,
        objective=base.objective,
        evaluation=base.evaluation,
        study=base.study,
        artifact=base.artifact,
        training=spine.training,
        split=spine.split,
        tuning=tuning,
        tuning_space=spine.tuning_space,
    )


def _resolve_evaluate_config(selection: EvaluateWorkflowSelection) -> EvaluateConfig:
    evaluation = _resolve_evaluation(_required(selection.evaluation, "evaluation"))
    if evaluation is None:
        raise ConfigResolutionError("evaluation is required")
    return EvaluateConfig(
        storage=_resolve_storage(
            None if selection.storage_root is None else StorageSpec(root=selection.storage_root)
        ),
        artifact_id=_required(selection.artifact_id, "artifact_id"),
        dataset_id=_required(selection.dataset_id, "dataset_id"),
        evaluation=evaluation,
        delay_seconds=selection.delay_seconds,
        batch_size=selection.batch_size or 256,
    )


def _required(value: str | None, label: str) -> str:
    if value is None:
        raise ConfigResolutionError(f"{label} is required")
    return value


def _validated_config_update(config: ConfigModelT, **updates: object) -> ConfigModelT:
    return type(config).model_validate(
        {
            **config.model_dump(mode="json", exclude_none=True),
            **updates,
        }
    )
