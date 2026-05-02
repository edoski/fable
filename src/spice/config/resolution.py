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
    ProviderSpec,
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
from .registry import (
    load_chain_spec,
    load_dataset_builder_config,
    load_dataset_spec,
    load_evaluator_config,
    load_features_config,
    load_model_config,
    load_named_group_payload,
    load_objective_config,
    load_prediction_config,
    load_problem_spec,
    load_provider_spec,
    load_split_config,
    load_training_config,
    load_tuning_config,
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
from .surfaces import SurfaceFrame, apply_surface_selection_overrides, load_surface_frame

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
        if selection.surface is None:
            raise ConfigResolutionError("surface is required")
        assert selection.surface is not None
        frame = apply_surface_selection_overrides(
            load_surface_frame(selection.surface),
            selection,
        )
        return _resolve_surface_frame(workflow_kind, frame, selection=selection)
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc


def _resolve_surface_frame(
    workflow: WorkflowTask,
    frame: SurfaceFrame,
    *,
    selection: WorkflowSelection,
) -> WorkflowConfig:
    if workflow is WorkflowTask.ACQUIRE:
        if not isinstance(selection, AcquireWorkflowSelection):
            raise ConfigResolutionError("acquire requires AcquireWorkflowSelection")
        return _resolve_acquire_config(frame, dry_run=selection.dry_run)
    if workflow is WorkflowTask.TRAIN:
        if not isinstance(selection, TrainWorkflowSelection):
            raise ConfigResolutionError("train requires TrainWorkflowSelection")
        return _resolve_train_config(frame, selection=selection)
    if workflow is WorkflowTask.TUNE:
        if not isinstance(selection, TuneWorkflowSelection):
            raise ConfigResolutionError("tune requires TuneWorkflowSelection")
        return _resolve_tune_config(
            frame,
            selection=selection,
            trial_count=selection.trial_count,
        )
    raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}")


def _validate_selection_kind(workflow: WorkflowTask, selection: WorkflowSelection) -> None:
    expected = workflow_selection_type(workflow)
    if not isinstance(selection, expected):
        raise ConfigResolutionError(
            f"{workflow.value} requires {expected.__name__}, got {type(selection).__name__}"
        )


def _resolve_dataset(name: str) -> DatasetSpec:
    return load_dataset_spec(name)


def _resolve_chain(name: str) -> ChainSpec:
    return load_chain_spec(name)


def _resolve_problem(name: str | ProblemSpec) -> ProblemSpec:
    if isinstance(name, ProblemSpec):
        return name
    return load_problem_spec(name)


def _resolve_features(name: str) -> FeaturesConfig:
    return load_features_config(name)


def _resolve_dataset_builder(name: str) -> DatasetBuilderConfig:
    return load_dataset_builder_config(name)


def _resolve_prediction(name: str) -> PredictionConfig:
    return load_prediction_config(name)


def _resolve_model(name: str) -> ModelConfig[str]:
    return load_model_config(name)


def _resolve_training(name: str) -> TrainingConfig:
    return load_training_config(name)


def _resolve_split(name: str) -> SplitConfig:
    return load_split_config(name)


def _resolve_tuning(name: str) -> TuningConfig:
    return load_tuning_config(name)


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


def _resolve_provider(name: str) -> ProviderSpec:
    return load_provider_spec(name)


def _resolve_model_workflow_base(
    frame: SurfaceFrame,
    *,
    validate_objective_benchmark: bool,
) -> ModelWorkflowBase:
    dataset = _resolve_dataset(frame.dataset)
    chain = _resolve_chain(frame.chain)
    storage = _resolve_storage(frame.storage)
    problem = _resolve_problem(frame.problem)
    model = _resolve_model(_required(frame.model, "model"))
    dataset_builder = _resolve_dataset_builder(frame.dataset_builder)
    features = _resolve_features(_required(frame.features, "features"))
    prediction = _resolve_prediction(frame.prediction)
    objective = _resolve_objective(
        _required(frame.objective, "objective"),
        evaluation_name=frame.evaluation_id if validate_objective_benchmark else None,
    )
    evaluation = _resolve_evaluation(frame.evaluation_id)
    study = _resolve_study(frame.study)
    artifact = _resolve_artifact(frame.artifact)
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
    frame: SurfaceFrame,
    *,
    model: ModelConfig[str],
    problem: ProblemSpec,
    artifact: ArtifactConfig,
    require_tuning: bool,
    allow_tuned_variant: bool,
) -> ModelWorkflowSpine:
    training = _resolve_training(frame.training_id)
    split = _resolve_split(frame.split)
    if require_tuning or (allow_tuned_variant and artifact.variant.value == "tuned"):
        tuning = _resolve_tuning(frame.tuning_id)
        tuning_space = load_named_tuning_space(
            _required(frame.tuning_space_id, "tuning.space"),
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


def _resolve_acquire_config(frame: SurfaceFrame, *, dry_run: bool | None) -> AcquireConfig:
    dataset = _resolve_dataset(frame.dataset)
    chain = _resolve_chain(frame.chain)
    storage = _resolve_storage(frame.storage)
    problem = _resolve_problem(frame.problem)
    provider = _resolve_provider(frame.provider)
    endpoint = provider.endpoint_config_for(chain.name)
    rpc_endpoint = ResolvedRpcEndpointConfig(
        provider_name=provider.name,
        url=endpoint.url,
        reference=endpoint.reference or endpoint.url,
        timeout_seconds=provider.transport.timeout_seconds,
        retry_count=provider.transport.retry_count,
        backoff_factor=provider.transport.backoff_factor,
    )
    features = _resolve_features(_required(frame.features, "features"))
    if provider.acquisition is None:
        raise ConfigResolutionError(
            f"provider {provider.name} must define acquisition settings"
        )
    acquisition = provider.acquisition
    if dry_run is not None:
        acquisition = _validated_config_update(acquisition, dry_run=dry_run)
    return AcquireConfig(
        chain=chain,
        dataset=dataset,
        storage=storage,
        problem=problem,
        features=features,
        rpc_endpoint=rpc_endpoint,
        acquisition=acquisition,
    )


def _resolve_train_config(
    frame: SurfaceFrame,
    *,
    selection: TrainWorkflowSelection,
) -> TrainConfig:
    base = _resolve_model_workflow_base(frame, validate_objective_benchmark=True)
    spine = _resolve_model_workflow_spine(
        frame,
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
        dataset_id=selection.dataset_id,
        study_id=selection.study_id,
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


def _resolve_tune_config(
    frame: SurfaceFrame,
    *,
    selection: TuneWorkflowSelection,
    trial_count: int | None,
) -> TuneConfig:
    base = _resolve_model_workflow_base(frame, validate_objective_benchmark=True)
    spine = _resolve_model_workflow_spine(
        frame,
        model=base.model,
        problem=base.problem,
        artifact=base.artifact,
        require_tuning=True,
        allow_tuned_variant=False,
    )
    assert spine.tuning is not None
    assert spine.tuning_space is not None
    tuning = spine.tuning
    if trial_count is not None:
        tuning = _validated_config_update(tuning, trial_count=trial_count)
    return TuneConfig(
        chain=base.chain,
        dataset=base.dataset,
        storage=base.storage,
        dataset_id=_required(selection.dataset_id, "dataset_id"),
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
