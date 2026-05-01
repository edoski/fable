"""Workflow selection resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

from pydantic import ValidationError

from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig, coerce_evaluator_config
from ..modeling.dataset_builders import coerce_dataset_builder_config
from ..modeling.families.base import ModelConfig
from ..modeling.families.registry import coerce_model_config
from ..modeling.tuned_config import coerce_tuning_space_config
from ..objectives import ObjectiveConfig, coerce_objective_config
from .models import (
    AcquireConfig,
    AcquisitionConfig,
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
    coerce_features_config,
    coerce_problem_spec,
)
from .registry import load_named_group
from .selections import (
    AcquireWorkflowSelection,
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
    WorkflowSelection,
    workflow_selection_type,
)
from .surfaces import SurfaceFrame, apply_selection_overrides, load_surface_frame
from .workflow_configs import coerce_resolved_workflow_config

_MODEL_GROUP = "model"
_TUNING_SPACE_GROUP = "tuning_space"

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


def load_named_tuning_space(
    name: str,
    *,
    model_config: ModelConfig[str],
    problem_config: ProblemSpec,
) -> TuningSpaceConfig:
    tuning_space = coerce_tuning_space_config(
        load_named_group(name, _TUNING_SPACE_GROUP),
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
        frame = apply_selection_overrides(
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
    if workflow is WorkflowTask.EVALUATE:
        if not isinstance(selection, EvaluateWorkflowSelection):
            raise ConfigResolutionError("evaluate requires EvaluateWorkflowSelection")
        return _resolve_evaluate_config(selection)
    raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}")


def _validate_selection_kind(workflow: WorkflowTask, selection: WorkflowSelection) -> None:
    expected = workflow_selection_type(workflow)
    if not isinstance(selection, expected):
        raise ConfigResolutionError(
            f"{workflow.value} requires {expected.__name__}, got {type(selection).__name__}"
        )


def _resolve_dataset(name: str) -> DatasetSpec:
    return DatasetSpec.model_validate(load_named_group(name, "dataset"))


def _resolve_chain(name: str) -> ChainSpec:
    return ChainSpec.model_validate(load_named_group(name, "chain"))


def _resolve_problem(name: str | ProblemSpec) -> ProblemSpec:
    if isinstance(name, ProblemSpec):
        return name
    return coerce_problem_spec(load_named_group(name, "problem"))


def _resolve_features(name: str) -> FeaturesConfig:
    return coerce_features_config(load_named_group(name, "features"))


def _resolve_dataset_builder(name: str) -> DatasetBuilderConfig:
    return coerce_dataset_builder_config(load_named_group(name, "dataset_builder"))


def _resolve_prediction(name: str) -> PredictionConfig:
    return PredictionConfig.model_validate(load_named_group(name, "prediction"))


def _resolve_model(name: str) -> ModelConfig[str]:
    return coerce_model_config(load_named_group(name, _MODEL_GROUP))


def _resolve_training(name: str) -> TrainingConfig:
    return TrainingConfig.model_validate(load_named_group(name, "training"))


def _resolve_split(name: str) -> SplitConfig:
    return SplitConfig.model_validate(load_named_group(name, "split"))


def _resolve_tuning(name: str) -> TuningConfig:
    return TuningConfig.model_validate(load_named_group(name, "tuning"))


def _resolve_evaluation(name: str | None) -> EvaluatorConfig | None:
    if name is None:
        return None
    return coerce_evaluator_config(load_named_group(name, "evaluation"))


def _resolve_objective(name: str, *, evaluation_name: str | None) -> ObjectiveConfig:
    raw_objective = load_named_group(name, "objective")
    expected_evaluation = _benchmark_evaluation_name(raw_objective)
    if (
        evaluation_name is not None
        and expected_evaluation is not None
        and expected_evaluation != evaluation_name
    ):
        raise ConfigResolutionError(
            f"objective {name} requires evaluation {expected_evaluation}, "
            f"got {evaluation_name}"
        )
    return coerce_objective_config(raw_objective)


def _benchmark_evaluation_name(payload: dict[str, object]) -> str | None:
    if payload.get("id") != "evaluation":
        return None
    benchmark_id = payload.get("benchmark_id")
    if not isinstance(benchmark_id, str):
        raise ConfigResolutionError("evaluation objective.benchmark_id must be a named benchmark")
    return benchmark_id


def _resolve_storage(storage: StorageSpec | None) -> StorageSpec:
    return storage or StorageSpec()


def _resolve_study(study: StudyConfig | None) -> StudyConfig:
    return study or StudyConfig()


def _resolve_artifact(artifact: ArtifactConfig | None) -> ArtifactConfig:
    return artifact or ArtifactConfig()


def _resolve_provider(name: str) -> ProviderSpec:
    return ProviderSpec.model_validate(load_named_group(name, "provider"))


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
    require_tuning: bool,
    allow_tuned_variant: bool,
) -> ModelWorkflowSpine:
    training = _resolve_training(frame.training_id)
    split = _resolve_split(frame.split)
    artifact = _resolve_artifact(frame.artifact)
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
        acquisition = AcquisitionConfig.model_validate(
            {**acquisition.model_dump(mode="json"), "dry_run": dry_run}
        )
    config = coerce_resolved_workflow_config(
        WorkflowTask.ACQUIRE,
        {
            "chain": chain,
            "dataset": dataset,
            "storage": storage,
            "problem": problem,
            "features": features,
            "rpc_endpoint": rpc_endpoint,
            "acquisition": acquisition,
        },
    )
    assert isinstance(config, AcquireConfig)
    return config


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
        require_tuning=False,
        allow_tuned_variant=True,
    )
    config = coerce_resolved_workflow_config(
        WorkflowTask.TRAIN,
        {
            "chain": base.chain,
            "dataset": base.dataset,
            "storage": base.storage,
            "dataset_id": selection.dataset_id,
            "study_id": selection.study_id,
            "problem": base.problem,
            "model": base.model,
            "dataset_builder": base.dataset_builder,
            "features": base.features,
            "prediction": base.prediction,
            "objective": base.objective,
            "evaluation": base.evaluation,
            "study": base.study,
            "artifact": base.artifact,
            "training": spine.training,
            "split": spine.split,
            "tuning": spine.tuning,
            "tuning_space": spine.tuning_space,
        },
    )
    assert isinstance(config, TrainConfig)
    return config


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
        require_tuning=True,
        allow_tuned_variant=False,
    )
    assert spine.tuning is not None
    assert spine.tuning_space is not None
    tuning = spine.tuning
    if trial_count is not None:
        tuning = TuningConfig.model_validate(
            {**tuning.model_dump(mode="json"), "trial_count": trial_count}
        )
    config = coerce_resolved_workflow_config(
        WorkflowTask.TUNE,
        {
            "chain": base.chain,
            "dataset": base.dataset,
            "storage": base.storage,
            "dataset_id": _required(selection.dataset_id, "dataset_id"),
            "problem": base.problem,
            "model": base.model,
            "dataset_builder": base.dataset_builder,
            "features": base.features,
            "prediction": base.prediction,
            "objective": base.objective,
            "evaluation": base.evaluation,
            "study": base.study,
            "artifact": base.artifact,
            "training": spine.training,
            "split": spine.split,
            "tuning": tuning,
            "tuning_space": spine.tuning_space,
        },
    )
    assert isinstance(config, TuneConfig)
    return config


def _resolve_evaluate_config(selection: EvaluateWorkflowSelection) -> EvaluateConfig:
    evaluation = _resolve_evaluation(_required(selection.evaluation, "evaluation"))
    if evaluation is None:
        raise ConfigResolutionError("evaluation is required")
    config = coerce_resolved_workflow_config(
        WorkflowTask.EVALUATE,
        {
            "storage": _resolve_storage(
                None if selection.storage_root is None else StorageSpec(root=selection.storage_root)
            ),
            "artifact_id": _required(selection.artifact_id, "artifact_id"),
            "dataset_id": _required(selection.dataset_id, "dataset_id"),
            "evaluation": evaluation,
            "delay_seconds": selection.delay_seconds,
            "batch_size": selection.batch_size or 256,
        },
    )
    assert isinstance(config, EvaluateConfig)
    return config


def _required(value: str | None, label: str) -> str:
    if value is None:
        raise ConfigResolutionError(f"{label} is required")
    return value
