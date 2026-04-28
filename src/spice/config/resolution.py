"""Workflow request handling and surface resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from pydantic import Field, ValidationError

from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig, coerce_evaluator_config
from ..modeling.dataset_builders import coerce_dataset_builder_config
from ..modeling.families.base import ConfigModel, ModelConfig
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
from .surfaces import SurfaceFrame, apply_request_overrides, load_surface_frame

_MODEL_GROUP = "model"
_TUNING_SPACE_GROUP = "tuning_space"

WorkflowConfig = AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig


class WorkflowRequestBase(ConfigModel):
    surface: str | None = None
    chain: str | None = None
    problem: str | None = None
    features: str | None = None
    storage_root: Path | None = None


class AcquireWorkflowRequest(WorkflowRequestBase):
    provider: str | None = None
    dry_run: bool | None = None


class ModelWorkflowRequestBase(WorkflowRequestBase):
    objective: str | None = None
    evaluation: str | None = None
    model: str | None = None
    tuning_space: str | None = None
    training: str | None = None
    split: str | None = None
    tuning: str | None = None
    study: str | None = None


class TrainWorkflowRequest(ModelWorkflowRequestBase):
    variant: str | None = None


class TuneWorkflowRequest(ModelWorkflowRequestBase):
    trial_count: int | None = Field(default=None, gt=0)


class EvaluateWorkflowRequest(ModelWorkflowRequestBase):
    variant: str | None = None
    delay_seconds: int | None = Field(default=None, gt=0)


WorkflowConfigRequest = (
    AcquireWorkflowRequest
    | TrainWorkflowRequest
    | TuneWorkflowRequest
    | EvaluateWorkflowRequest
)


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
    request: AcquireWorkflowRequest,
) -> AcquireConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.TRAIN],
    request: TrainWorkflowRequest,
) -> TrainConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.TUNE],
    request: TuneWorkflowRequest,
) -> TuneConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.EVALUATE],
    request: EvaluateWorkflowRequest,
) -> EvaluateConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: WorkflowTask,
    request: WorkflowConfigRequest,
) -> WorkflowConfig: ...


def resolve_workflow_config(
    workflow_kind: WorkflowTask,
    request: WorkflowConfigRequest,
) -> WorkflowConfig:
    """Resolve one workflow request into one validated workflow config."""

    if request.surface is None:
        raise ConfigResolutionError("surface is required")
    try:
        _validate_request_kind(workflow_kind, request)
        frame = apply_request_overrides(
            load_surface_frame(request.surface),
            chain=request.chain,
            problem=request.problem,
            features=request.features,
            objective=getattr(request, "objective", None),
            evaluation=getattr(request, "evaluation", None),
            model=getattr(request, "model", None),
            tuning_space=getattr(request, "tuning_space", None),
            provider=getattr(request, "provider", None),
            training=getattr(request, "training", None),
            split=getattr(request, "split", None),
            tuning=getattr(request, "tuning", None),
            study=getattr(request, "study", None),
            variant=getattr(request, "variant", None),
            delay_seconds=getattr(request, "delay_seconds", None),
            storage_root=request.storage_root,
        )
        return _resolve_surface_frame(workflow_kind, frame, request=request)
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc


def resolve_workflow_frame_config(
    workflow_kind: WorkflowTask,
    frame: SurfaceFrame,
    *,
    request: WorkflowConfigRequest,
) -> WorkflowConfig:
    try:
        _validate_request_kind(workflow_kind, request)
        return _resolve_surface_frame(workflow_kind, frame, request=request)
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc


def _resolve_surface_frame(
    workflow: WorkflowTask,
    frame: SurfaceFrame,
    *,
    request: WorkflowConfigRequest,
) -> WorkflowConfig:
    if workflow is WorkflowTask.ACQUIRE:
        if not isinstance(request, AcquireWorkflowRequest):
            raise ConfigResolutionError("acquire requires AcquireWorkflowRequest")
        return _resolve_acquire_config(frame, dry_run=request.dry_run)
    if workflow is WorkflowTask.TRAIN:
        return _resolve_train_config(frame)
    if workflow is WorkflowTask.TUNE:
        if not isinstance(request, TuneWorkflowRequest):
            raise ConfigResolutionError("tune requires TuneWorkflowRequest")
        return _resolve_tune_config(frame, trial_count=request.trial_count)
    if workflow is WorkflowTask.EVALUATE:
        return _resolve_evaluate_config(frame)
    raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}")


def _validate_request_kind(workflow: WorkflowTask, request: WorkflowConfigRequest) -> None:
    expected = workflow_request_type(workflow)
    if not isinstance(request, expected):
        raise ConfigResolutionError(
            f"{workflow.value} requires {expected.__name__}, got {type(request).__name__}"
        )


def workflow_request_type(workflow: WorkflowTask) -> type[WorkflowRequestBase]:
    if workflow is WorkflowTask.ACQUIRE:
        return AcquireWorkflowRequest
    if workflow is WorkflowTask.TRAIN:
        return TrainWorkflowRequest
    if workflow is WorkflowTask.TUNE:
        return TuneWorkflowRequest
    if workflow is WorkflowTask.EVALUATE:
        return EvaluateWorkflowRequest
    raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}")


def workflow_request_fields(workflow: WorkflowTask) -> tuple[str, ...]:
    return tuple(workflow_request_type(workflow).model_fields)


def workflow_request_payload(
    workflow: WorkflowTask,
    values: Mapping[str, object | None],
) -> dict[str, object]:
    fields = frozenset(workflow_request_fields(workflow))
    return {
        key: value
        for key, value in values.items()
        if key in fields and value is not None
    }


def hydrate_model_workflow_config(
    workflow: WorkflowTask,
    payload: Mapping[str, object],
) -> WorkflowConfig:
    try:
        if workflow not in {WorkflowTask.TRAIN, WorkflowTask.TUNE, WorkflowTask.EVALUATE}:
            raise ConfigResolutionError(f"Unsupported model workflow: {workflow.value}")
        resolved_payload = _model_workflow_payload(payload)
        if workflow is WorkflowTask.TRAIN:
            return TrainConfig.model_validate(resolved_payload)
        if workflow is WorkflowTask.TUNE:
            return TuneConfig.model_validate(resolved_payload)
        if workflow is WorkflowTask.EVALUATE:
            return EvaluateConfig.model_validate(resolved_payload)
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc
    raise ConfigResolutionError(f"Unsupported model workflow: {workflow.value}")


def _model_workflow_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw = dict(payload)
    problem = coerce_problem_spec(_mapping_field(raw, "problem"))
    model = coerce_model_config(_mapping_field(raw, "model"))
    tuning_space_payload = raw.get("tuning_space")
    tuning_space = (
        None
        if tuning_space_payload is None
        else coerce_tuning_space_config(
            _mapping_value(tuning_space_payload, label="tuning_space"),
            model_config=model,
            problem_config=problem,
        )
    )
    return {
        **raw,
        "chain": ChainSpec.model_validate(_mapping_field(raw, "chain")),
        "dataset": DatasetSpec.model_validate(_mapping_field(raw, "dataset")),
        "storage": StorageSpec.model_validate(_mapping_field(raw, "storage")),
        "problem": problem,
        "model": model,
        "dataset_builder": coerce_dataset_builder_config(_mapping_field(raw, "dataset_builder")),
        "features": coerce_features_config(_mapping_field(raw, "features")),
        "prediction": PredictionConfig.model_validate(_mapping_field(raw, "prediction")),
        "objective": coerce_objective_config(_mapping_field(raw, "objective")),
        "evaluation": _optional_evaluation(raw.get("evaluation")),
        "study": StudyConfig.model_validate(_mapping_field(raw, "study")),
        "artifact": ArtifactConfig.model_validate(_mapping_field(raw, "artifact")),
        "split": SplitConfig.model_validate(_mapping_field(raw, "split")),
        "training": TrainingConfig.model_validate(_mapping_field(raw, "training")),
        "tuning": _optional_tuning(raw.get("tuning")),
        "tuning_space": tuning_space,
    }


def _optional_evaluation(payload: object) -> EvaluatorConfig | None:
    if payload is None:
        return None
    return coerce_evaluator_config(_mapping_value(payload, label="evaluation"))


def _optional_tuning(payload: object) -> TuningConfig | None:
    if payload is None:
        return None
    return TuningConfig.model_validate(_mapping_value(payload, label="tuning"))


def _mapping_field(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _mapping_value(payload.get(key), label=key)


def _mapping_value(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ConfigResolutionError(f"resolved workflow snapshot field {label} must be a mapping")
    return payload


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
    return AcquireConfig(
        chain=chain,
        dataset=dataset,
        storage=storage,
        problem=problem,
        features=features,
        rpc_endpoint=rpc_endpoint,
        acquisition=acquisition,
    )


def _resolve_train_config(frame: SurfaceFrame) -> TrainConfig:
    base = _resolve_model_workflow_base(frame, validate_objective_benchmark=True)
    spine = _resolve_model_workflow_spine(
        frame,
        model=base.model,
        problem=base.problem,
        require_tuning=False,
        allow_tuned_variant=True,
    )
    return TrainConfig(
        chain=base.chain,
        dataset=base.dataset,
        storage=base.storage,
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


def _resolve_tune_config(frame: SurfaceFrame, *, trial_count: int | None) -> TuneConfig:
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
    return TuneConfig(
        chain=base.chain,
        dataset=base.dataset,
        storage=base.storage,
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


def _resolve_evaluate_config(frame: SurfaceFrame) -> EvaluateConfig:
    base = _resolve_model_workflow_base(frame, validate_objective_benchmark=False)
    spine = _resolve_model_workflow_spine(
        frame,
        model=base.model,
        problem=base.problem,
        require_tuning=False,
        allow_tuned_variant=True,
    )
    delay_seconds = frame.delay_seconds or base.problem.max_delay_seconds
    return EvaluateConfig(
        chain=base.chain,
        dataset=base.dataset,
        storage=base.storage,
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
        delay_seconds=delay_seconds,
        tuning=spine.tuning,
        tuning_space=spine.tuning_space,
    )


def _required(value: str | None, label: str) -> str:
    if value is None:
        raise ConfigResolutionError(f"{label} is required")
    return value
