"""Workflow request handling and surface resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from pydantic import Field, ValidationError

from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig
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
    FeatureSetConfig,
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
    coerce_feature_set_config,
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
    feature_set: str | None = None
    storage_root: Path | None = None


class AcquireWorkflowRequest(WorkflowRequestBase):
    acquisition: str | None = None
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
    feature_set: FeatureSetConfig
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
            feature_set=request.feature_set,
            objective=getattr(request, "objective", None),
            evaluation=getattr(request, "evaluation", None),
            model=getattr(request, "model", None),
            tuning_space=getattr(request, "tuning_space", None),
            acquisition=getattr(request, "acquisition", None),
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


def _resolve_dataset(name: str) -> DatasetSpec:
    return DatasetSpec.model_validate(load_named_group(name, "dataset"))


def _resolve_chain(name: str) -> ChainSpec:
    return ChainSpec.model_validate(load_named_group(name, "chain"))


def _resolve_problem(name: str) -> ProblemSpec:
    return coerce_problem_spec(load_named_group(name, "problem"))


def _resolve_feature_set(name: str) -> FeatureSetConfig:
    return coerce_feature_set_config(load_named_group(name, "feature_set"))


def _resolve_dataset_builder(name: str) -> DatasetBuilderConfig:
    return coerce_dataset_builder_config(load_named_group(name, "dataset_builder"))


def _resolve_prediction(name: str) -> PredictionConfig:
    return PredictionConfig.model_validate(load_named_group(name, "prediction"))


def _resolve_model(name: str) -> ModelConfig[str]:
    return coerce_model_config(load_named_group(name, _MODEL_GROUP))


def _resolve_acquisition(name: str) -> AcquisitionConfig:
    return AcquisitionConfig.model_validate(load_named_group(name, "acquisition"))


def _resolve_training(name: str) -> TrainingConfig:
    return TrainingConfig.model_validate(load_named_group(name, "training"))


def _resolve_split(name: str) -> SplitConfig:
    return SplitConfig.model_validate(load_named_group(name, "split"))


def _resolve_tuning(name: str) -> TuningConfig:
    return TuningConfig.model_validate(load_named_group(name, "tuning"))


def _resolve_evaluation(name: str | None) -> EvaluatorConfig | None:
    if name is None:
        return None
    return EvaluatorConfig.model_validate(load_named_group(name, "evaluation"))


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


def _resolve_rpc_endpoint(
    provider_name: str,
    *,
    chain: ChainSpec,
) -> ResolvedRpcEndpointConfig:
    provider = ProviderSpec.model_validate(load_named_group(provider_name, "provider"))
    endpoint = provider.endpoint_config_for(chain.name)
    return ResolvedRpcEndpointConfig(
        provider_name=provider.name,
        url=endpoint.url,
        reference=endpoint.reference or endpoint.url,
        timeout_seconds=provider.transport.timeout_seconds,
        retry_count=provider.transport.retry_count,
        backoff_factor=provider.transport.backoff_factor,
    )


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
    feature_set = _resolve_feature_set(_required(frame.feature_set, "feature_set"))
    prediction = _resolve_prediction(frame.prediction)
    objective = _resolve_objective(
        _required(frame.objective, "objective"),
        evaluation_name=frame.evaluation if validate_objective_benchmark else None,
    )
    evaluation = _resolve_evaluation(frame.evaluation)
    study = _resolve_study(frame.study)
    artifact = _resolve_artifact(frame.artifact)
    return ModelWorkflowBase(
        dataset=dataset,
        chain=chain,
        storage=storage,
        problem=problem,
        model=model,
        dataset_builder=dataset_builder,
        feature_set=feature_set,
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
    training = _resolve_training(frame.training)
    split = _resolve_split(frame.split)
    artifact = _resolve_artifact(frame.artifact)
    if require_tuning or (allow_tuned_variant and artifact.variant.value == "tuned"):
        tuning = _resolve_tuning(frame.tuning)
        tuning_space = load_named_tuning_space(
            _required(frame.tuning_space, "tuning_space"),
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
    rpc_endpoint = _resolve_rpc_endpoint(frame.provider, chain=chain)
    feature_set = _resolve_feature_set(_required(frame.feature_set, "feature_set"))
    acquisition = _resolve_acquisition(frame.acquisition)
    if dry_run is not None:
        acquisition = AcquisitionConfig.model_validate(
            {**acquisition.model_dump(mode="json"), "dry_run": dry_run}
        )
    return AcquireConfig(
        chain=chain,
        dataset=dataset,
        storage=storage,
        problem=problem,
        feature_set=feature_set,
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
        feature_set=base.feature_set,
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
        feature_set=base.feature_set,
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
    return EvaluateConfig(
        chain=base.chain,
        dataset=base.dataset,
        storage=base.storage,
        problem=base.problem,
        model=base.model,
        dataset_builder=base.dataset_builder,
        feature_set=base.feature_set,
        prediction=base.prediction,
        objective=base.objective,
        evaluation=base.evaluation,
        study=base.study,
        artifact=base.artifact,
        training=spine.training,
        split=spine.split,
        delay_seconds=_required_int(frame.delay_seconds, "delay_seconds"),
        tuning=spine.tuning,
        tuning_space=spine.tuning_space,
    )


def _required(value: str | None, label: str) -> str:
    if value is None:
        raise ConfigResolutionError(f"{label} is required")
    return value


def _required_int(value: int | None, label: str) -> int:
    if value is None:
        raise ConfigResolutionError(f"{label} is required")
    return value
