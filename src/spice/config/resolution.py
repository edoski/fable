"""Workflow request handling and preset resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

from pydantic import Field, ValidationError

from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig
from ..modeling.families.base import ConfigModel, ModelConfig
from ..modeling.families.registry import coerce_model_config
from ..modeling.tuned_config import coerce_tuning_space_config
from ..objectives import ObjectiveConfig, coerce_objective_config
from .models import (
    AcquireConfig,
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
    coerce_dataset_builder_config,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from .presets import PresetFrame, apply_request_overrides, load_preset_frame
from .registry import load_named_group

_MODEL_GROUP = "model"
_TUNING_SPACE_GROUP = "tuning_space"

WorkflowConfig = AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig


class WorkflowRequest(ConfigModel):
    preset: str | None = None
    chain: str | None = None
    study: str | None = None
    variant: str | None = None
    delay_seconds: int | None = Field(default=None, gt=0)
    trial_count: int | None = Field(default=None, gt=0)
    storage_root: Path | None = None
    dry_run: bool | None = None


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
    request: WorkflowRequest,
) -> AcquireConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.TRAIN],
    request: WorkflowRequest,
) -> TrainConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.TUNE],
    request: WorkflowRequest,
) -> TuneConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.EVALUATE],
    request: WorkflowRequest,
) -> EvaluateConfig: ...


def resolve_workflow_config(
    workflow_kind: WorkflowTask,
    request: WorkflowRequest,
) -> WorkflowConfig:
    """Resolve one workflow request into one validated workflow config."""

    if request.preset is None:
        raise ConfigResolutionError("preset is required")
    try:
        frame = apply_request_overrides(
            load_preset_frame(request.preset),
            workflow=workflow_kind,
            chain=request.chain,
            study=request.study,
            variant=request.variant,
            delay_seconds=request.delay_seconds,
            trial_count=request.trial_count,
            storage_root=request.storage_root,
            dry_run=request.dry_run,
        )
        return _resolve_preset_frame(workflow_kind, frame)
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc


def _resolve_preset_frame(
    workflow: WorkflowTask,
    frame: PresetFrame,
) -> WorkflowConfig:
    if workflow is WorkflowTask.ACQUIRE:
        return _resolve_acquire_config(frame)
    if workflow is WorkflowTask.TRAIN:
        return _resolve_train_config(frame)
    if workflow is WorkflowTask.TUNE:
        return _resolve_tune_config(frame)
    if workflow is WorkflowTask.EVALUATE:
        return _resolve_evaluate_config(frame)
    raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}")


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
    return coerce_prediction_config(load_named_group(name, "prediction"))


def _resolve_model(name: str) -> ModelConfig[str]:
    return coerce_model_config(load_named_group(name, _MODEL_GROUP))


def _resolve_evaluation(name: str | None) -> EvaluatorConfig | None:
    if name is None:
        return None
    return EvaluatorConfig.model_validate(load_named_group(name, "evaluation"))


def _resolve_objective(name: str, *, evaluation_name: str | None) -> ObjectiveConfig:
    raw_objective = load_named_group(name, "objective")
    expected_evaluation = _benchmark_evaluation_name(raw_objective)
    if expected_evaluation is not None and expected_evaluation != evaluation_name:
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
    frame: PresetFrame,
) -> tuple[
    DatasetSpec,
    ChainSpec,
    StorageSpec,
    ProblemSpec,
    ModelConfig[str],
    DatasetBuilderConfig,
    FeatureSetConfig,
    PredictionConfig,
    ObjectiveConfig,
    EvaluatorConfig | None,
    StudyConfig,
    ArtifactConfig,
]:
    dataset = _resolve_dataset(frame.dataset)
    chain = _resolve_chain(frame.chain)
    storage = _resolve_storage(frame.storage)
    problem = _resolve_problem(frame.problem)
    model = _resolve_model(frame.model)
    dataset_builder = _resolve_dataset_builder(frame.dataset_builder)
    feature_set = _resolve_feature_set(frame.feature_set)
    prediction = _resolve_prediction(frame.prediction)
    objective = _resolve_objective(frame.objective, evaluation_name=frame.evaluation)
    evaluation = _resolve_evaluation(frame.evaluation)
    study = _resolve_study(frame.study)
    artifact = _resolve_artifact(frame.artifact)
    return (
        dataset,
        chain,
        storage,
        problem,
        model,
        dataset_builder,
        feature_set,
        prediction,
        objective,
        evaluation,
        study,
        artifact,
    )


def _resolve_model_workflow_spine(
    frame: PresetFrame,
    *,
    model: ModelConfig[str],
    problem: ProblemSpec,
    prediction: PredictionConfig,
    require_tuning: bool,
    allow_tuned_variant: bool,
) -> tuple[TrainingConfig, SplitConfig, TuningConfig | None, TuningSpaceConfig | None]:
    training = frame.training
    split = frame.split
    artifact = _resolve_artifact(frame.artifact)
    if require_tuning or (allow_tuned_variant and artifact.variant.value == "tuned"):
        tuning = frame.tuning
        tuning_space = load_named_tuning_space(
            frame.tuning_space,
            model_config=model,
            problem_config=problem,
        )
        return training, split, tuning, tuning_space
    return training, split, None, None


def _resolve_acquire_config(frame: PresetFrame) -> AcquireConfig:
    dataset = _resolve_dataset(frame.dataset)
    chain = _resolve_chain(frame.chain)
    storage = _resolve_storage(frame.storage)
    problem = _resolve_problem(frame.problem)
    rpc_endpoint = _resolve_rpc_endpoint(frame.provider, chain=chain)
    feature_set = _resolve_feature_set(frame.feature_set)
    return AcquireConfig(
        chain=chain,
        dataset=dataset,
        storage=storage,
        problem=problem,
        feature_set=feature_set,
        rpc_endpoint=rpc_endpoint,
        acquisition=frame.acquisition,
    )


def _resolve_train_config(frame: PresetFrame) -> TrainConfig:
    (
        dataset,
        chain,
        storage,
        problem,
        model,
        dataset_builder,
        feature_set,
        prediction,
        objective,
        evaluation,
        study,
        artifact,
    ) = _resolve_model_workflow_base(frame)
    training, split, tuning, tuning_space = _resolve_model_workflow_spine(
        frame,
        model=model,
        problem=problem,
        prediction=prediction,
        require_tuning=False,
        allow_tuned_variant=True,
    )
    return TrainConfig(
        chain=chain,
        dataset=dataset,
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
        training=training,
        split=split,
        tuning=tuning,
        tuning_space=tuning_space,
    )


def _resolve_tune_config(frame: PresetFrame) -> TuneConfig:
    (
        dataset,
        chain,
        storage,
        problem,
        model,
        dataset_builder,
        feature_set,
        prediction,
        objective,
        evaluation,
        study,
        artifact,
    ) = _resolve_model_workflow_base(frame)
    training, split, tuning, tuning_space = _resolve_model_workflow_spine(
        frame,
        model=model,
        problem=problem,
        prediction=prediction,
        require_tuning=True,
        allow_tuned_variant=False,
    )
    assert tuning is not None
    assert tuning_space is not None
    return TuneConfig(
        chain=chain,
        dataset=dataset,
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
        training=training,
        split=split,
        tuning=tuning,
        tuning_space=tuning_space,
    )


def _resolve_evaluate_config(frame: PresetFrame) -> EvaluateConfig:
    (
        dataset,
        chain,
        storage,
        problem,
        model,
        dataset_builder,
        feature_set,
        prediction,
        objective,
        evaluation,
        study,
        artifact,
    ) = _resolve_model_workflow_base(frame)
    training, split, tuning, tuning_space = _resolve_model_workflow_spine(
        frame,
        model=model,
        problem=problem,
        prediction=prediction,
        require_tuning=False,
        allow_tuned_variant=True,
    )
    return EvaluateConfig(
        chain=chain,
        dataset=dataset,
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
        training=training,
        split=split,
        delay_seconds=frame.delay_seconds,
        tuning=tuning,
        tuning_space=tuning_space,
    )
