"""Workflow selection resolution."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar, overload

from pydantic import ValidationError

from ..core.config_model import ConfigModel
from ..core.errors import ConfigResolutionError
from ..modeling.families.base import ModelConfig
from ..modeling.tuned_config import coerce_tuning_space_config
from . import typed_groups as typed
from .group_catalog import ConfigGroup
from .groups import load_named_group_payload
from .models import (
    ArtifactConfig,
    ArtifactVariant,
    EvaluateConfig,
    ProblemSpec,
    StorageSpec,
    StudyConfig,
    TrainConfig,
    TuneConfig,
    TuningConfig,
    TuningSpaceConfig,
)
from .resolved_workflows import (
    ResolvedEvaluateWorkflowFields,
    ResolvedModelWorkflowFields,
    ResolvedTrainWorkflowFields,
    ResolvedTuneWorkflowFields,
    assemble_resolved_workflow_config,
    final_evaluate_batch_size,
)
from .selections import (
    EvaluateWorkflowSelection,
    SurfaceWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
    WorkflowSelection,
)
from .surfaces import SurfaceFrame

ConfigModelT = TypeVar("ConfigModelT", bound=ConfigModel)
SelectionValueT = TypeVar("SelectionValueT")

ResolvedSelectionConfig = TrainConfig | TuneConfig | EvaluateConfig


def load_named_tuning_space(
    name: str,
    *,
    model_config: ModelConfig[str],
    problem_config: ProblemSpec,
) -> TuningSpaceConfig:
    tuning_space = coerce_tuning_space_config(
        load_named_group_payload(name, ConfigGroup.TUNING_SPACE),
        model_config=model_config,
        problem_config=problem_config,
    )
    if tuning_space is None:
        raise ConfigResolutionError(f"tuning space {name} resolved to None")
    return tuning_space


def _load_surface(selection: SurfaceWorkflowSelection) -> SurfaceFrame:
    if selection.surface is None:
        raise ConfigResolutionError("surface is required")
    return typed.load(typed.SURFACE, selection.surface)


def _selected(
    override: SelectionValueT | None,
    default: SelectionValueT | None,
) -> SelectionValueT | None:
    return default if override is None else override


def _surface_storage(
    selection: SurfaceWorkflowSelection,
    frame: SurfaceFrame,
) -> StorageSpec | None:
    if selection.storage_root is None:
        return frame.storage
    return StorageSpec(root=selection.storage_root)


def _selected_model_artifact(
    selection: TrainWorkflowSelection | TuneWorkflowSelection,
    frame: SurfaceFrame,
) -> ArtifactConfig | None:
    if isinstance(selection, TrainWorkflowSelection) and selection.variant is not None:
        return ArtifactConfig(variant=ArtifactVariant(selection.variant))
    return frame.artifact


@overload
def resolve_workflow_config(
    selection: TrainWorkflowSelection,
) -> TrainConfig: ...


@overload
def resolve_workflow_config(
    selection: TuneWorkflowSelection,
) -> TuneConfig: ...


@overload
def resolve_workflow_config(
    selection: EvaluateWorkflowSelection,
) -> EvaluateConfig: ...


def resolve_workflow_config(selection: WorkflowSelection) -> ResolvedSelectionConfig:
    """Resolve one workflow selection into one validated workflow config."""

    try:
        if isinstance(selection, EvaluateWorkflowSelection):
            return _resolve_evaluate_config(selection)
        if isinstance(selection, TrainWorkflowSelection):
            return _resolve_train_config(selection)
        if isinstance(selection, TuneWorkflowSelection):
            return _resolve_tune_config(selection)
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc
    raise ConfigResolutionError(f"Unsupported workflow selection: {type(selection).__name__}")


def _resolve_problem(name: str | ProblemSpec) -> ProblemSpec:
    if isinstance(name, ProblemSpec):
        return name
    return typed.load(typed.PROBLEM, name)


def _resolve_evaluator(name: str | None):
    if name is None:
        return None
    return typed.load(typed.EVALUATOR, name)


def _resolve_training_cutoff(
    selection: TrainWorkflowSelection | TuneWorkflowSelection,
    frame: SurfaceFrame,
) -> int | None:
    evaluations_name = _selected(selection.evaluations, frame.evaluations)
    if evaluations_name is None:
        return None
    return typed.load(typed.EVALUATIONS, evaluations_name).training_cutoff_timestamp


def _resolve_storage(storage: StorageSpec | None) -> StorageSpec:
    return storage or StorageSpec()


def _resolve_study(study: StudyConfig | None) -> StudyConfig:
    return study or StudyConfig()


def _resolve_artifact(artifact: ArtifactConfig | None) -> ArtifactConfig:
    return artifact or ArtifactConfig()


def _resolve_model_workflow_fields(
    selection: TrainWorkflowSelection | TuneWorkflowSelection,
    frame: SurfaceFrame,
    *,
    require_tuning: bool,
    allow_tuned_variant: bool,
) -> ResolvedModelWorkflowFields:
    corpus = typed.load(typed.CORPUS, frame.corpus)
    chain = typed.load(
        typed.CHAIN,
        _required_value(_selected(selection.chain, frame.chain), "chain"),
    )
    storage = _resolve_storage(_surface_storage(selection, frame))
    problem = _resolve_problem(
        _required_value(_selected(selection.problem, frame.problem), "problem")
    )
    model = typed.load(typed.MODEL, _required(_selected(selection.model, frame.model), "model"))
    prediction = typed.load(typed.PREDICTION, frame.prediction)
    study = _resolve_study(
        frame.study if selection.study is None else StudyConfig(name=selection.study)
    )
    artifact = _resolve_artifact(_selected_model_artifact(selection, frame))
    training = typed.load(
        typed.TRAINING,
        _required(_selected(selection.training, frame.training.id), "training"),
    )
    split = typed.load(
        typed.SPLIT,
        _required(_selected(selection.split, frame.training.split), "split"),
    )
    tuning: TuningConfig | None = None
    tuning_space: TuningSpaceConfig | None = None
    if require_tuning or (allow_tuned_variant and artifact.variant is ArtifactVariant.TUNED):
        tuning = typed.load(
            typed.TUNING,
            _required(_selected(selection.tuning, frame.tuning.id), "tuning"),
        )
        tuning_space = load_named_tuning_space(
            _required(_selected(selection.tuning_space, frame.tuning.space), "tuning.space"),
            model_config=model,
            problem_config=problem,
        )
    return ResolvedModelWorkflowFields(
        corpus=corpus,
        chain=chain,
        storage=storage,
        problem=problem,
        model=model,
        prediction=prediction,
        study=study,
        artifact=artifact,
        training=training,
        split=split,
        tuning=tuning,
        tuning_space=tuning_space,
    )


def _resolve_train_config(selection: TrainWorkflowSelection) -> TrainConfig:
    frame = _load_surface(selection)
    model_fields = _resolve_model_workflow_fields(
        selection,
        frame,
        require_tuning=False,
        allow_tuned_variant=True,
    )
    return assemble_resolved_workflow_config(
        ResolvedTrainWorkflowFields(
            model_fields=model_fields,
            corpus_id=selection.corpus_id,
            study_id=selection.study_id,
            training_cutoff_timestamp=_resolve_training_cutoff(selection, frame),
        )
    )


def _resolve_tune_config(selection: TuneWorkflowSelection) -> TuneConfig:
    frame = _load_surface(selection)
    model_fields = _resolve_model_workflow_fields(
        selection,
        frame,
        require_tuning=True,
        allow_tuned_variant=False,
    )
    assert model_fields.tuning is not None
    assert model_fields.tuning_space is not None
    tuning = model_fields.tuning
    if selection.trial_count is not None:
        tuning = _validated_config_update(tuning, trial_count=selection.trial_count)
    return assemble_resolved_workflow_config(
        ResolvedTuneWorkflowFields(
            model_fields=replace(model_fields, tuning=tuning),
            corpus_id=_required(selection.corpus_id, "corpus_id"),
            training_cutoff_timestamp=_resolve_training_cutoff(selection, frame),
        )
    )


def _resolve_evaluate_config(selection: EvaluateWorkflowSelection) -> EvaluateConfig:
    evaluator = _resolve_evaluator(_required(selection.evaluator, "evaluator"))
    if evaluator is None:
        raise ConfigResolutionError("evaluator is required")
    return assemble_resolved_workflow_config(
        ResolvedEvaluateWorkflowFields(
            storage=_resolve_storage(
                None
                if selection.storage_root is None
                else StorageSpec(root=selection.storage_root)
            ),
            artifact_id=_required(selection.artifact_id, "artifact_id"),
            corpus_id=_required(selection.corpus_id, "corpus_id"),
            evaluation_window=_required_value(
                selection.evaluation_window,
                "evaluation_window",
            ),
            evaluator=evaluator,
            delay_seconds=selection.delay_seconds,
            batch_size=final_evaluate_batch_size(selection.batch_size),
        ),
    )


def _required(value: str | None, label: str) -> str:
    return _required_value(value, label)


def _required_value(value: SelectionValueT | None, label: str) -> SelectionValueT:
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
