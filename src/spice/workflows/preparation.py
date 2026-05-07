"""Prepared workflow inputs and root resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ..config.models import (
    ArtifactVariant,
    EvaluateConfig,
    TrainConfig,
    TuneConfig,
    TunedParameterSet,
    TunedProblemParams,
)
from ..corpus.coverage import training_coverage_requirement, validate_corpus_coverage
from ..corpus.metadata import DatasetManifest
from ..modeling.artifact_inference import (
    ArtifactInferenceContext,
    prepare_artifact_inference_context,
)
from ..modeling.pipeline import (
    TrainingSpec,
    build_artifact_training_spec,
    build_trial_training_spec,
)
from ..modeling.tuning import apply_study_best_params, apply_tuned_parameters
from ..storage.workflow_root_materialization import (
    materialize_evaluate_roots,
    materialize_train_roots,
    materialize_tune_roots,
)
from ..storage.workflow_roots import (
    EvaluateWorkflowRoots,
    TrainWorkflowRoots,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
)


@dataclass(frozen=True, slots=True)
class PreparedTrainWorkflow:
    requested_config: TrainConfig
    active_config: TrainConfig
    roots: TrainWorkflowRoots
    corpus_manifest: DatasetManifest
    spec: TrainingSpec


@dataclass(frozen=True, slots=True)
class PreparedTuneWorkflow:
    config: TuneConfig
    roots: TuneWorkflowRoots
    corpus_manifest: DatasetManifest


@dataclass(frozen=True, slots=True)
class PreparedEvaluateWorkflow:
    config: EvaluateConfig
    roots: EvaluateWorkflowRoots
    inference_context: ArtifactInferenceContext


def prepare_train(config: TrainConfig) -> PreparedTrainWorkflow:
    roots = materialize_train_roots(config)
    active_config = _active_train_config(config, roots)
    corpus_manifest = roots.corpus.load_manifest()
    spec = build_artifact_training_spec(
        active_config,
        corpus=roots.corpus,
        artifact=roots.artifact,
        corpus_manifest=corpus_manifest,
    )
    _validate_training_coverage(corpus_manifest, spec=spec)
    return PreparedTrainWorkflow(
        requested_config=config,
        active_config=active_config,
        roots=roots,
        corpus_manifest=corpus_manifest,
        spec=spec,
    )


def prepare_tune(config: TuneConfig) -> PreparedTuneWorkflow:
    roots = materialize_tune_roots(config)
    corpus_manifest = roots.corpus.load_manifest()
    coverage_spec = _build_tuning_coverage_spec(
        config,
        roots=roots,
        corpus_manifest=corpus_manifest,
    )
    _validate_training_coverage(corpus_manifest, spec=coverage_spec)
    return PreparedTuneWorkflow(
        config=config,
        roots=roots,
        corpus_manifest=corpus_manifest,
    )


def prepare_evaluate(config: EvaluateConfig) -> PreparedEvaluateWorkflow:
    roots = materialize_evaluate_roots(config)
    return PreparedEvaluateWorkflow(
        config=config,
        roots=roots,
        inference_context=prepare_artifact_inference_context(
            config,
            corpus=roots.corpus,
            artifact=roots.artifact,
        ),
    )


def _active_train_config(config: TrainConfig, roots: TrainWorkflowRoots) -> TrainConfig:
    if config.artifact.variant is not ArtifactVariant.TUNED:
        return config
    assert isinstance(roots, TunedTrainWorkflowRoots)
    applied = apply_study_best_params(
        config,
        study=roots.study,
        corpus=roots.corpus,
    )
    return cast(TrainConfig, applied.config)


def _build_tuning_coverage_spec(
    config: TuneConfig,
    *,
    roots: TuneWorkflowRoots,
    corpus_manifest: DatasetManifest,
) -> TrainingSpec:
    if (
        config.tuning_space.problem is None
        or config.tuning_space.problem.lookback_seconds is None
    ):
        return build_trial_training_spec(
            config,
            corpus=roots.corpus,
            study=roots.study,
            corpus_manifest=corpus_manifest,
        )
    return build_trial_training_spec(
        apply_tuned_parameters(
            config,
            TunedParameterSet(
                problem=TunedProblemParams(
                    lookback_seconds=max(config.tuning_space.problem.lookback_seconds)
                )
            ),
        ),
        corpus=roots.corpus,
        study=roots.study,
        corpus_manifest=corpus_manifest,
    )


def _validate_training_coverage(
    corpus_manifest: DatasetManifest,
    *,
    spec: TrainingSpec,
) -> None:
    validate_corpus_coverage(
        corpus_manifest,
        contract=spec.problem_contract,
        feature_contract=spec.feature_contract,
        requirement=training_coverage_requirement(spec.problem_contract),
    )
