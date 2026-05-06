"""Workflow training and tuning preflight."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import TrainConfig, TuneConfig, TunedParameterSet, TunedProblemParams
from ..corpus.coverage import training_coverage_requirement, validate_corpus_coverage
from ..corpus.metadata import DatasetManifest
from ..modeling.pipeline import (
    TrainingSpec,
    build_artifact_training_spec,
    build_trial_training_spec,
)
from ..modeling.tuning import apply_tuned_parameters
from ..storage.workflow_roots import TrainWorkflowRoots, TuneWorkflowRoots


@dataclass(frozen=True, slots=True)
class TrainPreflight:
    corpus_manifest: DatasetManifest
    spec: TrainingSpec


@dataclass(frozen=True, slots=True)
class TunePreflight:
    corpus_manifest: DatasetManifest


def prepare_train_preflight(
    config: TrainConfig,
    roots: TrainWorkflowRoots,
) -> TrainPreflight:
    corpus_manifest = roots.corpus.load_manifest()
    spec = build_artifact_training_spec(
        config,
        corpus=roots.corpus,
        artifact=roots.artifact,
        corpus_manifest=corpus_manifest,
    )
    validate_corpus_coverage(
        corpus_manifest,
        contract=spec.problem_contract,
        feature_contract=spec.feature_contract,
        requirement=training_coverage_requirement(spec.problem_contract),
    )
    return TrainPreflight(corpus_manifest=corpus_manifest, spec=spec)


def prepare_tune_preflight(
    config: TuneConfig,
    roots: TuneWorkflowRoots,
) -> TunePreflight:
    corpus_manifest = roots.corpus.load_manifest()
    coverage_spec = _build_tuning_coverage_spec(
        config,
        roots=roots,
        corpus_manifest=corpus_manifest,
    )
    validate_corpus_coverage(
        corpus_manifest,
        contract=coverage_spec.problem_contract,
        feature_contract=coverage_spec.feature_contract,
        requirement=training_coverage_requirement(coverage_spec.problem_contract),
    )
    return TunePreflight(corpus_manifest=corpus_manifest)


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
