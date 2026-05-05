"""Workflow training and tuning preflight."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import TrainConfig, TuneConfig
from ..corpus.coverage import training_coverage_requirement, validate_corpus_coverage
from ..corpus.metadata import DatasetManifest
from ..modeling.pipeline import TrainingSpec, build_artifact_training_spec
from ..modeling.tuning_execution import build_tuning_coverage_spec
from ..storage.workflow_roots import TrainWorkflowRoots, TuneWorkflowRoots


@dataclass(frozen=True, slots=True)
class TrainPreflight:
    corpus_manifest: DatasetManifest
    spec: TrainingSpec


@dataclass(frozen=True, slots=True)
class TunePreflight:
    corpus_manifest: DatasetManifest
    coverage_spec: TrainingSpec


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
    coverage_spec = build_tuning_coverage_spec(
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
    return TunePreflight(corpus_manifest=corpus_manifest, coverage_spec=coverage_spec)
