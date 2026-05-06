"""Prepared workflow inputs and root resolution."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import AcquireConfig, EvaluateConfig, TrainConfig, TuneConfig
from ..corpus.metadata import DatasetManifest
from ..modeling.artifact_inference import (
    ArtifactInferenceContext,
    prepare_artifact_inference_context,
)
from ..modeling.pipeline import TrainingSpec
from ..storage.workflow_root_materialization import (
    materialize_acquire_roots,
    materialize_evaluate_roots,
    materialize_train_roots,
    materialize_tune_roots,
)
from ..storage.workflow_roots import (
    AcquireWorkflowRoots,
    EvaluateWorkflowRoots,
    TrainWorkflowRoots,
    TuneWorkflowRoots,
)
from . import _active_config, _training_preflight


@dataclass(frozen=True, slots=True)
class PreparedAcquireWorkflow:
    config: AcquireConfig
    roots: AcquireWorkflowRoots


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


def prepare_acquire(config: AcquireConfig) -> PreparedAcquireWorkflow:
    return PreparedAcquireWorkflow(
        config=config,
        roots=materialize_acquire_roots(config),
    )


def prepare_train(config: TrainConfig) -> PreparedTrainWorkflow:
    roots = materialize_train_roots(config)
    active_config = _active_config.active_train_config(config, roots)
    preflight = _training_preflight.prepare_train_preflight(active_config, roots)
    return PreparedTrainWorkflow(
        requested_config=config,
        active_config=active_config,
        roots=roots,
        corpus_manifest=preflight.corpus_manifest,
        spec=preflight.spec,
    )


def prepare_tune(config: TuneConfig) -> PreparedTuneWorkflow:
    roots = materialize_tune_roots(config)
    preflight = _training_preflight.prepare_tune_preflight(config, roots)
    return PreparedTuneWorkflow(
        config=config,
        roots=roots,
        corpus_manifest=preflight.corpus_manifest,
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
