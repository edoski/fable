"""Prepared workflow inputs and root resolution."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import AcquireConfig, EvaluateConfig, TrainConfig, TuneConfig
from ..corpus.metadata import DatasetManifest
from ..modeling.artifact_inference import ArtifactInferenceContext
from ..modeling.pipeline import TrainingSpec
from ..storage.workflow_roots import (
    AcquireWorkflowRoots,
    EvaluateWorkflowRoots,
    TrainWorkflowRoots,
    TuneWorkflowRoots,
)
from . import _active_config, _inference_preparation, _root_preparation, _training_preflight


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
    coverage_spec: TrainingSpec


@dataclass(frozen=True, slots=True)
class PreparedEvaluateWorkflow:
    config: EvaluateConfig
    roots: EvaluateWorkflowRoots
    inference_context: ArtifactInferenceContext


def prepare_acquire(config: AcquireConfig) -> PreparedAcquireWorkflow:
    return PreparedAcquireWorkflow(
        config=config,
        roots=_root_preparation.prepare_acquire_roots(config),
    )


def prepare_train(config: TrainConfig) -> PreparedTrainWorkflow:
    roots = _root_preparation.prepare_train_roots(config)
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
    roots = _root_preparation.prepare_tune_roots(config)
    preflight = _training_preflight.prepare_tune_preflight(config, roots)
    return PreparedTuneWorkflow(
        config=config,
        roots=roots,
        corpus_manifest=preflight.corpus_manifest,
        coverage_spec=preflight.coverage_spec,
    )


def prepare_evaluate(config: EvaluateConfig) -> PreparedEvaluateWorkflow:
    roots = _root_preparation.prepare_evaluate_roots(config)
    return PreparedEvaluateWorkflow(
        config=config,
        roots=roots,
        inference_context=_inference_preparation.prepare_inference_context(config, roots),
    )
