"""Hidden remote workflow execution."""

from __future__ import annotations

import sys

import typer

from ...config import BaselineSource, TrainRequest
from ...corpus import load_corpus
from ...environment import resolve_storage_root
from ...evaluation import EvaluationDeployment, evaluate
from ...execution import (
    _CandidateProcessInput,
    _Deployment,
    _WorkflowEnvelope,
)
from ...modeling import FitDeployment, train
from ...temporal.history import prepare_fit_history
from ...tuning import run_candidate

app = typer.Typer(add_completion=False)


@app.command("workflow")
def workflow_command() -> None:
    envelope = _WorkflowEnvelope.model_validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    storage_root = resolve_storage_root()
    request = envelope.request
    profile = envelope.deployment

    if isinstance(request, TrainRequest):
        source = request.source
        experiment = (
            source.training_definition.experiment
            if isinstance(source, BaselineSource)
            else source.experiment
        )
        corpus = load_corpus(storage_root, source.corpus_id)
        prepared = prepare_fit_history(corpus, experiment)
        train(request, prepared, storage_root, _fit_deployment(profile))
    else:
        evaluate(request, storage_root, _evaluation_deployment(profile))


@app.command("candidate", hidden=True)
def candidate_command() -> None:
    candidate = _CandidateProcessInput.model_validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    storage_root = resolve_storage_root()
    run_candidate(
        storage_root,
        candidate.request,
        candidate.method,
        _fit_deployment(candidate.deployment),
    )


def _fit_deployment(profile: _Deployment) -> FitDeployment:
    return FitDeployment(
        deterministic=profile.deterministic,
        benchmark=profile.benchmark,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
        prefetch_factor=profile.prefetch_factor,
        persistent_workers=profile.persistent_workers,
        float32_matmul_precision=profile.float32_matmul_precision,
        cuda_matmul_allow_tf32=profile.cuda_matmul_allow_tf32,
        cudnn_allow_tf32=profile.cudnn_allow_tf32,
    )


def _evaluation_deployment(profile: _Deployment) -> EvaluationDeployment:
    return EvaluationDeployment(
        batch_size=profile.evaluation_batch_size,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
        prefetch_factor=profile.prefetch_factor,
        persistent_workers=profile.persistent_workers,
        deterministic=profile.deterministic,
        benchmark=profile.benchmark,
        float32_matmul_precision=profile.float32_matmul_precision,
        cuda_matmul_allow_tf32=profile.cuda_matmul_allow_tf32,
        cudnn_allow_tf32=profile.cudnn_allow_tf32,
    )
