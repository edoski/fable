"""Hidden remote workflow execution."""

from __future__ import annotations

import sys

import typer

from ...config import WORKFLOW_REQUEST_ADAPTER, BaselineSource, TrainRequest
from ...corpus import load_corpus
from ...environment import resolve_storage_root
from ...evaluation import evaluate
from ...execution import _CandidateProcessInput
from ...modeling import train
from ...temporal.history import prepare_fit_history
from ...tuning import run_candidate

app = typer.Typer(add_completion=False)


@app.command("workflow")
def workflow_command() -> None:
    request = WORKFLOW_REQUEST_ADAPTER.validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    storage_root = resolve_storage_root()

    if isinstance(request, TrainRequest):
        source = request.source
        experiment = (
            source.training_definition.experiment
            if isinstance(source, BaselineSource)
            else source.experiment
        )
        corpus = load_corpus(storage_root, source.corpus_id)
        prepared = prepare_fit_history(corpus, experiment)
        train(request, prepared, storage_root)
    else:
        evaluate(request, storage_root)


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
    )
