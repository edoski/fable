"""FABLE command-line application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer

from .config import (
    WORKFLOW_REQUEST_ADAPTER,
    Method,
    TrainRequest,
    TuneRequest,
    WorkflowRequest,
)
from .corpus import load_corpus
from .environment import resolve_storage_root
from .evaluation import evaluate
from .execution import CandidateProcessInput, submit_candidate
from .execution import submit as submit_workflow
from .modeling import train
from .study import publish_study
from .temporal.history import prepare_fit_history
from .tuning import run_candidate

app = typer.Typer(add_completion=False)
remote_app = typer.Typer(add_completion=False)
study_app = typer.Typer(add_completion=False)
app.add_typer(remote_app, name="remote", hidden=True)
app.add_typer(study_app, name="study")


@app.command("submit")
def submit_command(
    request_paths: Annotated[
        list[Path],
        typer.Argument(metavar="REQUEST.json"),
    ],
) -> None:
    requests: list[WorkflowRequest] = [
        WORKFLOW_REQUEST_ADAPTER.validate_json(path.read_bytes()) for path in request_paths
    ]
    for request in requests:
        typer.echo(submit_workflow(request))


@remote_app.command("workflow")
def workflow_command() -> None:
    request = WORKFLOW_REQUEST_ADAPTER.validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    storage_root = resolve_storage_root()

    if isinstance(request, TrainRequest):
        source = request.source
        corpus = load_corpus(storage_root, source.corpus_id)
        prepared = prepare_fit_history(corpus, source.experiment)
        train(request, prepared, storage_root)
    else:
        evaluate(request, storage_root)


@remote_app.command("candidate", hidden=True)
def candidate_command() -> None:
    candidate = CandidateProcessInput.model_validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    storage_root = resolve_storage_root()
    run_candidate(
        storage_root,
        candidate.request,
        candidate.method,
    )


@study_app.command("run")
def study_run_command(
    request_path: Annotated[Path, typer.Argument(metavar="TUNE_REQUEST.json")],
    method_path: Annotated[Path, typer.Argument(metavar="METHOD.json")],
) -> None:
    request = TuneRequest.model_validate_json(request_path.read_bytes(), strict=True)
    method = Method.model_validate_json(method_path.read_bytes(), strict=True)
    typer.echo(submit_candidate(request, method))


@study_app.command("finalize")
def study_finalize_command(
    study_id: Annotated[UUID, typer.Argument(metavar="STUDY_ID")],
) -> None:
    publish_study(resolve_storage_root(), study_id)
