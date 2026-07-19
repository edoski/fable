"""FABLE command-line application."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..config import WORKFLOW_REQUEST_ADAPTER, WorkflowRequest
from ..execution import submit as submit_workflow
from .commands.corpus import app as corpus_app
from .commands.remote import app as remote_app
from .commands.study import app as study_app

app = typer.Typer(add_completion=False)
app.add_typer(corpus_app, name="corpus")
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


def main() -> None:
    app()
