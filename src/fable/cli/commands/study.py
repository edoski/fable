"""Study candidate submission and publication commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer

from ...config import METHOD_ADAPTER, TuneRequest
from ...execution.submission import _submit_candidate
from ...study import publish_study

app = typer.Typer(add_completion=False)


@app.command("run")
def run_command(
    request_path: Annotated[Path, typer.Argument(metavar="TUNE_REQUEST.json")],
    method_path: Annotated[Path, typer.Argument(metavar="METHOD.json")],
) -> None:
    request = TuneRequest.model_validate_json(request_path.read_bytes(), strict=True)
    method = METHOD_ADAPTER.validate_json(method_path.read_bytes(), strict=True)
    typer.echo(_submit_candidate(request, method))


@app.command("finalize")
def finalize_command(
    study_id: Annotated[UUID, typer.Argument(metavar="STUDY_ID")],
) -> None:
    if study_id.version != 4:
        raise ValueError("STUDY_ID must be a UUIDv4")

    storage_root = Path(os.environ["STORAGE_ROOT"])
    if not storage_root.is_absolute():
        raise ValueError("STORAGE_ROOT must be an absolute path")

    publish_study(storage_root, study_id)
