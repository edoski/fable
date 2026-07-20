"""Corpus command routing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from ...acquisition import acquire_corpus
from ...config import CorpusRequest
from ...environment import resolve_storage_root

app = typer.Typer(
    help="Create one native Corpus.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("acquire")
def acquire(
    request_path: Annotated[Path, typer.Argument(metavar="REQUEST.json")],
    rpc_url: Annotated[str, typer.Option("--rpc-url", metavar="URL")],
    poa: Annotated[bool, typer.Option("--poa/--no-poa")],
) -> None:
    request = CorpusRequest.model_validate_json(request_path.read_bytes())
    storage_root = resolve_storage_root()
    asyncio.run(
        acquire_corpus(
            request,
            storage_root=storage_root,
            rpc_url=rpc_url,
            poa=poa,
        )
    )
