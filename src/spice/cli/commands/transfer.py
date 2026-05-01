"""Storage transfer command routing."""

from __future__ import annotations

from typing import Annotated

import typer

from ...execution.session import open_execution_session
from ...execution.transfer import (
    pull_artifact_from_cluster,
    push_dataset_to_cluster,
)
from ..options import (
    DEFAULT_REMOTE_TARGET,
    RemoteTargetOption,
    StorageRootReadOption,
    resolve_storage_root,
)

push_app = typer.Typer(
    help="Copy one local root into cluster storage.",
    no_args_is_help=True,
)
pull_app = typer.Typer(
    help="Copy one cluster root into local storage.",
    no_args_is_help=True,
)
app = typer.Typer(
    help="Copy storage roots between local and cluster storage.",
    no_args_is_help=True,
)
app.add_typer(push_app, name="push")
app.add_typer(pull_app, name="pull")
ReplaceOption = Annotated[
    bool,
    typer.Option("--replace", help="Replace an existing destination root."),
]


@push_app.command("dataset", short_help="Push one dataset root to cluster storage.")
def push_dataset_command(
    dataset_id: Annotated[
        str,
        typer.Option("--dataset-id", metavar="DATASET_ID", help="Push this dataset root."),
    ],
    storage_root: StorageRootReadOption = None,
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    replace: ReplaceOption = False,
) -> None:
    root = resolve_storage_root(storage_root)
    session = open_execution_session(target)
    record = push_dataset_to_cluster(
        storage_root=root,
        session=session,
        dataset_id=dataset_id,
        replace=replace,
    )
    typer.echo(f"push dataset={record.dataset_name} dataset_id={record.dataset_id}")


@pull_app.command("artifact", short_help="Pull one artifact root from cluster storage.")
def pull_artifact_command(
    artifact_id: Annotated[
        str,
        typer.Option("--artifact-id", metavar="ARTIFACT_ID", help="Pull this artifact root."),
    ],
    storage_root: StorageRootReadOption = None,
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    replace: ReplaceOption = False,
) -> None:
    root = resolve_storage_root(storage_root)
    session = open_execution_session(target)
    pulled = pull_artifact_from_cluster(
        storage_root=root,
        session=session,
        artifact_id=artifact_id,
        replace=replace,
    )
    record = pulled.local_record
    typer.echo(f"pull artifact={record.artifact_id}")
    if not pulled.dataset_present:
        typer.echo(
            (
                "warning: matching local dataset root is missing; "
                "local inspection still needs that dataset"
            ),
            err=True,
        )
