"""Transfer and refresh command routing."""

from __future__ import annotations

from typing import Annotated

import typer

from ...remote import (
    pull_artifact_from_remote,
    pull_study_from_remote,
    push_dataset_to_remote,
    push_study_to_remote,
)
from ...storage.roots import (
    DatasetSelector,
    refresh_catalog,
)
from ..options import (
    ChainFilterOption,
    DatasetFilterOption,
    FeatureSetFilterOption,
    ModelFilterOption,
    PredictionFilterOption,
    ProblemFilterOption,
    StorageRootReadOption,
    StudyFilterOption,
    VariantFilterOption,
    resolve_storage_root,
)
from ._selectors import artifact_selector, study_selector

push_app = typer.Typer(
    help="Copy one local root into the remote cluster storage.",
    no_args_is_help=True,
)
pull_app = typer.Typer(
    help="Copy one remote root into local storage.",
    no_args_is_help=True,
)
refresh_app = typer.Typer(
    help="Rebuild derived storage indexes.",
    no_args_is_help=True,
)

ReplaceOption = Annotated[
    bool,
    typer.Option("--replace", help="Replace an existing destination root."),
]


@push_app.command("dataset", short_help="Push one dataset root to remote storage.")
def push_dataset_command(
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    storage_root: StorageRootReadOption = None,
    replace: ReplaceOption = False,
) -> None:
    root = resolve_storage_root(storage_root)
    record = push_dataset_to_remote(
        storage_root=root,
        selector=DatasetSelector(chain_name=chain, dataset_name=dataset),
        replace=replace,
    )
    typer.echo(
        f"pushed dataset {record.dataset_name} to remote root {record.dataset_id}"
    )


@push_app.command("study", short_help="Push one study root to remote storage.")
def push_study_command(
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    feature_set: FeatureSetFilterOption = None,
    prediction: PredictionFilterOption = None,
    model: ModelFilterOption = None,
    problem: ProblemFilterOption = None,
    study: StudyFilterOption = None,
    storage_root: StorageRootReadOption = None,
    replace: ReplaceOption = False,
) -> None:
    root = resolve_storage_root(storage_root)
    record = push_study_to_remote(
        storage_root=root,
        selector=study_selector(
            chain=chain,
            dataset=dataset,
            feature_set=feature_set,
            prediction=prediction,
            model=model,
            problem=problem,
            study=study,
        ),
        replace=replace,
    )
    typer.echo(f"pushed study {record.study_name} to remote root {record.study_id}")


@pull_app.command("artifact", short_help="Pull one artifact root from remote storage.")
def pull_artifact_command(
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    feature_set: FeatureSetFilterOption = None,
    prediction: PredictionFilterOption = None,
    model: ModelFilterOption = None,
    problem: ProblemFilterOption = None,
    variant: VariantFilterOption = None,
    study: StudyFilterOption = None,
    storage_root: StorageRootReadOption = None,
    replace: ReplaceOption = False,
) -> None:
    root = resolve_storage_root(storage_root)
    record, dataset_present = pull_artifact_from_remote(
        storage_root=root,
        selector=artifact_selector(
            chain=chain,
            dataset=dataset,
            feature_set=feature_set,
            prediction=prediction,
            model=model,
            problem=problem,
            variant=variant,
            study=study,
        ),
        replace=replace,
    )
    typer.echo(f"pulled artifact {record.artifact_id} into local storage")
    if not dataset_present:
        typer.echo(
            (
                "warning: matching local dataset root is missing; "
                "local evaluate still needs that corpus"
            ),
            err=True,
        )


@pull_app.command("study", short_help="Pull one study root from remote storage.")
def pull_study_command(
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    feature_set: FeatureSetFilterOption = None,
    prediction: PredictionFilterOption = None,
    model: ModelFilterOption = None,
    problem: ProblemFilterOption = None,
    study: StudyFilterOption = None,
    storage_root: StorageRootReadOption = None,
    replace: ReplaceOption = False,
) -> None:
    root = resolve_storage_root(storage_root)
    record = pull_study_from_remote(
        storage_root=root,
        selector=study_selector(
            chain=chain,
            dataset=dataset,
            feature_set=feature_set,
            prediction=prediction,
            model=model,
            problem=problem,
            study=study,
        ),
        replace=replace,
    )
    typer.echo(f"pulled study {record.study_id} into local storage")


@refresh_app.command("catalog", short_help="Rebuild the derived storage catalog.")
def refresh_catalog_command(
    storage_root: StorageRootReadOption = None,
) -> None:
    root = resolve_storage_root(storage_root)
    summary = refresh_catalog(root)
    typer.echo(
        " ".join(
            [
                "catalog refreshed",
                f"datasets={summary.dataset_roots}",
                f"studies={summary.study_roots}",
                f"artifacts={summary.artifact_roots}",
            ]
        )
    )
