"""Storage sync command routing."""

from __future__ import annotations

from typing import Annotated

import typer

from ...storage.roots import ArtifactSelector, DatasetSelector, StudySelector
from ...storage.sync import (
    pull_artifact_from_cluster,
    pull_study_from_cluster,
    push_dataset_to_cluster,
    push_study_to_cluster,
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

push_app = typer.Typer(
    help="Copy one local root into cluster storage.",
    no_args_is_help=True,
)
pull_app = typer.Typer(
    help="Copy one cluster root into local storage.",
    no_args_is_help=True,
)
ReplaceOption = Annotated[
    bool,
    typer.Option("--replace", help="Replace an existing destination root."),
]


@push_app.command("dataset", short_help="Push one dataset root to cluster storage.")
def push_dataset_command(
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    storage_root: StorageRootReadOption = None,
    replace: ReplaceOption = False,
) -> None:
    root = resolve_storage_root(storage_root)
    record = push_dataset_to_cluster(
        storage_root=root,
        selector=DatasetSelector(chain_name=chain, dataset_name=dataset),
        replace=replace,
    )
    typer.echo(f"push dataset={record.dataset_name} dataset_id={record.dataset_id}")


@push_app.command("study", short_help="Push one study root to cluster storage.")
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
    record = push_study_to_cluster(
        storage_root=root,
        selector=StudySelector(
            chain_name=chain,
            dataset_name=dataset,
            feature_set_id=feature_set,
            prediction_id=prediction,
            model_id=model,
            problem_id=problem,
            study_name=study,
        ),
        replace=replace,
    )
    typer.echo(f"push study={record.study_name} study_id={record.study_id}")


@pull_app.command("artifact", short_help="Pull one artifact root from cluster storage.")
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
    record, dataset_present = pull_artifact_from_cluster(
        storage_root=root,
        selector=ArtifactSelector(
            chain_name=chain,
            dataset_name=dataset,
            feature_set_id=feature_set,
            prediction_id=prediction,
            model_id=model,
            problem_id=problem,
            variant=variant,
            study_name=study,
        ),
        replace=replace,
    )
    typer.echo(f"pull artifact={record.artifact_id}")
    if not dataset_present:
        typer.echo(
            (
                "warning: matching local dataset root is missing; "
                "local evaluate still needs that corpus"
            ),
            err=True,
        )


@pull_app.command("study", short_help="Pull one study root from cluster storage.")
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
    record = pull_study_from_cluster(
        storage_root=root,
        selector=StudySelector(
            chain_name=chain,
            dataset_name=dataset,
            feature_set_id=feature_set,
            prediction_id=prediction,
            model_id=model,
            problem_id=problem,
            study_name=study,
        ),
        replace=replace,
    )
    typer.echo(f"pull study_id={record.study_id}")
