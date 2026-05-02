"""Storage command routing."""

from __future__ import annotations

from typing import Annotated, NoReturn

import typer

from ...core.errors import SpiceOperatorError
from ...storage.operator import (
    ArtifactInspectionDetail,
    DatasetInspectionDetail,
    StorageDeleteCommand,
    StorageDeleteCompleted,
    StorageDeleteFailure,
    StorageShowFailure,
    StorageShowQuery,
    StorageShowRendered,
    StudyInspectionDetail,
    delete_storage,
    refresh_storage_catalog,
    render_catalog_refresh,
    show_storage,
)
from ...storage.selectors import ArtifactSelector, DatasetSelector, StudySelector
from ..options import (
    ChainFilterOption,
    DatasetFilterOption,
    FeaturesFilterOption,
    ModelFilterOption,
    PredictionFilterOption,
    ProblemFilterOption,
    StorageRootDeleteOption,
    StorageRootReadOption,
    StudyFilterOption,
    VariantFilterOption,
    print_sections,
    resolve_storage_root,
)

show_app = typer.Typer(
    help="Query stored datasets, studies, and artifacts.",
    epilog=(
        "Example:\n"
        "  spice show artifact --chain ethereum --dataset icdcs_2026 --detail epochs"
    ),
    no_args_is_help=True,
)
delete_app = typer.Typer(
    help="Delete stored datasets, studies, and artifacts.",
    no_args_is_help=True,
)
refresh_app = typer.Typer(
    help="Rebuild derived storage indexes.",
    no_args_is_help=True,
)


DatasetDetailOption = Annotated[
    DatasetInspectionDetail | None,
    typer.Option("--detail", metavar="DETAIL", help="Show one detail table: runs."),
]
StudyDetailOption = Annotated[
    StudyInspectionDetail | None,
    typer.Option(
        "--detail",
        metavar="DETAIL",
        help="Show one detail table: trials or config.",
    ),
]
ArtifactDetailOption = Annotated[
    ArtifactInspectionDetail | None,
    typer.Option(
        "--detail",
        metavar="DETAIL",
        help="Show one detail table: epochs or runs.",
    ),
]


_SELECTOR_FLAGS: dict[str, dict[str, str]] = {
    "dataset": {
        "chain_name": "--chain",
        "dataset_name": "--dataset",
    },
    "study": {
        "chain_name": "--chain",
        "dataset_name": "--dataset",
        "features_id": "--features",
        "prediction_id": "--prediction",
        "model_id": "--model",
        "problem_id": "--problem",
        "study_name": "--study",
    },
    "artifact": {
        "chain_name": "--chain",
        "dataset_name": "--dataset",
        "features_id": "--features",
        "prediction_id": "--prediction",
        "model_id": "--model",
        "problem_id": "--problem",
        "variant": "--variant",
        "study_name": "--study",
    },
}


def _dataset_selector(
    *,
    dataset_id: str | None = None,
    chain: str | None,
    dataset: str | None,
) -> DatasetSelector:
    return DatasetSelector(dataset_id=dataset_id, chain_name=chain, dataset_name=dataset)


def _raise_show_failure(outcome: StorageShowFailure) -> NoReturn:
    for diagnostic in outcome.diagnostics:
        print_sections(diagnostic.title, diagnostic.sections, err=True)
    raise SpiceOperatorError(
        f"{outcome.message}{_narrowing_guidance(outcome.narrowing_attributes)}"
    )


def _raise_delete_failure(outcome: StorageDeleteFailure) -> NoReturn:
    for diagnostic in outcome.diagnostics:
        print_sections(diagnostic.title, diagnostic.sections, err=True)
    raise SpiceOperatorError(
        f"{outcome.message}"
        f"{_cascade_guidance(outcome)}"
        f"{_narrowing_guidance(outcome.narrowing_attributes)}"
    )


def _narrowing_guidance(attributes: tuple[str, ...]) -> str:
    flags = [
        flag for kind_flags in _SELECTOR_FLAGS.values() for attribute in attributes
        if (flag := kind_flags.get(attribute)) is not None
    ]
    if not flags:
        return ""
    return f". Try {', '.join(dict.fromkeys(flags))}."


def _cascade_guidance(outcome: StorageDeleteFailure) -> str:
    return " Re-run with --cascade." if outcome.cascade_available else ""


def _render_show(outcome: StorageShowRendered | StorageShowFailure) -> None:
    if isinstance(outcome, StorageShowFailure):
        _raise_show_failure(outcome)
    print_sections(outcome.renderable.title, outcome.renderable.sections)


def _handle_delete(outcome: StorageDeleteCompleted | StorageDeleteFailure) -> None:
    if isinstance(outcome, StorageDeleteFailure):
        _raise_delete_failure(outcome)


@show_app.command(
    "dataset",
    short_help="Show datasets.",
    help="List datasets or show one dataset in detail.",
)
def show_dataset_command(
    dataset_id: Annotated[
        str | None,
        typer.Option("--dataset-id", metavar="DATASET_ID", help="Show this dataset root."),
    ] = None,
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    storage_root: StorageRootReadOption = None,
    detail: DatasetDetailOption = None,
) -> None:
    root = resolve_storage_root(storage_root)
    selector = _dataset_selector(dataset_id=dataset_id, chain=chain, dataset=dataset)
    _render_show(
        show_storage(
            StorageShowQuery(
                storage_root=root,
                kind="dataset",
                selector=selector,
                has_filters=dataset_id is not None or chain is not None or dataset is not None,
                detail=None if detail is None else detail.value,
            )
        )
    )


@show_app.command(
    "study",
    short_help="Show studies.",
    help="List studies or show one study in detail.",
)
def show_study_command(
    study_id: Annotated[
        str | None,
        typer.Option("--study-id", metavar="STUDY_ID", help="Show this study root."),
    ] = None,
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    features: FeaturesFilterOption = None,
    prediction: PredictionFilterOption = None,
    model: ModelFilterOption = None,
    problem: ProblemFilterOption = None,
    study: StudyFilterOption = None,
    storage_root: StorageRootReadOption = None,
    detail: StudyDetailOption = None,
) -> None:
    root = resolve_storage_root(storage_root)
    selector = StudySelector(
        study_id=study_id,
        chain_name=chain,
        dataset_name=dataset,
        features_id=features,
        prediction_id=prediction,
        model_id=model,
        problem_id=problem,
        study_name=study,
    )
    _render_show(
        show_storage(
            StorageShowQuery(
                storage_root=root,
                kind="study",
                selector=selector,
                has_filters=any(
                    value is not None
                    for value in (
                        study_id,
                        chain,
                        dataset,
                        features,
                        prediction,
                        model,
                        problem,
                        study,
                    )
                ),
                detail=None if detail is None else detail.value,
            )
        )
    )


@show_app.command(
    "artifact",
    short_help="Show artifacts.",
    help="List artifacts or show one artifact in detail.",
)
def show_artifact_command(
    artifact_id: Annotated[
        str | None,
        typer.Option("--artifact-id", metavar="ARTIFACT_ID", help="Show this artifact root."),
    ] = None,
    dataset_id: Annotated[
        str | None,
        typer.Option("--dataset-id", metavar="DATASET_ID", help="Filter by dataset root."),
    ] = None,
    study_id: Annotated[
        str | None,
        typer.Option("--study-id", metavar="STUDY_ID", help="Filter by study root."),
    ] = None,
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    features: FeaturesFilterOption = None,
    prediction: PredictionFilterOption = None,
    model: ModelFilterOption = None,
    problem: ProblemFilterOption = None,
    variant: VariantFilterOption = None,
    study: StudyFilterOption = None,
    storage_root: StorageRootReadOption = None,
    detail: ArtifactDetailOption = None,
) -> None:
    root = resolve_storage_root(storage_root)
    selector = ArtifactSelector(
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        study_id=study_id,
        chain_name=chain,
        dataset_name=dataset,
        features_id=features,
        prediction_id=prediction,
        model_id=model,
        problem_id=problem,
        variant=variant,
        study_name=study,
    )
    _render_show(
        show_storage(
            StorageShowQuery(
                storage_root=root,
                kind="artifact",
                selector=selector,
                has_filters=any(
                    value is not None
                    for value in (
                        artifact_id,
                        dataset_id,
                        study_id,
                        chain,
                        dataset,
                        features,
                        prediction,
                        model,
                        problem,
                        variant,
                        study,
                    )
                ),
                detail=None if detail is None else detail.value,
            )
        )
    )


@delete_app.command(
    "artifact",
    short_help="Delete one artifact.",
    help="Delete exactly one artifact.",
)
def delete_artifact_command(
    artifact_id: Annotated[
        str,
        typer.Option("--artifact-id", metavar="ARTIFACT_ID", help="Delete this artifact root."),
    ],
    storage_root: StorageRootDeleteOption = None,
) -> None:
    root = resolve_storage_root(storage_root)
    _handle_delete(
        delete_storage(
            StorageDeleteCommand(
                storage_root=root,
                kind="artifact",
                selector=ArtifactSelector(artifact_id=artifact_id),
            )
        )
    )


@delete_app.command(
    "study",
    short_help="Delete one study.",
    help=("Delete exactly one study. Use --cascade to also delete dependent tuned artifacts."),
)
def delete_study_command(
    study_id: Annotated[
        str,
        typer.Option("--study-id", metavar="STUDY_ID", help="Delete this study root."),
    ],
    storage_root: StorageRootDeleteOption = None,
    cascade: Annotated[
        bool,
        typer.Option("--cascade", help="Also delete dependent tuned artifacts."),
    ] = False,
) -> None:
    root = resolve_storage_root(storage_root)
    _handle_delete(
        delete_storage(
            StorageDeleteCommand(
                storage_root=root,
                kind="study",
                selector=StudySelector(study_id=study_id),
                cascade=cascade,
            )
        )
    )


@refresh_app.command("catalog", short_help="Rebuild the derived storage catalog.")
def refresh_catalog_command(
    storage_root: StorageRootReadOption = None,
) -> None:
    root = resolve_storage_root(storage_root)
    typer.echo(render_catalog_refresh(refresh_storage_catalog(root)))


@delete_app.command(
    "dataset",
    short_help="Delete one dataset.",
    help=(
        "Delete exactly one dataset. Use --cascade to also delete dependent studies and artifacts."
    ),
)
def delete_dataset_command(
    dataset_id: Annotated[
        str,
        typer.Option("--dataset-id", metavar="DATASET_ID", help="Delete this dataset root."),
    ],
    storage_root: StorageRootDeleteOption = None,
    cascade: Annotated[
        bool,
        typer.Option("--cascade", help="Also delete dependent studies and artifacts."),
    ] = False,
) -> None:
    root = resolve_storage_root(storage_root)
    _handle_delete(
        delete_storage(
            StorageDeleteCommand(
                storage_root=root,
                kind="dataset",
                selector=DatasetSelector(dataset_id=dataset_id),
                cascade=cascade,
            )
        )
    )
