"""Shared CLI utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from ..core.reporting import Reporter

ChainFilterOption = Annotated[
    str | None,
    typer.Option("--chain", metavar="CHAIN", help="Filter by chain."),
]
DatasetFilterOption = Annotated[
    str | None,
    typer.Option("--dataset", metavar="DATASET", help="Filter by dataset name."),
]
FeatureSetFilterOption = Annotated[
    str | None,
    typer.Option("--feature-set", metavar="FEATURE_SET", help="Filter by feature set."),
]
PredictionFilterOption = Annotated[
    str | None,
    typer.Option("--prediction", metavar="PREDICTION", help="Filter by prediction config."),
]
ModelFilterOption = Annotated[
    str | None,
    typer.Option("--model", metavar="MODEL", help="Filter by model."),
]
ProblemFilterOption = Annotated[
    str | None,
    typer.Option("--problem", metavar="PROBLEM", help="Filter by problem."),
]
StudyFilterOption = Annotated[
    str | None,
    typer.Option("--study", metavar="STUDY", help="Filter by study name."),
]
VariantFilterOption = Annotated[
    str | None,
    typer.Option("--variant", metavar="VARIANT", help="Filter by artifact variant."),
]
StorageRootReadOption = Annotated[
    Path | None,
    typer.Option("--storage-root", metavar="PATH", help="Read from a non-default output root."),
]
StorageRootDeleteOption = Annotated[
    Path | None,
    typer.Option("--storage-root", metavar="PATH", help="Delete from a non-default output root."),
]


def resolve_storage_root(storage_root: Path | None) -> Path:
    return storage_root or Path("outputs")


def print_sections(
    title: str,
    sections: list[tuple[str, list[tuple[str, str]]]],
    *,
    err: bool = False,
) -> None:
    reporter = Reporter(stream=sys.stdout, error_stream=sys.stderr)
    if err:
        reporter.diagnostic_sections(title, sections)
    else:
        reporter.sections(title, sections)
