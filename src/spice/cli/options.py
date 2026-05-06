"""Shared CLI utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from ..core.reporting import Reporter

DEFAULT_REMOTE_TARGET = "disi_l40"


def workflow_selection_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Selection")


def workflow_execution_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Execution")


def workflow_output_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Outputs")


ChainFilterOption = Annotated[
    str | None,
    typer.Option("--chain", metavar="CHAIN", help="Filter by chain."),
]
DatasetFilterOption = Annotated[
    str | None,
    typer.Option("--dataset", metavar="DATASET", help="Filter by dataset name."),
]
FeaturesFilterOption = Annotated[
    str | None,
    typer.Option("--features", metavar="FEATURES", help="Filter by features."),
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
RemoteTargetOption = Annotated[
    str,
    typer.Option(
        "--target",
        metavar="TARGET",
        help="Use a named execution target.",
        rich_help_panel="Execution",
    ),
]
StorageRootReadOption = Annotated[
    Path | None,
    typer.Option("--storage-root", metavar="PATH", help="Read from a non-default output root."),
]
StorageRootDeleteOption = Annotated[
    Path | None,
    typer.Option("--storage-root", metavar="PATH", help="Delete from a non-default output root."),
]
WorkflowSurfaceOption = Annotated[
    str | None,
    workflow_selection_option(
        "--surface",
        metavar="SURFACE",
        help="Resolve a named workflow surface.",
    ),
]
WorkflowChainOption = Annotated[
    str | None,
    workflow_selection_option("--chain", metavar="CHAIN", help="Override the target chain."),
]
WorkflowProblemOption = Annotated[
    str | None,
    workflow_selection_option("--problem", metavar="PROBLEM", help="Override the problem spec."),
]
WorkflowFeaturesOption = Annotated[
    str | None,
    workflow_selection_option(
        "--features",
        metavar="FEATURES",
        help="Override the features spec.",
    ),
]
WorkflowProviderOption = Annotated[
    str | None,
    workflow_selection_option(
        "--provider",
        metavar="PROVIDER",
        help="Override the RPC provider spec.",
    ),
]
WorkflowObjectiveOption = Annotated[
    str | None,
    workflow_selection_option(
        "--objective",
        metavar="OBJECTIVE",
        help="Override the objective spec.",
    ),
]
WorkflowEvaluationOverrideOption = Annotated[
    str | None,
    workflow_selection_option(
        "--evaluation",
        metavar="EVALUATION",
        help="Override the evaluation spec.",
    ),
]
WorkflowEvaluationSpecOption = Annotated[
    str | None,
    workflow_selection_option(
        "--evaluation",
        metavar="EVALUATION",
        help="Use this evaluator spec.",
    ),
]
WorkflowModelOption = Annotated[
    str | None,
    workflow_selection_option("--model", metavar="MODEL", help="Override the model spec."),
]
WorkflowTuningSpaceOption = Annotated[
    str | None,
    workflow_selection_option(
        "--tuning-space",
        metavar="TUNING_SPACE",
        help="Override the tuning-space spec.",
    ),
]
WorkflowTrainingOption = Annotated[
    str | None,
    workflow_selection_option(
        "--training",
        metavar="TRAINING",
        help="Override the training spec.",
    ),
]
WorkflowSplitOption = Annotated[
    str | None,
    workflow_selection_option("--split", metavar="SPLIT", help="Override the split spec."),
]
WorkflowTuningOption = Annotated[
    str | None,
    workflow_selection_option("--tuning", metavar="TUNING", help="Override the tuning spec."),
]
WorkflowStudyOption = Annotated[
    str | None,
    workflow_selection_option("--study", metavar="STUDY", help="Override the study name."),
]
WorkflowVariantOption = Annotated[
    str | None,
    workflow_selection_option(
        "--variant",
        metavar="VARIANT",
        help="Override the artifact variant.",
    ),
]
WorkflowDatasetConsumerOption = Annotated[
    str | None,
    workflow_selection_option(
        "--dataset-id",
        metavar="DATASET_ID",
        help="Consume this corpus root.",
    ),
]
WorkflowEvaluationDatasetOption = Annotated[
    str | None,
    workflow_selection_option(
        "--dataset-id",
        metavar="DATASET_ID",
        help="Evaluate on this corpus root.",
    ),
]
WorkflowStudyConsumerOption = Annotated[
    str | None,
    workflow_selection_option("--study-id", metavar="STUDY_ID", help="Consume this study root."),
]
WorkflowArtifactConsumerOption = Annotated[
    str | None,
    workflow_selection_option(
        "--artifact-id",
        metavar="ARTIFACT_ID",
        help="Consume this artifact root.",
    ),
]
WorkflowStorageRootWriteOption = Annotated[
    Path | None,
    workflow_output_option(
        "--storage-root",
        metavar="PATH",
        help="Store outputs under a non-default root.",
    ),
]
WorkflowDryRunOption = Annotated[
    bool | None,
    typer.Option(
        "--dry-run/--no-dry-run",
        help="Skip persistence and RPC side effects.",
        rich_help_panel="Execution",
    ),
]
WorkflowDependencyOption = Annotated[
    str | None,
    workflow_execution_option(
        "--dependency",
        metavar="DEPENDENCY",
        help="Pass one Slurm dependency spec such as afterok:12345.",
    ),
]
WorkflowDetachOption = Annotated[
    bool,
    typer.Option(
        "--detach",
        help="Submit and exit without following the job.",
        rich_help_panel="Execution",
    ),
]
WorkflowTrialCountOption = Annotated[
    int | None,
    workflow_execution_option(
        "--trial-count",
        metavar="COUNT",
        help="Override the requested trial count.",
    ),
]
WorkflowDelaySecondsOption = Annotated[
    int | None,
    workflow_execution_option(
        "--delay-seconds",
        metavar="SECONDS",
        help="Override the evaluation delay in seconds.",
    ),
]
WorkflowBatchSizeOption = Annotated[
    int | None,
    workflow_execution_option(
        "--batch-size",
        metavar="COUNT",
        help="Override the evaluation batch size.",
    ),
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
