"""Shared CLI utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

DEFAULT_REMOTE_TARGET = "disi_l40"
DEFAULT_STORAGE_ROOT = Path("outputs")
DEFAULT_BENCHMARK_RUNS_ROOT = DEFAULT_STORAGE_ROOT / "benchmarks" / "runs"
DEFAULT_BENCHMARK_INDEX_PATH = Path("benchmarks") / "results.sqlite"


def _workflow_option(
    *param_decls: str,
    metavar: str,
    help: str,
    panel: str,
) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel=panel)


def workflow_selection_option(*param_decls: str, metavar: str, help: str) -> object:
    return _workflow_option(*param_decls, metavar=metavar, help=help, panel="Selection")


def workflow_execution_option(*param_decls: str, metavar: str, help: str) -> object:
    return _workflow_option(*param_decls, metavar=metavar, help=help, panel="Execution")


def workflow_output_option(*param_decls: str, metavar: str, help: str) -> object:
    return _workflow_option(*param_decls, metavar=metavar, help=help, panel="Outputs")


ChainFilterOption = Annotated[
    str | None,
    typer.Option("--chain", metavar="CHAIN", help="Filter by chain."),
]
CorpusFilterOption = Annotated[
    str | None,
    typer.Option("--corpus", metavar="CORPUS", help="Filter by corpus name."),
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
WorkflowEvaluatorSpecOption = Annotated[
    str | None,
    workflow_selection_option(
        "--evaluator",
        metavar="EVALUATOR",
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
WorkflowCorpusConsumerOption = Annotated[
    str | None,
    workflow_selection_option(
        "--corpus-id",
        metavar="CORPUS_ID",
        help="Consume this corpus root.",
    ),
]
WorkflowEvaluationCorpusOption = Annotated[
    str | None,
    workflow_selection_option(
        "--corpus-id",
        metavar="CORPUS_ID",
        help="Evaluate on this corpus root.",
    ),
]
WorkflowEvaluationWindowStartOption = Annotated[
    str | None,
    workflow_selection_option(
        "--evaluation-start",
        metavar="UTC_TIMESTAMP",
        help="Start of the scored evaluation window.",
    ),
]
WorkflowEvaluationWindowDurationOption = Annotated[
    int | None,
    workflow_selection_option(
        "--evaluation-duration-seconds",
        metavar="SECONDS",
        help="Duration of the scored evaluation window.",
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
    return storage_root or DEFAULT_STORAGE_ROOT
