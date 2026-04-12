"""Shared workflow helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ..config import ArtifactVariant, SimulateConfig, TrainConfig, TuneConfig, WorkflowTask
from ..core.console import ConsoleRuntime, Reporter, create_console_runtime
from ..modeling.evaluation import EpochMetrics
from ..modeling.pipeline import TrainingSpec
from ..planning.contracts import resolve_task_contract
from ..state.study import load_best_params
from ._tuning import apply_tuned_parameters


def build_training_spec(config: TrainConfig | TuneConfig) -> TrainingSpec:
    variant = selected_artifact_variant(config)
    contract = resolve_task_contract(
        chain=config.chain,
        task=config.task,
        feature_set=config.feature_set,
    )
    return TrainingSpec(
        chain=config.chain,
        dataset_id=config.dataset.id,
        task=config.task,
        contract=contract,
        feature_set=config.feature_set,
        model=config.model,
        variant=variant,
        study=config.study if variant is ArtifactVariant.TUNED else None,
        split=config.split,
        training=config.training,
    )


def epoch_metrics_to_dict(metrics: EpochMetrics) -> dict[str, float]:
    return {
        "loss": metrics.total_loss,
        "accuracy": metrics.accuracy,
        "cost_over_optimum": metrics.mean_cost_over_optimum,
        "profit_over_baseline": metrics.mean_profit_over_baseline,
    }


@dataclass(slots=True)
class WorkflowSession:
    runtime: ConsoleRuntime
    reporter: Reporter


def selected_artifact_variant(config: TrainConfig | TuneConfig | SimulateConfig) -> ArtifactVariant:
    if config.workflow is WorkflowTask.TUNE:
        return ArtifactVariant.TUNED
    return config.artifact.variant


@contextmanager
def abort_cleanup(
    reporter: Reporter,
    *,
    label: str,
    cleanup: Callable[[], None],
) -> Iterator[None]:
    try:
        yield
    except KeyboardInterrupt:
        cleanup()
        reporter.close()
        reporter.log(f"{label} interrupted; partial outputs removed", level="warning")
        raise


@contextmanager
def managed_workflow(
    config: object,
    *,
    run_name: str,
    runtime: ConsoleRuntime | None = None,
    reporter: Reporter | None = None,
    default_runtime_factory: Callable[..., ConsoleRuntime] = create_console_runtime,
    nested: bool = False,
) -> Iterator[WorkflowSession]:
    del config, run_name, nested
    active_runtime = runtime or default_runtime_factory(reporter=reporter)
    owns_runtime = runtime is None
    try:
        with active_runtime.activate():
            yield WorkflowSession(
                runtime=active_runtime,
                reporter=active_runtime.reporter,
            )
    finally:
        if owns_runtime:
            active_runtime.close()


def apply_study_best_params(config: TrainConfig) -> TrainConfig:
    path = config.paths.study_state_db
    if path is None:
        raise ValueError("study_state_db is required for tuned artifacts")
    try:
        params = load_best_params(
            path,
            study_name=config.study.id,
            model_id=config.model.id,
        )
    except OSError as exc:
        raise FileNotFoundError(
            f"Best tuning params are required but missing: {path}"
        ) from exc
    return apply_tuned_parameters(config, params)
