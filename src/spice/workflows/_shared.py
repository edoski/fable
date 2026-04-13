"""Shared workflow helpers."""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ..config import ArtifactVariant, SimulateConfig, TrainConfig, TuneConfig, WorkflowTask
from ..core.console import ConsoleRuntime, Reporter, create_console_runtime
from ..modeling.evaluation import EpochMetrics
from ..modeling.pipeline import TrainingSpec
from ..modeling.tuning import apply_tuned_parameters
from ..planning.contracts import resolve_task_contract
from ..state.study import load_best_params, load_study_manifest, validate_tuned_train_request


def build_training_spec(config: TrainConfig | TuneConfig) -> TrainingSpec:
    variant = selected_artifact_variant(config)
    contract = resolve_task_contract(
        task=config.task,
        feature_set=config.feature_set,
    )
    return TrainingSpec(
        chain=config.chain,
        dataset_id=config.paths.corpus_id,
        dataset_name=config.dataset.name,
        artifact_id=(
            config.paths.artifact_id
            if config.paths.artifact_id is not None
            else config.paths.study_id or "trial"
        ),
        task=config.task,
        contract=contract,
        feature_set=config.feature_set,
        model=config.model,
        variant=variant,
        study=config.study if variant is ArtifactVariant.TUNED else None,
        study_id=config.paths.study_id if variant is ArtifactVariant.TUNED else None,
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


@dataclass(slots=True)
class _InterruptState:
    interrupted: bool = False


def selected_artifact_variant(config: TrainConfig | TuneConfig | SimulateConfig) -> ArtifactVariant:
    if config.workflow is WorkflowTask.TUNE:
        return ArtifactVariant.TUNED
    return config.artifact.variant


@contextmanager
def _capture_sigint() -> Iterator[_InterruptState]:
    state = _InterruptState()
    if not hasattr(signal, "SIGINT"):
        yield state
        return
    previous_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame) -> None:
        state.interrupted = True
        if previous_handler == signal.SIG_IGN:
            return
        if previous_handler == signal.SIG_DFL:
            signal.default_int_handler(signum, frame)
            return
        if callable(previous_handler):
            previous_handler(signum, frame)

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except ValueError:
        yield state
        return
    try:
        yield state
    finally:
        try:
            signal.signal(signal.SIGINT, previous_handler)
        except ValueError:
            pass


def _cleanup_after_interrupt(
    reporter: Reporter,
    *,
    label: str,
    cleanup: Callable[[], None],
) -> None:
    cleanup()
    reporter.close()
    reporter.log(f"{label} cancelled; partial outputs removed", level="warning")


@contextmanager
def abort_cleanup(
    reporter: Reporter,
    *,
    label: str,
    cleanup: Callable[[], None],
) -> Iterator[None]:
    with _capture_sigint() as interrupt_state:
        try:
            yield
        except BaseException as exc:
            if interrupt_state.interrupted or isinstance(exc, KeyboardInterrupt):
                _cleanup_after_interrupt(reporter, label=label, cleanup=cleanup)
            raise
        if interrupt_state.interrupted:
            _cleanup_after_interrupt(reporter, label=label, cleanup=cleanup)
            raise KeyboardInterrupt


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
    manifest = load_study_manifest(path)
    validate_tuned_train_request(config, manifest=manifest)
    try:
        params = load_best_params(path, study_name=config.study.name)
    except OSError as exc:
        raise FileNotFoundError(
            f"Best tuning params are required but missing: {path}"
        ) from exc
    return apply_tuned_parameters(config, params)
