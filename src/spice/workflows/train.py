"""Training workflow."""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, cast
from uuid import uuid4

from ..config.models import ArtifactVariant, TrainConfig
from ..core.errors import ConfigResolutionError
from ..core.files import promote_paths_atomic, prune_empty_directories, remove_path
from ..core.rendering import metric_string
from ..core.reporting import Reporter
from ..modeling.persisted_training import run_persisted_training
from ..modeling.pipeline import build_training_spec
from ..modeling.summary import training_result_fields
from ..modeling.training import TrainingEpochProgress
from ..modeling.tuning import apply_study_best_params
from ..storage.engine import ARTIFACT_ROOT_KIND
from ..storage.layout import resolve_workflow_paths
from ..storage.roots import reindex_root

SignalHandler = Callable[[int, FrameType | None], Any] | int | None


def _build_staged_artifact_root(artifact_root: Path) -> Path:
    return artifact_root.parent / f".{artifact_root.name}.staging.{uuid4().hex}"


def _cleanup_staged_artifact_root(staged_root: Path, *, prune_stop_at: Path) -> None:
    remove_path(staged_root)
    prune_empty_directories(staged_root.parent, stop_at=prune_stop_at)


@contextmanager
def _abort_cleanup(
    reporter: Reporter,
    *,
    label: str,
    cleanup: Callable[[], None],
) -> Iterator[None]:
    interrupted = False
    signal_ids = [
        getattr(signal, name)
        for name in ("SIGINT", "SIGTERM")
        if hasattr(signal, name)
    ]
    previous_handlers: dict[int, SignalHandler] = {}

    def _run_cleanup() -> None:
        cleanup()
        reporter.milestone(f"{label} cancelled; partial outputs removed", level="warning")

    def _handle_interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True
        previous_handler = previous_handlers.get(signum, signal.SIG_DFL)
        if previous_handler == signal.SIG_IGN:
            return
        if previous_handler == signal.SIG_DFL:
            raise KeyboardInterrupt
        if callable(previous_handler):
            previous_handler(signum, frame)
            return
        raise KeyboardInterrupt

    registered: list[int] = []
    try:
        for signum in signal_ids:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_interrupt)
            registered.append(signum)
    except ValueError:
        registered = []
    try:
        yield
    except BaseException as exc:
        if interrupted or isinstance(exc, KeyboardInterrupt):
            _run_cleanup()
        raise
    finally:
        for signum in registered:
            try:
                signal.signal(signum, cast(signal.Handlers, previous_handlers[signum]))
            except ValueError:
                pass
    if interrupted:
        _run_cleanup()
        raise KeyboardInterrupt


def _workflow_facts(config: TrainConfig) -> list[tuple[str, str]]:
    facts = [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("problem", config.problem.id),
        ("prediction", config.prediction.id),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
    ]
    if config.artifact.variant is ArtifactVariant.TUNED:
        facts.append(("study", config.study.name))
    return facts


def _fit_epoch_message(
    progress: TrainingEpochProgress,
    *,
    primary_metric_id: str,
) -> str:
    fields = [f"epoch={progress.epoch}/{progress.max_epochs}"]
    if progress.objective_metric_id in progress.objective_metrics.values:
        fields.append(
            f"objective.{progress.objective_metric_id}="
            f"{metric_string(progress.objective_metrics.values[progress.objective_metric_id])}"
        )
    if primary_metric_id in progress.validation_metrics.values:
        fields.append(
            f"validation.{primary_metric_id}="
            f"{metric_string(progress.validation_metrics.values[primary_metric_id])}"
        )
    fields.append(f"best_epoch={progress.best_epoch}")
    fields.append(
        f"best.{progress.objective_metric_id}={metric_string(progress.best_objective_value)}"
    )
    return "fit " + " ".join(fields)


def run(config: TrainConfig, *, reporter: Reporter | None = None) -> None:
    active_reporter = reporter or Reporter()
    active_config = config
    study_id: str | None = None
    if config.artifact.variant is ArtifactVariant.TUNED:
        applied = apply_study_best_params(config)
        active_config = applied.config
        study_id = applied.study_id
    paths = resolve_workflow_paths(active_config, study_id=study_id)
    active_reporter.header("train", _workflow_facts(active_config))
    spec = build_training_spec(active_config)
    artifact_dir = paths.artifact_root
    history_block_path = paths.history_dir
    if artifact_dir is None:
        raise ConfigResolutionError("training workflow requires artifact output paths")
    staged_artifact_root = _build_staged_artifact_root(artifact_dir)
    prune_stop_at = artifact_dir.parent.parent
    with _abort_cleanup(
        active_reporter,
        label="train",
        cleanup=lambda: _cleanup_staged_artifact_root(
            staged_artifact_root,
            prune_stop_at=prune_stop_at,
        ),
    ):
        _cleanup_staged_artifact_root(
            staged_artifact_root,
            prune_stop_at=prune_stop_at,
        )
        persisted = run_persisted_training(
            history_block_path,
            spec=spec,
            artifact_dir=staged_artifact_root,
            state_root_kind=ARTIFACT_ROOT_KIND,
            on_prepare_complete=lambda prepared: active_reporter.milestone(
                f"prepare rows={prepared.n_rows_used} samples={prepared.sample_count}"
            ),
            on_fit_start=lambda: active_reporter.milestone(
                f"fit started epochs={spec.training.max_epochs}"
            ),
            on_epoch_end=lambda progress: active_reporter.milestone(
                _fit_epoch_message(
                    progress,
                    primary_metric_id=spec.prediction_contract.primary_metric_id,
                )
            ),
            on_early_stop=lambda epoch, best_epoch: active_reporter.milestone(
                f"fit early_stop epoch={epoch} best_epoch={best_epoch}"
            ),
        )
        artifact_root = paths.artifact_root
        artifact_state_db = paths.artifact_state_db
        artifact_id = paths.artifact_id
        if artifact_root is None or artifact_state_db is None or artifact_id is None:
            raise ConfigResolutionError("training workflow requires artifact output paths")
        promote_paths_atomic([(artifact_root, staged_artifact_root)])
        reindex_root(paths.output_root, root_path=artifact_root)
    active_reporter.result(
        "train",
        training_result_fields(
            persisted.summary,
            artifact_dir=artifact_root,
        ),
    )
