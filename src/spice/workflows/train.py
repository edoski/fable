"""Training workflow."""

from __future__ import annotations

from ..config.models import TrainConfig
from ..core.reporting import Reporter
from ..modeling.persisted_training import run_persisted_training
from ..storage.transactions import commit_artifact_root
from .preparation import prepare_train
from .reporting import (
    report_train_early_stop,
    report_train_epoch,
    report_train_fit_start,
    report_train_prepare_complete,
    report_train_result,
    train_workflow_facts,
)


def run(config: TrainConfig, *, reporter: Reporter | None = None) -> None:
    active_reporter = reporter or Reporter()
    prepared = prepare_train(config)
    roots = prepared.roots
    spec = prepared.spec
    active_reporter.header("train", train_workflow_facts(prepared.active_config, roots))
    artifact_dir = roots.artifact.root_path
    history_block_path = roots.corpus.history_dir
    committed = commit_artifact_root(
        roots.artifact,
        writer=lambda staged_root: run_persisted_training(
            history_block_path,
            spec=spec,
            artifact_dir=staged_root,
            on_prepare_complete=lambda prepared: report_train_prepare_complete(
                active_reporter,
                n_rows_used=prepared.n_rows_used,
                sample_count=prepared.sample_count,
            ),
            on_fit_start=lambda: report_train_fit_start(
                active_reporter,
                max_epochs=spec.training.max_epochs,
            ),
            on_epoch_end=lambda progress: report_train_epoch(
                active_reporter,
                progress,
                primary_metric_id=spec.prediction_contract.primary_metric_id,
            ),
            on_early_stop=lambda epoch, best_epoch: report_train_early_stop(
                active_reporter,
                epoch=epoch,
                best_epoch=best_epoch,
            ),
        )
    )
    persisted = committed.result
    report_train_result(
        active_reporter,
        summary=persisted.summary,
        artifact_dir=artifact_dir,
    )
