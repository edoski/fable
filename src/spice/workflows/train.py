"""Training workflow."""

from __future__ import annotations

from ..config.models import TrainConfig
from ..core.reporting import Reporter
from ..modeling.persisted_training import run_persisted_training
from ..storage.transactions import commit_artifact_root
from .preparation import prepare_train
from .reporting import (
    report_train_result,
    train_reporting_callbacks,
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
            callbacks=train_reporting_callbacks(active_reporter, spec=spec),
        )
    )
    persisted = committed.result
    report_train_result(
        active_reporter,
        summary=persisted.summary,
        artifact_dir=artifact_dir,
    )
