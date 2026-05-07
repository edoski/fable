"""Tuning workflow."""

from __future__ import annotations

from ..config.models import TuneConfig
from ..core.reporting import Reporter
from ..modeling.tuning_execution import open_tuning_execution, run_tuning_execution
from ..storage.transactions import record_study_root_mutation
from .preparation import prepare_tune
from .reporting import (
    report_tune_result,
    tune_reporting_callbacks,
    tune_workflow_facts,
)


def run(config: TuneConfig, *, reporter: Reporter | None = None) -> None:
    active_reporter = reporter or Reporter()
    prepared = prepare_tune(config)
    roots = prepared.roots
    corpus_manifest = prepared.corpus_manifest
    active_reporter.header("tune", tune_workflow_facts(config, roots))

    opened_commit = record_study_root_mutation(
        roots.study,
        mutation=lambda: open_tuning_execution(
            config,
            roots=roots,
            corpus_manifest=corpus_manifest,
        ),
    )
    opened = opened_commit.result
    summary_commit = record_study_root_mutation(
        roots.study,
        mutation=lambda: run_tuning_execution(
            opened,
            config=config,
            roots=roots,
            corpus_manifest=corpus_manifest,
            callbacks=tune_reporting_callbacks(active_reporter),
        ),
    )
    summary = summary_commit.result
    report_tune_result(active_reporter, summary=summary)
