"""Tuning workflow."""

from __future__ import annotations

from ..config.models import TuneConfig
from ..core.reporting import Reporter
from ..modeling.tuning_execution import (
    TuningExecutionCallbacks,
    open_tuning_execution,
    run_tuning_execution,
)
from ..storage.transactions import record_study_root_mutation
from .preparation import prepare_tune
from .reporting import (
    report_tune_best,
    report_tune_result,
    report_tune_resume,
    report_tune_study_start,
    report_tune_trial,
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
            callbacks=TuningExecutionCallbacks(
                on_resume=lambda existing, target: report_tune_resume(
                    active_reporter,
                    existing=existing,
                    target=target,
                ),
                on_study_start=lambda remaining: report_tune_study_start(
                    active_reporter,
                    remaining=remaining,
                ),
                on_trial_complete=lambda progress: report_tune_trial(
                    active_reporter,
                    progress,
                ),
                on_best_improved=lambda progress: report_tune_best(
                    active_reporter,
                    progress,
                ),
            ),
        ),
    )
    summary = summary_commit.result
    report_tune_result(active_reporter, summary=summary)
