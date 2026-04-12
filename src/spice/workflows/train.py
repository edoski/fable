"""Training workflow."""

from __future__ import annotations

from ..core.config import ArtifactVariant, ExperimentConfig
from ..core.console import Reporter
from ..core.constants import ARTIFACT_MANIFEST_FILENAME, MODEL_STATE_FILENAME
from ..core.files import remove_path
from ..core.tracking import log_artifacts, log_epoch_history
from ..modeling.execution import run_persisted_training
from ._cli import load_cli_config
from ._shared import (
    abort_cleanup,
    apply_study_best_params,
    build_training_spec,
    epoch_metrics_to_dict,
    managed_workflow,
)


def _chain_label(chain_name: str) -> str:
    return chain_name.replace("_", " ").title()


def _format_train_summary_sections(
    config: ExperimentConfig,
    persisted,
) -> list[tuple[str, list[tuple[str, str]]]]:
    report = persisted.report
    result = persisted.training_run.training_result
    best_validation = persisted.best_validation_metrics
    return [
        (
            "dataset",
            [
                ("id", report.dataset_id),
                ("chain", _chain_label(report.chain)),
                ("family", report.family),
                ("delay", f"{report.max_delay_seconds}s"),
            ],
        ),
        (
            "provenance",
            [
                ("variant", report.variant.value),
                *([] if report.study is None else [("study", report.study.id)]),
                ("artifact", str(report.artifact_dir)),
            ],
        ),
        (
            "runtime",
            [
                ("lookback", f"{report.lookback_seconds}s"),
                ("best epoch", str(report.best_epoch)),
                ("device", result.resolved_device),
                ("precision", result.resolved_precision),
                ("compile", "on" if result.compiled else "off"),
            ],
        ),
        (
            "metrics",
            [
                (
                    "split sizes",
                    (
                        f"train={report.split_sizes.train_samples:,} "
                        f"validation={report.split_sizes.validation_samples:,} "
                        f"test={report.split_sizes.test_samples:,}"
                    ),
                ),
                ("validation loss", f"{best_validation.total_loss:.4f}"),
                ("validation accuracy", f"{best_validation.accuracy:.3f}"),
                (
                    "test profit over baseline",
                    f"{report.test_metrics.mean_profit_over_baseline:.4f}",
                ),
            ],
        ),
    ]


def _clean_training_outputs(config: ExperimentConfig, *, prune_empty_root: bool) -> None:
    for path in (
        config.paths.checkpoint_dir,
        config.paths.artifact_root / ARTIFACT_MANIFEST_FILENAME,
        config.paths.artifact_root / MODEL_STATE_FILENAME,
        config.paths.train_report_path,
        config.paths.simulation_report_path,
    ):
        remove_path(path)
    if prune_empty_root and config.paths.artifact_root.exists():
        try:
            next(config.paths.artifact_root.iterdir())
        except StopIteration:
            config.paths.artifact_root.rmdir()


def run(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    with managed_workflow(
        config,
        run_name=(
            "train-"
            f"{config.chain.name.value}-{config.model.family.value}-"
            f"{config.dataset.temporal.max_delay_seconds}s"
        ),
        reporter=reporter,
    ) as session:
        session.reporter.log(f"variant: {config.artifact.variant.value}")
        active_config = config
        if config.artifact.variant is ArtifactVariant.TUNED:
            active_config = apply_study_best_params(config)
        spec = build_training_spec(active_config)
        artifact_dir = active_config.paths.artifact_root
        report_path = active_config.paths.train_report_path
        history_block_path = active_config.paths.history_dir
        with abort_cleanup(
            session.reporter,
            label="train",
            cleanup=lambda: _clean_training_outputs(active_config, prune_empty_root=True),
        ):
            _clean_training_outputs(active_config, prune_empty_root=True)
            persisted = run_persisted_training(
                history_block_path,
                spec=spec,
                artifact_dir=artifact_dir,
                report_path=report_path,
                reporter=session.reporter,
                runtime=session.runtime,
            )
        session.runtime.log_sectioned_summary(
            "training summary",
            _format_train_summary_sections(active_config, persisted),
        )
        if session.tracking_enabled:
            import mlflow

            mlflow.log_metrics(
                {
                    "test_loss": persisted.report.test_metrics.total_loss,
                    "test_accuracy": persisted.report.test_metrics.accuracy,
                    "test_cost_over_optimum": persisted.report.test_metrics.mean_cost_over_optimum,
                    "test_profit_over_baseline": (
                        persisted.report.test_metrics.mean_profit_over_baseline
                    ),
                    "best_epoch": float(persisted.training_run.training_result.best_epoch),
                }
            )
            log_epoch_history(
                prefix="train",
                metrics_history=[
                    epoch_metrics_to_dict(item)
                    for item in persisted.training_run.training_result.train_history
                ],
            )
            log_epoch_history(
                prefix="validation",
                metrics_history=[
                    epoch_metrics_to_dict(item)
                    for item in persisted.training_run.training_result.validation_history
                ],
            )
            log_artifacts(persisted.artifact_paths)


def main(argv: list[str] | None = None) -> None:
    run(load_cli_config("train", prog="spice-train", argv=argv))


if __name__ == "__main__":
    main()
