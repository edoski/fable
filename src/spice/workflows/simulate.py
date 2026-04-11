"""Hydra entrypoint for simulation runs."""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

from ..core.config import ExperimentConfig, coerce_config
from ..core.console import Reporter, RichReporter
from ..core.tracking import configure_mlflow, log_artifacts, log_config
from ..data.datasets import derive_dataset_geometry
from ..data.io import load_enriched_block_frame
from ..modeling.artifacts import load_training_artifact
from ..modeling.inference import predict_class_offsets
from ..modeling.pipeline import prepare_inference_dataset
from ..modeling.reporting import build_simulation_report, write_json_report
from ..modeling.simulation import run_temporal_simulation
from ._shared import start_run_if_enabled


def run(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    artifact_dir = Path(config.paths.artifact_root)
    history_block_path = Path(config.paths.enriched_history_dir)
    evaluation_block_path = Path(config.paths.enriched_evaluation_dir)
    if config.tracking.enabled:
        configure_mlflow(config)

    active_reporter = reporter or RichReporter()
    run_context = start_run_if_enabled(
        config,
        run_name=(
            f"simulate-{config.chain.name.value}-{config.model.family.value}"
            f"-{config.max_delay_seconds}s"
        ),
    )
    try:
        if run_context is not None:
            run_context.__enter__()
            log_config(config)
            mlflow.set_tags(config.tracking.tags)

        loaded_artifact = load_training_artifact(artifact_dir)
        history_blocks = load_enriched_block_frame(history_block_path)
        evaluation_blocks = load_enriched_block_frame(evaluation_block_path)
        prepared = prepare_inference_dataset(
            history_blocks,
            evaluation_blocks,
            geometry=derive_dataset_geometry(
                lookback_seconds=loaded_artifact.manifest.lookback_seconds,
                max_delay_seconds=loaded_artifact.manifest.max_delay_seconds,
                block_time_seconds=loaded_artifact.manifest.chain.block_time_seconds,
            ),
            scaler=loaded_artifact.manifest.scaler,
            evaluation_start_timestamp=config.dataset.evaluation_start_timestamp,
            evaluation_end_timestamp=config.dataset.evaluation_end_timestamp,
        )
        predictions = predict_class_offsets(
            loaded_artifact.model,
            store=prepared.store,
            sample_indices=prepared.sample_indices,
            lookback_steps=prepared.geometry.lookback_steps,
            effective_batch_size=config.training.effective_batch_size,
            device=config.training.device,
        )
        simulation = run_temporal_simulation(
            prepared.store,
            predictions,
            sample_indices=prepared.sample_indices,
            window_seconds=config.simulation.window_seconds,
            arrival_rate_per_second=config.simulation.arrival_rate_per_second,
            repetitions=config.simulation.repetitions,
            seed=config.simulation.seed,
        )
        report = build_simulation_report(
            loaded_artifact,
            artifact_dir=artifact_dir,
            history_block_path=history_block_path,
            evaluation_block_path=evaluation_block_path,
            prepared=prepared,
            simulation=simulation,
            window_seconds=config.simulation.window_seconds,
            arrival_rate_per_second=config.simulation.arrival_rate_per_second,
            repetitions=config.simulation.repetitions,
        )
        report_path = Path(config.paths.simulation_report_path)
        if report_path.exists():
            report_path.unlink()
        write_json_report(report_path, report)
        active_reporter.log(f"simulation finished: {report_path}")

        if config.tracking.enabled:
            mlflow.log_metrics(
                {
                    "simulation_profit_over_baseline_mean": report.profit_over_baseline.mean,
                    "simulation_profit_over_baseline_std": report.profit_over_baseline.std,
                    "simulation_cost_over_optimum_mean": report.cost_over_optimum.mean,
                    "simulation_cost_over_optimum_std": report.cost_over_optimum.std,
                    "simulation_baseline_cost_over_optimum_mean": (
                        report.baseline_cost_over_optimum.mean
                    ),
                    "simulation_total_events": float(report.total_events),
                }
            )
            log_artifacts([report_path])
    finally:
        if run_context is not None:
            run_context.__exit__(None, None, None)
        if reporter is None:
            active_reporter.close()


@hydra.main(version_base=None, config_path="../conf", config_name="simulate")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="simulate"))


if __name__ == "__main__":
    main()
