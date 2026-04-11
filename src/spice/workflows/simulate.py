"""Hydra entrypoint for simulation runs."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from ..core.config import ExperimentConfig, coerce_config
from ..core.console import Reporter
from ..core.tracking import log_artifacts
from ..data.datasets import derive_dataset_geometry
from ..data.io import load_block_frame
from ..modeling.artifacts import load_training_artifact
from ..modeling.inference import predict_class_offsets
from ..modeling.pipeline import prepare_inference_dataset
from ..modeling.reporting import build_simulation_report, write_json_report
from ..modeling.simulation import run_temporal_simulation
from ._shared import managed_workflow


def run(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    artifact_dir = config.paths.artifact_root
    history_block_path = config.paths.history_dir
    evaluation_block_path = config.paths.evaluation_dir
    with managed_workflow(
        config,
        run_name=(
            f"simulate-{config.chain.name.value}-{config.model.family.value}"
            f"-{config.dataset.temporal.max_delay_seconds}s"
        ),
        reporter=reporter,
    ) as session:
        load_task = session.reporter.start_task("load inference inputs")
        loaded_artifact = load_training_artifact(artifact_dir)
        history_blocks = load_block_frame(history_block_path)
        evaluation_blocks = load_block_frame(evaluation_block_path)
        session.reporter.finish_task(
            load_task,
            message=f"artifact={artifact_dir} evaluation={evaluation_block_path}",
        )
        prepare_task = session.reporter.start_task("prepare inference dataset")
        prepared = prepare_inference_dataset(
            history_blocks,
            evaluation_blocks,
            geometry=derive_dataset_geometry(
                lookback_seconds=loaded_artifact.manifest.lookback_seconds,
                max_delay_seconds=loaded_artifact.manifest.max_delay_seconds,
                block_time_seconds=loaded_artifact.manifest.chain.block_time_seconds,
            ),
            scaler=loaded_artifact.manifest.scaler,
            window_start_timestamp=config.dataset.window.start_timestamp,
            window_end_timestamp=config.dataset.window.end_timestamp,
        )
        session.reporter.finish_task(
            prepare_task,
            message=f"examples={prepared.n_examples_total}",
        )
        predictions = predict_class_offsets(
            loaded_artifact.model,
            store=prepared.store,
            sample_indices=prepared.sample_indices,
            lookback_steps=prepared.geometry.lookback_steps,
            batch_size=config.training.batch_size,
            device=config.training.device,
            reporter=session.reporter,
        )
        simulation = run_temporal_simulation(
            prepared.store,
            predictions,
            sample_indices=prepared.sample_indices,
            window_seconds=config.simulation.window_seconds,
            arrival_rate_per_second=config.simulation.arrival_rate_per_second,
            repetitions=config.simulation.repetitions,
            seed=config.simulation.seed,
            reporter=session.reporter,
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
        report_path = config.paths.simulation_report_path
        report_task = session.reporter.start_task("write simulation report")
        write_json_report(report_path, report)
        session.reporter.finish_task(report_task, message=str(report_path))
        session.reporter.log(f"simulation finished: {report_path}")

        if session.tracking_enabled:
            import mlflow

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


@hydra.main(version_base=None, config_path="../conf", config_name="simulate")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="simulate"))


if __name__ == "__main__":
    main()
