"""Simulation workflow."""

from __future__ import annotations

from ..config import SimulateConfig
from ..core.console import Reporter
from ..core.files import remove_path
from ..data.io import load_block_frame
from ..features import feature_warmup_blocks
from ..modeling.artifacts import load_training_artifact, validate_artifact_feature_graph
from ..modeling.inference import predict_class_offsets
from ..modeling.pipeline import prepare_inference_dataset
from ..modeling.reporting import build_simulation_report, write_json_report
from ..modeling.simulation import run_temporal_simulation
from ..planning.geometry import derive_dataset_geometry, minimum_history_context_blocks
from ._shared import abort_cleanup, managed_workflow


def _workflow_facts(config: SimulateConfig) -> list[tuple[str, str]]:
    facts = [
        ("dataset", config.dataset.id),
        ("chain", config.chain.name),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
    ]
    if config.artifact.variant.value == "tuned":
        facts.append(("study", config.study.id))
    return facts


def run(config: SimulateConfig, *, reporter: Reporter | None = None) -> None:
    artifact_dir = config.paths.artifact_root
    history_block_path = config.paths.history_dir
    evaluation_block_path = config.paths.evaluation_dir
    report_path = config.paths.simulation_report_path
    if artifact_dir is None or report_path is None:
        raise ValueError("simulation workflow requires artifact output paths")
    with managed_workflow(
        config,
        run_name=(
            f"simulate-{config.chain.name}-{config.model.id}"
            f"-{config.dataset.temporal.max_delay_seconds}s"
        ),
        reporter=reporter,
    ) as session:
        session.runtime.configure_workflow("simulate", _workflow_facts(config))
        load_reporter = session.runtime.stage_reporter("load", label="load")
        prepare_reporter = session.runtime.stage_reporter("prepare", label="prepare")
        predict_reporter = session.runtime.stage_reporter("predict", label="predict")
        simulation_reporter = session.runtime.stage_reporter("simulate", label="simulate")
        write_reporter = session.runtime.stage_reporter(
            "write",
            label="write",
            running_status="writing",
        )
        with abort_cleanup(
            session.reporter,
            label="simulate",
            cleanup=lambda: remove_path(report_path),
        ):
            load_task = load_reporter.start_task("load inference inputs")
            loaded_artifact = load_training_artifact(artifact_dir)
            selection = validate_artifact_feature_graph(
                loaded_artifact.manifest,
                requested_feature_set_id=config.feature_set.id,
            )
            history_blocks = load_block_frame(history_block_path)
            evaluation_blocks = load_block_frame(evaluation_block_path)
            load_reporter.finish_task(
                load_task,
                message=f"artifact={artifact_dir} evaluation={evaluation_block_path}",
            )
            prepare_task = prepare_reporter.start_task("prepare inference dataset")
            minimum_context = minimum_history_context_blocks(
                lookback_seconds=loaded_artifact.manifest.lookback_seconds,
                block_time_seconds=loaded_artifact.manifest.chain.runtime.block_time_seconds,
                feature_warmup_blocks=feature_warmup_blocks(selection.feature_names),
            )
            if minimum_context > config.dataset.history_context_blocks:
                raise ValueError(
                    "Configured dataset.history_context_blocks "
                    "is too small for the selected feature set: "
                    f"need at least {minimum_context}, got {config.dataset.history_context_blocks}"
                )
            prepared = prepare_inference_dataset(
                history_blocks,
                evaluation_blocks,
                selection=selection,
                geometry=derive_dataset_geometry(
                    lookback_seconds=loaded_artifact.manifest.lookback_seconds,
                    max_delay_seconds=loaded_artifact.manifest.max_delay_seconds,
                    block_time_seconds=loaded_artifact.manifest.chain.runtime.block_time_seconds,
                    history_context_blocks=minimum_context,
                ),
                scaler=loaded_artifact.manifest.scaler,
                window_start_timestamp=config.evaluation_window_start_timestamp,
                window_end_timestamp=config.evaluation_window_end_timestamp,
            )
            prepare_reporter.finish_task(
                prepare_task,
                message=f"samples={prepared.sample_count}",
            )
            predictions = predict_class_offsets(
                loaded_artifact.model,
                store=prepared.store,
                sample_indices=prepared.sample_indices,
                lookback_steps=prepared.geometry.lookback_steps,
                batch_size=config.training.batch_size,
                device=config.training.device,
                reporter=predict_reporter,
            )
            simulation = run_temporal_simulation(
                prepared.store,
                predictions,
                sample_indices=prepared.sample_indices,
                window_seconds=config.simulation.window_seconds,
                arrival_rate_per_second=config.simulation.arrival_rate_per_second,
                repetitions=config.simulation.repetitions,
                seed=config.simulation.seed,
                reporter=simulation_reporter,
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
            report_task = write_reporter.start_task("write simulation report")
            write_json_report(report_path, report)
            write_reporter.finish_task(report_task, message=str(report_path), silent=True)
        session.runtime.log_sectioned_summary(
            "simulation summary",
            [
                (
                    "dataset",
                    [
                        ("id", report.dataset_id),
                        ("chain", report.chain),
                        ("model", report.model_id),
                        ("delay", f"{report.max_delay_seconds}s"),
                    ],
                ),
                (
                    "provenance",
                    [
                        ("variant", report.variant.value),
                        *([] if report.study is None else [("study", report.study.id)]),
                        ("report", str(report_path)),
                    ],
                ),
                (
                    "simulation",
                    [
                        ("window", f"{report.simulation_window_seconds}s"),
                        ("repetitions", str(report.repetitions)),
                        ("events", f"{report.total_events:,}"),
                    ],
                ),
                (
                    "results",
                    [
                        (
                            "profit over baseline",
                            (
                                f"{report.profit_over_baseline.mean:.4f} +/- "
                                f"{report.profit_over_baseline.std:.4f}"
                            ),
                        ),
                        (
                            "cost over optimum",
                            (
                                f"{report.cost_over_optimum.mean:.4f} +/- "
                                f"{report.cost_over_optimum.std:.4f}"
                            ),
                        ),
                    ],
                ),
            ],
        )
