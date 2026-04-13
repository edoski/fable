"""Simulation workflow."""

from __future__ import annotations

from ..config import SimulateConfig
from ..core.console import Reporter
from ..data.io import load_block_frame
from ..modeling.artifacts import load_training_artifact, validate_artifact_feature_graph
from ..modeling.inference import predict_class_offsets
from ..modeling.pipeline import prepare_inference_dataset
from ..modeling.reporting import build_simulation_summary_record
from ..modeling.simulation import run_temporal_simulation
from ..planning.contracts import resolve_feature_contract
from ..state import ARTIFACT_ROOT_KIND
from ..state.artifact import write_simulation_state
from ._shared import abort_cleanup, managed_workflow


def _workflow_facts(config: SimulateConfig) -> list[tuple[str, str]]:
    facts = [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("task", config.task.id),
        ("execution", config.execution.id),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
    ]
    if config.artifact.variant.value == "tuned":
        facts.append(("study", config.study.name))
    return facts


def _state_root_kind(config: SimulateConfig) -> str:
    del config
    return ARTIFACT_ROOT_KIND


def run(config: SimulateConfig, *, reporter: Reporter | None = None) -> None:
    artifact_dir = config.paths.artifact_root
    history_block_path = config.paths.history_dir
    evaluation_block_path = config.paths.evaluation_dir
    if artifact_dir is None:
        raise ValueError("simulation workflow requires artifact output paths")
    with managed_workflow(
        config,
        run_name=(
            f"simulate-{config.chain.name}-{config.model.id}"
            f"-{config.task.id}-{config.execution.id}"
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
            cleanup=lambda: None,
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
            if loaded_artifact.manifest.task_id != config.task.id:
                raise ValueError(
                    "Configured task.id does not match the trained artifact: "
                    f"expected {loaded_artifact.manifest.task_id}, got {config.task.id}"
                )
            if (
                loaded_artifact.manifest.max_supported_delay_seconds
                != config.task.max_supported_delay_seconds
            ):
                raise ValueError(
                    "Configured task.max_supported_delay_seconds does not match "
                    "the trained artifact"
                )
            if (
                loaded_artifact.manifest.lookback_seconds != config.task.lookback_seconds
                or loaded_artifact.manifest.sample_count != config.task.sample_count
            ):
                raise ValueError("Configured task does not match the trained artifact contract")
            contract = resolve_feature_contract(
                task=config.task,
                feature_set_id=selection.feature_set_id,
                feature_names=selection.feature_names,
            )
            if (
                config.execution.requested_delay_seconds
                > loaded_artifact.manifest.max_supported_delay_seconds
            ):
                raise ValueError(
                    "execution.requested_delay_seconds exceeds artifact capability: "
                    f"{config.execution.requested_delay_seconds} > "
                    f"{loaded_artifact.manifest.max_supported_delay_seconds}"
                )
            prepared = prepare_inference_dataset(
                history_blocks,
                evaluation_blocks,
                selection=selection,
                window=contract.window_for_delay(config.execution.requested_delay_seconds),
                scaler=loaded_artifact.manifest.scaler,
                max_candidate_slots=loaded_artifact.manifest.max_candidate_slots,
                window_start_timestamp=config.evaluation_window_start_timestamp,
                window_end_timestamp=config.evaluation_window_end_timestamp,
            )
            prepare_reporter.finish_task(
                prepare_task,
                message=f"samples={prepared.sample_count}",
            )
            predictions = predict_class_offsets(
                loaded_artifact.model,
                model_id=loaded_artifact.manifest.model.id,
                store=prepared.store,
                sample_indices=prepared.sample_indices,
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
            summary = build_simulation_summary_record(
                loaded_artifact,
                prepared=prepared,
                simulation=simulation,
                requested_delay_seconds=config.execution.requested_delay_seconds,
                window_seconds=config.simulation.window_seconds,
                arrival_rate_per_second=config.simulation.arrival_rate_per_second,
                repetitions=config.simulation.repetitions,
            )
            report_task = write_reporter.start_task("write simulation state")
            write_simulation_state(
                artifact_dir / ".spice" / "state.sqlite",
                root_kind=_state_root_kind(config),
                summary=summary,
            )
            write_reporter.finish_task(report_task, message=str(artifact_dir), silent=True)
        session.runtime.log_sectioned_summary(
            "simulation summary",
            [
                (
                    "dataset",
                    [
                        ("name", summary.dataset_name),
                        ("storage id", summary.dataset_id),
                        ("chain", summary.chain),
                        ("model", summary.model_id),
                        ("task", summary.task_id),
                    ],
                ),
                (
                    "provenance",
                    [
                        ("artifact id", summary.artifact_id),
                        ("variant", summary.variant.value),
                        *([] if summary.study is None else [("study", summary.study.name)]),
                        ("capability", f"{summary.max_supported_delay_seconds}s"),
                        ("requested", f"{summary.requested_delay_seconds}s"),
                    ],
                ),
                (
                    "simulation",
                    [
                        ("window", f"{summary.simulation_window_seconds}s"),
                        ("repetitions", str(summary.repetitions)),
                        ("events", f"{summary.total_events:,}"),
                    ],
                ),
                (
                    "results",
                    [
                        (
                            "profit over baseline",
                            (
                                f"{summary.profit_over_baseline.mean:.4f} +/- "
                                f"{summary.profit_over_baseline.std:.4f}"
                            ),
                        ),
                        (
                            "cost over optimum",
                            (
                                f"{summary.cost_over_optimum.mean:.4f} +/- "
                                f"{summary.cost_over_optimum.std:.4f}"
                            ),
                        ),
                    ],
                ),
            ],
        )
