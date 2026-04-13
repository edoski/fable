"""Simulation workflow."""

from __future__ import annotations

from ..config import SimulateConfig
from ..core.reporting import Reporter
from ..corpus.io import load_block_frame
from ..modeling.artifacts import load_training_artifact, validate_artifact_feature_graph
from ..modeling.inference import predict_with_model
from ..modeling.pipeline import prepare_inference_dataset
from ..modeling.results import build_simulation_summary_record
from ..modeling.simulation import run_prediction_replay
from ..modeling.summary import simulation_summary_sections
from ..prediction import compile_prediction_contract
from ..storage import ARTIFACT_ROOT_KIND, RootKind
from ..storage.artifact import write_simulation_state
from ..temporal.contracts import resolve_feature_contract
from ._shared import abort_cleanup, managed_workflow


def _workflow_facts(config: SimulateConfig) -> list[tuple[str, str]]:
    facts = [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("problem", config.problem.id),
        ("prediction", config.prediction.id),
        ("execution", config.execution.id),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
    ]
    if config.artifact.variant.value == "tuned":
        facts.append(("study", config.study.name))
    return facts


def _state_root_kind(config: SimulateConfig) -> RootKind:
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
            f"-{config.problem.id}-{config.execution.id}"
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
            if (
                loaded_artifact.manifest.problem.model_dump(mode="json")
                != config.problem.model_dump(mode="json")
            ):
                raise ValueError(
                    "Configured problem does not match the trained artifact semantics"
                )
            if (
                loaded_artifact.manifest.prediction.model_dump(mode="json")
                != config.prediction.model_dump(mode="json")
            ):
                raise ValueError(
                    "Configured prediction does not match the trained artifact semantics"
                )
            contract = resolve_feature_contract(
                problem=config.problem,
                selection=selection,
            )
            prediction_contract = compile_prediction_contract(
                prediction_id=config.prediction.id,
                family_config=config.prediction.family,
            )
            if "simulate" not in prediction_contract.supported_workflows:
                raise ValueError(
                    f"prediction family {prediction_contract.prediction_family_id} "
                    "does not support simulate"
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
                contract=contract,
                requested_delay_seconds=config.execution.requested_delay_seconds,
                compiler_runtime_metadata=loaded_artifact.manifest.compiler_runtime_metadata,
                scaler=loaded_artifact.manifest.scaler,
                max_candidate_slots=loaded_artifact.manifest.max_candidate_slots,
                window_start_timestamp=config.evaluation_window_start_timestamp,
                window_end_timestamp=config.evaluation_window_end_timestamp,
            )
            prepare_reporter.finish_task(
                prepare_task,
                message=f"samples={prepared.sample_count}",
            )
            predictions = predict_with_model(
                loaded_artifact.model,
                model_id=loaded_artifact.manifest.model.id,
                prediction_contract=prediction_contract,
                store=prepared.store,
                sample_indices=prepared.sample_indices,
                batch_size=config.training.batch_size,
                device=config.training.device,
                reporter=predict_reporter,
            )
            simulation = run_prediction_replay(
                prediction_contract,
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
                loaded_artifact.manifest,
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
            simulation_summary_sections(summary),
        )
