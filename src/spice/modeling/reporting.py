"""Structured report artifacts for training and simulation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .artifacts import LoadedTrainingArtifact, TrainingArtifactManifest
from .pipeline import (
    PreparedInferenceDataset,
    PreparedTrainingDataset,
    TrainingRunResult,
)
from .simulation import SimulationSummary


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitSizes(ReportModel):
    train_examples: int
    validation_examples: int
    test_examples: int


class TestMetricsReport(ReportModel):
    total_loss: float
    accuracy: float
    mean_cost_over_optimum: float
    mean_profit_over_baseline: float


class TrainingRunReport(ReportModel):
    kind: Literal["training_run_report"] = "training_run_report"
    history_block_path: str
    artifact_dir: str
    chain: str
    family: str
    max_delay_seconds: int
    device_requested: str
    lookback_seconds: int
    block_time_seconds: float
    target_anchor_count: int
    n_blocks_available: int
    n_blocks_used: int
    n_examples_total: int
    lookback_steps: int
    max_extra_wait_steps: int
    action_count: int
    n_features: int
    split_sizes: SplitSizes
    best_epoch: int
    test_metrics: TestMetricsReport


class SimulationAggregateReport(ReportModel):
    mean: float
    std: float


class SimulationRunReport(ReportModel):
    window_start_timestamp: float
    window_end_timestamp: float
    n_arrivals: int
    n_events: int
    profit_over_baseline: float
    cost_over_optimum: float
    baseline_cost_over_optimum: float


class SimulationReport(ReportModel):
    kind: Literal["simulation_report"] = "simulation_report"
    artifact_dir: str
    history_block_path: str
    evaluation_block_path: str
    chain: str
    family: str
    max_delay_seconds: int
    lookback_seconds: int
    block_time_seconds: float
    lookback_steps: int
    max_extra_wait_steps: int
    action_count: int
    simulation_window_seconds: int
    arrival_rate_per_second: float
    repetitions: int
    n_history_context_blocks: int
    n_evaluation_blocks: int
    n_examples_total: int
    profit_over_baseline: SimulationAggregateReport
    cost_over_optimum: SimulationAggregateReport
    baseline_cost_over_optimum: SimulationAggregateReport
    total_events: int
    runs: list[SimulationRunReport]


def build_training_run_report(
    result: TrainingRunResult,
    *,
    target_anchor_count: int,
    max_delay_seconds: int,
    lookback_seconds: int,
    chain_name: str,
    family: str,
    block_time_seconds: float,
    manifest: TrainingArtifactManifest,
    prepared: PreparedTrainingDataset,
    artifact_dir: Path,
    history_block_path: Path,
    device_requested: str,
) -> TrainingRunReport:
    metrics = result.test_metrics
    return TrainingRunReport(
        history_block_path=str(history_block_path),
        artifact_dir=str(artifact_dir),
        chain=chain_name,
        family=family,
        max_delay_seconds=max_delay_seconds,
        device_requested=device_requested,
        lookback_seconds=lookback_seconds,
        block_time_seconds=block_time_seconds,
        target_anchor_count=target_anchor_count,
        n_blocks_available=prepared.n_blocks_available,
        n_blocks_used=prepared.n_blocks_used,
        n_examples_total=prepared.n_examples_total,
        lookback_steps=manifest.lookback_steps,
        max_extra_wait_steps=manifest.max_extra_wait_steps,
        action_count=manifest.action_count,
        n_features=manifest.n_features,
        split_sizes=SplitSizes(
            train_examples=int(prepared.split_indices.train.shape[0]),
            validation_examples=int(prepared.split_indices.validation.shape[0]),
            test_examples=int(prepared.split_indices.test.shape[0]),
        ),
        best_epoch=result.training_result.best_epoch,
        test_metrics=TestMetricsReport(
            total_loss=metrics.total_loss,
            accuracy=metrics.accuracy,
            mean_cost_over_optimum=metrics.mean_cost_over_optimum,
            mean_profit_over_baseline=metrics.mean_profit_over_baseline,
        ),
    )


def build_simulation_report(
    loaded_artifact: LoadedTrainingArtifact,
    *,
    artifact_dir: Path,
    history_block_path: Path,
    evaluation_block_path: Path,
    prepared: PreparedInferenceDataset,
    simulation: SimulationSummary,
    window_seconds: int,
    arrival_rate_per_second: float,
    repetitions: int,
) -> SimulationReport:
    manifest = loaded_artifact.manifest
    return SimulationReport(
        artifact_dir=str(artifact_dir),
        history_block_path=str(history_block_path),
        evaluation_block_path=str(evaluation_block_path),
        chain=manifest.chain.name.value,
        family=manifest.model.family.value,
        max_delay_seconds=manifest.max_delay_seconds,
        lookback_seconds=manifest.lookback_seconds,
        block_time_seconds=manifest.chain.block_time_seconds,
        lookback_steps=manifest.lookback_steps,
        max_extra_wait_steps=manifest.max_extra_wait_steps,
        action_count=manifest.action_count,
        simulation_window_seconds=window_seconds,
        arrival_rate_per_second=arrival_rate_per_second,
        repetitions=repetitions,
        n_history_context_blocks=prepared.n_history_context_blocks,
        n_evaluation_blocks=prepared.n_evaluation_blocks,
        n_examples_total=prepared.n_examples_total,
        profit_over_baseline=SimulationAggregateReport(
            mean=simulation.mean_profit_over_baseline,
            std=simulation.std_profit_over_baseline,
        ),
        cost_over_optimum=SimulationAggregateReport(
            mean=simulation.mean_cost_over_optimum,
            std=simulation.std_cost_over_optimum,
        ),
        baseline_cost_over_optimum=SimulationAggregateReport(
            mean=simulation.mean_baseline_cost_over_optimum,
            std=simulation.std_baseline_cost_over_optimum,
        ),
        total_events=simulation.total_events,
        runs=[
            SimulationRunReport(
                window_start_timestamp=run.window_start_timestamp,
                window_end_timestamp=run.window_end_timestamp,
                n_arrivals=run.n_arrivals,
                n_events=run.n_events,
                profit_over_baseline=run.profit_over_baseline,
                cost_over_optimum=run.cost_over_optimum,
                baseline_cost_over_optimum=run.baseline_cost_over_optimum,
            )
            for run in simulation.runs
        ],
    )


def write_json_report(path: Path, report: TrainingRunReport | SimulationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
