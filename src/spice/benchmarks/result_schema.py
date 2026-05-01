# pyright: strict

"""SQLite schema for the benchmark result index."""

from __future__ import annotations

from sqlalchemy import Column, Float, Integer, MetaData, String, Table

metadata = MetaData()

benchmark_runs = Table(
    "benchmark_runs",
    metadata,
    Column("run_dir", String, primary_key=True),
    Column("benchmark", String, nullable=False),
    Column("target", String, nullable=False),
    Column("created_at_utc", String, nullable=False),
    Column("updated_at", Integer, nullable=False),
)

result_observations = Table(
    "result_observations",
    metadata,
    Column("observation_id", String, primary_key=True),
    Column("run_dir", String, nullable=False),
    Column("benchmark", String, nullable=False),
    Column("run_id", String, nullable=False),
    Column("case_id", String, nullable=False),
    Column("step_id", String, nullable=False),
    Column("git_commit", String, nullable=False),
    Column("execution_ref", String, nullable=False),
    Column("artifact_id", String, nullable=False),
    Column("evaluation_storage_id", String, nullable=False),
    Column("dataset_id", String, nullable=False),
    Column("chain_name", String, nullable=False),
    Column("dataset_name", String, nullable=False),
    Column("surface", String, nullable=False),
    Column("features_id", String, nullable=False),
    Column("model_id", String, nullable=False),
    Column("problem_id", String, nullable=False),
    Column("prediction_id", String, nullable=False),
    Column("objective_id", String, nullable=False),
    Column("evaluation_id", String, nullable=False),
    Column("delay_seconds", Integer, nullable=False),
    Column("variant", String, nullable=False),
    Column("study_name", String),
    Column("sample_count", Integer, nullable=False),
    Column("total_events", Integer, nullable=False),
    Column("recorded_at_utc", String, nullable=False),
    Column("payload_json", String, nullable=False),
    Column("updated_at", Integer, nullable=False),
)

metric_values = Table(
    "metric_values",
    metadata,
    Column("observation_id", String, primary_key=True),
    Column("source", String, primary_key=True),
    Column("metric_id", String, primary_key=True),
    Column("value", Float(), nullable=False),
)
