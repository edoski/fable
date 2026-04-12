"""SQLAlchemy Core schema for SPICE-owned workflow state."""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, Column, Float, Integer, MetaData, String, Table

STATE_SCHEMA_VERSION = 2

metadata = MetaData()

spice_meta = Table(
    "spice_meta",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("root_kind", String, nullable=False),
    Column("schema_version", Integer, nullable=False),
    Column("created_at", Integer, nullable=False),
    Column("updated_at", Integer, nullable=False),
)

dataset_summary = Table(
    "dataset_summary",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("dataset_id", String, nullable=False),
    Column("chain_name", String, nullable=False),
    Column("chain_id", Integer, nullable=False),
    Column("provider_name", String, nullable=False),
    Column("provider_reference", String, nullable=False),
    Column("provider_endpoint_fingerprint", String, nullable=False),
    Column("history_request_start_timestamp", Integer, nullable=False),
    Column("history_request_end_timestamp", Integer, nullable=False),
    Column("evaluation_request_start_timestamp", Integer, nullable=False),
    Column("evaluation_request_end_timestamp", Integer, nullable=False),
    Column("history_coverage_start_timestamp", Integer, nullable=False),
    Column("history_coverage_end_timestamp", Integer, nullable=False),
    Column("evaluation_coverage_start_timestamp", Integer, nullable=False),
    Column("evaluation_coverage_end_timestamp", Integer, nullable=False),
    Column("history_validation", JSON, nullable=False),
    Column("evaluation_validation", JSON, nullable=False),
)

acquire_runs = Table(
    "acquire_runs",
    metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("recorded_at", Integer, nullable=False),
    Column("provider_name", String, nullable=False),
    Column("provider_reference", String, nullable=False),
    Column("provider_endpoint_fingerprint", String, nullable=False),
    Column("task_id", String, nullable=False),
    Column("feature_set_id", String, nullable=False),
    Column("lookback_seconds", Integer, nullable=False),
    Column("sample_count", Integer, nullable=False),
    Column("max_supported_delay_seconds", Integer, nullable=False),
    Column("required_history_context_blocks", Integer, nullable=False),
    Column("required_history_blocks", Integer, nullable=False),
    Column("chunk_size", Integer, nullable=False),
    Column("rpc_batch_size", Integer, nullable=False),
    Column("rpc_concurrency", Integer, nullable=False),
    Column("rpc_min_batch_size", Integer, nullable=False),
    Column("rpc_concurrency_rungs", JSON, nullable=False),
    Column("configured_batch_size", Integer, nullable=False),
    Column("final_batch_size", Integer, nullable=False),
    Column("min_batch_size", Integer, nullable=False),
    Column("configured_concurrency", Integer, nullable=False),
    Column("final_concurrency", Integer, nullable=False),
    Column("concurrency_rungs", JSON, nullable=False),
    Column("oversize_error_count", Integer, nullable=False),
    Column("transient_error_count", Integer, nullable=False),
    Column("oversize_backoffs", Integer, nullable=False),
    Column("transient_backoffs", Integer, nullable=False),
    Column("concurrency_recoveries", Integer, nullable=False),
)

artifact_manifest = Table(
    "artifact_manifest",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("chain_name", String, nullable=False),
    Column("chain_block_time_seconds", Float, nullable=False),
    Column("dataset_id", String, nullable=False),
    Column("task_id", String, nullable=False),
    Column("variant", String, nullable=False),
    Column("study_id", String),
    Column("max_supported_delay_seconds", Integer, nullable=False),
    Column("lookback_seconds", Integer, nullable=False),
    Column("sample_count", Integer, nullable=False),
    Column("feature_set_id", String, nullable=False),
    Column("feature_names", JSON, nullable=False),
    Column("feature_graph_fingerprint", String, nullable=False),
    Column("model", JSON, nullable=False),
    Column("scaler", JSON, nullable=False),
)

training_summary = Table(
    "training_summary",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("chain_name", String, nullable=False),
    Column("dataset_id", String, nullable=False),
    Column("variant", String, nullable=False),
    Column("study_id", String),
    Column("model_id", String, nullable=False),
    Column("task_id", String, nullable=False),
    Column("max_supported_delay_seconds", Integer, nullable=False),
    Column("lookback_seconds", Integer, nullable=False),
    Column("sample_count", Integer, nullable=False),
    Column("n_blocks_available", Integer, nullable=False),
    Column("n_blocks_used", Integer, nullable=False),
    Column("train_samples", Integer, nullable=False),
    Column("validation_samples", Integer, nullable=False),
    Column("test_samples", Integer, nullable=False),
    Column("best_epoch", Integer, nullable=False),
    Column("resolved_device", String, nullable=False),
    Column("resolved_precision", String, nullable=False),
    Column("compiled", Boolean, nullable=False),
    Column("best_validation_metrics", JSON, nullable=False),
    Column("test_metrics", JSON, nullable=False),
)

training_epochs = Table(
    "training_epochs",
    metadata,
    Column("epoch", Integer, primary_key=True),
    Column("train_metrics", JSON, nullable=False),
    Column("validation_metrics", JSON, nullable=False),
)

simulation_summary = Table(
    "simulation_summary",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("chain_name", String, nullable=False),
    Column("dataset_id", String, nullable=False),
    Column("variant", String, nullable=False),
    Column("study_id", String),
    Column("model_id", String, nullable=False),
    Column("task_id", String, nullable=False),
    Column("max_supported_delay_seconds", Integer, nullable=False),
    Column("requested_delay_seconds", Integer, nullable=False),
    Column("lookback_seconds", Integer, nullable=False),
    Column("simulation_window_seconds", Integer, nullable=False),
    Column("arrival_rate_per_second", Float, nullable=False),
    Column("repetitions", Integer, nullable=False),
    Column("n_history_context_blocks", Integer, nullable=False),
    Column("n_evaluation_blocks", Integer, nullable=False),
    Column("sample_count", Integer, nullable=False),
    Column("profit_over_baseline", JSON, nullable=False),
    Column("cost_over_optimum", JSON, nullable=False),
    Column("baseline_cost_over_optimum", JSON, nullable=False),
    Column("total_events", Integer, nullable=False),
)

simulation_runs = Table(
    "simulation_runs",
    metadata,
    Column("ordinal", Integer, primary_key=True),
    Column("window_start_timestamp", Float, nullable=False),
    Column("window_end_timestamp", Float, nullable=False),
    Column("n_arrivals", Integer, nullable=False),
    Column("n_events", Integer, nullable=False),
    Column("profit_over_baseline", Float, nullable=False),
    Column("cost_over_optimum", Float, nullable=False),
    Column("baseline_cost_over_optimum", Float, nullable=False),
)

DATASET_TABLES = (dataset_summary, acquire_runs)
ARTIFACT_TABLES = (
    artifact_manifest,
    training_summary,
    training_epochs,
    simulation_summary,
    simulation_runs,
)
