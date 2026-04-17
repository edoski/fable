"""SQLAlchemy Core schema for SPICE-owned workflow state."""

from __future__ import annotations

from sqlalchemy import JSON, Column, Integer, MetaData, String, Table

metadata = MetaData()

spice_meta = Table(
    "spice_meta",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("root_kind", String, nullable=False),
    Column("created_at", Integer, nullable=False),
    Column("updated_at", Integer, nullable=False),
)

dataset_manifest = Table(
    "dataset_manifest",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("payload", JSON, nullable=False),
)

acquire_runs = Table(
    "acquire_runs",
    metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("recorded_at", Integer, nullable=False),
    Column("payload", JSON, nullable=False),
)

artifact_manifest = Table(
    "artifact_manifest",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("payload", JSON, nullable=False),
)

training_summary = Table(
    "training_summary",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("payload", JSON, nullable=False),
)

training_epochs = Table(
    "training_epochs",
    metadata,
    Column("epoch", Integer, primary_key=True),
    Column("payload", JSON, nullable=False),
)

evaluation_summary = Table(
    "evaluation_summary",
    metadata,
    Column("evaluation_id", String, primary_key=True),
    Column("recorded_at", Integer, nullable=False),
    Column("payload", JSON, nullable=False),
)

evaluation_runs = Table(
    "evaluation_runs",
    metadata,
    Column("evaluation_id", String, primary_key=True),
    Column("ordinal", Integer, primary_key=True),
    Column("payload", JSON, nullable=False),
)

study_manifest = Table(
    "study_manifest",
    metadata,
    Column("singleton", Integer, primary_key=True),
    Column("payload", JSON, nullable=False),
)

DATASET_TABLES = (dataset_manifest, acquire_runs)
STUDY_TABLES = (study_manifest,)
ARTIFACT_TABLES = (
    artifact_manifest,
    training_summary,
    training_epochs,
    evaluation_summary,
    evaluation_runs,
)
