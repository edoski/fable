"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_START_TS = 1762646400  # 2025-11-09 00:00:00 UTC
EVALUATION_END_TS = 1762732800  # 2025-11-10 00:00:00 UTC

ARTIFACT_MANIFEST_FILENAME = "artifact.json"
MODEL_STATE_FILENAME = "model.pt"
TRAIN_REPORT_FILENAME = "train_report.json"
SIMULATION_REPORT_FILENAME = "simulation_report.json"
