from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from spice.config import TrainConfig, WorkflowTask
from spice.storage.layout import resolve_workflow_paths
from tests.dataset_helpers import make_history_rows, write_dataset_dir

PRESET = "icdcs_2026"


@pytest.fixture
def load_test_train_config(tmp_path: Path, load_workflow_config):
    def _load(
        tmp_path_arg: Path | None = None,
        *,
        override: dict[str, object] | None = None,
    ) -> TrainConfig:
        workspace = tmp_path if tmp_path_arg is None else tmp_path_arg
        return cast(
            TrainConfig,
            load_workflow_config(
                WorkflowTask.TRAIN,
                workspace=workspace,
                preset=PRESET,
                override=override,
            ),
        )

    return _load


@pytest.fixture
def seed_history_dataset():
    def _seed(config: TrainConfig) -> Path:
        return write_dataset_dir(
            resolve_workflow_paths(config).history_dir,
            make_history_rows(config),
        )

    return _seed
