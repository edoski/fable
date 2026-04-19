from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from spice.config import AcquireConfig, WorkflowTask
from tests.dataset_helpers import make_block_rows as _make_block_rows
from tests.dataset_helpers import write_dataset_dir as _write_dataset_dir

PRESET = "icdcs_2026"


@pytest.fixture
def load_test_acquire_config(tmp_path: Path, load_workflow_config):
    def _load(
        tmp_path_arg: Path | None = None,
        *,
        override: dict[str, object] | None = None,
        chain: str | None = None,
        provider: str | None = None,
    ) -> AcquireConfig:
        workspace = tmp_path if tmp_path_arg is None else tmp_path_arg
        return cast(
            AcquireConfig,
            load_workflow_config(
                WorkflowTask.ACQUIRE,
                workspace=workspace,
                preset=PRESET,
                override=override,
                chain=chain,
                provider=provider,
            ),
        )

    return _load


@pytest.fixture
def make_block_rows():
    def _build_block_rows(
        count: int,
        *,
        start_block: int,
        start_timestamp: int,
        chain_id: int = 1,
        block_interval_seconds: int = 12,
    ) -> list[dict[str, int]]:
        return _make_block_rows(
            count,
            start_block=start_block,
            start_timestamp=start_timestamp,
            chain_id=chain_id,
            block_interval_seconds=block_interval_seconds,
        )

    return _build_block_rows


@pytest.fixture
def write_dataset_dir():
    def _write(dataset_dir: Path, rows: list[dict[str, int]]) -> Path:
        return _write_dataset_dir(dataset_dir, rows)

    return _write
