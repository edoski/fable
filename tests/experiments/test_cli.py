from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest

_ROOT = Path(__file__).parents[2]
_UUID1 = UUID("10000000-0000-1000-8000-000000000001")
_UPSTREAM_ID = UUID("20000000-0000-4000-8000-000000000001")


@pytest.mark.parametrize(
    ("script_name", "upstream_ids"),
    [
        pytest.param("feature_ablation", (), id="feature-ablation"),
        pytest.param("c_study", (_UPSTREAM_ID,), id="context"),
        pytest.param("hpo", (_UPSTREAM_ID,), id="hpo"),
        pytest.param("k_study", (_UPSTREAM_ID,), id="horizon"),
        pytest.param("held_out", (_UPSTREAM_ID, _UPSTREAM_ID), id="held-out"),
    ],
)
def test_prepare_rejects_explicit_uuid1_before_authoring(
    script_name: str,
    upstream_ids: tuple[UUID, ...],
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "experiments" / f"{script_name}.py"),
            "prepare",
            str(tmp_path),
            *(str(identifier) for identifier in upstream_ids),
            "--experiment-id",
            str(_UUID1),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "experiment_id must be a UUIDv4" in result.stderr
    assert not (tmp_path / "experiments").exists()
