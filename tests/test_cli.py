from __future__ import annotations

from typer.testing import CliRunner

from spice.cli import app
from tests.support import make_history_rows, write_config, write_dataset_dir


def test_cli_blocks_validate_smoke(tmp_path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "artifacts"
    raw_dir = output_root / "raw" / "ethereum" / "history"
    write_config(config_path, output_root=output_root)
    from tests.support import write_raw_chunk

    write_raw_chunk(raw_dir, chain_name="ethereum", rows=make_history_rows(4))

    result = runner.invoke(app, ["blocks", "validate", str(config_path), "ethereum", "history"])

    assert result.exit_code == 0
    assert "status=clean" in result.stdout


def test_cli_train_smoke(tmp_path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "artifacts"
    history_dir = tmp_path / "history"
    artifact_dir = tmp_path / "run"
    write_config(config_path, output_root=output_root)
    write_dataset_dir(history_dir, make_history_rows())

    result = runner.invoke(
        app,
        [
            "train",
            str(config_path),
            str(history_dir),
            str(artifact_dir),
            "ethereum",
            "lstm",
            "36",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert "best_epoch=" in result.stdout
