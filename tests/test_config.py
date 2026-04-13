from __future__ import annotations

from typer.testing import CliRunner

from spice.cli import app
from spice.identifiers import corpus_storage_id
from tests.support import write_override

runner = CliRunner()


def test_acquire_cli_loads_specs_and_applies_override_precedence(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = write_override(
        tmp_path,
        {
            "chain": "polygon",
            "provider": "direct",
            "task": {
                "sample_count": 640,
            },
        },
    )
    monkeypatch.setenv("AVALANCHE_RPC_URL", "https://avax.example.test")
    captured: dict[str, object] = {}

    def _capture(**kwargs) -> None:
        from spice.config import load_acquire_config

        captured["config"] = load_acquire_config(
            preset=kwargs["preset"],
            config_path=kwargs["config"],
            dataset=kwargs["dataset"],
            task=kwargs["task"],
            chain=kwargs["chain"],
            provider=kwargs["provider"],
            feature_set=kwargs["feature_set"],
            acquisition=kwargs["acquisition_profile"],
            storage_root=kwargs["storage_root"],
            dry_run=kwargs["dry_run"],
        )

    monkeypatch.setattr("spice.cli._run_acquire", _capture)

    result = runner.invoke(
        app,
        [
            "acquire",
            "--preset",
            "icdcs_2026",
            "--config",
            str(config_path),
            "--chain",
            "avalanche",
            "--storage-root",
            str(tmp_path / "outputs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    config = captured["config"]
    assert config.chain.name == "avalanche"
    assert config.chain.runtime.chain_id == 43114
    assert config.provider.name == "direct"
    assert config.provider.rpc.timeout_seconds == 30.0
    assert config.provider.endpoint_for(config.chain.name) == "https://avax.example.test"
    assert config.acquisition.rpc.batch_size == 256
    assert config.dataset.name == "icdcs_2026"
    assert config.task.sample_count == 640
    assert config.feature_set.id == "icdcs_2026"
    assert config.paths.output_root == tmp_path / "outputs"
    assert config.paths.history_dir == (
        tmp_path
        / "outputs"
        / "corpora"
        / "avalanche"
        / corpus_storage_id(chain_name="avalanche", dataset_name="icdcs_2026")
        / "history"
    )
