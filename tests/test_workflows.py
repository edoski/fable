from spice.acquisition.cryo import CryoRunResult
from spice.acquisition.provenance import source_manifest_path_for
from spice.acquisition.rpc_providers import RpcProviderName
from spice.api import (
    load_artifact,
    load_config,
    promote_blocks,
    pull_blocks,
    run_simulation_workflow,
    run_training_workflow,
)
from spice.core.constants import SIMULATION_REPORT_FILENAME, TRAIN_REPORT_FILENAME
from tests.support import (
    make_block_rows,
    make_evaluation_rows,
    make_history_rows,
    write_config,
    write_dataset_dir,
    write_raw_chunk,
)


def test_training_and_simulation_workflows_write_artifacts(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "artifacts"
    history_dir = tmp_path / "history"
    evaluation_dir = tmp_path / "evaluation"
    artifact_dir = tmp_path / "run"
    write_config(config_path, output_root=output_root)
    write_dataset_dir(history_dir, make_history_rows())
    write_dataset_dir(evaluation_dir, make_evaluation_rows())

    config = load_config(config_path)
    train_report = run_training_workflow(
        config,
        history_dir,
        artifact_dir,
        "ethereum",
        "lstm",
        36,
        device="cpu",
    )
    loaded = load_artifact(artifact_dir)
    simulation_report = run_simulation_workflow(
        config,
        artifact_dir,
        history_dir,
        evaluation_dir,
        device="cpu",
    )

    assert train_report.chain == "ethereum"
    assert loaded.manifest.max_delay_seconds == 36
    assert simulation_report.total_events > 0
    assert (artifact_dir / TRAIN_REPORT_FILENAME).is_file()
    assert (artifact_dir / SIMULATION_REPORT_FILENAME).is_file()


def test_pull_blocks_writes_source_manifest_and_validation(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "artifacts"
    write_config(config_path, output_root=output_root)
    config = load_config(config_path)

    def fake_run_cryo(*args, **kwargs):
        output_dir = kwargs["output_dir"] if "output_dir" in kwargs else args[2]
        timestamps = kwargs["timestamps"] if "timestamps" in kwargs else args[3]
        write_raw_chunk(
            output_dir,
            chain_name="ethereum",
            rows=make_block_rows(
                4,
                start_block=1,
                start_timestamp=timestamps.start,
                include_gas_limit=False,
            ),
        )
        return CryoRunResult(command="cryo blocks ...", completed_chunks=1, expected_chunks=1)

    monkeypatch.setattr("spice.api.run_cryo", fake_run_cryo)

    result = pull_blocks(
        config,
        "ethereum",
        "history",
        rpc_provider=RpcProviderName.PUBLICNODE,
        dry_run=False,
        validate_on_success=True,
        config_path=config_path,
    )

    assert result.validation is not None
    assert result.validation.status == "clean"
    assert result.source_manifest_path == source_manifest_path_for(result.output_dir)


def test_promote_blocks_moves_validated_dataset_into_canonical_output(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "artifacts"
    source_dir = tmp_path / "staging" / "publicnode" / "raw" / "ethereum" / "history"
    write_config(config_path, output_root=output_root)
    config = load_config(config_path)

    rows = make_history_rows(4)
    rows = [{key: value for key, value in row.items() if key != "gas_limit"} for row in rows]
    write_raw_chunk(source_dir, chain_name="ethereum", rows=rows)
    from spice.acquisition.cryo import history_range_for_chain
    from spice.acquisition.provenance import write_source_manifest
    from spice.acquisition.rpc_providers import resolve_rpc_provider
    from spice.core.config import BlockSegment

    write_source_manifest(
        source_dir,
        config_path=config_path,
        chain=config.resolve_chain("ethereum"),
        segment=BlockSegment.HISTORY,
        timestamps=history_range_for_chain(config.resolve_chain("ethereum")),
        provider=resolve_rpc_provider(
            "publicnode",
            chains=(config.resolve_chain("ethereum").name,),
        ),
        pull=config.pull,
        overwrite=False,
        validation=None,
    )

    result = promote_blocks(
        config,
        "ethereum",
        "history",
        source_dir,
        allow_warnings=True,
    )

    assert result.output_dir == output_root / "raw" / "ethereum" / "history"
    assert result.source_manifest_path == source_manifest_path_for(result.output_dir)
    assert result.output_dir.is_dir()
