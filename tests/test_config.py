from __future__ import annotations

import pytest
import yaml

from spice.core.config import ArtifactVariant, WorkflowTask, load_params_config
from spice.workflows.dvc import load_stage_config
from tests.support import (
    REPO_ROOT,
    TEST_WINDOW_END_TIMESTAMP,
    TEST_WINDOW_START_TIMESTAMP,
    base_overrides,
    compose_experiment,
)


def test_params_train_config_resolves_paths_after_overrides(tmp_path) -> None:
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path)
        + ["dataset.temporal.max_delay_seconds=24", "presets.model=transformer"],
    )

    assert config.task is WorkflowTask.TRAIN
    assert config.dataset.temporal.max_delay_seconds == 24
    assert config.model.family.value == "transformer"
    assert config.dataset.id == "icdcs_2025_11_09"
    assert config.dataset.sampling.sample_count == 48
    assert config.acquisition.history_sample_budget == 48
    assert config.effective_history_sample_budget == 48
    assert config.artifact.variant is ArtifactVariant.BASELINE
    assert config.paths.artifact_root.as_posix().endswith(
        "/ethereum/icdcs_2025_11_09/transformer/24s/baseline/default"
    )
    assert config.paths.history_dir.as_posix().endswith(
        "/datasets/ethereum/icdcs_2025_11_09/history"
    )
    assert config.paths.metadata_root.as_posix().endswith(
        "/datasets/ethereum/icdcs_2025_11_09/.spice"
    )
    assert config.paths.dataset_metadata_path.as_posix().endswith(
        "/datasets/ethereum/icdcs_2025_11_09/.spice/metadata.json"
    )


def test_variant_paths_resolve_baseline_and_tuned_lineages(tmp_path) -> None:
    baseline = compose_experiment("train", overrides=base_overrides(tmp_path))
    tuned = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path) + ["artifact.variant=tuned", "study.id=fee-sweep-a"],
    )

    assert baseline.paths.artifact_root.as_posix().endswith("/lstm/36s/baseline/default")
    assert baseline.paths.tuning_root.as_posix().endswith("/lstm/36s/tuned/default/tuning")
    assert tuned.paths.artifact_root.as_posix().endswith("/lstm/36s/tuned/fee-sweep-a")
    assert tuned.paths.train_report_path.as_posix().endswith(
        "/lstm/36s/tuned/fee-sweep-a/train_report.json"
    )
    assert tuned.paths.tuning_best_params_path.as_posix().endswith(
        "/lstm/36s/tuned/fee-sweep-a/tuning/best_params.json"
    )


def test_direct_provider_reads_env_interpolation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ETHEREUM_RPC_URL", "https://eth.example.test")
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["presets.provider=direct"],
    )

    assert config.provider.endpoint_for(config.chain.name) == "https://eth.example.test"
    assert config.provider.reference_for(config.chain.name) == "$ETHEREUM_RPC_URL"
    assert config.acquisition.chunk_size == 8192
    assert config.acquisition.rpc_batch_size == 256
    assert config.acquisition.rpc_concurrency == 48
    assert config.acquisition.rpc_min_batch_size == 64
    assert config.acquisition.rpc_concurrency_rungs == [8, 16, 24, 32, 48]


def test_train_config_does_not_require_direct_provider_endpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ETHEREUM_RPC_URL", raising=False)
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path) + ["presets.provider=direct"],
    )

    assert config.provider.reference_for(config.chain.name) == "$ETHEREUM_RPC_URL"


def test_acquire_config_requires_direct_provider_endpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ETHEREUM_RPC_URL", raising=False)

    with pytest.raises(ValueError, match="Missing RPC endpoint for chain: ethereum"):
        compose_experiment(
            "acquire",
            overrides=base_overrides(tmp_path) + ["presets.provider=direct"],
        )


@pytest.mark.parametrize(
    ("chain_name", "uses_poa_extra_data", "rpc_batch_size", "chunk_size"),
    [
        ("ethereum", False, 256, 8192),
        ("polygon", True, 256, 8192),
        ("avalanche", True, 256, 8192),
    ],
)
def test_publicnode_chain_matrix_applies_expected_chain_settings(
    tmp_path,
    chain_name: str,
    uses_poa_extra_data: bool,
    rpc_batch_size: int,
    chunk_size: int,
) -> None:
    overrides = [
        f"runtime.output_root={tmp_path / 'artifacts'}",
        "tracking.enabled=false",
    ]
    if chain_name != "ethereum":
        overrides.append(f"presets.chain={chain_name}")

    config = compose_experiment("acquire", overrides=overrides)

    assert config.chain.name.value == chain_name
    assert config.chain.uses_poa_extra_data is uses_poa_extra_data
    assert config.acquisition.rpc_batch_size == rpc_batch_size
    assert config.acquisition.chunk_size == chunk_size
    assert config.acquisition.rpc_concurrency == 48
    assert config.acquisition.rpc_min_batch_size == 64
    assert config.acquisition.rpc_concurrency_rungs == [8, 16, 24, 32, 48]


def test_invalid_rpc_concurrency_config_fails_early(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="rpc_concurrency must be present in rpc_concurrency_rungs",
    ):
        compose_experiment(
            "acquire",
            overrides=base_overrides(tmp_path) + ["acquisition.rpc_concurrency=12"],
        )


def test_invalid_transformer_config_fails_early(tmp_path) -> None:
    with pytest.raises(ValueError, match="d_model must be divisible by nhead"):
        compose_experiment(
            "train",
            overrides=base_overrides(tmp_path)
            + ["presets.model=transformer", "model.d_model=126", "model.nhead=8"],
        )


def test_model_schema_rejects_family_irrelevant_fields(tmp_path) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        compose_experiment(
            "train",
            overrides=base_overrides(tmp_path)
            + ["presets.model=transformer", "model.hidden_size=128"],
        )


def test_span_and_evaluation_duration_resolve_to_derived_windows(tmp_path) -> None:
    config = compose_experiment("simulate", overrides=base_overrides(tmp_path))

    assert config.evaluation_window_start_timestamp == TEST_WINDOW_START_TIMESTAMP
    assert config.evaluation_window_end_timestamp == TEST_WINDOW_END_TIMESTAMP
    assert config.history_window_end_timestamp == TEST_WINDOW_START_TIMESTAMP


def test_dvc_acquire_stage_params_are_scoped_to_acquisition_inputs() -> None:
    payload = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stages = payload["stages"]
    acquire_params = stages["acquire"]["params"][0]["params.yaml"]

    assert "training" not in acquire_params
    assert "model" not in acquire_params
    assert "dataset.span" in acquire_params
    assert "dataset.temporal" in acquire_params
    assert "dataset.sampling.sample_count" in acquire_params
    assert "acquisition.history_sample_budget" in acquire_params
    assert "evaluation.duration_days" in acquire_params


def test_history_sample_budget_cannot_be_smaller_than_sample_count(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="acquisition.history_sample_budget must be at least dataset.sampling.sample_count",
    ):
        compose_experiment(
            "acquire",
            overrides=base_overrides(tmp_path)
            + ["acquisition.history_sample_budget=32"],
        )


def test_invalid_tuning_direction_fails_early(tmp_path) -> None:
    with pytest.raises(ValueError, match="maximize|minimize"):
        compose_experiment(
            "tune",
            overrides=base_overrides(tmp_path) + ["tuning.direction=sideways"],
        )


def test_invalid_tuning_objective_fails_early(tmp_path) -> None:
    with pytest.raises(ValueError, match="validation_loss|validation_accuracy"):
        compose_experiment(
            "tune",
            overrides=base_overrides(tmp_path) + ["tuning.objective_metric=validation_magic"],
        )


def test_tuning_search_space_rejects_empty_candidate_list(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        compose_experiment(
            "tune",
            overrides=base_overrides(tmp_path)
            + ["tuning.search_space.training.learning_rate=[]"],
        )


def test_tuning_search_space_rejects_unsupported_field(tmp_path) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        compose_experiment(
            "tune",
            overrides=base_overrides(tmp_path)
            + ["+tuning.search_space.training.momentum=[0.9]"],
        )


def test_dvc_runner_loads_generated_params_without_forcing_tuned_training() -> None:
    config = load_stage_config("train", REPO_ROOT / "params.yaml")

    assert config.task is WorkflowTask.TRAIN
    assert config.artifact.variant is ArtifactVariant.BASELINE
    assert config.study.id == "default"
    assert config.paths.artifact_root.as_posix().endswith(
        "/models/ethereum/icdcs_2025_11_09/lstm/36s/baseline/default"
    )


def test_direct_and_dvc_loaders_share_the_same_baseline_params() -> None:
    direct_config = load_params_config("train", params_path=REPO_ROOT / "params.yaml")
    dvc_config = load_stage_config("train", REPO_ROOT / "params.yaml")

    assert direct_config.dataset.sampling.sample_count == dvc_config.dataset.sampling.sample_count
    assert (
        direct_config.dataset.temporal.lookback_seconds
        == dvc_config.dataset.temporal.lookback_seconds
    )
    assert direct_config.training.precision == dvc_config.training.precision
    assert direct_config.training.compile == dvc_config.training.compile
    assert direct_config.artifact.variant == dvc_config.artifact.variant
    assert direct_config.study == dvc_config.study
    assert direct_config.paths.artifact_root == dvc_config.paths.artifact_root
