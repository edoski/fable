from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from spice.core.config import CompileMode, ModelFamily, TrainingPrecision
from spice.core.constants import ARTIFACT_MANIFEST_FILENAME, MODEL_STATE_FILENAME
from spice.core.files import write_path_atomic
from spice.modeling._runtime import (
    build_sequence_loader,
    resolve_compile_enabled,
    resolve_trainer_precision,
)
from spice.modeling.artifacts import load_training_artifact
from spice.modeling.evaluation import compute_temporal_batch_metrics
from spice.modeling.execution import run_persisted_training
from spice.modeling.models import ModelOutputs
from spice.modeling.pipeline import TrainingSpec
from spice.modeling.reporting import TrainingRunReport
from spice.modeling.torch_datasets import SequenceBatch
from spice.workflows.train import run as run_train
from tests.support import base_overrides, compose_experiment, make_history_rows, write_dataset_dir


def test_run_persisted_training_writes_canonical_training_outputs(tmp_path) -> None:
    config = compose_experiment("train", overrides=base_overrides(tmp_path))
    history_dir = config.paths.history_dir
    artifact_dir = config.paths.artifact_root
    report_path = config.paths.train_report_path
    write_dataset_dir(history_dir, make_history_rows())

    persisted = run_persisted_training(
        history_dir,
        spec=TrainingSpec(
            chain=config.chain,
            dataset_id=config.dataset.id,
            model=config.model,
            max_delay_seconds=config.dataset.temporal.max_delay_seconds,
            lookback_seconds=config.dataset.temporal.lookback_seconds,
            sample_count=config.dataset.sampling.sample_count,
            split=config.split,
            training=config.training,
        ),
        artifact_dir=artifact_dir,
        report_path=report_path,
    )

    loaded = load_training_artifact(artifact_dir)
    report = TrainingRunReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    assert persisted.artifact_dir == artifact_dir
    assert persisted.report_path == report_path
    assert artifact_dir / ARTIFACT_MANIFEST_FILENAME in persisted.artifact_paths
    assert artifact_dir / MODEL_STATE_FILENAME in persisted.artifact_paths
    assert report_path in persisted.artifact_paths
    assert loaded.manifest.dataset_id == config.dataset.id
    assert loaded.manifest.model.family == config.model.family
    assert loaded.manifest.variant.value == "baseline"
    assert loaded.manifest.study is None
    assert report.dataset_id == config.dataset.id
    assert report.variant.value == "baseline"
    assert report.study is None
    assert report.artifact_dir == artifact_dir
    assert report.best_epoch == persisted.training_run.training_result.best_epoch


def test_temporal_batch_metrics_use_shared_loss_path(tmp_path) -> None:
    config = compose_experiment("train", overrides=base_overrides(tmp_path))
    outputs = ModelOutputs(
        logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
        fee_hat=torch.tensor([1.0, 3.0], dtype=torch.float32),
    )
    batch = SequenceBatch(
        inputs=torch.zeros((2, 4, 3), dtype=torch.float32),
        class_label=torch.tensor([0, 1], dtype=torch.long),
        target_log_fee=torch.tensor([1.5, 2.0], dtype=torch.float32),
        action_log_fees=torch.tensor([[1.0, 4.0], [5.0, 3.0]], dtype=torch.float32),
        next_block_log_fee=torch.tensor([1.0, 5.0], dtype=torch.float32),
        optimal_log_fee=torch.tensor([1.0, 3.0], dtype=torch.float32),
    )
    class_weights = torch.ones(2, dtype=torch.float32)

    total_loss, metrics = compute_temporal_batch_metrics(
        outputs,
        batch,
        class_weights=class_weights,
        training_config=config.training,
    )
    expected_loss = config.training.action_loss_weight * F.cross_entropy(
        outputs.logits,
        batch.class_label,
        weight=class_weights,
    ) + config.training.fee_loss_weight * F.smooth_l1_loss(
        outputs.fee_hat,
        batch.target_log_fee,
    )

    assert torch.isclose(total_loss, expected_loss)
    assert metrics.correct_count == 2


def test_sequence_loader_matches_manual_temporal_batch_construction(tmp_path) -> None:
    config = compose_experiment("train", overrides=base_overrides(tmp_path))
    history_dir = config.paths.history_dir
    write_dataset_dir(history_dir, make_history_rows())
    persisted = run_persisted_training(
        history_dir,
        spec=TrainingSpec(
            chain=config.chain,
            dataset_id=config.dataset.id,
            model=config.model,
            max_delay_seconds=config.dataset.temporal.max_delay_seconds,
            lookback_seconds=config.dataset.temporal.lookback_seconds,
            sample_count=config.dataset.sampling.sample_count,
            split=config.split,
            training=config.training,
        ),
        artifact_dir=config.paths.artifact_root,
        report_path=config.paths.train_report_path,
    )

    loader = build_sequence_loader(
        persisted.training_run.prepared.store,
        persisted.training_run.prepared.split_indices.train,
        lookback_steps=persisted.training_run.prepared.geometry.lookback_steps,
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    batch = next(iter(loader))
    store = persisted.training_run.prepared.store
    lookback_steps = persisted.training_run.prepared.geometry.lookback_steps
    expected_sample_indices = persisted.training_run.prepared.split_indices.train[
        : config.training.batch_size
    ]
    expected_inputs = []
    for sample_index in expected_sample_indices:
        sample_row_index = int(store.sample_row_indices[int(sample_index)])
        sequence_start = sample_row_index - lookback_steps + 1
        expected_inputs.append(store.feature_matrix[sequence_start : sample_row_index + 1])
    expected_inputs_tensor = torch.from_numpy(
        np.stack(expected_inputs).astype(np.float32, copy=False)
    )

    assert isinstance(batch, SequenceBatch)
    assert batch.inputs.ndim == 3
    assert batch.class_label.ndim == 1
    assert torch.equal(batch.inputs, expected_inputs_tensor)
    assert torch.equal(
        batch.class_label,
        torch.from_numpy(store.class_labels[expected_sample_indices].astype(np.int64, copy=False)),
    )


def test_runtime_policy_prefers_fp32_for_lstm_on_mps_and_compile_for_mps(tmp_path) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    config = compose_experiment("train", overrides=base_overrides(tmp_path))
    config.training.device = "mps"
    config.training.precision = TrainingPrecision.AUTO
    config.training.compile = CompileMode.AUTO
    device = torch.device("mps")

    assert resolve_trainer_precision(
        config.training,
        device=device,
        family=ModelFamily.LSTM,
    ) == "32-true"
    transformer_precision = resolve_trainer_precision(
        config.training,
        device=device,
        family=ModelFamily.TRANSFORMER,
    )
    assert resolve_trainer_precision(
        config.training,
        device=device,
        family=ModelFamily.TRANSFORMER,
    ) == "bf16-mixed"
    assert resolve_compile_enabled(
        config.training,
        device=device,
        precision="32-true",
        family=ModelFamily.LSTM,
    ) is True
    assert resolve_compile_enabled(
        config.training,
        device=device,
        precision=transformer_precision,
        family=ModelFamily.TRANSFORMER,
    ) is False
    assert resolve_compile_enabled(
        config.training,
        device=device,
        precision="32-true",
        family=ModelFamily.TRANSFORMER,
    ) is False


def test_train_runs_with_compile_and_auto_precision_on_mps(tmp_path) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path)
        + [
            "presets.model=lstm",
            "training.device=mps",
            "training.precision=auto",
            "training.compile=on",
        ],
    )
    write_dataset_dir(config.paths.history_dir, make_history_rows())

    run_train(config)

    assert (config.paths.artifact_root / ARTIFACT_MANIFEST_FILENAME).is_file()


def test_atomic_writer_preserves_previous_file_on_failure(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text("stable", encoding="utf-8")

    def _failing_writer(tmp_path_arg):
        tmp_path_arg.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        write_path_atomic(path, _failing_writer)

    assert path.read_text(encoding="utf-8") == "stable"
    assert not list(tmp_path.glob(".*.tmp"))
