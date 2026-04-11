from __future__ import annotations

import torch
import torch.nn.functional as F

from spice.core.constants import ARTIFACT_MANIFEST_FILENAME, MODEL_STATE_FILENAME
from spice.modeling._runtime import build_sequence_loader
from spice.modeling.artifacts import load_training_artifact
from spice.modeling.evaluation import compute_temporal_batch_metrics
from spice.modeling.execution import run_persisted_training
from spice.modeling.models import ModelOutputs
from spice.modeling.pipeline import TrainingSpec
from spice.modeling.reporting import TrainingRunReport
from spice.modeling.torch_datasets import SequenceBatch
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
            anchor_count=config.dataset.sampling.anchor_count,
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
    assert report.dataset_id == config.dataset.id
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


def test_sequence_loader_collates_to_named_tuple_batch(tmp_path) -> None:
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
            anchor_count=config.dataset.sampling.anchor_count,
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
        device=torch.device("cpu"),
    )
    batch = next(iter(loader))

    assert isinstance(batch, SequenceBatch)
    assert batch.inputs.ndim == 3
    assert batch.class_label.ndim == 1
