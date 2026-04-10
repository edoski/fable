import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from spice_temporal.artifacts import (
    ARTIFACT_MANIFEST_FILENAME,
    MODEL_STATE_FILENAME,
    build_training_artifact_manifest,
    load_training_artifact,
    write_training_artifact,
)
from spice_temporal.config import ChainConfig, ModelConfig, SplitConfig, TrainingConfig
from spice_temporal.pipeline import run_training
from spice_temporal.specs import TrainingSpec
from tests.support import make_history_block


class ArtifactRoundTripTestCase(unittest.TestCase):
    def test_write_and_load_training_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            history_blocks_path = tmp_path / "history.json"
            history_blocks_path.write_text(
                json.dumps([asdict(make_history_block(index)) for index in range(420)]),
                encoding="utf-8",
            )
            spec = TrainingSpec(
                chain=ChainConfig(
                    name="ethereum",
                    chain_id=1,
                    block_time_seconds=12.0,
                    history_days=1,
                ),
                model=ModelConfig(family="lstm"),
                max_delay_seconds=36,
                lookback_seconds=600,
                target_anchor_count=64,
                split=SplitConfig(),
                training=TrainingConfig(max_epochs=2, effective_batch_size=8, device="cpu"),
            )
            result = run_training(
                history_block_path=history_blocks_path,
                spec=spec,
            )
            manifest = build_training_artifact_manifest(
                result.prepared,
                spec=spec,
            )
            artifact_dir = tmp_path / "artifact"
            write_training_artifact(artifact_dir, manifest=manifest, model=result.model)
            loaded = load_training_artifact(artifact_dir)

            self.assertTrue((artifact_dir / ARTIFACT_MANIFEST_FILENAME).exists())
            self.assertTrue((artifact_dir / MODEL_STATE_FILENAME).exists())
            self.assertEqual(loaded.manifest.max_delay_seconds, 36)
            self.assertEqual(loaded.manifest.action_count, result.prepared.action_count)
            with torch.no_grad():
                first_train_index = int(result.prepared.split_indices.train[0])
                anchor_row = int(result.prepared.store.anchor_row_indices[first_train_index])
                sample = torch.from_numpy(
                    result.prepared.store.feature_matrix[
                        anchor_row - result.prepared.geometry.lookback_steps + 1 : anchor_row + 1
                    ]
                ).unsqueeze(0)
                outputs = loaded.model(sample)
            self.assertEqual(outputs.logits.shape[-1], result.prepared.action_count)


if __name__ == "__main__":
    unittest.main()
