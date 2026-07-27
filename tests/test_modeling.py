from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import polars as pl
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

import fable.modeling as modeling
from fable.addresses import (
    artifact_checkpoint_path,
)
from fable.config import (
    BaselineSource,
    BlockWindow,
    CorpusDefinition,
    CorpusRequest,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    TrainingDefinition,
    TrainRequest,
    TransformerDefinition,
    TransformerLstmDefinition,
    TuneRequest,
)
from fable.corpus import BlockFrame, Corpus, FinalizedAnchor
from fable.min_block_fee import TargetState, min_block_fee_loss
from fable.modeling import (
    ArtifactAssociation,
    load_artifact,
    train,
)
from fable.temporal.features import FeatureState
from fable.temporal.history import prepare_fit_history
from tests.helpers import modeling_method

ARTIFACT_ID = UUID("10000000-0000-4000-8000-000000000001")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")
_BASE_FEES = np.array(
    [11, 12, 10, 4, 9, 4, 8, 3, 5, 6, 10, 6, 2, 2],
    dtype=np.int64,
)


@pytest.fixture(autouse=True)
def _use_single_process_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modeling._runtime, "NUM_WORKERS", 0)


def _experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=BlockWindow(
            first_parent_block=12,
            last_parent_block=15,
        ),
        validation_window=BlockWindow(
            first_parent_block=20,
            last_parent_block=21,
        ),
        context_blocks=3,
        horizon_blocks=2,
        ordered_features=("log_base_fee_per_gas", "gas_utilization"),
    )


def _request() -> TrainRequest:
    definition = TrainingDefinition(
        experiment=_experiment(),
        method=modeling_method(),
    )
    return TrainRequest(
        workflow="train",
        artifact_id=ARTIFACT_ID,
        source=BaselineSource(
            kind="baseline",
            corpus_id=CORPUS_ID,
            training_definition=definition,
        ),
    )


def _corpus() -> Corpus:
    blocks = np.arange(10, 24, dtype=np.int64)
    request = _corpus_request()
    return Corpus(
        request=request,
        finalized_anchor=FinalizedAnchor(block_number=23, block_hash="a" * 64),
        blocks=BlockFrame(
            pl.DataFrame(
                {
                    "block_number": blocks,
                    "timestamp": blocks * 11,
                    "chain_id": np.ones(blocks.size, dtype=np.int64),
                    "base_fee_per_gas": _BASE_FEES,
                    "gas_used": 30 + np.arange(blocks.size, dtype=np.int64),
                    "gas_limit": np.full(blocks.size, 100, dtype=np.int64),
                    "tx_count": 4 + np.arange(blocks.size, dtype=np.int64),
                    "effective_priority_fee_per_gas_p50": np.arange(blocks.size, dtype=np.int64),
                }
            ),
            request.definition,
        ),
    )


def _corpus_request() -> CorpusRequest:
    return CorpusRequest(
        corpus_id=CORPUS_ID,
        definition=CorpusDefinition(chain_id=1, first_block=10, last_block=23),
    )


def _candidate_request(method: Method) -> TuneRequest:
    return TuneRequest(
        workflow="tune",
        study_id=UUID("40000000-0000-4000-8000-000000000001"),
        corpus_id=CORPUS_ID,
        experiment=_experiment(),
        methods=(method,),
    )


def _definition(
    model: LstmDefinition | TransformerDefinition | TransformerLstmDefinition,
) -> TrainingDefinition:
    return TrainingDefinition(
        experiment=_experiment(),
        method=Method(
            model=model,
            fit=FitMethod(
                learning_rate=0.002,
                weight_decay=0.003,
                accumulation=1,
                gradient_clip_norm=0.8,
                seed=29,
                max_epochs=1,
                validate_every_completed_epoch=1,
                patience=0,
                min_delta=0.0,
            ),
        ),
    )


def _train_request(
    artifact_id: UUID,
    model: LstmDefinition | TransformerDefinition | TransformerLstmDefinition,
) -> TrainRequest:
    return TrainRequest(
        workflow="train",
        artifact_id=artifact_id,
        source=BaselineSource(
            kind="baseline",
            corpus_id=CORPUS_ID,
            training_definition=_definition(model),
        ),
    )


def test_artifact_association_round_trips_strict_json() -> None:
    association = ArtifactAssociation(
        request=_request(),
        feature_state=FeatureState(
            means=(1.0, 2.0),
            standard_deviations=(0.5, 0.25),
        ),
        target_state=TargetState(mean=3.0, standard_deviation=0.75),
    )

    assert (
        ArtifactAssociation.model_validate_json(
            association.model_dump_json(exclude_none=True),
            strict=True,
        )
        == association
    )


def test_artifact_association_rejects_only_owned_mismatches() -> None:
    feature_state = FeatureState(
        means=(1.0, 2.0),
        standard_deviations=(0.5, 0.25),
    )
    target_state = TargetState(mean=3.0, standard_deviation=0.75)
    method = modeling_method()

    with pytest.raises(ValidationError, match="baseline artifacts"):
        ArtifactAssociation(
            request=_request(),
            feature_state=feature_state,
            target_state=target_state,
            study_result_index=0,
            method=method,
        )
    with pytest.raises(ValidationError, match="feature state width"):
        ArtifactAssociation(
            request=_request(),
            feature_state=FeatureState(
                means=(1.0,),
                standard_deviations=(0.5,),
            ),
            target_state=target_state,
        )


def test_transformer_encoder_layers_have_independent_matrix_initialization() -> None:
    torch.manual_seed(71)
    encoder = modeling._encoder(
        width=4,
        heads=2,
        feedforward=7,
        layers=2,
        dropout=0.1,
    )
    matrices = [
        [parameter for parameter in layer.parameters() if parameter.ndim > 1]
        for layer in encoder.layers
    ]

    assert matrices[0]
    assert all(
        not torch.equal(first, second)
        for first, second in zip(matrices[0], matrices[1], strict=True)
    )


def test_epoch_logs_weight_short_batches_in_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_fit_history(_corpus(), _experiment())
    association = ArtifactAssociation(
        request=_request(),
        feature_state=prepared.feature_state,
        target_state=prepared.target_state,
    )
    torch.manual_seed(89)
    module = modeling._FitModule(modeling._json_association(association)).eval()
    batches = list(DataLoader(prepared.training, batch_size=3, shuffle=False))
    complete = next(iter(DataLoader(prepared.training, batch_size=4, shuffle=False)))
    with torch.no_grad():
        expected = float(module._loss(complete).mean_total)
    logged: dict[str, list[tuple[torch.Tensor, dict[str, Any]]]] = {
        "training_total_loss": [],
        "validation_total_loss": [],
    }

    def capture(name: str, value: torch.Tensor, **kwargs: Any) -> None:
        logged[name].append((value, kwargs))

    monkeypatch.setattr(module, "log", capture)
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            module.training_step(batch, batch_index)
            module.validation_step(batch, batch_index)

    for entries in logged.values():
        assert [kwargs["batch_size"] for _, kwargs in entries] == [3, 1]
        assert all(value.dtype == torch.float64 for value, _ in entries)
        assert all(kwargs["on_step"] is False for _, kwargs in entries)
        assert all(kwargs["on_epoch"] is True for _, kwargs in entries)
        assert all(kwargs["logger"] is False for _, kwargs in entries)
        weighted = sum(float(value) * int(kwargs["batch_size"]) for value, kwargs in entries) / 4
        unweighted = sum(float(value) for value, _ in entries) / 2
        assert weighted == pytest.approx(expected)
        assert unweighted != pytest.approx(expected)


def test_gradient_clipping_uses_trainer_value_and_rejects_nonfinite() -> None:
    association = ArtifactAssociation(
        request=_request(),
        feature_state=FeatureState(
            means=(1.0, 2.0),
            standard_deviations=(0.5, 0.25),
        ),
        target_state=TargetState(mean=3.0, standard_deviation=0.75),
    )
    module = modeling._FitModule(modeling._json_association(association))
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    parameter.grad = torch.tensor([3.0, 4.0])
    module.configure_gradient_clipping(optimizer, 2.0, "norm")
    torch.testing.assert_close(parameter.grad.norm(), torch.tensor(2.0))

    parameter.grad = torch.tensor([3.0, 4.0])
    module.configure_gradient_clipping(optimizer, 0.0, "norm")
    torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 4.0]))

    parameter.grad = torch.tensor([math.inf, 0.0])
    with pytest.raises(RuntimeError, match="non-finite"):
        module.configure_gradient_clipping(optimizer, 0.0, "norm")


@pytest.mark.parametrize(
    ("artifact_id", "model"),
    [
        (
            UUID("30000000-0000-4000-8000-000000000001"),
            LstmDefinition(
                family="lstm",
                hidden=5,
                layers=1,
                head_hidden=3,
                dropout=0.1,
            ),
        ),
        (
            UUID("30000000-0000-4000-8000-000000000002"),
            TransformerDefinition(
                family="transformer",
                model_width=4,
                attention_heads=2,
                transformer_layers=1,
                feedforward_width=7,
                head_hidden=3,
                dropout=0.1,
            ),
        ),
        (
            UUID("30000000-0000-4000-8000-000000000003"),
            TransformerLstmDefinition(
                family="transformer_lstm",
                model_width=4,
                attention_heads=2,
                transformer_layers=1,
                feedforward_width=7,
                lstm_hidden=5,
                lstm_layers=1,
                head_hidden=3,
                dropout=0.1,
            ),
        ),
    ],
)
def test_all_three_models_train_load_and_apply_direct_loss(
    artifact_id: UUID,
    model: LstmDefinition | TransformerDefinition | TransformerLstmDefinition,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_fit_history(_corpus(), _experiment())
    request = _train_request(artifact_id, model)
    real_trainer: Any = modeling.pl.Trainer

    def cpu_trainer(**kwargs: Any) -> Any:
        kwargs["accelerator"] = "cpu"
        return real_trainer(**kwargs)

    monkeypatch.setattr(modeling.pl, "Trainer", cpu_trainer)

    checkpoint = artifact_checkpoint_path(tmp_path, artifact_id)
    hidden = checkpoint.with_name(f".{checkpoint.name}")
    cleanup_attempted = False
    real_unlink = Path.unlink
    if isinstance(model, LstmDefinition):

        def fail_hidden_cleanup(
            path: Path,
            missing_ok: bool = False,
        ) -> None:
            nonlocal cleanup_attempted
            if path == hidden:
                cleanup_attempted = True
                raise OSError("cleanup failed")
            real_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", fail_hidden_cleanup)

    train(request, prepared, tmp_path)
    association, loaded_model = load_artifact(tmp_path, artifact_id)

    assert association.request == request
    assert checkpoint == tmp_path / "artifacts" / f"{artifact_id}.ckpt"
    assert checkpoint.is_file()
    if isinstance(model, LstmDefinition):
        assert cleanup_attempted
        assert hidden.is_file()
        contents = checkpoint.read_bytes()
        with pytest.raises(FileExistsError):
            train(request, prepared, tmp_path)
        assert checkpoint.read_bytes() == contents
    else:
        assert not hidden.exists()

    batches = list(DataLoader(prepared.training, batch_size=3, shuffle=False))
    for batch in batches:
        output = loaded_model(batch["inputs"])
        assert output.action_logits.shape == (batch["inputs"].shape[0], 2)
        assert output.minimum_fee_z.shape == (batch["inputs"].shape[0],)
        assert torch.isfinite(output.action_logits).all()
        assert torch.isfinite(output.minimum_fee_z).all()
        loss = min_block_fee_loss(
            output,
            label=batch["label"],
            target=batch["target"],
        )
        assert torch.isfinite(loss.mean_total)

    if isinstance(model, LstmDefinition):
        mismatched_id = UUID("30000000-0000-4000-8000-000000000009")
        checkpoint.rename(artifact_checkpoint_path(tmp_path, mismatched_id))
        with pytest.raises(ValueError, match="embedded artifact ID"):
            load_artifact(tmp_path, mismatched_id)


def test_full_checkpoint_resume_preserves_selection_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    method = Method(
        model=LstmDefinition(
            family="lstm",
            hidden=5,
            layers=1,
            head_hidden=3,
            dropout=0.0,
        ),
        fit=FitMethod(
            learning_rate=0.004,
            weight_decay=0.002,
            accumulation=1,
            gradient_clip_norm=0.0,
            seed=37,
            max_epochs=4,
            validate_every_completed_epoch=2,
            patience=10,
            min_delta=0.0,
        ),
    )
    request = _candidate_request(method)
    prepared = prepare_fit_history(_corpus(), request.experiment)
    real_trainer: Any = modeling.pl.Trainer
    fit_kwargs: list[dict[str, object]] = []

    class TrainerSpy:
        def __init__(self, **kwargs: Any) -> None:
            kwargs["accelerator"] = "cpu"
            if not fit_kwargs:
                kwargs["max_epochs"] = 2
            self._trainer = real_trainer(**kwargs)

        def fit(self, module: Any, **kwargs: Any) -> None:
            fit_kwargs.append(dict(kwargs))
            self._trainer.fit(module, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self._trainer, name)

    monkeypatch.setattr(modeling.pl, "Trainer", TrainerSpy)
    scratch = tmp_path / "candidate"
    association = modeling._CandidateAssociation(
        request=request,
        method=method,
        feature_state=prepared.feature_state,
        target_state=prepared.target_state,
    )

    def progress() -> list[tuple[int, float]]:
        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("epoch=")]
        return [
            (
                int(epoch.removeprefix("epoch=")),
                float(loss.removeprefix("validation_total_loss=")),
            )
            for epoch, loss in (line.split() for line in lines)
        ]

    first = modeling._fit(association, prepared, scratch)
    first_progress = progress()
    second = modeling._fit(association, prepared, scratch)
    second_progress = progress()

    assert [epoch for epoch, _ in first_progress] == [2]
    assert [epoch for epoch, _ in second_progress] == [4]
    validation_progress = first_progress + second_progress
    assert all(math.isfinite(loss) for _, loss in validation_progress)
    assert first.objective == first_progress[0][1]
    assert first.selected_epoch == 2
    assert first.completed_epochs == 2
    assert second.completed_epochs == method.fit.max_epochs
    assert second.objective == min(loss for _, loss in validation_progress)
    assert second.selected_epoch == next(
        epoch for epoch, loss in validation_progress if loss == second.objective
    )
    assert fit_kwargs[0]["ckpt_path"] is None
    assert fit_kwargs[1]["ckpt_path"] == scratch / "last.ckpt"
    assert "weights_only" not in fit_kwargs[1]
    best_path = scratch / f"best-{second.selected_epoch - 1:02d}.ckpt"
    assert sorted(path.name for path in scratch.iterdir()) == [best_path.name, "last.ckpt"]
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    last_checkpoint = torch.load(scratch / "last.ckpt", map_location="cpu", weights_only=True)
    assert "optimizer_states" not in best_checkpoint
    assert "optimizer_states" in last_checkpoint
