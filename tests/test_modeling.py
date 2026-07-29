from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import polars as pl
import pytest
import torch
from lightning.pytorch.callbacks import Callback
from pydantic import ValidationError
from torch.utils.data import DataLoader

import fable.modeling as modeling
from fable.addresses import (
    artifact_checkpoint_path,
    corpus_blocks_path,
    corpus_directory,
    corpus_json_path,
    study_json_path,
)
from fable.config import (
    BlockWindow,
    CorpusDefinition,
    CorpusRequest,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    SelectedStudySource,
    TrainingDefinition,
    TrainRequest,
    TransformerDefinition,
    TransformerLstmDefinition,
    TuneRequest,
)
from fable.corpus import BlockFrame, Corpus, FinalizedAnchor
from fable.min_block_fee import MinBlockFeeOutput, TargetState, min_block_fee_loss
from fable.modeling import (
    ArtifactAssociation,
    load_artifact,
    train,
)
from fable.study import RetainedResult, Study
from fable.temporal import FeatureState, prepare_fit_history
from tests.helpers import modeling_method

ARTIFACT_ID = UUID("10000000-0000-4000-8000-000000000001")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")
STUDY_ID = UUID("40000000-0000-4000-8000-000000000001")
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
                    "effective_priority_fee_per_gas_p90": 2
                    * np.arange(blocks.size, dtype=np.int64),
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


def _write_corpus(storage_root: Path) -> None:
    corpus = _corpus()
    corpus_directory(storage_root, CORPUS_ID).mkdir(parents=True)
    corpus_json_path(storage_root, CORPUS_ID).write_text(
        json.dumps(
            {
                "request": corpus.request.model_dump(mode="json"),
                "finalized_anchor": corpus.finalized_anchor.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    corpus.blocks.to_polars().write_parquet(corpus_blocks_path(storage_root, CORPUS_ID))


def _candidate_request(method: Method) -> TuneRequest:
    return TuneRequest(
        workflow="tune",
        study_id=STUDY_ID,
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


def _train_request(artifact_id: UUID = ARTIFACT_ID) -> TrainRequest:
    return TrainRequest(
        workflow="train",
        artifact_id=artifact_id,
        source=SelectedStudySource(
            corpus_id=CORPUS_ID,
            study_id=STUDY_ID,
            study_result_index=0,
            experiment=_experiment(),
        ),
    )


def _write_selected_study(
    storage_root: Path,
    request: TrainRequest,
    method: Method,
) -> None:
    source = request.source
    study = Study(
        request=TuneRequest(
            workflow="tune",
            study_id=source.study_id,
            corpus_id=source.corpus_id,
            experiment=source.experiment,
            methods=(method,),
        ),
        trials=(
            RetainedResult(
                objective=0.5,
                selected_epoch=1,
                completed_epochs=1,
            ),
        ),
    )
    path = study_json_path(storage_root, source.study_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(study.model_dump_json(), encoding="utf-8")


def test_artifact_association_round_trips_strict_json() -> None:
    association = ArtifactAssociation(
        request=_train_request(),
        feature_state=FeatureState(
            means=(1.0, 2.0),
            standard_deviations=(0.5, 0.25),
        ),
        target_state=TargetState(mean=3.0, standard_deviation=0.75),
        method=modeling_method(),
    )

    assert (
        ArtifactAssociation.model_validate_json(
            association.model_dump_json(exclude_none=True),
            strict=True,
        )
        == association
    )


def test_artifact_association_rejects_feature_width_mismatch() -> None:
    target_state = TargetState(mean=3.0, standard_deviation=0.75)
    with pytest.raises(ValidationError, match="feature state width"):
        ArtifactAssociation(
            request=_train_request(),
            feature_state=FeatureState(
                means=(1.0,),
                standard_deviations=(0.5,),
            ),
            target_state=target_state,
            method=modeling_method(),
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
        request=_train_request(),
        feature_state=prepared.feature_state,
        target_state=prepared.target_state,
        method=modeling_method(),
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
        "validation_base_fee_optimality_gap": [],
    }

    def capture(name: str, value: torch.Tensor, **kwargs: Any) -> None:
        logged[name].append((value, kwargs))

    monkeypatch.setattr(module, "log", capture)
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            module.training_step(batch, batch_index)
            module.validation_step(batch, batch_index)

    for name in ("training_total_loss", "validation_total_loss"):
        entries = logged[name]
        assert [kwargs["batch_size"] for _, kwargs in entries] == [3, 1]
        assert all(value.dtype == torch.float64 for value, _ in entries)
        assert all(kwargs["on_step"] is False for _, kwargs in entries)
        assert all(kwargs["on_epoch"] is True for _, kwargs in entries)
        assert all(kwargs["logger"] is False for _, kwargs in entries)
        weighted = sum(float(value) * int(kwargs["batch_size"]) for value, kwargs in entries) / 4
        unweighted = sum(float(value) for value, _ in entries) / 2
        assert weighted == pytest.approx(expected)
        assert unweighted != pytest.approx(expected)

    gap_entries = logged["validation_base_fee_optimality_gap"]
    assert [kwargs["batch_size"] for _, kwargs in gap_entries] == [3, 1]
    output = module(complete["inputs"])
    actions = output.action_logits.argmax(dim=1)
    selected = complete["base_fees"].gather(1, actions.unsqueeze(1)).squeeze(1)
    minimum = complete["base_fees"].amin(dim=1)
    expected_gap = float(((selected - minimum).to(torch.float64) / minimum).mean())
    weighted_gap = (
        sum(float(value) * int(kwargs["batch_size"]) for value, kwargs in gap_entries) / 4
    )
    assert weighted_gap == pytest.approx(expected_gap)


def test_validation_logs_mean_base_fee_cost_over_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_fit_history(_corpus(), _experiment())
    association = ArtifactAssociation(
        request=_train_request(),
        feature_state=prepared.feature_state,
        target_state=prepared.target_state,
        method=modeling_method(),
    )
    module = modeling._FitModule(modeling._json_association(association)).eval()
    batch = {
        "inputs": torch.zeros((2, 3, 2)),
        "label": torch.tensor([0, 1]),
        "target": torch.zeros(2),
        "base_fees": torch.tensor([[4, 2], [3, 5]]),
    }
    output = MinBlockFeeOutput(
        action_logits=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        minimum_fee_z=torch.zeros(2),
    )
    logged: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}

    monkeypatch.setattr(module, "forward", lambda _inputs: output)
    monkeypatch.setattr(
        module,
        "log",
        lambda name, value, **kwargs: logged.__setitem__(name, (value, kwargs)),
    )

    module.validation_step(batch, 0)

    selected = batch["base_fees"][torch.arange(2), torch.tensor([1, 0])]
    minimum = batch["base_fees"].amin(dim=1)
    expected = ((selected - minimum) / minimum).mean(dtype=torch.float64)
    value, options = logged["validation_base_fee_optimality_gap"]
    torch.testing.assert_close(value, expected)
    assert options["batch_size"] == 2
    assert options["on_step"] is False
    assert options["on_epoch"] is True


def test_gradient_clipping_uses_trainer_value_and_rejects_nonfinite() -> None:
    association = ArtifactAssociation(
        request=_train_request(),
        feature_state=FeatureState(
            means=(1.0, 2.0),
            standard_deviations=(0.5, 0.25),
        ),
        target_state=TargetState(mean=3.0, standard_deviation=0.75),
        method=modeling_method(),
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
    method = _definition(model).method
    request = _train_request(artifact_id)
    _write_corpus(tmp_path)
    _write_selected_study(tmp_path, request, method)
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

    train(request, tmp_path)
    association, loaded_model = load_artifact(tmp_path, artifact_id)

    assert association.request == request
    assert checkpoint == tmp_path / "artifacts" / f"{artifact_id}.ckpt"
    assert checkpoint.is_file()
    if isinstance(model, LstmDefinition):
        assert cleanup_attempted
        assert hidden.is_file()
        contents = checkpoint.read_bytes()
        with pytest.raises(FileExistsError):
            train(request, tmp_path)
        assert checkpoint.read_bytes() == contents
    else:
        assert not hidden.exists()

    application_history = prepare_fit_history(_corpus(), _experiment())
    batches = list(DataLoader(application_history.training, batch_size=3, shuffle=False))
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
    _write_corpus(tmp_path)
    real_trainer: Any = modeling.pl.Trainer
    fit_kwargs: list[dict[str, object]] = []

    class InterruptAfterEpoch(Callback):
        def on_train_batch_start(
            self,
            trainer: Any,
            *_args: object,
        ) -> None:
            if trainer.current_epoch == 1:
                raise RuntimeError("simulated interruption")

    class TrainerSpy:
        def __init__(self, **kwargs: Any) -> None:
            kwargs["accelerator"] = "cpu"
            if not fit_kwargs:
                kwargs["callbacks"].append(InterruptAfterEpoch())
            self._trainer = real_trainer(**kwargs)

        def fit(self, module: Any, **kwargs: Any) -> None:
            fit_kwargs.append(dict(kwargs))
            self._trainer.fit(module, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self._trainer, name)

    monkeypatch.setattr(modeling.pl, "Trainer", TrainerSpy)
    scratch = tmp_path / "candidate"

    def progress() -> list[tuple[int, float, float]]:
        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("epoch=")]
        return [
            (
                int(epoch.removeprefix("epoch=")),
                float(loss.removeprefix("validation_total_loss=")),
                float(gap.removeprefix("validation_base_fee_optimality_gap=")),
            )
            for epoch, loss, gap in (line.split() for line in lines)
        ]

    with pytest.raises(RuntimeError, match="simulated interruption"):
        modeling.fit_candidate(request, 0, tmp_path, scratch)
    first_progress = progress()
    assert (scratch / "last.ckpt").is_file()

    second = modeling.fit_candidate(request, 0, tmp_path, scratch)
    second_progress = progress()

    assert first_progress == []
    assert [epoch for epoch, _, _ in second_progress] == [2, 4]
    validation_progress = first_progress + second_progress
    assert all(math.isfinite(loss) and math.isfinite(gap) for _, loss, gap in validation_progress)
    assert second.completed_epochs == method.fit.max_epochs
    assert second.objective == min(gap for _, _, gap in validation_progress)
    assert second.selected_epoch == next(
        epoch for epoch, _, gap in validation_progress if gap == second.objective
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
