from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from uuid import UUID

import polars as pl
import pytest
from torch import nn

import experiments.context_history as context_history_module
from experiments.context_history import write_context_history_evidence
from spice.addresses import evaluation_directory, evaluation_json_path
from spice.config import (
    AdamWMethod,
    BaselineSource,
    CorpusDefinition,
    CorpusRequest,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LossDefinition,
    LstmCapacity,
    LstmDefinition,
    LstmMethod,
    OriginWindow,
    SelectedStudySource,
    TrainingDefinition,
    TrainRequest,
    TransformerDefinition,
)
from spice.corpus import Corpus, FinalizedAnchor
from spice.min_block_fee import TargetState
from spice.modeling.artifacts import ArtifactAssociation
from spice.temporal.features import FeatureState

_CHAINS = (1, 137, 43_114)
_CONTEXTS = (50, 100, 200, 400)
_HORIZONS = (2, 3, 4, 5, 10, 15, 30, 50, 100, 200)
_METRICS = (
    "earliest_hindsight_label_cross_entropy_loss",
    "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mae",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mse",
    "multitask_total_loss",
    "earliest_hindsight_label_accuracy",
    "earliest_hindsight_label_macro_f1",
    "finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0",
    "finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0",
    "finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0",
    "signed_captured_hindsight_opportunity_ratio",
    "mean_origin_target_base_fee_per_gas_savings_fraction_vs_immediate_k0",
    "mean_origin_selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k",
    "mean_origin_immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k",
    "harmful_action_rate",
    "mean_extra_wait_block_opportunities_vs_immediate_k0",
    "mean_selected_action_wait_seconds",
    "mean_full_horizon_elapsed_seconds",
)
_COLUMNS = (
    "evaluation_id",
    "artifact_id",
    "corpus_id",
    "chain_id",
    "model_family",
    "context_blocks",
    "horizon_blocks",
    "ordered_features",
    "classification_loss",
    "training_first_parent_block",
    "training_last_parent_block",
    "training_origin_count",
    "training_examples_per_epoch",
    "training_minibatches_per_epoch",
    "training_optimizer_updates_per_epoch",
    "training_context_span_seconds_minimum",
    "training_context_span_seconds_median",
    "training_context_span_seconds_mean",
    "training_context_span_seconds_maximum",
    "validation_first_parent_block",
    "validation_last_parent_block",
    "validation_origin_count",
    "validation_context_span_seconds_minimum",
    "validation_context_span_seconds_median",
    "validation_context_span_seconds_mean",
    "validation_context_span_seconds_maximum",
    "testing_first_parent_block",
    "testing_last_parent_block",
    "testing_origin_count",
    "testing_context_span_seconds_minimum",
    "testing_context_span_seconds_median",
    "testing_context_span_seconds_mean",
    "testing_context_span_seconds_maximum",
    *(name for metric in _METRICS for name in (metric, f"{metric}_delta_vs_same_chain_c200")),
    "final_k_horizon_blocks",
    "final_k_artifact_ids",
)


@dataclass
class _Matrix:
    storage_root: Path
    context_ids: tuple[UUID, ...]
    context_requests: tuple[EvaluateRequest, ...]
    final_ids: tuple[UUID, ...]
    artifacts: dict[UUID, ArtifactAssociation]
    corpora: dict[UUID, Corpus]
    reductions: dict[UUID, pl.DataFrame]
    artifact_calls: list[UUID]
    corpus_calls: list[UUID]
    reduction_calls: list[UUID]


def _uuid(namespace: int, index: int) -> UUID:
    return UUID(f"{namespace:08x}-0000-4000-8000-{index:012x}")


def _fit(*, accumulation: int = 1) -> FitMethod:
    return FitMethod(
        accumulation=accumulation,
        gradient_clip_norm=1.0,
        scheduler="none",
        seed=2026,
        max_epochs=36,
        validate_every_completed_epoch=1,
        patience=8,
        min_delta=0.0,
        improvement="strict_lower",
        restore="earliest_best",
    )


def _optimizer() -> AdamWMethod:
    return AdamWMethod(learning_rate=0.0003, weight_decay=0.0001)


def _loss() -> LossDefinition:
    return LossDefinition(
        classification_algorithm="cross_entropy",
        classification_weighting="unweighted",
        regression_algorithm="smooth_l1",
        regression_threshold=1.0,
        classification_scale=1.0,
        regression_scale=1.0,
    )


def _features(chain_id: int) -> tuple[str, ...]:
    common = ("log_base_fee_per_gas", "gas_utilization")
    if chain_id == 1:
        common += ("log_exact_forming_base_fee_per_gas",)
    return (*common, "hour_sin", "hour_cos")


def _experiment(first_block: int, context: int, horizon: int, chain_id: int) -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=OriginWindow(
            role="training",
            first_parent_block=first_block + context - 1,
            last_parent_block=first_block + 448,
        ),
        validation_window=OriginWindow(
            role="validation",
            first_parent_block=first_block + 700,
            last_parent_block=first_block + 703,
        ),
        context_blocks=context,
        horizon_blocks=horizon,
        ordered_features=_features(chain_id),
        loss=_loss(),
    )


def _corpus(corpus_id: UUID, chain_id: int, first_block: int) -> Corpus:
    offsets = list(range(1_001))
    timestamps = list(accumulate([0, *(0 if offset % 2 else 10 for offset in offsets[1:])]))
    return Corpus(
        request=CorpusRequest(
            corpus_id=corpus_id,
            definition=CorpusDefinition(
                chain_id=chain_id,
                first_block=first_block,
                last_block=first_block + 1_000,
            ),
        ),
        finalized_anchor=FinalizedAnchor(block_number=first_block + 1_000, block_hash="a" * 64),
        blocks=pl.DataFrame(
            {
                "block_number": range(first_block, first_block + 1_001),
                "timestamp": timestamps,
                "chain_id": [chain_id] * 1_001,
                "base_fee_per_gas": [100] * 1_001,
                "gas_used": [50] * 1_001,
                "gas_limit": [100] * 1_001,
                "tx_count": [1] * 1_001,
            },
            schema={
                "block_number": pl.Int64,
                "timestamp": pl.Int64,
                "chain_id": pl.Int64,
                "base_fee_per_gas": pl.Int64,
                "gas_used": pl.Int64,
                "gas_limit": pl.Int64,
                "tx_count": pl.Int64,
            },
        ),
    )


def _context_association(
    artifact_id: UUID,
    corpus_id: UUID,
    experiment: ExperimentSemantics,
) -> ArtifactAssociation:
    definition = TrainingDefinition(
        experiment=experiment,
        model=LstmDefinition(family="lstm", hidden=4, layers=1, head_hidden=3, dropout=0.0),
        optimizer=_optimizer(),
        training_batch=64,
        fit=_fit(),
    )
    return ArtifactAssociation(
        request=TrainRequest(
            workflow="train",
            artifact_id=artifact_id,
            source=BaselineSource(
                kind="baseline",
                corpus_id=corpus_id,
                training_definition=definition,
            ),
        ),
        feature_state=FeatureState(
            means=(0.0,) * len(experiment.ordered_features),
            standard_deviations=(1.0,) * len(experiment.ordered_features),
        ),
        target_state=TargetState(mean=0.0, standard_deviation=1.0),
    )


def _final_association(
    artifact_id: UUID,
    corpus_id: UUID,
    study_id: UUID,
    experiment: ExperimentSemantics,
) -> ArtifactAssociation:
    method = LstmMethod(
        family="lstm",
        capacity=LstmCapacity(hidden=8, layers=2, head_hidden=5),
        dropout=0.1,
        optimizer=AdamWMethod(learning_rate=0.0007, weight_decay=0.0002),
        training_batch=32,
        fit=_fit(accumulation=2),
    )
    return ArtifactAssociation(
        request=TrainRequest(
            workflow="train",
            artifact_id=artifact_id,
            source=SelectedStudySource(
                kind="selected_study",
                corpus_id=corpus_id,
                study_id=study_id,
                study_result_index=0,
                experiment=experiment,
            ),
        ),
        feature_state=FeatureState(
            means=(0.0,) * len(experiment.ordered_features),
            standard_deviations=(1.0,) * len(experiment.ordered_features),
        ),
        target_state=TargetState(mean=0.0, standard_deviation=1.0),
        study_result_index=0,
        method=method,
    )


def _matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Matrix:
    context_ids: list[UUID] = []
    context_requests: list[EvaluateRequest] = []
    final_ids: list[UUID] = []
    artifacts: dict[UUID, ArtifactAssociation] = {}
    corpora: dict[UUID, Corpus] = {}
    reductions: dict[UUID, pl.DataFrame] = {}

    for chain_index, chain_id in enumerate(_CHAINS):
        first_block = 1_000 * (chain_index + 1)
        corpus_id = _uuid(3, chain_index + 1)
        study_id = _uuid(4, chain_index + 1)
        corpora[corpus_id] = _corpus(corpus_id, chain_id, first_block)
        for context_index, context in enumerate(_CONTEXTS):
            evaluation_id = _uuid(1, chain_index * 10 + context_index + 1)
            artifact_id = _uuid(2, chain_index * 10 + context_index + 1)
            experiment = _experiment(first_block, context, 5, chain_id)
            context_ids.append(evaluation_id)
            artifacts[artifact_id] = _context_association(artifact_id, corpus_id, experiment)
            request = EvaluateRequest(
                workflow="evaluate",
                evaluation_id=evaluation_id,
                artifact_id=artifact_id,
                corpus_id=corpus_id,
                window=OriginWindow(
                    role="testing",
                    first_parent_block=first_block + 900,
                    last_parent_block=first_block + 903,
                ),
            )
            context_requests.append(request)
            evaluation_directory(tmp_path, evaluation_id).mkdir(parents=True)
            evaluation_json_path(tmp_path, evaluation_id).write_text(
                request.model_dump_json(),
                encoding="utf-8",
            )
            metric_values = {
                metric: (
                    None
                    if metric == "signed_captured_hindsight_opportunity_ratio"
                    and chain_index == 0
                    and context == 50
                    else float(chain_index * 100 + metric_index + context / 100)
                )
                for metric_index, metric in enumerate(_METRICS)
            }
            reductions[evaluation_id] = pl.DataFrame(
                {
                    "evaluation_id": [str(evaluation_id)],
                    **{key: [value] for key, value in metric_values.items()},
                }
            )

        for horizon_index, horizon in enumerate(_HORIZONS):
            artifact_id = _uuid(5, chain_index * 20 + horizon_index + 1)
            final_ids.append(artifact_id)
            artifacts[artifact_id] = _final_association(
                artifact_id,
                corpus_id,
                study_id,
                _experiment(first_block, 200, horizon, chain_id),
            )

    artifact_calls: list[UUID] = []
    corpus_calls: list[UUID] = []
    reduction_calls: list[UUID] = []

    def load_artifact(
        _storage_root: Path,
        artifact_id: UUID,
    ) -> tuple[ArtifactAssociation, nn.Module]:
        artifact_calls.append(artifact_id)
        return artifacts[artifact_id], nn.Identity().eval()

    def load_corpus(_storage_root: Path, corpus_id: UUID) -> Corpus:
        corpus_calls.append(corpus_id)
        return corpora[corpus_id]

    def reduce_evaluation(_storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
        reduction_calls.append(evaluation_id)
        return reductions[evaluation_id]

    monkeypatch.setattr(context_history_module, "load_artifact", load_artifact)
    monkeypatch.setattr(context_history_module, "load_corpus", load_corpus)
    monkeypatch.setattr(context_history_module, "reduce_evaluation", reduce_evaluation)
    return _Matrix(
        storage_root=tmp_path,
        context_ids=tuple(context_ids),
        context_requests=tuple(context_requests),
        final_ids=tuple(final_ids),
        artifacts=artifacts,
        corpora=corpora,
        reductions=reductions,
        artifact_calls=artifact_calls,
        corpus_calls=corpus_calls,
        reduction_calls=reduction_calls,
    )


def _write_request(matrix: _Matrix, request: EvaluateRequest) -> None:
    evaluation_json_path(matrix.storage_root, request.evaluation_id).write_text(
        request.model_dump_json(),
        encoding="utf-8",
    )


def _replace_context_definition(
    matrix: _Matrix,
    index: int,
    definition: TrainingDefinition,
) -> None:
    request = matrix.context_requests[index]
    current = matrix.artifacts[request.artifact_id]
    matrix.artifacts[request.artifact_id] = ArtifactAssociation(
        request=TrainRequest(
            workflow="train",
            artifact_id=request.artifact_id,
            source=BaselineSource(
                kind="baseline",
                corpus_id=request.corpus_id,
                training_definition=definition,
            ),
        ),
        feature_state=current.feature_state,
        target_state=current.target_state,
        classification_state=current.classification_state,
    )


def test_write_context_history_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix(tmp_path, monkeypatch)
    destination = tmp_path / "context-history.tsv"

    write_context_history_evidence(
        matrix.storage_root,
        matrix.context_ids,
        matrix.final_ids,
        destination,
    )

    assert matrix.reduction_calls == list(matrix.context_ids)
    assert matrix.artifact_calls == [
        *(request.artifact_id for request in matrix.context_requests),
        *matrix.final_ids,
    ]
    assert matrix.corpus_calls == list(matrix.corpora)
    assert not destination.with_name(f".{destination.name}").exists()

    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 12
    assert tuple(rows[0]) == _COLUMNS

    expected_schema = pl.Schema(
        {
            **{name: pl.String for name in _COLUMNS[:3]},
            "chain_id": pl.Int64,
            "model_family": pl.String,
            "context_blocks": pl.Int64,
            "horizon_blocks": pl.Int64,
            "ordered_features": pl.String,
            "classification_loss": pl.String,
            **{name: pl.Int64 for name in _COLUMNS[9:16]},
            "training_context_span_seconds_median": pl.Float64,
            "training_context_span_seconds_mean": pl.Float64,
            "training_context_span_seconds_maximum": pl.Int64,
            **{name: pl.Int64 for name in _COLUMNS[19:23]},
            "validation_context_span_seconds_median": pl.Float64,
            "validation_context_span_seconds_mean": pl.Float64,
            "validation_context_span_seconds_maximum": pl.Int64,
            **{name: pl.Int64 for name in _COLUMNS[26:30]},
            "testing_context_span_seconds_median": pl.Float64,
            "testing_context_span_seconds_mean": pl.Float64,
            "testing_context_span_seconds_maximum": pl.Int64,
            **{name: pl.Float64 for name in _COLUMNS[33:69]},
            "final_k_horizon_blocks": pl.String,
            "final_k_artifact_ids": pl.String,
        }
    )
    assert pl.read_csv(destination, separator="\t", null_values="").schema == expected_schema

    for row_index, row in enumerate(rows):
        chain_index, context_index = divmod(row_index, len(_CONTEXTS))
        chain_id = _CHAINS[chain_index]
        context = _CONTEXTS[context_index]
        first_block = 1_000 * (chain_index + 1)
        assert int(row["chain_id"]) == chain_id
        assert row["model_family"] == "lstm"
        assert int(row["context_blocks"]) == context
        assert int(row["horizon_blocks"]) == 5
        assert json.loads(row["ordered_features"]) == list(_features(chain_id))
        assert row["classification_loss"] == "unweighted"
        assert int(row["training_first_parent_block"]) == first_block + context - 1
        assert int(row["training_last_parent_block"]) == first_block + 448
        assert int(row["training_origin_count"]) == 450 - context
        assert int(row["training_examples_per_epoch"]) == 450 - context
        expected_updates = math.ceil((450 - context) / 64)
        assert int(row["training_minibatches_per_epoch"]) == expected_updates
        assert int(row["training_optimizer_updates_per_epoch"]) == expected_updates
        for role in ("training", "validation", "testing"):
            assert int(row[f"{role}_context_span_seconds_minimum"]) == context * 5 - 10
            assert float(row[f"{role}_context_span_seconds_median"]) == context * 5 - 5
            assert float(row[f"{role}_context_span_seconds_mean"]) == context * 5 - 5
            assert int(row[f"{role}_context_span_seconds_maximum"]) == context * 5

        reduction = matrix.reductions[matrix.context_ids[row_index]].row(0, named=True)
        baseline = matrix.reductions[matrix.context_ids[chain_index * 4 + 2]].row(0, named=True)
        for metric in _METRICS:
            value = reduction[metric]
            expected_delta = (
                None
                if value is None or baseline[metric] is None
                else value - baseline[metric]
            )
            assert row[metric] == ("" if value is None else str(value))
            delta_name = f"{metric}_delta_vs_same_chain_c200"
            assert row[delta_name] == ("" if expected_delta is None else str(expected_delta))

        if context == 200:
            assert json.loads(row["final_k_horizon_blocks"]) == list(_HORIZONS)
            start = chain_index * len(_HORIZONS)
            assert json.loads(row["final_k_artifact_ids"]) == [
                str(value) for value in matrix.final_ids[start : start + len(_HORIZONS)]
            ]
        else:
            assert row["final_k_horizon_blocks"] == "[]"
            assert row["final_k_artifact_ids"] == "[]"


@pytest.mark.parametrize(
    ("case", "expected_error"),
    (
        ("incomplete_context", ValueError),
        ("context_order", ValueError),
        ("final_k_order", ValueError),
        ("context_corpus", ValueError),
        ("context_family", ValueError),
        ("context_c", ValueError),
        ("context_k", ValueError),
        ("final_k_source", ValueError),
        ("final_k_window", ValueError),
        ("occupied_destination", FileExistsError),
        ("late_owner_failure", KeyError),
    ),
)
def test_write_context_history_evidence_rejects_invalid_matrix(
    case: str,
    expected_error: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix(tmp_path, monkeypatch)
    context_ids = list(matrix.context_ids)
    final_ids = list(matrix.final_ids)
    destination = tmp_path / "context-history.tsv"

    if case == "incomplete_context":
        context_ids.pop()
    elif case == "context_order":
        context_ids[0], context_ids[1] = context_ids[1], context_ids[0]
    elif case == "final_k_order":
        final_ids[0], final_ids[1] = final_ids[1], final_ids[0]
    elif case == "context_corpus":
        request = matrix.context_requests[0]
        _write_request(matrix, request.model_copy(update={"corpus_id": _uuid(8, 1)}))
    elif case == "context_family":
        request = matrix.context_requests[0]
        association = matrix.artifacts[request.artifact_id]
        source = association.request.source
        assert isinstance(source, BaselineSource)
        current = source.training_definition
        _replace_context_definition(
            matrix,
            0,
            TrainingDefinition(
                experiment=current.experiment,
                model=TransformerDefinition(
                    family="transformer",
                    model_width=4,
                    attention_heads=2,
                    transformer_layers=1,
                    feedforward_width=8,
                    head_hidden=3,
                    dropout=0.0,
                ),
                optimizer=current.optimizer,
                training_batch=current.training_batch,
                fit=current.fit,
            ),
        )
    elif case in {"context_c", "context_k"}:
        request = matrix.context_requests[0]
        first_block = matrix.corpora[request.corpus_id].request.definition.first_block
        experiment = _experiment(
            first_block,
            100 if case == "context_c" else 50,
            6 if case == "context_k" else 5,
            1,
        )
        current = matrix.artifacts[request.artifact_id].request.source
        assert isinstance(current, BaselineSource)
        definition = current.training_definition
        _replace_context_definition(
            matrix,
            0,
            TrainingDefinition(
                experiment=experiment,
                model=definition.model,
                optimizer=definition.optimizer,
                training_batch=definition.training_batch,
                fit=definition.fit,
            ),
        )
    elif case == "final_k_source":
        final_id = final_ids[0]
        c200 = matrix.artifacts[matrix.context_requests[2].artifact_id]
        source = c200.request.source
        assert isinstance(source, BaselineSource)
        matrix.artifacts[final_id] = _context_association(
            final_id,
            source.corpus_id,
            _experiment(1_000, 200, 2, 1),
        )
    elif case == "final_k_window":
        final_id = final_ids[0]
        association = matrix.artifacts[final_id]
        source = association.request.source
        assert isinstance(source, SelectedStudySource)
        experiment = source.experiment
        changed = ExperimentSemantics(
            training_window=OriginWindow(
                role="training",
                first_parent_block=experiment.training_window.first_parent_block + 1,
                last_parent_block=experiment.training_window.last_parent_block,
            ),
            validation_window=experiment.validation_window,
            context_blocks=experiment.context_blocks,
            horizon_blocks=experiment.horizon_blocks,
            ordered_features=experiment.ordered_features,
            loss=experiment.loss,
        )
        matrix.artifacts[final_id] = _final_association(
            final_id,
            source.corpus_id,
            source.study_id,
            changed,
        )
    elif case == "occupied_destination":
        destination.write_text("existing", encoding="utf-8")
    elif case == "late_owner_failure":
        matrix.reductions.pop(matrix.context_ids[-1])

    with pytest.raises(expected_error):
        write_context_history_evidence(
            matrix.storage_root,
            tuple(context_ids),
            tuple(final_ids),
            destination,
        )

    if case == "occupied_destination":
        assert destination.read_text(encoding="utf-8") == "existing"
        assert not destination.with_name(f".{destination.name}").exists()
    elif case == "late_owner_failure":
        assert not destination.exists()
        assert not destination.with_name(f".{destination.name}").exists()
