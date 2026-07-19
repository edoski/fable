from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal
from uuid import UUID

import polars as pl
import pytest
from torch import nn

import fable.evaluation.report as report_module
from fable.addresses import evaluation_json_path
from fable.config import (
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
from fable.corpus import Corpus, FinalizedAnchor
from fable.evaluation import write_sealed_report
from fable.min_block_fee import ClassificationLossState, TargetState
from fable.modeling.artifacts import ArtifactAssociation
from fable.temporal.features import FeatureState

_EVALUATION_IDS = (
    UUID("10000000-0000-4000-8000-000000000003"),
    UUID("10000000-0000-4000-8000-000000000001"),
    UUID("10000000-0000-4000-8000-000000000002"),
)
_ARTIFACT_IDS = (
    UUID("20000000-0000-4000-8000-000000000003"),
    UUID("20000000-0000-4000-8000-000000000001"),
    UUID("20000000-0000-4000-8000-000000000002"),
)
_CORPUS_IDS = (
    UUID("30000000-0000-4000-8000-000000000001"),
    UUID("30000000-0000-4000-8000-000000000002"),
)
_STUDY_ID = UUID("40000000-0000-4000-8000-000000000001")

_Weighting = Literal["unweighted", "corrected_inverse_frequency"]

_CONTEXT_SCHEMA = pl.Schema(
    {
        "evaluation_id": pl.String,
        "artifact_id": pl.String,
        "corpus_id": pl.String,
        "chain_id": pl.Int64,
        "window_role": pl.String,
        "first_parent_block": pl.Int64,
        "last_parent_block": pl.Int64,
        "corpus_endpoint_block": pl.Int64,
        "testing_candidate_origin_count": pl.Int64,
        "testing_incomplete_kmax_outcome_exclusion_count": pl.Int64,
        "testing_elapsed_seconds": pl.Int64,
        "source_kind": pl.String,
        "study_id": pl.String,
        "study_result_index": pl.Int64,
        "model_family": pl.String,
        "context_blocks": pl.Int64,
        "horizon_blocks": pl.Int64,
        "ordered_features": pl.List(pl.String),
        "classification_loss": pl.String,
        "trainable_parameter_count": pl.Int64,
    }
)
_REDUCTION_SCHEMA = pl.Schema(
    {
        "evaluation_id": pl.String,
        "eligible_origin_count": pl.Int64,
        "earliest_hindsight_label_correct_count": pl.Int64,
        "earliest_hindsight_label_cross_entropy_loss_sum": pl.Float64,
        "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss_sum": pl.Float64,
        "hindsight_minimum_base_fee_per_gas_within_k_natural_log_absolute_error_sum": pl.Float64,
        "hindsight_minimum_base_fee_per_gas_within_k_natural_log_squared_error_sum": pl.Float64,
        "earliest_hindsight_label_cross_entropy_loss": pl.Float64,
        "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss": pl.Float64,
        "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mae": pl.Float64,
        "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mse": pl.Float64,
        "multitask_total_loss": pl.Float64,
        "earliest_hindsight_label_accuracy": pl.Float64,
        "earliest_hindsight_label_macro_f1": pl.Float64,
        "immediate_k0_base_fee_per_gas_sum": pl.Float64,
        "finite_target_base_fee_per_gas_savings_sum": pl.Float64,
        "finite_target_base_fee_per_gas_hindsight_opportunity_sum": pl.Float64,
        "finite_target_base_fee_per_gas_hindsight_regret_sum": pl.Float64,
        "finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0": pl.Float64,
        "finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0": pl.Float64,
        "finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0": pl.Float64,
        "signed_captured_hindsight_opportunity_ratio": pl.Float64,
        "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_sum": pl.Float64,
        "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_defined_origin_count": pl.Int64,
        (
            "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_"
            "zero_denominator_exclusion_count"
        ): pl.Int64,
        "mean_origin_target_base_fee_per_gas_savings_fraction_vs_immediate_k0": pl.Float64,
        (
            "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum"
        ): pl.Float64,
        (
            "selected_target_base_fee_per_gas_increase_fraction_vs_"
            "hindsight_best_within_k_defined_origin_count"
        ): pl.Int64,
        (
            "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_"
            "zero_denominator_exclusion_count"
        ): pl.Int64,
        (
            "mean_origin_selected_target_base_fee_per_gas_increase_fraction_"
            "vs_hindsight_best_within_k"
        ): pl.Float64,
        (
            "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum"
        ): pl.Float64,
        (
            "immediate_k0_base_fee_per_gas_increase_fraction_vs_"
            "hindsight_best_within_k_defined_origin_count"
        ): pl.Int64,
        (
            "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_"
            "zero_denominator_exclusion_count"
        ): pl.Int64,
        (
            "mean_origin_immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k"
        ): pl.Float64,
        "harmful_action_count": pl.Int64,
        "harmful_action_rate": pl.Float64,
        "selected_action_count_by_k": pl.List(pl.Int64),
        "extra_wait_block_opportunities_vs_immediate_k0_sum": pl.Float64,
        "mean_extra_wait_block_opportunities_vs_immediate_k0": pl.Float64,
        "selected_action_wait_seconds_sum": pl.Float64,
        "mean_selected_action_wait_seconds": pl.Float64,
        "full_horizon_elapsed_seconds_sum": pl.Float64,
        "mean_full_horizon_elapsed_seconds": pl.Float64,
    }
)
_TSV_SCHEMA = pl.Schema(
    {
        **{
            name: pl.String if dtype == pl.List(pl.String) else dtype
            for name, dtype in _CONTEXT_SCHEMA.items()
        },
        **{
            name: pl.String if dtype == pl.List(pl.Int64) else dtype
            for name, dtype in tuple(_REDUCTION_SCHEMA.items())[1:]
        },
    }
)


def _loss(weighting: _Weighting) -> LossDefinition:
    return LossDefinition(
        classification_algorithm="cross_entropy",
        classification_weighting=weighting,
        regression_algorithm="smooth_l1",
        regression_threshold=1.0,
        classification_scale=1.0,
        regression_scale=1.0,
    )


def _fit() -> FitMethod:
    return FitMethod(
        accumulation=1,
        gradient_clip_norm=1.0,
        scheduler="none",
        seed=2026,
        max_epochs=3,
        validate_every_completed_epoch=1,
        patience=1,
        min_delta=0.0,
        improvement="strict_lower",
        restore="earliest_best",
    )


def _experiment(
    first_block: int,
    *,
    context_blocks: int,
    horizon_blocks: int,
    ordered_features: tuple[str, ...],
    weighting: _Weighting,
) -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=OriginWindow(
            role="training",
            first_parent_block=first_block,
            last_parent_block=first_block + 1,
        ),
        validation_window=OriginWindow(
            role="validation",
            first_parent_block=first_block + 5,
            last_parent_block=first_block + 6,
        ),
        context_blocks=context_blocks,
        horizon_blocks=horizon_blocks,
        ordered_features=ordered_features,
        loss=_loss(weighting),
    )


def _method() -> LstmMethod:
    return LstmMethod(
        family="lstm",
        capacity=LstmCapacity(hidden=4, layers=1, head_hidden=3),
        dropout=0.0,
        optimizer=AdamWMethod(learning_rate=0.01, weight_decay=0.0),
        training_batch=8,
        fit=_fit(),
    )


def _association(
    artifact_id: UUID,
    corpus_id: UUID,
    experiment: ExperimentSemantics,
    *,
    selected: bool,
    transformer: bool = False,
) -> ArtifactAssociation:
    classification_state = (
        None
        if experiment.loss.classification_weighting == "unweighted"
        else ClassificationLossState(
            class_support=(1,) * experiment.horizon_blocks,
        )
    )
    common = {
        "feature_state": FeatureState(
            means=(0.0,) * len(experiment.ordered_features),
            standard_deviations=(1.0,) * len(experiment.ordered_features),
        ),
        "target_state": TargetState(mean=0.0, standard_deviation=1.0),
        "classification_state": classification_state,
    }
    if selected:
        method = _method()
        return ArtifactAssociation(
            request=TrainRequest(
                workflow="train",
                artifact_id=artifact_id,
                source=SelectedStudySource(
                    kind="selected_study",
                    corpus_id=corpus_id,
                    study_id=_STUDY_ID,
                    study_result_index=2,
                    experiment=experiment,
                ),
            ),
            study_result_index=2,
            method=method,
            **common,
        )

    model = (
        TransformerDefinition(
            family="transformer",
            model_width=4,
            attention_heads=2,
            transformer_layers=1,
            feedforward_width=8,
            head_hidden=3,
            dropout=0.0,
        )
        if transformer
        else LstmDefinition(
            family="lstm",
            hidden=4,
            layers=1,
            head_hidden=3,
            dropout=0.0,
        )
    )
    return ArtifactAssociation(
        request=TrainRequest(
            workflow="train",
            artifact_id=artifact_id,
            source=BaselineSource(
                kind="baseline",
                corpus_id=corpus_id,
                training_definition=TrainingDefinition(
                    experiment=experiment,
                    model=model,
                    optimizer=AdamWMethod(learning_rate=0.01, weight_decay=0.0),
                    training_batch=8,
                    fit=_fit(),
                ),
            ),
        ),
        **common,
    )


def _corpus(corpus_id: UUID, *, chain_id: int, first_block: int) -> Corpus:
    block_numbers = list(range(first_block, first_block + 15))
    timestamps = [1_000_000 + offset * offset + offset for offset in range(15)]
    return Corpus(
        request=CorpusRequest(
            corpus_id=corpus_id,
            definition=CorpusDefinition(
                chain_id=chain_id,
                first_block=first_block,
                last_block=first_block + 14,
            ),
        ),
        finalized_anchor=FinalizedAnchor(
            block_number=first_block + 14,
            block_hash="a" * 64,
        ),
        blocks=pl.DataFrame(
            {
                "block_number": block_numbers,
                "timestamp": timestamps,
                "chain_id": [chain_id] * 15,
                "base_fee_per_gas": [100 + offset for offset in range(15)],
                "gas_used": [50] * 15,
                "gas_limit": [100] * 15,
                "tx_count": [10] * 15,
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


def _reduction(evaluation_id: UUID, marker: int, *, zero_opportunity: bool) -> pl.DataFrame:
    values: list[object] = []
    for index, (name, dtype) in enumerate(_REDUCTION_SCHEMA.items()):
        if name == "evaluation_id":
            value: object = str(evaluation_id)
        elif name == "signed_captured_hindsight_opportunity_ratio" and zero_opportunity:
            value = None
        elif (
            name == "finite_target_base_fee_per_gas_hindsight_opportunity_sum" and zero_opportunity
        ):
            value = 0.0
        elif dtype == pl.Int64:
            value = marker * 100 + index
        elif dtype == pl.Float64:
            value = marker + index / 100
        else:
            value = [marker, marker + 1, marker + 2]
        values.append(value)
    return pl.DataFrame([values], schema=_REDUCTION_SCHEMA, orient="row")


def _arrange_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    tuple[EvaluateRequest, ...],
    dict[UUID, ArtifactAssociation],
    dict[str, list[UUID]],
]:
    experiments = (
        _experiment(
            100,
            context_blocks=5,
            horizon_blocks=3,
            ordered_features=("log_base_fee_per_gas", "gas_utilization"),
            weighting="corrected_inverse_frequency",
        ),
        _experiment(
            200,
            context_blocks=7,
            horizon_blocks=3,
            ordered_features=("log_base_fee_per_gas",),
            weighting="unweighted",
        ),
        _experiment(
            100,
            context_blocks=9,
            horizon_blocks=3,
            ordered_features=("log_base_fee_per_gas", "gas_utilization"),
            weighting="unweighted",
        ),
    )
    corpus_ids = (_CORPUS_IDS[0], _CORPUS_IDS[1], _CORPUS_IDS[0])
    requests = tuple(
        EvaluateRequest(
            workflow="evaluate",
            evaluation_id=evaluation_id,
            artifact_id=artifact_id,
            corpus_id=corpus_id,
            window=OriginWindow(
                role="testing",
                first_parent_block=experiment.training_window.first_parent_block + 10,
                last_parent_block=experiment.training_window.first_parent_block + 12,
            ),
        )
        for evaluation_id, artifact_id, corpus_id, experiment in zip(
            _EVALUATION_IDS,
            _ARTIFACT_IDS,
            corpus_ids,
            experiments,
            strict=True,
        )
    )
    for request in requests:
        path = evaluation_json_path(tmp_path, request.evaluation_id)
        path.parent.mkdir(parents=True)
        path.write_text(request.model_dump_json(), encoding="utf-8")

    associations = {
        _ARTIFACT_IDS[0]: _association(
            _ARTIFACT_IDS[0],
            _CORPUS_IDS[0],
            experiments[0],
            selected=True,
        ),
        _ARTIFACT_IDS[1]: _association(
            _ARTIFACT_IDS[1],
            _CORPUS_IDS[1],
            experiments[1],
            selected=False,
            transformer=True,
        ),
        _ARTIFACT_IDS[2]: _association(
            _ARTIFACT_IDS[2],
            _CORPUS_IDS[0],
            experiments[2],
            selected=False,
        ),
    }
    frozen = nn.Embedding(2, 3)
    frozen.weight.requires_grad_(False)
    models: dict[UUID, nn.Module] = {
        _ARTIFACT_IDS[0]: nn.ModuleList([nn.Linear(2, 3, bias=False), frozen]),
        _ARTIFACT_IDS[1]: nn.Linear(3, 4),
        _ARTIFACT_IDS[2]: nn.Sequential(
            nn.Linear(1, 2, bias=False),
            nn.Linear(2, 1, bias=False),
        ),
    }
    corpora = {
        _CORPUS_IDS[0]: _corpus(_CORPUS_IDS[0], chain_id=1, first_block=100),
        _CORPUS_IDS[1]: _corpus(_CORPUS_IDS[1], chain_id=137, first_block=200),
    }
    reductions = {
        evaluation_id: _reduction(
            evaluation_id,
            marker,
            zero_opportunity=marker == 1,
        )
        for marker, evaluation_id in enumerate(_EVALUATION_IDS, start=1)
    }
    calls: dict[str, list[UUID]] = {"reduce": [], "artifact": [], "corpus": []}

    def reduce_evaluation(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
        assert storage_root == tmp_path
        calls["reduce"].append(evaluation_id)
        return reductions[evaluation_id]

    def load_artifact(storage_root: Path, artifact_id: UUID):
        assert storage_root == tmp_path
        calls["artifact"].append(artifact_id)
        return associations[artifact_id], models[artifact_id]

    def load_corpus(storage_root: Path, corpus_id: UUID) -> Corpus:
        assert storage_root == tmp_path
        calls["corpus"].append(corpus_id)
        return corpora[corpus_id]

    monkeypatch.setattr(report_module, "reduce_evaluation", reduce_evaluation)
    monkeypatch.setattr(report_module, "load_artifact", load_artifact)
    monkeypatch.setattr(report_module, "load_corpus", load_corpus)
    return requests, associations, calls


def test_write_sealed_report_preserves_order_and_publishes_exact_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, calls = _arrange_report(tmp_path, monkeypatch)
    destination = tmp_path / "sealed.tsv"
    renames: list[tuple[Path, Path]] = []
    real_rename = Path.rename

    def rename(source: Path, target: Path) -> Path:
        renames.append((source, target))
        return real_rename(source, target)

    monkeypatch.setattr(Path, "rename", rename)

    write_sealed_report(tmp_path, _EVALUATION_IDS, destination)

    hidden = destination.with_name(f".{destination.name}")
    assert renames == [(hidden, destination)]
    assert not hidden.exists()
    assert calls == {
        "reduce": list(_EVALUATION_IDS),
        "artifact": list(_ARTIFACT_IDS),
        "corpus": [_CORPUS_IDS[0], _CORPUS_IDS[1]],
    }

    report = pl.read_csv(destination, separator="\t", null_values="")
    assert report.schema == _TSV_SCHEMA
    assert report["evaluation_id"].to_list() == [str(value) for value in _EVALUATION_IDS]

    with destination.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["artifact_id"] for row in rows] == [str(value) for value in _ARTIFACT_IDS]
    assert [row["corpus_id"] for row in rows] == [
        str(_CORPUS_IDS[0]),
        str(_CORPUS_IDS[1]),
        str(_CORPUS_IDS[0]),
    ]
    assert [row["chain_id"] for row in rows] == ["1", "137", "1"]
    assert [row["source_kind"] for row in rows] == ["selected_study", "baseline", "baseline"]
    assert [row["study_id"] for row in rows] == [str(_STUDY_ID), "", ""]
    assert [row["study_result_index"] for row in rows] == ["2", "", ""]
    assert [row["model_family"] for row in rows] == ["lstm", "transformer", "lstm"]
    assert [row["context_blocks"] for row in rows] == ["5", "7", "9"]
    assert [row["horizon_blocks"] for row in rows] == ["3", "3", "3"]
    assert [row["ordered_features"] for row in rows] == [
        '["log_base_fee_per_gas","gas_utilization"]',
        '["log_base_fee_per_gas"]',
        '["log_base_fee_per_gas","gas_utilization"]',
    ]
    assert [row["classification_loss"] for row in rows] == [
        "corrected_inverse_frequency",
        "unweighted",
        "unweighted",
    ]
    assert [row["trainable_parameter_count"] for row in rows] == ["6", "16", "4"]
    assert [row["corpus_endpoint_block"] for row in rows] == ["114", "214", "114"]
    assert [row["testing_candidate_origin_count"] for row in rows] == ["5", "5", "5"]
    assert [row["testing_incomplete_kmax_outcome_exclusion_count"] for row in rows] == [
        "2",
        "2",
        "2",
    ]
    assert [row["testing_elapsed_seconds"] for row in rows] == ["46", "46", "46"]
    assert [row["eligible_origin_count"] for row in rows] == ["101", "201", "301"]
    assert [row["earliest_hindsight_label_cross_entropy_loss_sum"] for row in rows] == [
        "1.03",
        "2.03",
        "3.03",
    ]
    assert [row["selected_action_count_by_k"] for row in rows] == [
        "[1,2,3]",
        "[2,3,4]",
        "[3,4,5]",
    ]
    assert rows[0]["signed_captured_hindsight_opportunity_ratio"] == ""
    assert rows[0]["finite_target_base_fee_per_gas_hindsight_opportunity_sum"] == "0.0"


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("empty", ValueError),
        ("duplicate", ValueError),
        ("request_id", ValueError),
        ("non_testing", ValueError),
        ("corpus", ValueError),
        ("destination", FileExistsError),
        ("hidden", FileExistsError),
        ("later_row", RuntimeError),
    ],
)
def test_write_sealed_report_rejects_invalid_input_without_partial_publication(
    case: str,
    error: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, associations, _ = _arrange_report(tmp_path, monkeypatch)
    destination = tmp_path / "sealed.tsv"
    hidden = destination.with_name(f".{destination.name}")
    evaluation_ids = _EVALUATION_IDS
    if case == "empty":
        evaluation_ids = ()
    elif case == "duplicate":
        evaluation_ids = (_EVALUATION_IDS[0], _EVALUATION_IDS[0])
    elif case == "request_id":
        evaluation_json_path(tmp_path, _EVALUATION_IDS[0]).write_text(
            requests[0].model_copy(update={"evaluation_id": _EVALUATION_IDS[1]}).model_dump_json(),
            encoding="utf-8",
        )
    elif case == "non_testing":
        evaluation_json_path(tmp_path, _EVALUATION_IDS[0]).write_text(
            requests[0]
            .model_copy(
                update={
                    "window": OriginWindow(
                        role="validation",
                        first_parent_block=105,
                        last_parent_block=106,
                    )
                }
            )
            .model_dump_json(),
            encoding="utf-8",
        )
    elif case == "corpus":
        association = associations[_ARTIFACT_IDS[0]]
        associations[_ARTIFACT_IDS[0]] = association.model_copy(
            update={
                "request": association.request.model_copy(
                    update={
                        "source": association.request.source.model_copy(
                            update={"corpus_id": _CORPUS_IDS[1]}
                        )
                    }
                )
            }
        )
    elif case == "destination":
        destination.write_text("existing", encoding="utf-8")
    elif case == "hidden":
        hidden.write_text("existing", encoding="utf-8")
    elif case == "later_row":
        original = report_module.reduce_evaluation

        def fail_later(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
            if evaluation_id == _EVALUATION_IDS[1]:
                raise RuntimeError("later row failed")
            return original(storage_root, evaluation_id)

        monkeypatch.setattr(report_module, "reduce_evaluation", fail_later)

    with pytest.raises(error):
        write_sealed_report(tmp_path, evaluation_ids, destination)

    if case == "destination":
        assert destination.read_text(encoding="utf-8") == "existing"
        assert not hidden.exists()
    elif case == "hidden":
        assert not destination.exists()
        assert hidden.read_text(encoding="utf-8") == "existing"
    elif case == "later_row":
        assert not destination.exists()
        assert not hidden.exists()
