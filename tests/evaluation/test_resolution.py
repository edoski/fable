from __future__ import annotations

import math
from pathlib import Path
from typing import Literal
from uuid import UUID

import polars as pl
import pytest
from torch import nn

import fable.evaluation.resolution as resolution_module
from fable.addresses import evaluation_directory
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
)
from fable.corpus import BlockFrame, Corpus, FinalizedAnchor
from fable.evaluation import reduce_evaluation, resolve_evaluations
from fable.min_block_fee import ClassificationLossState, TargetState
from fable.modeling import ArtifactAssociation
from fable.temporal.features import FeatureState

_EVALUATION_ID = UUID("10000000-0000-4000-8000-000000000001")
_OTHER_EVALUATION_ID = UUID("10000000-0000-4000-8000-000000000002")
_ARTIFACT_ID = UUID("20000000-0000-4000-8000-000000000001")
_CORPUS_ID = UUID("30000000-0000-4000-8000-000000000001")
_OTHER_CORPUS_ID = UUID("30000000-0000-4000-8000-000000000002")
_STUDY_ID = UUID("40000000-0000-4000-8000-000000000001")

_Weighting = Literal["unweighted", "corrected_inverse_frequency"]

_LOG_ABSOLUTE_SUM = "hindsight_minimum_base_fee_per_gas_within_k_natural_log_absolute_error_sum"
_LOG_SQUARED_SUM = "hindsight_minimum_base_fee_per_gas_within_k_natural_log_squared_error_sum"
_SAVINGS_EXCLUDED = (
    "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_zero_denominator_exclusion_count"
)
_SELECTED_FRACTION_SUM = (
    "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum"
)
_SELECTED_DEFINED = (
    "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_"
    "defined_origin_count"
)
_SELECTED_EXCLUDED = (
    "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_"
    "zero_denominator_exclusion_count"
)
_SELECTED_MEAN = (
    "mean_origin_selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k"
)
_IMMEDIATE_FRACTION_SUM = (
    "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum"
)
_IMMEDIATE_DEFINED = (
    "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_"
    "defined_origin_count"
)
_IMMEDIATE_EXCLUDED = (
    "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_"
    "zero_denominator_exclusion_count"
)
_IMMEDIATE_MEAN = (
    "mean_origin_immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k"
)

_RESULT_SCHEMA = pl.Schema(
    {
        "evaluation_id": pl.String,
        "eligible_origin_count": pl.Int64,
        "earliest_hindsight_label_correct_count": pl.Int64,
        "earliest_hindsight_label_cross_entropy_loss_sum": pl.Float64,
        "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss_sum": pl.Float64,
        _LOG_ABSOLUTE_SUM: pl.Float64,
        _LOG_SQUARED_SUM: pl.Float64,
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
        _SAVINGS_EXCLUDED: pl.Int64,
        "mean_origin_target_base_fee_per_gas_savings_fraction_vs_immediate_k0": pl.Float64,
        _SELECTED_FRACTION_SUM: pl.Float64,
        _SELECTED_DEFINED: pl.Int64,
        _SELECTED_EXCLUDED: pl.Int64,
        _SELECTED_MEAN: pl.Float64,
        _IMMEDIATE_FRACTION_SUM: pl.Float64,
        _IMMEDIATE_DEFINED: pl.Int64,
        _IMMEDIATE_EXCLUDED: pl.Int64,
        _IMMEDIATE_MEAN: pl.Float64,
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

_OBSERVATION_SCHEMA = pl.Schema(
    {
        "origin_block": pl.Int64,
        "origin_timestamp": pl.Int64,
        "selected_action_k": pl.Int64,
        "earliest_hindsight_action_k": pl.Int64,
        "classification_loss_contribution": pl.Float64,
        "predicted_hindsight_minimum_base_fee_z": pl.Float32,
        "previous_closed_parent_base_fee_per_gas": pl.Int64,
        "closed_parent_base_fee_per_gas": pl.Int64,
        "immediate_k0_base_fee_per_gas": pl.Int64,
        "selected_target_base_fee_per_gas": pl.Int64,
        "hindsight_minimum_base_fee_per_gas": pl.Int64,
        "selected_action_wait_seconds": pl.Int64,
        "full_horizon_elapsed_seconds": pl.Int64,
    }
)


def _loss(weighting: _Weighting) -> LossDefinition:
    return LossDefinition(
        classification_algorithm="cross_entropy",
        classification_weighting=weighting,
        regression_algorithm="smooth_l1",
        regression_threshold=0.75,
        classification_scale=2.0,
        regression_scale=0.5,
    )


def _experiment(weighting: _Weighting) -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=OriginWindow(
            role="training",
            first_parent_block=1,
            last_parent_block=5,
        ),
        validation_window=OriginWindow(
            role="validation",
            first_parent_block=20,
            last_parent_block=24,
        ),
        context_blocks=3,
        horizon_blocks=4,
        ordered_features=("log_base_fee_per_gas",),
        loss=_loss(weighting),
    )


def _method() -> LstmMethod:
    return LstmMethod(
        family="lstm",
        capacity=LstmCapacity(hidden=4, layers=1, head_hidden=3),
        dropout=0.0,
        optimizer=AdamWMethod(learning_rate=0.01, weight_decay=0.0),
        training_batch=7,
        fit=FitMethod(
            accumulation=1,
            gradient_clip_norm=1.0,
            scheduler="none",
            seed=17,
            max_epochs=2,
            validate_every_completed_epoch=1,
            patience=1,
            min_delta=0.0,
            improvement="strict_lower",
            restore="earliest_best",
        ),
    )


def _association(
    source_kind: Literal["baseline", "selected"],
    weighting: _Weighting,
    *,
    corpus_id: UUID = _CORPUS_ID,
    target_state: TargetState | None = None,
) -> ArtifactAssociation:
    experiment = _experiment(weighting)
    if source_kind == "baseline":
        source = BaselineSource(
            kind="baseline",
            corpus_id=corpus_id,
            training_definition=TrainingDefinition(
                experiment=experiment,
                model=LstmDefinition(
                    family="lstm",
                    hidden=4,
                    layers=1,
                    head_hidden=3,
                    dropout=0.0,
                ),
                optimizer=_method().optimizer,
                training_batch=7,
                fit=_method().fit,
            ),
        )
        return ArtifactAssociation(
            request=TrainRequest(
                workflow="train",
                artifact_id=_ARTIFACT_ID,
                source=source,
            ),
            feature_state=FeatureState(means=(0.0,), standard_deviations=(1.0,)),
            target_state=target_state or TargetState(mean=0.0, standard_deviation=math.log(2.0)),
            classification_state=(
                None
                if weighting == "unweighted"
                else ClassificationLossState(class_support=(2, 3, 4, 5))
            ),
        )
    else:
        source = SelectedStudySource(
            kind="selected_study",
            corpus_id=corpus_id,
            study_id=_STUDY_ID,
            study_result_index=2,
            experiment=experiment,
        )
        study_result_index = 2
    classification_state = (
        None if weighting == "unweighted" else ClassificationLossState(class_support=(2, 3, 4, 5))
    )
    return ArtifactAssociation(
        request=TrainRequest(
            workflow="train",
            artifact_id=_ARTIFACT_ID,
            source=source,
        ),
        feature_state=FeatureState(means=(0.0,), standard_deviations=(1.0,)),
        target_state=target_state or TargetState(mean=0.0, standard_deviation=math.log(2.0)),
        classification_state=classification_state,
        study_result_index=study_result_index,
        method=_method(),
    )


def _request(
    *,
    evaluation_id: UUID = _EVALUATION_ID,
    corpus_id: UUID = _CORPUS_ID,
    window: OriginWindow | None = None,
) -> EvaluateRequest:
    return EvaluateRequest(
        workflow="evaluate",
        evaluation_id=evaluation_id,
        artifact_id=_ARTIFACT_ID,
        corpus_id=corpus_id,
        window=window or _experiment("unweighted").validation_window,
    )


def _observations() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_block": [20, 21, 22, 23, 24],
            "origin_timestamp": [1_000, 1_010, 1_020, 1_030, 1_040],
            "selected_action_k": [0, 2, 0, 2, 0],
            "earliest_hindsight_action_k": [0, 0, 1, 1, 0],
            "classification_loss_contribution": [1.0, 2.0, 3.0, 4.0, 5.0],
            "predicted_hindsight_minimum_base_fee_z": [6.5, 6.0, 8.75, 11.0, 9.75],
            "previous_closed_parent_base_fee_per_gas": [55, 64, 160, 300, 600],
            "closed_parent_base_fee_per_gas": [64, 160, 300, 600, 1_024],
            "immediate_k0_base_fee_per_gas": [64, 160, 300, 600, 1_024],
            "selected_target_base_fee_per_gas": [64, 144, 300, 650, 1_024],
            "hindsight_minimum_base_fee_per_gas": [64, 128, 256, 512, 1_024],
            "selected_action_wait_seconds": [0, 20, 0, 24, 0],
            "full_horizon_elapsed_seconds": [40, 44, 48, 52, 56],
        },
        schema=_OBSERVATION_SCHEMA,
    )


def _publish_evaluation(
    storage_root: Path,
    request: EvaluateRequest,
    observations: pl.DataFrame,
    *,
    evaluation_id: UUID | None = None,
) -> None:
    directory = evaluation_directory(storage_root, evaluation_id or request.evaluation_id)
    directory.mkdir(parents=True)
    (directory / "evaluation.json").write_text(request.model_dump_json(), encoding="utf-8")
    observations.write_parquet(directory / "observations.parquet")


def _stub_artifact(
    monkeypatch: pytest.MonkeyPatch,
    association: ArtifactAssociation,
) -> list[tuple[Path, UUID]]:
    calls: list[tuple[Path, UUID]] = []

    def load_artifact(storage_root: Path, artifact_id: UUID):
        calls.append((storage_root, artifact_id))
        return association, nn.Identity()

    monkeypatch.setattr(resolution_module, "load_artifact", load_artifact)
    return calls


@pytest.mark.parametrize(
    ("source_kind", "weighting"),
    [("baseline", "corrected_inverse_frequency"), ("selected", "unweighted")],
)
def test_reduce_evaluation_returns_all_scientific_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: Literal["baseline", "selected"],
    weighting: _Weighting,
) -> None:
    request = _request()
    _publish_evaluation(tmp_path, request, _observations())
    artifact_calls = _stub_artifact(monkeypatch, _association(source_kind, weighting))

    result = reduce_evaluation(tmp_path, _EVALUATION_ID)

    assert artifact_calls == [(tmp_path, _ARTIFACT_ID)]
    assert result.schema == _RESULT_SCHEMA
    assert result.height == 1
    expected: dict[str, str | int | float | list[int]] = {
        "evaluation_id": str(_EVALUATION_ID),
        "eligible_origin_count": 5,
        "earliest_hindsight_label_correct_count": 2,
        "earliest_hindsight_label_cross_entropy_loss_sum": 15.0,
        "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss_sum": 1.4166666697710752,
        _LOG_ABSOLUTE_SUM: 3.1191623125197543,
        _LOG_SQUARED_SUM: 2.8226614567694335,
        "earliest_hindsight_label_cross_entropy_loss": 3.0,
        "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss": 0.28333333395421506,
        "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mae": 0.6238324625039509,
        "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mse": 0.5645322913538867,
        "multitask_total_loss": 3.283333333954215,
        "earliest_hindsight_label_accuracy": 0.4,
        "earliest_hindsight_label_macro_f1": 2.0 / 9.0,
        "immediate_k0_base_fee_per_gas_sum": 2_148.0,
        "finite_target_base_fee_per_gas_savings_sum": -34.0,
        "finite_target_base_fee_per_gas_hindsight_opportunity_sum": 164.0,
        "finite_target_base_fee_per_gas_hindsight_regret_sum": 198.0,
        "finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0": -34.0 / 2_148.0,
        "finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0": 164.0
        / 2_148.0,
        "finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0": 198.0 / 2_148.0,
        "signed_captured_hindsight_opportunity_ratio": -34.0 / 164.0,
        "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_sum": 1.0 / 60.0,
        "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_defined_origin_count": 5,
        _SAVINGS_EXCLUDED: 0,
        "mean_origin_target_base_fee_per_gas_savings_fraction_vs_immediate_k0": 1.0 / 300.0,
        _SELECTED_FRACTION_SUM: 145.0 / 256.0,
        _SELECTED_DEFINED: 5,
        _SELECTED_EXCLUDED: 0,
        _SELECTED_MEAN: 29.0 / 256.0,
        _IMMEDIATE_FRACTION_SUM: 19.0 / 32.0,
        _IMMEDIATE_DEFINED: 5,
        _IMMEDIATE_EXCLUDED: 0,
        _IMMEDIATE_MEAN: 19.0 / 160.0,
        "harmful_action_count": 1,
        "harmful_action_rate": 0.2,
        "selected_action_count_by_k": [3, 0, 2, 0],
        "extra_wait_block_opportunities_vs_immediate_k0_sum": 4.0,
        "mean_extra_wait_block_opportunities_vs_immediate_k0": 0.8,
        "selected_action_wait_seconds_sum": 44.0,
        "mean_selected_action_wait_seconds": 8.8,
        "full_horizon_elapsed_seconds_sum": 240.0,
        "mean_full_horizon_elapsed_seconds": 48.0,
    }
    actual = result.row(0, named=True)
    for name, value in expected.items():
        if isinstance(value, float):
            assert actual[name] == pytest.approx(value, rel=1e-12, abs=1e-12), name
        else:
            assert actual[name] == value, name


def test_reduce_evaluation_keeps_only_captured_opportunity_nullable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = OriginWindow(role="testing", first_parent_block=30, last_parent_block=30)
    request = _request(window=window)
    fee = 100_000_000_000
    observations = pl.DataFrame(
        {
            "origin_block": [30],
            "origin_timestamp": [2_000],
            "selected_action_k": [1],
            "earliest_hindsight_action_k": [0],
            "classification_loss_contribution": [0.25],
            "predicted_hindsight_minimum_base_fee_z": [0.0],
            "previous_closed_parent_base_fee_per_gas": [fee],
            "closed_parent_base_fee_per_gas": [fee],
            "immediate_k0_base_fee_per_gas": [fee],
            "selected_target_base_fee_per_gas": [fee + 1],
            "hindsight_minimum_base_fee_per_gas": [fee],
            "selected_action_wait_seconds": [12],
            "full_horizon_elapsed_seconds": [20],
        },
        schema=_OBSERVATION_SCHEMA,
    )
    _publish_evaluation(
        tmp_path,
        request,
        observations,
        evaluation_id=_EVALUATION_ID,
    )
    association = _association(
        "selected",
        "unweighted",
        target_state=TargetState(mean=math.log(fee), standard_deviation=1.0),
    )
    _stub_artifact(monkeypatch, association)

    result = reduce_evaluation(tmp_path, _EVALUATION_ID)

    row = result.row(0, named=True)
    assert row["finite_target_base_fee_per_gas_savings_sum"] == -1.0
    assert row["finite_target_base_fee_per_gas_hindsight_opportunity_sum"] == 0.0
    assert row["finite_target_base_fee_per_gas_hindsight_regret_sum"] == 1.0
    assert row["signed_captured_hindsight_opportunity_ratio"] is None
    assert row["harmful_action_count"] == 1
    assert sum(result.null_count().row(0)) == 1


@pytest.mark.parametrize(
    "case",
    [
        "evaluation_id",
        "source_corpus",
        "window",
        "schema",
        "null",
        "origins",
        "action",
        "fee",
        "wait",
        "stored_nonfinite",
        "derived_nonfinite",
        "origin_timestamp",
        "adjacent_fee",
    ],
)
def test_reduce_evaluation_rejects_owned_invalid_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    request = _request()
    observations = _observations()
    association = _association("baseline", "unweighted")

    if case == "evaluation_id":
        request = _request(evaluation_id=_OTHER_EVALUATION_ID)
    elif case == "source_corpus":
        association = _association("baseline", "unweighted", corpus_id=_OTHER_CORPUS_ID)
    elif case == "window":
        request = _request(
            window=OriginWindow(
                role="validation",
                first_parent_block=19,
                last_parent_block=23,
            )
        )
    elif case == "schema":
        observations = observations.select(
            "origin_timestamp",
            "origin_block",
            *list(_OBSERVATION_SCHEMA.names())[2:],
        )
    elif case == "null":
        observations = observations.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise("origin_timestamp")
            .cast(pl.Int64)
            .alias("origin_timestamp")
        )
    elif case == "origins":
        observations = observations.with_columns(
            pl.Series("origin_block", [20, 21, 23, 22, 24], dtype=pl.Int64)
        )
    elif case == "action":
        observations = observations.with_columns(
            pl.Series("selected_action_k", [4, 2, 0, 2, 0], dtype=pl.Int64)
        )
    elif case == "fee":
        observations = observations.with_columns(
            pl.Series(
                "selected_target_base_fee_per_gas",
                [63, 144, 300, 650, 1_024],
                dtype=pl.Int64,
            )
        )
    elif case == "wait":
        observations = observations.with_columns(
            pl.Series("selected_action_wait_seconds", [-1, 20, 0, 24, 0], dtype=pl.Int64),
        )
    elif case == "stored_nonfinite":
        observations = observations.with_columns(
            pl.Series(
                "classification_loss_contribution",
                [math.inf, 2.0, 3.0, 4.0, 5.0],
                dtype=pl.Float64,
            )
        )
    elif case == "derived_nonfinite":
        association = _association(
            "baseline",
            "unweighted",
            target_state=TargetState(mean=0.0, standard_deviation=1e308),
        )
    elif case == "origin_timestamp":
        observations = observations.with_columns(
            pl.Series("origin_timestamp", [-1, 1_010, 1_020, 1_030, 1_040], dtype=pl.Int64)
        )
    elif case == "adjacent_fee":
        observations = observations.with_columns(
            pl.Series(
                "previous_closed_parent_base_fee_per_gas",
                [0, 64, 160, 300, 600],
                dtype=pl.Int64,
            )
        )

    _publish_evaluation(
        tmp_path,
        request,
        observations,
        evaluation_id=_EVALUATION_ID,
    )
    _stub_artifact(monkeypatch, association)

    with pytest.raises(ValueError):
        reduce_evaluation(tmp_path, _EVALUATION_ID)


def test_resolve_evaluations_preserves_order_and_shares_durable_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_request = _request(evaluation_id=_EVALUATION_ID)
    second_request = _request(evaluation_id=_OTHER_EVALUATION_ID)
    _publish_evaluation(tmp_path, first_request, _observations())
    _publish_evaluation(tmp_path, second_request, _observations())
    association = _association("baseline", "unweighted")
    artifact_calls = _stub_artifact(monkeypatch, association)
    definition = CorpusDefinition(chain_id=1, first_block=1, last_block=30)
    corpus = Corpus(
        request=CorpusRequest(
            corpus_id=_CORPUS_ID,
            definition=definition,
        ),
        finalized_anchor=FinalizedAnchor(block_number=30, block_hash="a" * 64),
        blocks=BlockFrame(
            pl.DataFrame(
                {
                    "block_number": range(1, 31),
                    "timestamp": range(1, 31),
                    "chain_id": [1] * 30,
                    "base_fee_per_gas": [1] * 30,
                    "gas_used": [0] * 30,
                    "gas_limit": [1] * 30,
                    "tx_count": [0] * 30,
                }
            ),
            definition,
        ),
    )
    corpus_calls: list[tuple[Path, UUID]] = []

    def load_corpus(storage_root: Path, corpus_id: UUID) -> Corpus:
        corpus_calls.append((storage_root, corpus_id))
        return corpus

    monkeypatch.setattr(resolution_module, "load_corpus", load_corpus)

    resolved = resolve_evaluations(
        tmp_path,
        (_OTHER_EVALUATION_ID, _EVALUATION_ID, _OTHER_EVALUATION_ID),
    )

    assert tuple(item.request.evaluation_id for item in resolved) == (
        _OTHER_EVALUATION_ID,
        _EVALUATION_ID,
        _OTHER_EVALUATION_ID,
    )
    assert resolved[0] is resolved[2]
    assert resolved[0].corpus is corpus
    source = association.request.source
    assert isinstance(source, BaselineSource)
    assert resolved[0].training_definition == source.training_definition
    assert resolved[0].observations.collect_schema() == _OBSERVATION_SCHEMA
    assert resolved[0].reduction.schema == _RESULT_SCHEMA
    assert resolved[0].trainable_parameter_count == 0
    assert artifact_calls == [(tmp_path, _ARTIFACT_ID)]
    assert corpus_calls == [(tmp_path, _CORPUS_ID)]
    assert resolve_evaluations(tmp_path, ()) == ()
