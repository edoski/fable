"""Reduction of one canonical evaluation to transient scientific facts."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import polars as pl

from ..addresses import evaluation_json_path, evaluation_observations_path
from ..config import BaselineSource, EvaluateRequest
from ..modeling.artifacts import load_artifact

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

_RESULT_COLUMNS = (
    "evaluation_id",
    "eligible_origin_count",
    "earliest_hindsight_label_correct_count",
    "earliest_hindsight_label_cross_entropy_loss_sum",
    "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss_sum",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_absolute_error_sum",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_squared_error_sum",
    "earliest_hindsight_label_cross_entropy_loss",
    "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mae",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mse",
    "multitask_total_loss",
    "earliest_hindsight_label_accuracy",
    "earliest_hindsight_label_macro_f1",
    "immediate_k0_base_fee_per_gas_sum",
    "finite_target_base_fee_per_gas_savings_sum",
    "finite_target_base_fee_per_gas_hindsight_opportunity_sum",
    "finite_target_base_fee_per_gas_hindsight_regret_sum",
    "finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0",
    "finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0",
    "finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0",
    "signed_captured_hindsight_opportunity_ratio",
    "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_sum",
    "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_defined_origin_count",
    "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_zero_denominator_exclusion_count",
    "mean_origin_target_base_fee_per_gas_savings_fraction_vs_immediate_k0",
    "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum",
    "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_defined_origin_count",
    "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_zero_denominator_exclusion_count",
    "mean_origin_selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k",
    "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum",
    "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_defined_origin_count",
    "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_zero_denominator_exclusion_count",
    "mean_origin_immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k",
    "harmful_action_count",
    "harmful_action_rate",
    "selected_action_count_by_k",
    "extra_wait_block_opportunities_vs_immediate_k0_sum",
    "mean_extra_wait_block_opportunities_vs_immediate_k0",
    "selected_action_wait_seconds_sum",
    "mean_selected_action_wait_seconds",
    "full_horizon_elapsed_seconds_sum",
    "mean_full_horizon_elapsed_seconds",
)

_FLOAT_RESULT_COLUMNS = (
    "earliest_hindsight_label_cross_entropy_loss_sum",
    "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss_sum",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_absolute_error_sum",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_squared_error_sum",
    "earliest_hindsight_label_cross_entropy_loss",
    "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mae",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mse",
    "multitask_total_loss",
    "earliest_hindsight_label_accuracy",
    "earliest_hindsight_label_macro_f1",
    "immediate_k0_base_fee_per_gas_sum",
    "finite_target_base_fee_per_gas_savings_sum",
    "finite_target_base_fee_per_gas_hindsight_opportunity_sum",
    "finite_target_base_fee_per_gas_hindsight_regret_sum",
    "finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0",
    "finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0",
    "finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0",
    "signed_captured_hindsight_opportunity_ratio",
    "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_sum",
    "mean_origin_target_base_fee_per_gas_savings_fraction_vs_immediate_k0",
    "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum",
    "mean_origin_selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k",
    "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum",
    "mean_origin_immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k",
    "harmful_action_rate",
    "extra_wait_block_opportunities_vs_immediate_k0_sum",
    "mean_extra_wait_block_opportunities_vs_immediate_k0",
    "selected_action_wait_seconds_sum",
    "mean_selected_action_wait_seconds",
    "full_horizon_elapsed_seconds_sum",
    "mean_full_horizon_elapsed_seconds",
)


def reduce_evaluation(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
    """Reduce one canonical evaluation to one fixed scientific-facts row."""

    request = EvaluateRequest.model_validate_json(
        evaluation_json_path(storage_root, evaluation_id).read_text(encoding="utf-8"),
        strict=True,
    )
    if request.evaluation_id != evaluation_id:
        raise ValueError("evaluation request ID must match the requested evaluation")

    association, _ = load_artifact(storage_root, request.artifact_id)
    source = association.request.source
    if source.corpus_id != request.corpus_id:
        raise ValueError("artifact source Corpus must match the evaluation Corpus")
    experiment = (
        source.training_definition.experiment
        if isinstance(source, BaselineSource)
        else source.experiment
    )
    if request.window.role == "validation":
        if request.window != experiment.validation_window:
            raise ValueError("validation window must match the artifact experiment")
    elif (
        experiment.validation_window.last_parent_block + experiment.horizon_blocks
        >= request.window.first_parent_block
    ):
        raise ValueError("testing window must follow complete validation outcomes")

    observations_path = evaluation_observations_path(storage_root, evaluation_id)
    if pl.read_parquet_schema(observations_path) != _OBSERVATION_SCHEMA:
        raise ValueError("observations must have the canonical ordered schema")
    observations = pl.scan_parquet(observations_path)

    count = request.window.last_parent_block - request.window.first_parent_block + 1
    horizon = experiment.horizon_blocks
    loss = experiment.loss
    target = association.target_state

    immediate = pl.col("immediate_k0_base_fee_per_gas")
    selected = pl.col("selected_target_base_fee_per_gas")
    hindsight = pl.col("hindsight_minimum_base_fee_per_gas")
    target_log = hindsight.cast(pl.Float64).log()
    target_z = (
        (target_log - pl.lit(target.mean, dtype=pl.Float64))
        / pl.lit(target.standard_deviation, dtype=pl.Float64)
    ).cast(pl.Float32)
    predicted_z = pl.col("predicted_hindsight_minimum_base_fee_z")
    error = predicted_z - pl.col("_target_z")
    absolute_error = error.abs()
    half = pl.lit(0.5, dtype=pl.Float32)
    threshold = pl.lit(loss.regression_threshold, dtype=pl.Float32)
    regression_scale = pl.lit(loss.regression_scale, dtype=pl.Float32)
    smooth_l1 = (
        pl.when(absolute_error < threshold)
        .then(half * error * error / threshold)
        .otherwise(absolute_error - half * threshold)
        * regression_scale
    )
    predicted_log = pl.lit(target.mean, dtype=pl.Float64) + pl.lit(
        target.standard_deviation,
        dtype=pl.Float64,
    ) * predicted_z.cast(pl.Float64)

    mapped = observations.with_columns(
        (immediate - selected).alias("_savings"),
        (immediate - hindsight).alias("_opportunity"),
        (selected - hindsight).alias("_regret"),
        (selected > immediate).alias("_harmful"),
        target_z.alias("_target_z"),
        (predicted_log - target_log).alias("_log_error"),
    ).with_columns(smooth_l1.alias("_smooth_l1"))

    support_by_k = [
        (pl.col("earliest_hindsight_action_k") == k).sum().cast(pl.Int64) for k in range(horizon)
    ]
    prediction_by_k = [
        (pl.col("selected_action_k") == k).sum().cast(pl.Int64) for k in range(horizon)
    ]
    true_positive_by_k = [
        ((pl.col("selected_action_k") == k) & (pl.col("earliest_hindsight_action_k") == k))
        .sum()
        .cast(pl.Int64)
        for k in range(horizon)
    ]
    f1_by_k: list[pl.Expr] = []
    active_by_k: list[pl.Expr] = []
    for support, prediction, true_positive in zip(
        support_by_k,
        prediction_by_k,
        true_positive_by_k,
        strict=True,
    ):
        denominator = support + prediction
        active = denominator > 0
        active_by_k.append(active.cast(pl.Int64))
        f1_by_k.append(
            pl.when(active)
            .then(pl.lit(2.0) * true_positive.cast(pl.Float64) / denominator.cast(pl.Float64))
            .otherwise(pl.lit(0.0))
        )

    no_nulls = (
        pl.sum_horizontal([pl.col(name).null_count() for name in _OBSERVATION_SCHEMA.names()]) == 0
    )
    expected_origins = pl.int_range(0, pl.len(), dtype=pl.Int64) + pl.lit(
        request.window.first_parent_block,
        dtype=pl.Int64,
    )
    valid_inputs = pl.all_horizontal(
        pl.len() == count,
        no_nulls,
        (pl.col("origin_block") == expected_origins).all(),
        pl.col("classification_loss_contribution").is_finite().all(),
        (pl.col("classification_loss_contribution") >= 0.0).all(),
        predicted_z.is_finite().all(),
        (pl.col("selected_action_k") >= 0).all(),
        (pl.col("selected_action_k") < horizon).all(),
        (pl.col("earliest_hindsight_action_k") >= 0).all(),
        (pl.col("earliest_hindsight_action_k") < horizon).all(),
        (immediate > 0).all(),
        (selected > 0).all(),
        (hindsight > 0).all(),
        (hindsight <= immediate).all(),
        (hindsight <= selected).all(),
        (pl.col("selected_action_wait_seconds") >= 0).all(),
        (pl.col("full_horizon_elapsed_seconds") >= 0).all(),
        ((pl.col("selected_action_k") != 0) | (pl.col("selected_action_wait_seconds") == 0)).all(),
        (pl.col("selected_action_wait_seconds") <= pl.col("full_horizon_elapsed_seconds")).all(),
    )

    float64 = pl.Float64
    aggregated = mapped.select(
        pl.lit(str(evaluation_id)).alias("evaluation_id"),
        pl.len().cast(pl.Int64).alias("eligible_origin_count"),
        (pl.col("selected_action_k") == pl.col("earliest_hindsight_action_k"))
        .sum()
        .cast(pl.Int64)
        .alias("earliest_hindsight_label_correct_count"),
        pl.col("classification_loss_contribution")
        .sum()
        .alias("earliest_hindsight_label_cross_entropy_loss_sum"),
        pl.col("_smooth_l1")
        .cast(float64)
        .sum()
        .alias("hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss_sum"),
        pl.col("_log_error")
        .abs()
        .sum()
        .alias("hindsight_minimum_base_fee_per_gas_within_k_natural_log_absolute_error_sum"),
        (pl.col("_log_error") * pl.col("_log_error"))
        .sum()
        .alias("hindsight_minimum_base_fee_per_gas_within_k_natural_log_squared_error_sum"),
        immediate.cast(float64).sum().alias("immediate_k0_base_fee_per_gas_sum"),
        pl.col("_savings").cast(float64).sum().alias("finite_target_base_fee_per_gas_savings_sum"),
        pl.col("_opportunity")
        .cast(float64)
        .sum()
        .alias("finite_target_base_fee_per_gas_hindsight_opportunity_sum"),
        pl.col("_regret")
        .cast(float64)
        .sum()
        .alias("finite_target_base_fee_per_gas_hindsight_regret_sum"),
        pl.col("_opportunity").sum().alias("_exact_opportunity_sum"),
        (pl.col("_savings").cast(float64) / immediate.cast(float64))
        .sum()
        .alias("target_base_fee_per_gas_savings_fraction_vs_immediate_k0_sum"),
        pl.len()
        .cast(pl.Int64)
        .alias("target_base_fee_per_gas_savings_fraction_vs_immediate_k0_defined_origin_count"),
        pl.lit(0, dtype=pl.Int64).alias(
            "target_base_fee_per_gas_savings_fraction_vs_immediate_k0_zero_denominator_exclusion_count"
        ),
        (pl.col("_regret").cast(float64) / hindsight.cast(float64))
        .sum()
        .alias("selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum"),
        pl.len()
        .cast(pl.Int64)
        .alias(
            "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_defined_origin_count"
        ),
        pl.lit(0, dtype=pl.Int64).alias(
            "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_zero_denominator_exclusion_count"
        ),
        (pl.col("_opportunity").cast(float64) / hindsight.cast(float64))
        .sum()
        .alias("immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum"),
        pl.len()
        .cast(pl.Int64)
        .alias(
            "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_defined_origin_count"
        ),
        pl.lit(0, dtype=pl.Int64).alias(
            "immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_zero_denominator_exclusion_count"
        ),
        pl.col("_harmful").sum().cast(pl.Int64).alias("harmful_action_count"),
        pl.concat_list(prediction_by_k).alias("selected_action_count_by_k"),
        pl.col("selected_action_k")
        .cast(float64)
        .sum()
        .alias("extra_wait_block_opportunities_vs_immediate_k0_sum"),
        pl.col("selected_action_wait_seconds")
        .cast(float64)
        .sum()
        .alias("selected_action_wait_seconds_sum"),
        pl.col("full_horizon_elapsed_seconds")
        .cast(float64)
        .sum()
        .alias("full_horizon_elapsed_seconds_sum"),
        (pl.sum_horizontal(f1_by_k) / pl.sum_horizontal(active_by_k)).alias(
            "earliest_hindsight_label_macro_f1"
        ),
        valid_inputs.alias("_valid_inputs"),
    )

    n = pl.col("eligible_origin_count").cast(float64)
    savings_sum = pl.col("finite_target_base_fee_per_gas_savings_sum")
    opportunity_sum = pl.col("finite_target_base_fee_per_gas_hindsight_opportunity_sum")
    regret_sum = pl.col("finite_target_base_fee_per_gas_hindsight_regret_sum")
    immediate_sum = pl.col("immediate_k0_base_fee_per_gas_sum")
    result_plan = aggregated.with_columns(
        (pl.col("earliest_hindsight_label_cross_entropy_loss_sum") / n).alias(
            "earliest_hindsight_label_cross_entropy_loss"
        ),
        (pl.col("hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss_sum") / n).alias(
            "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss"
        ),
        (
            pl.col("hindsight_minimum_base_fee_per_gas_within_k_natural_log_absolute_error_sum") / n
        ).alias("hindsight_minimum_base_fee_per_gas_within_k_natural_log_mae"),
        (
            pl.col("hindsight_minimum_base_fee_per_gas_within_k_natural_log_squared_error_sum") / n
        ).alias("hindsight_minimum_base_fee_per_gas_within_k_natural_log_mse"),
        (
            (
                pl.col("earliest_hindsight_label_cross_entropy_loss_sum")
                + pl.col("hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss_sum")
            )
            / n
        ).alias("multitask_total_loss"),
        (pl.col("earliest_hindsight_label_correct_count").cast(float64) / n).alias(
            "earliest_hindsight_label_accuracy"
        ),
        (savings_sum / immediate_sum).alias(
            "finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0"
        ),
        (opportunity_sum / immediate_sum).alias(
            "finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0"
        ),
        (regret_sum / immediate_sum).alias(
            "finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0"
        ),
        pl.when(pl.col("_exact_opportunity_sum") == 0)
        .then(pl.lit(None, dtype=float64))
        .otherwise(savings_sum / opportunity_sum)
        .alias("signed_captured_hindsight_opportunity_ratio"),
        (pl.col("target_base_fee_per_gas_savings_fraction_vs_immediate_k0_sum") / n).alias(
            "mean_origin_target_base_fee_per_gas_savings_fraction_vs_immediate_k0"
        ),
        (
            pl.col(
                "selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum"
            )
            / n
        ).alias(
            "mean_origin_selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k"
        ),
        (
            pl.col("immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum")
            / n
        ).alias(
            "mean_origin_immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k"
        ),
        (pl.col("harmful_action_count").cast(float64) / n).alias("harmful_action_rate"),
        (pl.col("extra_wait_block_opportunities_vs_immediate_k0_sum") / n).alias(
            "mean_extra_wait_block_opportunities_vs_immediate_k0"
        ),
        (pl.col("selected_action_wait_seconds_sum") / n).alias("mean_selected_action_wait_seconds"),
        (pl.col("full_horizon_elapsed_seconds_sum") / n).alias("mean_full_horizon_elapsed_seconds"),
    ).with_columns(
        pl.all_horizontal(
            [pl.col(name).is_finite().fill_null(True) for name in _FLOAT_RESULT_COLUMNS]
        ).alias("_finite_results"),
        pl.all_horizontal(
            [
                pl.col(name).is_not_null()
                for name in _RESULT_COLUMNS
                if name != "signed_captured_hindsight_opportunity_ratio"
            ]
        ).alias("_nonnull_results"),
    )

    result = result_plan.select(
        *_RESULT_COLUMNS,
        "_valid_inputs",
        "_finite_results",
        "_nonnull_results",
    ).collect()
    if not all(
        result[name].item() is True
        for name in ("_valid_inputs", "_finite_results", "_nonnull_results")
    ):
        raise ValueError("evaluation observations contain invalid scientific facts")
    result.drop_in_place("_valid_inputs")
    result.drop_in_place("_finite_results")
    result.drop_in_place("_nonnull_results")
    return result


__all__ = ["reduce_evaluation"]
