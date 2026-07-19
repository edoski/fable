"""Primary K=5 fee-condition evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl

from fable.addresses import evaluation_json_path, evaluation_observations_path
from fable.config import EvaluateRequest, SelectedStudySource
from fable.corpus import load_corpus
from fable.evaluation.reduction import reduce_evaluation
from fable.modeling.artifacts import load_artifact

_RAW_DESCRIPTOR = "closed_parent_base_fee_per_gas"
_LOG_DESCRIPTOR = "signed_one_block_base_fee_log_change"
_OBSERVATION_COLUMNS = (
    "previous_closed_parent_base_fee_per_gas",
    _RAW_DESCRIPTOR,
    "selected_action_k",
    "earliest_hindsight_action_k",
    "immediate_k0_base_fee_per_gas",
    "selected_target_base_fee_per_gas",
    "hindsight_minimum_base_fee_per_gas",
)
_SCHEMA = pl.Schema(
    {
        "evaluation_id": pl.String,
        "artifact_id": pl.String,
        "corpus_id": pl.String,
        "chain_id": pl.Int64,
        "first_parent_block": pl.Int64,
        "last_parent_block": pl.Int64,
        "horizon_blocks": pl.Int64,
        "descriptor": pl.String,
        "quartile": pl.Int64,
        "closed_parent_base_fee_per_gas_cutpoint_25": pl.Int64,
        "closed_parent_base_fee_per_gas_cutpoint_50": pl.Int64,
        "closed_parent_base_fee_per_gas_cutpoint_75": pl.Int64,
        "signed_one_block_base_fee_log_change_cutpoint_25": pl.Float64,
        "signed_one_block_base_fee_log_change_cutpoint_50": pl.Float64,
        "signed_one_block_base_fee_log_change_cutpoint_75": pl.Float64,
        "closed_parent_base_fee_per_gas_cell_median": pl.Float64,
        "signed_one_block_base_fee_log_change_cell_median": pl.Float64,
        "condition_origin_count": pl.Int64,
        "earliest_hindsight_label_correct_count": pl.Int64,
        "immediate_k0_base_fee_per_gas_sum": pl.Float64,
        "finite_target_base_fee_per_gas_savings_sum": pl.Float64,
        "finite_target_base_fee_per_gas_hindsight_opportunity_sum": pl.Float64,
        "finite_target_base_fee_per_gas_hindsight_regret_sum": pl.Float64,
        "finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0": pl.Float64,
        "finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0": pl.Float64,
        "finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0": pl.Float64,
        "earliest_hindsight_label_accuracy": pl.Float64,
    }
)


def _evaluation_rows(
    storage_root: Path,
    evaluation_id: UUID,
    expected_chain_id: int,
) -> list[dict[str, Any]]:
    request = EvaluateRequest.model_validate_json(
        evaluation_json_path(storage_root, evaluation_id).read_bytes(),
        strict=True,
    )
    if request.evaluation_id != evaluation_id:
        raise ValueError("evaluation request ID must match the requested evaluation")
    if request.window.role != "testing":
        raise ValueError("fee-condition evidence requires a testing evaluation")

    reduced = reduce_evaluation(storage_root, evaluation_id).row(0, named=True)
    association, _ = load_artifact(storage_root, request.artifact_id)
    source = association.request.source
    if not isinstance(source, SelectedStudySource):
        raise ValueError("fee-condition evidence requires a SelectedStudySource")
    experiment = source.experiment
    if experiment.context_blocks != 200:
        raise ValueError("fee-condition evidence requires C200")
    if experiment.horizon_blocks != 5:
        raise ValueError("fee-condition evidence requires K=5")

    corpus = load_corpus(storage_root, request.corpus_id)
    chain_id = corpus.request.definition.chain_id
    if chain_id != expected_chain_id:
        raise ValueError("evaluation chain does not match the required caller order")

    observations = (
        pl.scan_parquet(evaluation_observations_path(storage_root, evaluation_id))
        .select(_OBSERVATION_COLUMNS)
        .with_columns(
            (
                pl.col(_RAW_DESCRIPTOR).cast(pl.Float64)
                / pl.col("previous_closed_parent_base_fee_per_gas").cast(pl.Float64)
            )
            .log()
            .alias(_LOG_DESCRIPTOR),
            pl.col("immediate_k0_base_fee_per_gas").alias("_immediate"),
            (
                pl.col("immediate_k0_base_fee_per_gas") - pl.col("selected_target_base_fee_per_gas")
            ).alias("_savings"),
            (
                pl.col("immediate_k0_base_fee_per_gas")
                - pl.col("hindsight_minimum_base_fee_per_gas")
            ).alias("_opportunity"),
            (
                pl.col("selected_target_base_fee_per_gas")
                - pl.col("hindsight_minimum_base_fee_per_gas")
            ).alias("_regret"),
            (pl.col("selected_action_k") == pl.col("earliest_hindsight_action_k")).alias(
                "_correct"
            ),
        )
        .collect()
    )
    invalid_fees = observations.select(
        (
            (pl.col("previous_closed_parent_base_fee_per_gas") <= 0)
            | (pl.col(_RAW_DESCRIPTOR) <= 0)
        ).any()
    ).item()
    if invalid_fees:
        raise ValueError("fee descriptors require positive adjacent base fees")
    if not observations[_LOG_DESCRIPTOR].is_finite().all():
        raise ValueError("signed fee-log changes must be finite")

    count = reduced["eligible_origin_count"]
    absolute_sums = observations.select(
        pl.col("_immediate").cast(pl.Float64).abs().sum().alias("B"),
        pl.col("_savings").cast(pl.Float64).abs().sum().alias("S"),
        pl.col("_opportunity").cast(pl.Float64).abs().sum().alias("G"),
        pl.col("_regret").cast(pl.Float64).abs().sum().alias("Q"),
    ).row(0, named=True)

    rows: list[dict[str, Any]] = []
    for descriptor in (_RAW_DESCRIPTOR, _LOG_DESCRIPTOR):
        ordered = observations[descriptor].sort()
        cutpoints = (
            ordered[(count + 3) // 4 - 1],
            ordered[(count + 1) // 2 - 1],
            ordered[(3 * count + 3) // 4 - 1],
        )
        values = observations[descriptor]
        q25, q50, q75 = cutpoints
        cells = (
            values <= q25,
            (values > q25) & (values <= q50),
            (values > q50) & (values <= q75),
            values > q75,
        )
        cell_sums: list[dict[str, Any]] = []
        cell_count = 0
        correct_count = 0
        for quartile, mask in enumerate(cells, start=1):
            cell = observations.filter(mask)
            origin_count = cell.height
            cell_correct = cell.select(pl.col("_correct").sum()).row(0)[0]
            median = (
                cell.select(pl.col(descriptor).cast(pl.Float64).median()).row(0)[0]
                if origin_count
                else None
            )
            sums = cell.select(
                pl.col("_immediate").cast(pl.Float64).sum().alias("B"),
                pl.col("_savings").cast(pl.Float64).sum().alias("S"),
                pl.col("_opportunity").cast(pl.Float64).sum().alias("G"),
                pl.col("_regret").cast(pl.Float64).sum().alias("Q"),
            ).row(0, named=True)
            cell_sums.append(sums)
            cell_count += origin_count
            correct_count += cell_correct
            immediate = sums["B"]
            savings = sums["S"]
            opportunity = sums["G"]
            regret = sums["Q"]
            rows.append(
                {
                    "evaluation_id": str(evaluation_id),
                    "artifact_id": str(request.artifact_id),
                    "corpus_id": str(request.corpus_id),
                    "chain_id": chain_id,
                    "first_parent_block": request.window.first_parent_block,
                    "last_parent_block": request.window.last_parent_block,
                    "horizon_blocks": experiment.horizon_blocks,
                    "descriptor": descriptor,
                    "quartile": quartile,
                    "closed_parent_base_fee_per_gas_cutpoint_25": (
                        cutpoints[0] if descriptor == _RAW_DESCRIPTOR else None
                    ),
                    "closed_parent_base_fee_per_gas_cutpoint_50": (
                        cutpoints[1] if descriptor == _RAW_DESCRIPTOR else None
                    ),
                    "closed_parent_base_fee_per_gas_cutpoint_75": (
                        cutpoints[2] if descriptor == _RAW_DESCRIPTOR else None
                    ),
                    "signed_one_block_base_fee_log_change_cutpoint_25": (
                        cutpoints[0] if descriptor == _LOG_DESCRIPTOR else None
                    ),
                    "signed_one_block_base_fee_log_change_cutpoint_50": (
                        cutpoints[1] if descriptor == _LOG_DESCRIPTOR else None
                    ),
                    "signed_one_block_base_fee_log_change_cutpoint_75": (
                        cutpoints[2] if descriptor == _LOG_DESCRIPTOR else None
                    ),
                    "closed_parent_base_fee_per_gas_cell_median": (
                        median if descriptor == _RAW_DESCRIPTOR else None
                    ),
                    "signed_one_block_base_fee_log_change_cell_median": (
                        median if descriptor == _LOG_DESCRIPTOR else None
                    ),
                    "condition_origin_count": origin_count,
                    "earliest_hindsight_label_correct_count": cell_correct,
                    "immediate_k0_base_fee_per_gas_sum": immediate,
                    "finite_target_base_fee_per_gas_savings_sum": savings,
                    "finite_target_base_fee_per_gas_hindsight_opportunity_sum": opportunity,
                    "finite_target_base_fee_per_gas_hindsight_regret_sum": regret,
                    "finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0": (
                        savings / immediate if origin_count else None
                    ),
                    "finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0": (
                        opportunity / immediate if origin_count else None
                    ),
                    "finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0": (
                        regret / immediate if origin_count else None
                    ),
                    "earliest_hindsight_label_accuracy": (
                        cell_correct / origin_count if origin_count else None
                    ),
                }
            )

        if cell_count != count:
            raise ValueError("condition cell counts must recombine to the S14 origin count")
        if correct_count != reduced["earliest_hindsight_label_correct_count"]:
            raise ValueError("condition correct counts must recombine to the S14 correct count")
        unit_roundoff = 2**-53
        gamma = ((count + 3) * unit_roundoff) / (1 - (count + 3) * unit_roundoff)
        for name, full_name in (
            (
                "B",
                "immediate_k0_base_fee_per_gas_sum",
            ),
            (
                "S",
                "finite_target_base_fee_per_gas_savings_sum",
            ),
            (
                "G",
                "finite_target_base_fee_per_gas_hindsight_opportunity_sum",
            ),
            (
                "Q",
                "finite_target_base_fee_per_gas_hindsight_regret_sum",
            ),
        ):
            q1, q2, q3, q4 = (sums[name] for sums in cell_sums)
            combined = ((q1 + q2) + q3) + q4
            full = reduced[full_name]
            absolute = absolute_sums[name]
            if absolute == 0.0:
                valid = combined == 0.0 and full == 0.0
            else:
                valid = abs(combined - full) <= 3 * gamma * absolute
            if not valid:
                raise ValueError(f"{descriptor} {full_name} does not recombine to S14")
    return rows


def write_k5_fee_condition_evidence(
    storage_root: Path,
    evaluation_ids: tuple[UUID, UUID, UUID],
    destination: Path,
) -> None:
    """Write the fixed three-chain primary K=5 fee-condition table."""

    if len(evaluation_ids) != 3:
        raise ValueError("fee-condition evidence requires exactly three evaluation IDs")
    if len(set(evaluation_ids)) != 3:
        raise ValueError("fee-condition evaluation IDs must be distinct")

    rows = [
        row
        for index, chain_id in enumerate((1, 137, 43_114))
        for row in _evaluation_rows(storage_root, evaluation_ids[index], chain_id)
    ]
    evidence = pl.from_dicts(rows, schema=_SCHEMA)
    for name, dtype in _SCHEMA.items():
        if dtype == pl.Float64 and not evidence[name].drop_nulls().is_finite().all():
            raise ValueError("derived Float64 evidence values must be finite")

    hidden = destination.with_name(f".{destination.name}")
    if destination.exists():
        raise ValueError("fee-condition destination must be absent")
    if hidden.exists():
        raise ValueError("fee-condition hidden sibling must be absent")
    evidence.write_csv(hidden, separator="\t", null_value="")
    hidden.rename(destination)


__all__ = ["write_k5_fee_condition_evidence"]
