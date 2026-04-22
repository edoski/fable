"""End-to-end audit for the current temporal parity path."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl

from ..config.models import TrainConfig, WorkflowTask
from ..config.resolution import WorkflowRequest, resolve_workflow_config
from ..evaluation import compile_evaluator_contract
from ..features import compile_feature_contract
from ..prediction import compile_prediction_contract
from .models import AuditFinding, CurrentParityAudit
from .reference import load_reference_predictions, summarize_reference_raw_datasets


def run_current_parity_audit(
    *,
    preset: str,
    reference_root: Path,
    storage_root: Path,
) -> CurrentParityAudit:
    config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(
                preset=preset,
                storage_root=storage_root,
            ),
        ),
    )
    feature_contract = compile_feature_contract(feature_set=config.feature_set)
    prediction_contract = compile_prediction_contract(
        prediction_id=config.prediction.id,
        family_id=config.prediction.family_id,
    )
    evaluator_contract = (
        None
        if config.evaluation is None
        else compile_evaluator_contract(config.evaluation)
    )
    reference_raw = summarize_reference_raw_datasets(reference_root)
    reference_metrics = load_reference_predictions(reference_root)
    findings = _build_findings(
        config=config,
        reference_raw=reference_raw,
        evaluator_contract=evaluator_contract,
    )
    return CurrentParityAudit(
        preset=preset,
        dataset=config.dataset.model_dump(mode="json"),
        chain=config.chain.model_dump(mode="json"),
        problem=config.problem.model_dump(mode="json"),
        feature_set={
            **config.feature_set.model_dump(mode="json"),
            "feature_names": list(feature_contract.feature_names),
            "feature_prerequisites": feature_contract.feature_prerequisites.model_dump(
                mode="json"
            ),
        },
        dataset_builder=config.dataset_builder.model_dump(mode="json"),
        prediction=config.prediction.model_dump(mode="json"),
        model=config.model.model_dump(mode="json"),
        evaluation=None if config.evaluation is None else config.evaluation.model_dump(mode="json"),
        compiler_runtime=config.problem.compiler.model_dump(mode="json"),
        realization_policy=config.problem.realization_policy.model_dump(mode="json"),
        compiled_prediction={
            "prediction_family_id": prediction_contract.prediction_family_id,
            "primary_metric_id": prediction_contract.primary_metric_id,
            "direction": prediction_contract.direction,
            "training_metric_ids": [
                descriptor.id for descriptor in prediction_contract.training_metric_descriptors
            ],
        },
        compiled_evaluator=(
            None
            if evaluator_contract is None
            else {
                "evaluation_id": evaluator_contract.evaluation_id,
                "primary_metric_id": evaluator_contract.primary_metric_id,
                "direction": evaluator_contract.direction,
                "metric_ids": [
                    descriptor.id for descriptor in evaluator_contract.metric_descriptors
                ],
                "config_payload": evaluator_contract.config_payload,
            }
        ),
        local_corpora=_summarize_local_corpora(storage_root / "corpora"),
        reference_raw=[dataset.payload() for dataset in reference_raw],
        reference_metrics=[row.payload() for row in reference_metrics],
        findings=[finding.payload() for finding in findings],
    )


def _build_findings(
    *,
    config: TrainConfig,
    reference_raw,
    evaluator_contract,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    short_raw = [
        dataset for dataset in reference_raw if dataset.row_count < config.problem.sample_count
    ]
    if short_raw:
        summary = ", ".join(f"{item.chain}={item.row_count}" for item in short_raw)
        findings.append(
            AuditFinding(
                area="corpus",
                status="gap",
                detail=(
                    "Shared raw reference CSVs are shorter than the current training sample count "
                    f"{config.problem.sample_count}; exact upstream multi-day split construction "
                    f"remains missing ({summary})."
                ),
            )
        )

    if config.problem.compiler.id == "estimated_block":
        findings.append(
            AuditFinding(
                area="compiler",
                status="mismatch",
                detail=(
                    "The active problem still uses the estimated_block compiler, while the "
                    "reference labels appear to be built from timestamp-native future windows over "
                    "actual rows."
                ),
            )
        )

    if config.dataset_builder.id == "professor_temporal":
        findings.append(
            AuditFinding(
                area="dataset_builder",
                status="gap",
                detail=(
                    "professor_temporal currently tail-selects one corpus and applies fixed "
                    "chronological fractions, but the reference repo expects missing prebuilt "
                    "train/validation/test CSVs."
                ),
            )
        )

    if config.prediction.family_id == "min_block_fee_multitask":
        findings.append(
            AuditFinding(
                area="prediction",
                status="mismatch",
                detail=(
                    "The multitask loss is currently weighted 0.5 classification / 0.5 regression, "
                    "while the reference classification path uses 1.0 classification / 0.5 "
                    "regression and log1p(minBaseFee) regression targets."
                ),
            )
        )

    if {"time_since_start", "base_fee_trend"} & set(config.feature_set.outputs):
        findings.append(
            AuditFinding(
                area="features",
                status="inferred",
                detail=(
                    "The feature set already includes the ambiguous reference columns "
                    "time_since_start and base_fee_trend, but their exact formulas still "
                    "need to be recovered from notebook evidence."
                ),
            )
        )

    if evaluator_contract is not None and evaluator_contract.evaluation_id == "paper_replay_2h":
        findings.append(
            AuditFinding(
                area="evaluation",
                status="aligned",
                detail=(
                    "The current evaluator is already using paper_replay_2h with 2h Poisson replay "
                    "and the standard economic metrics."
                ),
            )
        )
        findings.append(
            AuditFinding(
                area="evaluation",
                status="inferred",
                detail=(
                    "The evaluator is structurally aligned, but it has not yet been numerically "
                    "cross-checked against the shared distributed evaluation repository."
                ),
            )
        )

    return findings


def _summarize_local_corpora(corpora_root: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    if not corpora_root.is_dir():
        return summaries
    for chain_dir in sorted(path for path in corpora_root.iterdir() if path.is_dir()):
        for corpus_dir in sorted(
            path for path in chain_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        ):
            history_rows = _count_parquet_rows(corpus_dir / "history")
            evaluation_rows = _count_parquet_rows(corpus_dir / "evaluation")
            summaries.append(
                {
                    "chain": chain_dir.name,
                    "corpus_id": corpus_dir.name,
                    "history_rows": history_rows,
                    "evaluation_rows": evaluation_rows,
                }
            )
    return summaries


def _count_parquet_rows(directory: Path) -> int:
    parquet_paths = sorted(directory.glob("*.parquet"))
    if not parquet_paths:
        return 0
    lazy = pl.scan_parquet([str(path) for path in parquet_paths])
    frame = lazy.select(pl.len().alias("count")).collect()
    return int(frame.item())
