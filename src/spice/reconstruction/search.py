"""Candidate search for reference label, split, and feature assumptions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .features import (
    feature_warmup_rows,
    generate_feature_candidates,
    materialize_feature_candidate,
)
from .labels import generate_label_candidates, materialize_candidate_labels
from .models import FeatureCandidateSummary, LabelSearchResult
from .reference import (
    expected_unique_class_count,
    load_reference_predictions,
    load_reference_raw_blocks,
)
from .splits import apply_split_candidate, generate_split_candidates

DEFAULT_CHAINS = ("ethereum", "polygon", "avalanche")
DEFAULT_DELAYS = (12, 24, 36)


@dataclass(frozen=True, slots=True)
class ReconstructionSearchResult:
    label_results: dict[str, list[LabelSearchResult]]
    feature_results: dict[str, list[FeatureCandidateSummary]]


def run_reference_search(
    *,
    reference_root: Path,
    chains: Iterable[str] = DEFAULT_CHAINS,
    delays: Iterable[int] = DEFAULT_DELAYS,
) -> ReconstructionSearchResult:
    reference_metrics = load_reference_predictions(reference_root)
    label_results: dict[str, list[LabelSearchResult]] = {}
    feature_results: dict[str, list[FeatureCandidateSummary]] = {}
    for chain in chains:
        blocks = load_reference_raw_blocks(reference_root, chain)
        for delay_seconds in delays:
            key = _result_key(chain=chain, delay_seconds=delay_seconds)
            label_results[key] = run_label_search(
                blocks=blocks,
                chain=chain,
                delay_seconds=delay_seconds,
                reference_metrics=reference_metrics,
            )
            feature_results[key] = run_feature_search(
                blocks=blocks,
                chain=chain,
                delay_seconds=delay_seconds,
            )
    return ReconstructionSearchResult(
        label_results=label_results,
        feature_results=feature_results,
    )


def run_label_search(
    *,
    blocks,
    chain: str,
    delay_seconds: int,
    reference_metrics,
) -> list[LabelSearchResult]:
    expected_classes = expected_unique_class_count(
        reference_metrics,
        chain=chain,
        delay_seconds=delay_seconds,
    )
    results: list[LabelSearchResult] = []
    for label_candidate in generate_label_candidates():
        materialized = materialize_candidate_labels(
            blocks,
            delay_seconds=delay_seconds,
            candidate=label_candidate,
        )
        for split_candidate in generate_split_candidates():
            try:
                split = apply_split_candidate(
                    materialized.table,
                    candidate=split_candidate,
                    warmup_rows=feature_warmup_rows(),
                )
            except ValueError:
                continue
            train_unique = int(split.train["minBlock"].n_unique()) if split.train.height else 0
            validation_unique = (
                int(split.validation["minBlock"].n_unique()) if split.validation.height else 0
            )
            test_unique = int(split.test["minBlock"].n_unique()) if split.test.height else 0
            invalid_ratio = 0.0
            if materialized.rows_total > 0:
                invalid_ratio = materialized.rows_dropped / materialized.rows_total
            score = _label_result_score(
                expected_unique_classes=expected_classes,
                train_unique_classes=train_unique,
                validation_unique_classes=validation_unique,
                test_unique_classes=test_unique,
                invalid_ratio=invalid_ratio,
                uses_clip_tail=label_candidate.tail_policy == "clip",
            )
            results.append(
                LabelSearchResult(
                    chain=chain,
                    delay_seconds=delay_seconds,
                    label_candidate=label_candidate,
                    split_candidate=split_candidate,
                    expected_unique_classes=expected_classes,
                    train_unique_classes=train_unique,
                    validation_unique_classes=validation_unique,
                    test_unique_classes=test_unique,
                    rows_total=materialized.rows_total,
                    rows_kept=materialized.rows_kept,
                    rows_dropped=materialized.rows_dropped,
                    invalid_ratio=invalid_ratio,
                    score=score,
                )
            )
    return sorted(results, key=lambda item: item.score)


def run_feature_search(
    *,
    blocks,
    chain: str,
    delay_seconds: int,
) -> list[FeatureCandidateSummary]:
    results: list[FeatureCandidateSummary] = []
    for candidate in generate_feature_candidates():
        _, summary = materialize_feature_candidate(
            blocks,
            chain=chain,
            delay_seconds=delay_seconds,
            candidate=candidate,
        )
        results.append(summary)
    return sorted(results, key=lambda item: item.score)


def best_candidates(
    result: ReconstructionSearchResult,
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    all_keys = sorted(set(result.label_results) | set(result.feature_results))
    for key in all_keys:
        best_label = result.label_results.get(key, [None])[0]
        best_feature = result.feature_results.get(key, [None])[0]
        summary[key] = {
            "best_label": None if best_label is None else best_label.payload(),
            "best_feature": None if best_feature is None else best_feature.payload(),
        }
    return summary


def _label_result_score(
    *,
    expected_unique_classes: int | None,
    train_unique_classes: int,
    validation_unique_classes: int,
    test_unique_classes: int,
    invalid_ratio: float,
    uses_clip_tail: bool,
) -> float:
    score = invalid_ratio * 5.0
    if expected_unique_classes is not None:
        score += (
            abs(train_unique_classes - expected_unique_classes) / max(expected_unique_classes, 1)
        ) * 10.0
    score += (
        abs(validation_unique_classes - train_unique_classes) / max(train_unique_classes, 1)
    ) * 2.0
    score += (
        abs(test_unique_classes - train_unique_classes) / max(train_unique_classes, 1)
    ) * 2.0
    if uses_clip_tail:
        score += 0.25
    return score


def _result_key(*, chain: str, delay_seconds: int) -> str:
    return f"{chain}_{delay_seconds}s"
