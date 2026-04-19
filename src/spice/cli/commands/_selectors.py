"""Shared storage selector builders for CLI commands."""

from __future__ import annotations

from ...storage.roots import ArtifactSelector, StudySelector


def study_selector(
    *,
    chain: str | None,
    dataset: str | None,
    feature_set: str | None,
    prediction: str | None,
    model: str | None,
    problem: str | None,
    study: str | None,
) -> StudySelector:
    return StudySelector(
        chain_name=chain,
        dataset_name=dataset,
        feature_set_id=feature_set,
        prediction_id=prediction,
        model_id=model,
        problem_id=problem,
        study_name=study,
    )


def artifact_selector(
    *,
    chain: str | None,
    dataset: str | None,
    feature_set: str | None,
    prediction: str | None,
    model: str | None,
    problem: str | None,
    variant: str | None,
    study: str | None,
) -> ArtifactSelector:
    return ArtifactSelector(
        chain_name=chain,
        dataset_name=dataset,
        feature_set_id=feature_set,
        prediction_id=prediction,
        model_id=model,
        problem_id=problem,
        variant=variant,
        study_name=study,
    )
