"""Workflow inference preflight."""

from __future__ import annotations

from ..config.models import EvaluateConfig
from ..modeling.artifact_inference import (
    ArtifactInferenceContext,
    prepare_artifact_inference_context,
)
from ..storage.workflow_roots import EvaluateWorkflowRoots


def prepare_inference_context(
    config: EvaluateConfig,
    roots: EvaluateWorkflowRoots,
) -> ArtifactInferenceContext:
    return prepare_artifact_inference_context(
        config,
        corpus=roots.corpus,
        artifact=roots.artifact,
    )
