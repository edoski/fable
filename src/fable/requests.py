"""Fresh request constructors."""

from __future__ import annotations

from uuid import UUID, uuid4

from .config import (
    BlockWindow,
    EvaluateRequest,
    ExperimentSemantics,
    Method,
    SelectedStudySource,
    TrainRequest,
    TuneRequest,
)


def fresh_train_request(source: SelectedStudySource) -> TrainRequest:
    return TrainRequest(workflow="train", artifact_id=uuid4(), source=source)


def fresh_tune_request(
    corpus_id: UUID,
    experiment: ExperimentSemantics,
    methods: tuple[Method, ...],
) -> TuneRequest:
    return TuneRequest(
        workflow="tune",
        study_id=uuid4(),
        corpus_id=corpus_id,
        experiment=experiment,
        methods=methods,
    )


def fresh_evaluate_request(
    artifact_id: UUID,
    corpus_id: UUID,
    testing_window: BlockWindow,
) -> EvaluateRequest:
    return EvaluateRequest(
        workflow="evaluate",
        evaluation_id=uuid4(),
        artifact_id=artifact_id,
        corpus_id=corpus_id,
        testing_window=testing_window,
    )
