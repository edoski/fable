"""Fresh request constructors."""

from __future__ import annotations

from hashlib import sha256
from uuid import UUID, uuid4

from ..config import (
    CorpusDefinition,
    CorpusRequest,
    EvaluateRequest,
    OriginWindow,
    StudyDefinition,
    TrainingSource,
    TrainRequest,
    TuneRequest,
)

_DIGEST_LENGTH = 20


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    return f"{prefix}_{digest}"


def fresh_corpus_request(definition: CorpusDefinition) -> CorpusRequest:
    return CorpusRequest(corpus_id=uuid4(), definition=definition)


def fresh_train_request(source: TrainingSource) -> TrainRequest:
    return TrainRequest(workflow="train", artifact_id=uuid4(), source=source)


def fresh_tune_request(corpus_id: UUID, study_definition: StudyDefinition) -> TuneRequest:
    return TuneRequest(
        workflow="tune",
        study_id=uuid4(),
        corpus_id=corpus_id,
        study_definition=study_definition,
    )


def fresh_evaluate_request(
    artifact_id: UUID,
    corpus_id: UUID,
    window: OriginWindow,
) -> EvaluateRequest:
    return EvaluateRequest(
        workflow="evaluate",
        evaluation_id=uuid4(),
        artifact_id=artifact_id,
        corpus_id=corpus_id,
        window=window,
    )
