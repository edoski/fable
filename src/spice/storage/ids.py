"""Deterministic storage identifiers for typed canonical identities."""

from __future__ import annotations

import json
from hashlib import sha256

from .identity import ArtifactStorageIdentity, IdentityModel, StudyStorageIdentity, identity_payload

_DIGEST_LENGTH = 20


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    return f"{prefix}_{digest}"


def _canonical_identity(identity: IdentityModel) -> str:
    return json.dumps(
        identity_payload(identity),
        sort_keys=True,
        separators=(",", ":"),
    )


def corpus_storage_id(
    *,
    chain_name: str,
    dataset_name: str,
    evaluation_date: object,
) -> str:
    return _stable_id("cor", chain_name, dataset_name, str(evaluation_date))


def study_storage_id(*, identity: StudyStorageIdentity) -> str:
    return _stable_id("std", _canonical_identity(identity))


def artifact_storage_id(*, identity: ArtifactStorageIdentity) -> str:
    return _stable_id("art", _canonical_identity(identity))
