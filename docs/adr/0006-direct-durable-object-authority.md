# ADR 0006: Direct Durable Object Authority

## Status

Accepted.

## Context

FABLE (Fee Analysis through Blockchain Learning and Estimation) objects must preserve enough authority to interpret corpora, studies, artifacts, and evaluations directly. Their requests and associations contain that authority at the canonical object address.

## Decision

UUIDv4 values identify instances. Each completed object owns its exact typed request once at a direct canonical address:

- `corpora/<corpus_id>/corpus.json` and `blocks.parquet`;
- `studies/<study_id>.json`;
- `artifacts/<artifact_id>.ckpt`;
- `evaluations/<evaluation_id>/evaluation.json` and `observations.parquet`.

Typed requests, embedded associations, and the selected Study result index plus exact Method establish meaning. Completed objects are loaded directly and validated against the requested UUID and association.

A completed evaluation owns its exact `EvaluateRequest` plus sufficient canonical prediction and outcome observations. Loading it validates the strict request identity, exact schema, and ordered window coverage. Atomic publication owns observation value consistency, so transient reduction trusts those values and is recomputed directly from the completed evaluation object; Artifact and Corpus availability is not required after publication. Selection remains recomputed from its canonical Study object.

Artifact fitting and Study assembly use owner-local hidden scratch. The selected checkpoint or
assembled Study remains inside that scratch while `os.link()` creates the canonical path without
overwrite. Scratch is removed only after the canonical link exists. Failed cleanup can therefore
leave scratch beside a valid canonical object; there is no separate cleanup recovery path. Corpus
and evaluation directories retain owner-local scratch and direct rename. The mobile exporter rejects
an existing output before lowering, builds a hidden sibling directory, checks again immediately
before rename, and removes scratch after failure.

## Consequences

Callers supply the typed UUID they intend to use. Durable schemas stay focused, and each transient operation depends only on the completed object that owns its required authority.
