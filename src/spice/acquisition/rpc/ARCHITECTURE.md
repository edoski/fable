# RPC Acquisition Architecture

## Purpose

`acquisition.rpc` implements the JSON-RPC block-source Adapter. It resolves timestamp windows, fetches block rows, maps provider failures to generic acquisition errors, and canonicalizes RPC blocks.

## Theory

RPC providers are latency-bound and failure-prone. The architecture separates transport from block semantics so provider behavior can change without changing the canonical corpus format or Corpus Assembly policy.

## Invariants

RPC payload handling stays inside the Adapter. Generic acquisition types and pull scheduling live outside `acquisition.rpc`. Downstream packages should not know which RPC provider produced a row.

## Extension Points

Tune batching and concurrency in acquisition config through `acquisition.pull`. Add provider-specific transport behavior and requirement-to-RPC mapping behind the transport/client seam, not in corpus or workflow code.

## Module Map

```text
rpc/
  transport.py   request transport and retry mechanics
  client.py      typed block-fetch client
```

## RPC Flow

```text
block numbers
    |
    v
batched JSON-RPC calls
    |
    +-- optional eth_feeHistory batches for priority_fee_percentiles
    |
    v
provider responses
    |
    v
RpcBlock objects
    |
    v
CanonicalBlockRow
```

The boundary is canonicalization. Once a row is canonical, downstream code should not care whether it came from PublicNode, another RPC endpoint, or a future Adapter.

The RPC client maps corpus source requirements onto provider calls during adapter construction. Core block columns come from `eth_getBlockByNumber`. When source requirements include the `priority_fee_percentiles` enrichment, the client also requests `eth_feeHistory` and fills priority-fee percentile columns in canonical rows. Unknown enrichments fail at construction. Required fee-history payloads that are unsupported or malformed fail acquisition instead of publishing null priority-fee facts.

## Failure Policy

Transport failures belong in transport/client code and are mapped to generic acquisition errors when pull scheduling can retry them. Data-shape failures belong in canonicalization. Required source facts are materiality checks: if the provider cannot return the required RPC payload, the adapter raises an unsupported-source error. Workflow-level failures should mention acquisition context, not low-level HTTP details unless needed for debugging.
