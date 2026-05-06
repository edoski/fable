# Concrete RPC Acquisition

The RPC acquisition implementation downloads canonical block rows from JSON-RPC endpoints. It handles timestamp-to-block lookup, batch requests, provider transport, and provider-specific payload conversion.

## Mental Model

Blockchain RPC is a remote database with rate limits, transient failures, and variable latency. Acquisition must be fast, but it must not corrupt row order or silently skip blocks.

```text
timestamp window
  -> binary search block boundaries
  -> half-open block ranges
  -> batch requests
  -> canonical rows
```

## `BlockRpcClient`

The client owns chain-specific block lookup and canonical row fetching.

Main operations:

| Operation | Behavior |
| --- | --- |
| Resolve first block at/after timestamp | Binary search over block timestamps. |
| Resolve `[start, end)` range | Convert time interval into block interval. |
| Estimate recent interval | Sample recent blocks to estimate seconds per block. |
| Fetch block batch | Use Web3 batch requests, then canonicalize rows. |

`eth_feeHistory` is only requested when corpus source requirements include the feature-owned `priority_fee_percentiles` enrichment. Gas utilization is derived from canonical block fields, so no-priority datasets do not need fee-history RPC calls.

Binary search is needed because block timestamps are monotonic enough for range lookup, but the chain does not expose "block at timestamp" as a native RPC call.

## Transport

`ManagedAsyncHTTPProvider` owns the async HTTP session. It retries allowed transport failures, force-closes TCP sessions during shutdown, sorts batch responses by JSON-RPC id, and injects proof-of-authority middleware for chains that need it.

```text
web3 batch request
  -> JSON-RPC response list
  -> sort by request id
  -> validate response shape
  -> block payloads
```

## Adaptive Controller

`AcquisitionPullController` adjusts batch size and concurrency in the generic acquisition pull scheduler.

| Signal | Response |
| --- | --- |
| Oversized request | Halve batch size down to configured minimum. |
| Transient failure streak | Drop to a lower concurrency rung. |
| Sustained success | Recover one concurrency rung after success threshold. |

This keeps acquisition productive on healthy endpoints and less aggressive when an endpoint pushes back.

## Pull Scheduler

`acquisition.pull.pull_block_range` splits the block range into batch requests and runs them concurrently. It depends on the generic `BlockSource` Interface, so it is not an RPC Adapter.

Important behavior:

```text
in-flight batches may finish out of order
  -> scheduler buffers completed rows
  -> writer emits only contiguous next range
```

That rule preserves materialized order. Oversized batches are split and retried. Transient failures are retried up to the configured attempt limit. On exit, in-flight tasks are cancelled.

## Corpus Assembly Link

Corpus Assembly uses RPC acquisition through the `BlockSource` Interface:

1. Download history, then count valid temporal samples.
2. If valid samples are short, refill history with bounded larger lookback estimates.
3. Download evaluation-day rows.
4. Write dataset state and promote corpus paths atomically.

The refill exists because nominal block timing can underestimate how much real history is needed after feature warmup, problem warmup, and row filtering.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Empty pull plan | Range resolution produced no blocks. |
| Oversized at minimum batch | Endpoint cannot serve even the smallest allowed batch. |
| Retry exhaustion | Endpoint stayed unavailable for a range. |
| Row-count mismatch | RPC returned incomplete or unexpected payloads. |
| Scheduler stall | No request can make progress. |
| Unsupported payload shape | RPC block response cannot become canonical row. |

## Extension Pattern

A new RPC provider should keep the same canonical row output and controller snapshot metadata. Endpoint-specific behavior belongs in transport/client setup; Corpus Assembly should still receive ordered canonical rows through `BlockSource`.
